#!/usr/bin/env python3
"""Tracker evaluation against the 2D identity GT of a label segment (docs/13).

Runs the pipeline over the segment (with --warmup seconds before it, not scored) and scores, at the GT stamps,
exactly the boxes the GUI draws: box2d for tracks updated this frame, otherwise the projected class-size box at
the 3D position (coasting / lost / radar-only = what the user sees as ghosts). Per class and overall:
MOTA, IDF1/IDP/IDR, IDsw, Frag, MT/ML, HOTA/DetA/AssA/LocA, FP split (duplicate / mislocated / ghost),
predicted ids per GT id. Ignore-flagged GT absorbs predictions (neither TP nor FP).

v2 (docs/13 "평가 지표 v2", gt_design §D.7), when the GT carries the schema-v2 keys: IDSWR / FNR / FPR per class,
MOTP[m] (2D-IoU-matched TP pairs vs box ref3d range, by range bin and reference quality), worker recall by ref3d
range bin and by occlusion level, the metrics split by meta.ego_state (static vs moving ego), and with --match bev2m
an auxiliary scoring by BEV centre distance <= 2 m between the track position and ref3d.xy_level.

  python3 scripts/eval_mot_gt.py data/labels/day_halt_short_085 [--set tracker.max_radar_only_s=1.0] [--tag x]
      [--states confirmed,lost,radar_only] [--out data/labels/eval_log.jsonl] [--match bev2m]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.gt import GT  # noqa: E402
from rcmot.labeling.metrics import RANGE_BINS, Frame, evaluate, evaluate_bev, motp_range, near_zone, range_bin_name, rates  # noqa: E402
from rcmot.labeling.provenance import ref3d_of  # noqa: E402
from rcmot.pipeline import build_pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


def set_cfg(cfg: dict, kv: str):
    k, v = kv.split("=", 1)
    d = cfg
    for p in k.split(".")[:-1]:
        d = d.setdefault(p, {})
    d[k.split(".")[-1]] = yaml.safe_load(v)


def drawn_boxes(res, calib, states: set[str], W=1280, H=720):
    """(tid, cls, state, xyxy, speed, static, range_m, xy, tsu, t_cam) of every box the GUI would draw for this
    FrameResult (speed = horizontal |velocity| shown to the user [m/s], static = standstill flag, range_m = horizontal
    range of the 3D position from the camera centre [m] (same origin as the GT ref3d), xy = BEV position, level frame,
    tsu / t_cam = time since any / since a camera-backed update [s]: the alarm counting rules, docs/38 §10)."""
    from rcmot.gui.viewer import projected_box

    C = calib.camera_center()
    out = []
    for t in res.tracks:
        if t.state.name.lower() not in states:
            continue
        box = t.box2d if t.time_since_update == 0 else projected_box(calib, t.cls, t.pos)
        if box is None:
            continue
        b = np.clip(np.asarray(box, float), [0, 0, 0, 0], [W, H, W, H])
        if b[2] - b[0] < 2 or b[3] - b[1] < 2:
            continue
        v = np.asarray(t.vel, float).ravel()
        pos = np.asarray(t.pos, float).ravel()
        rng = float(np.hypot(pos[0] - C[0], pos[1] - C[1])) if pos.size >= 2 and np.isfinite(pos[:2]).all() else np.nan
        out.append((int(t.tid), t.cls, t.state.name, b, float(np.hypot(v[0], v[1])) if v.size >= 2 else np.nan,
                    bool(getattr(t, "static", False)), rng, pos[:2].copy() if pos.size >= 2 else np.full(2, np.nan),
                    float(t.time_since_update), float(getattr(t, "time_since_camera", 0.0))))
    return out


def _pred_range(p):
    return float(p[6]) if len(p) > 6 else np.nan


def _pred_xy(p):
    return np.asarray(p[7], float) if len(p) > 7 else np.full(2, np.nan)


def make_frame(gb: list, pb: list) -> Frame:
    """metrics.Frame of one GT frame: 2D boxes plus the v2 3D columns (ref3d of the GT, range / xy of the tracks)."""
    refs = [ref3d_of(b) for b in gb]
    return Frame(np.array([b["id"] for b in gb]), np.array([b["xyxy"] for b in gb], float).reshape(-1, 4),
                 np.array([p[0] for p in pb]), np.array([p[3] for p in pb], float).reshape(-1, 4),
                 np.array(["ignore" in b["flags"] for b in gb], bool),
                 gt_range=np.array([np.nan if r is None else r["range_m"] for r in refs], float),
                 gt_xy=np.array([[np.nan, np.nan] if r is None or r.get("xy_level") is None else r["xy_level"] for r in refs],
                                float).reshape(-1, 2),
                 gt_quality=np.array(["none" if r is None else r["quality"] for r in refs], object),
                 pr_range=np.array([_pred_range(p) for p in pb], float),
                 pr_xy=np.array([_pred_xy(p) for p in pb], float).reshape(-1, 2))


def class_frames(gt, pred: dict, cls: str, idx=None) -> list[Frame]:
    """Frames of one class ("all" = every class) over the GT frame indices idx (default: all)."""
    out = []
    for k in (range(len(gt.frames)) if idx is None else idx):
        f = gt.frames[k]
        gb = [b for b in f["boxes"] if cls == "all" or b["cls"] == cls]
        pb = [p for p in pred.get(k, []) if cls == "all" or p[1] == cls]
        out.append(make_frame(gb, pb))
    return out


def ego_state_frames(gt) -> dict[str, list[int]] | None:
    """GT frame indices per ego state from meta.ego_state ([[f0, f1, state]]): the raw states plus the pooled
    "moving" (drive / swing / drive+swing). None when the key is absent."""
    segs = gt.meta.get("ego_state")
    if not segs:
        return None
    out: dict[str, list[int]] = {}
    for f0, f1, st in segs:
        for k in range(int(f0), min(int(f1), len(gt.frames) - 1) + 1):
            out.setdefault(st, []).append(k)
            if st != "static":
                out.setdefault("moving", []).append(k)
    return {k: sorted(set(v)) for k, v in out.items()}


def has_ref3d(gt) -> bool:
    return any(ref3d_of(b) is not None for f in gt.frames for b in f["boxes"])


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match_frame(gboxes, pboxes, thr=0.5):
    """Greedy IoU matching of one frame: {gt index: pred index}."""
    pairs = sorted(((_iou(g["xyxy"], p[3]), i, j) for i, g in enumerate(gboxes) for j, p in enumerate(pboxes)),
                   reverse=True)
    out, used = {}, set()
    for v, i, j in pairs:
        if v < thr:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def motion_stats(gt, pred, thr=0.5, moving=0.3):
    """Speed shown on GT-static / GT-walking objects (docs/21; ids[ID]["motion"] = [[f0, f1, state]]) and the
    standstill entry delay (GT static segment start -> first matched frame with the static flag). "static" / "walk"
    and the entry delay pool ALL GT classes; "static_worker", "static_heavy_equipment", ... split the speeds by class."""
    lab = {}
    for tid, d in gt.ids.items():
        for f0, f1, st in d.get("motion", []):
            for k in range(f0, f1 + 1):
                lab[(k, tid)] = st
    if not lab:
        return None
    sp = {"static": [], "walk": []}
    flag = {"static": [], "walk": []}
    by_cls: dict = {}                                            # (state, GT class) -> shown speeds
    for k, f in enumerate(gt.frames):
        gb = [b for b in f["boxes"] if "ignore" not in b["flags"]]
        mm = match_frame(gb, pred.get(k, []), thr)
        for i, j in mm.items():
            st = lab.get((k, gb[i]["id"]))
            if st not in sp:
                continue
            p = pred[k][j]
            if np.isfinite(p[4]):
                sp[st].append(p[4])
                by_cls.setdefault((st, gb[i]["cls"]), []).append(p[4])
            flag[st].append(p[5])
    delays, never = [], 0
    for tid, d in gt.ids.items():
        for f0, f1, st in d.get("motion", []):
            if st != "static" or f1 - f0 < 5:
                continue
            t_first = None
            for k in range(f0, f1 + 1):
                gb = [b for b in gt.frames[k]["boxes"] if b["id"] == tid and "ignore" not in b["flags"]]
                if not gb:
                    continue
                mm = match_frame(gb, pred.get(k, []), thr)
                if 0 in mm and pred[k][mm[0]][5]:
                    t_first = gt.frames[k]["stamp"]
                    break
            if t_first is None:
                never += 1
            else:
                delays.append(t_first - gt.frames[f0]["stamp"])
    out = {}
    for st in ("static", "walk"):
        a = np.asarray(sp[st], float)
        out[st] = {"n": int(a.size), "p50": round(float(np.median(a)), 3) if a.size else None,
                   "p90": round(float(np.percentile(a, 90)), 3) if a.size else None,
                   f"frac_gt{moving}": round(float((a > moving).mean()), 3) if a.size else None,
                   "frac_static_flag": round(float(np.mean(flag[st])), 3) if flag[st] else None}
    for (st, cls), v in sorted(by_cls.items()):
        a = np.asarray(v, float)
        out[f"{st}_{cls}"] = {"n": int(a.size), "p50": round(float(np.median(a)), 3),
                              "p90": round(float(np.percentile(a, 90)), 3),
                              f"frac_gt{moving}": round(float((a > moving).mean()), 3)}
    out["static_entry"] = {"segments": len(delays) + never, "never": never,
                           "delay_p50_s": round(float(np.median(delays)), 2) if delays else None,
                           "delay_p90_s": round(float(np.percentile(delays, 90)), 2) if delays else None}
    return out


def recall_by_height(gt, pred, thr=0.5, bins=(0, 60, 120, 1e9), cls="worker"):
    """Worker recall per GT box height [px] (far / mid / near) - range proxy for the detector study (docs/21 E1)."""
    hit, tot = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    for k, f in enumerate(gt.frames):
        gb = [b for b in f["boxes"] if b["cls"] == cls and "ignore" not in b["flags"]]
        mm = match_frame(gb, [p for p in pred.get(k, []) if p[1] == cls], thr)
        for i, b in enumerate(gb):
            h = b["xyxy"][3] - b["xyxy"][1]
            q = int(np.searchsorted(bins, h, side="right") - 1)
            tot[q] += 1
            hit[q] += i in mm
    return {f"h{int(bins[q])}-{int(bins[q + 1]) if bins[q + 1] < 1e8 else 'inf'}":
            {"n": int(tot[q]), "recall": round(float(hit[q] / tot[q]), 3) if tot[q] else None}
            for q in range(len(bins) - 1)}


def recall_by_range(gt, pred, thr=0.5, bins=RANGE_BINS, cls="worker"):
    """Worker recall per ref3d range bin [m] (GT boxes without a reference are counted in "no_ref")."""
    hit, tot = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    no_ref = [0, 0]
    for k, f in enumerate(gt.frames):
        gb = [b for b in f["boxes"] if b["cls"] == cls and "ignore" not in b["flags"]]
        mm = match_frame(gb, [p for p in pred.get(k, []) if p[1] == cls], thr)
        for i, b in enumerate(gb):
            r = ref3d_of(b)
            if r is None:
                no_ref[0] += 1
                no_ref[1] += i in mm
                continue
            q = int(np.searchsorted(bins, r["range_m"], side="right") - 1)
            tot[q] += 1
            hit[q] += i in mm
    out = {f"r{range_bin_name(bins, q)}": {"n": int(tot[q]), "recall": round(float(hit[q] / tot[q]), 3) if tot[q] else None}
           for q in range(len(bins) - 1)}
    out["no_ref"] = {"n": no_ref[0], "recall": round(no_ref[1] / no_ref[0], 3) if no_ref[0] else None}
    return out


def recall_by_occ(gt, pred, thr=0.5, cls="worker", levels=("none", "partial", "full")):
    """Worker recall per occlusion level (box "occ", schema v2). None when no box carries the key."""
    if not any("occ" in b for f in gt.frames for b in f["boxes"]):
        return None
    hit, tot = {q: 0 for q in levels}, {q: 0 for q in levels}
    for k, f in enumerate(gt.frames):
        gb = [b for b in f["boxes"] if b["cls"] == cls and "ignore" not in b["flags"]]
        mm = match_frame(gb, [p for p in pred.get(k, []) if p[1] == cls], thr)
        for i, b in enumerate(gb):
            q = b.get("occ", "none")
            if q not in tot:
                continue
            tot[q] += 1
            hit[q] += i in mm
    return {q: {"n": tot[q], "recall": round(hit[q] / tot[q], 3) if tot[q] else None} for q in levels}


def _motp_clean(frames: list[Frame], thr: float) -> dict:
    """MOTP[m] on the same ignore-cleaned frames and CLEAR matching that evaluate() scores."""
    from rcmot.labeling.metrics import drop_ignored
    return motp_range([drop_ignored(f, thr)[0] for f in frames], thr)


def brief(r: dict) -> dict:
    """Compact per-split summary (ego-state tables)."""
    out = {k: r[k] for k in ("HOTA", "MOTA", "IDF1", "IDsw", "IDSWR", "FNR", "FPR_gt", "TP", "FP", "FN", "n_gt_boxes") if k in r}
    if "MOTP_m" in r:
        out["MOTP_m"] = {k: r["MOTP_m"][k] for k in ("n", "mean", "median", "p90")}
    return out


def table(name, r):
    fb = r["FP_breakdown"]
    return (f"{name:16s} HOTA {r['HOTA']:.3f} DetA {r['DetA']:.3f} AssA {r['AssA']:.3f} | MOTA {r['MOTA']:.3f} "
            f"IDF1 {r['IDF1']:.3f} (IDP {r['IDP']:.2f} IDR {r['IDR']:.2f}) | IDsw {r['IDsw']} Frag {r['Frag']} "
            f"MT {r['MT']}/{r['n_gt_ids']} ML {r['ML']} | TP {r['TP']} FP {r['FP']} (dup {fb['duplicate']}, "
            f"misloc {fb['mislocated']}, ghost {fb['ghost']}) FN {r['FN']} | pred ids {r['n_pred_ids']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--config")
    ap.add_argument("--calib")
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--states", default="confirmed,lost,radar_only", help="track states that count as output")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stamp-tol", type=float, default=0.02, help="max |pipeline - GT stamp| for a frame match [s]")
    ap.add_argument("--match", default="iou", choices=["iou", "bev2m"],
                    help="bev2m: additionally score with the BEV centre-distance rule (<= 2 m, GT with ref3d only)")
    ap.add_argument("--out", default=str(ROOT / "data/labels/eval_log.jsonl"))
    ap.add_argument("--dump-pred", help="write the per-GT-frame predictions (drawn boxes) to this JSON (diagnosis)")
    ap.add_argument("--near", default="20,30", help="near-zone views [m] (docs/38): people beyond R are ignore, outputs "
                    "beyond R + 2 m on nobody in the zone are not counted; '' = off")
    a = ap.parse_args()
    gt = GT(Path(a.seg) / "gt.json")
    m = gt.meta
    cfg = yaml.safe_load(open(a.config or ROOT / m["config"]))
    for kv in a.set:
        set_cfg(cfg, kv)
    calib = RigCalib.load(str(ROOT / (a.calib or m["calib"])))
    pipe = build_pipeline(cfg, calib, ROOT)
    if a.dump_pred:
        pipe.tracker.relink_log = []
    states = set(a.states.split(","))
    gt_st = np.array([f["stamp"] for f in gt.frames])
    # sensor-clock radars (Continental): iterate_bag maps the header clock onto the bag clock with one offset per bag
    # (radar_clock_offset, or meta.radar_clock_offset_s when the GT recorded it); match on the radar message's bag
    # receive time (exact, same message) when every GT frame carries bag_t, else nearest stamp within --stamp-tol
    gt_bt = np.array([f.get("bag_t", np.nan) for f in gt.frames], float)
    use_bt = bool(len(gt_bt)) and not np.isnan(gt_bt).any()
    best: dict[int, float] = {}                         # GT frame -> |stamp error| of the prediction kept for it
    pred: dict[int, list] = {}
    tp = cfg["topics"]
    start = max(m["start"] - a.warmup, 0.0)
    t0 = time.time()
    tms: list[dict] = []                                # per-step stage timings (pipeline FrameResult.timing_ms)
    tmap: list[tuple] = []                              # (tracker time, scan stamp): relink-log entries -> stamps
    imu = m.get("imu_topic") or tp.get("imu")         # the GT's IMU (154405: Bosch IMU, as its GUI profile; 11-04 audit)
    for img, rf, yaw in iterate_bag(ROOT / m["bag"], calib, tp["radar"], tp["image"], imu, start=start,
                                    duration=m["end"] - start + 0.5, radar_clock_offset_s=m.get("radar_clock_offset_s")):
        res = pipe.step(img, rf, yaw)
        tms.append(res.timing_ms)
        tmap.append((pipe.tracker.t, rf.stamp))
        if use_bt and getattr(rf, "bag_t", None) is not None:
            k = int(np.argmin(np.abs(gt_bt - rf.bag_t)))
            err, tol = abs(gt_bt[k] - rf.bag_t), 2e-3
        else:
            k = int(np.argmin(np.abs(gt_st - rf.stamp)))
            err, tol = abs(gt_st[k] - rf.stamp), a.stamp_tol
        if err <= tol and err < best.get(k, np.inf):
            best[k] = err
            pred[k] = drawn_boxes(res, calib, states, W=img.shape[1], H=img.shape[0])
    missing = len(gt.frames) - len(pred)
    if a.dump_pred:
        json.dump({str(k): v for k, v in pred.items()}, open(a.dump_pred, "w"),
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        tt = np.array([a for a, _ in tmap]) if tmap else np.zeros(1)
        ts = np.array([b for _, b in tmap]) if tmap else np.zeros(1)
        rl = [{**{k: v for k, v in e.items() if k != "track"}, "new_tid": int(e["track"].tid) if "track" in e else None,
               "stamp": round(float(np.interp(e.get("t", 0.0), tt, ts)), 3)}
              for e in (pipe.tracker.relink_log or [])]
        json.dump(rl, open(str(a.dump_pred).replace(".json", "") + ".relink.json", "w"))
    results, bev, by_ego = {}, {}, {}
    ref_ok = has_ref3d(gt)
    ego = ego_state_frames(gt)
    for cls in ("worker", "heavy_equipment", "all"):
        frames = class_frames(gt, pred, cls)
        if not any(len(f.gt_ids) for f in frames) and not any(len(f.pr_ids) for f in frames):
            continue
        r = evaluate(frames, a.iou)
        r.update(rates(r))
        if ref_ok:
            r["MOTP_m"] = _motp_clean(frames, a.iou)
            if a.match == "bev2m":
                bev[cls] = evaluate_bev(frames, a.iou)
        results[cls] = r
        if ego:
            for st, idx in ego.items():
                fr = class_frames(gt, pred, cls, idx)
                if not any(len(f.gt_ids) for f in fr):
                    continue
                e = evaluate(fr, a.iou)
                e.update(rates(e))
                if ref_ok:
                    e["MOTP_m"] = _motp_clean(fr, a.iou)
                by_ego.setdefault(st, {})[cls] = {"frames": len(idx), **brief(e)}
    near = {}
    if ref_ok and a.near:
        for R in [float(x) for x in a.near.split(",") if x.strip()]:
            fr = [near_zone(f, R, thr=a.iou) for f in class_frames(gt, pred, "worker")]
            if not any(len(f.gt_ids) for f in fr):
                continue
            e = evaluate(fr, a.iou)
            e.update(rates(e))
            nz = {"worker": {k: e[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDsw", "TP", "FP", "FN",
                                                "n_gt_boxes", "n_pred_boxes", "pred_in_ignore") if k in e}}
            if a.match == "bev2m":
                b = evaluate_bev(fr, a.iou)
                nz["bev2m_worker"] = {k: b[k] for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDsw", "TP", "FP", "FN")
                                      if k in b}
                nz["bev2m_worker"]["MOTP_m"] = {k: b["MOTP_m"][k] for k in ("n", "mean", "median", "p90")}
            near[f"{R:g}"] = nz
    # per-state view of the false positives (all classes): which states produce ghosts
    st_fp = {}
    for s in sorted(states):
        frames = []
        for k, f in enumerate(gt.frames):
            pb = [p for p in pred.get(k, []) if p[2].lower() == s]
            frames.append(Frame(np.array([b["id"] for b in f["boxes"]]),
                                np.array([b["xyxy"] for b in f["boxes"]], float).reshape(-1, 4),
                                np.array([p[0] for p in pb]), np.array([p[3] for p in pb], float).reshape(-1, 4),
                                np.array(["ignore" in b["flags"] for b in f["boxes"]], bool)))
        r = evaluate(frames, a.iou)
        st_fp[s] = {"boxes": r["n_pred_boxes"], "TP": r["TP"], "FP": r["FP"], **r["FP_breakdown"]}
    mo, rh = motion_stats(gt, pred, a.iou), recall_by_height(gt, pred, a.iou)
    rr = recall_by_range(gt, pred, a.iou) if ref_ok else None
    ro = recall_by_occ(gt, pred, a.iou)
    summ = {"tag": a.tag, "seg": Path(a.seg).name, "time": time.strftime("%Y-%m-%d %H:%M"), "set": a.set,
            "states": a.states, "frames": len(gt.frames), "frames_missing_pred": missing,
            "runtime_s": round(time.time() - t0, 1), "results": results, "by_state": st_fp,
            "motion": mo, "recall_by_height": rh, "recall_by_range": rr, "recall_by_occ": ro,
            "by_ego_state": by_ego or None, "bev2m": bev or None, "match": a.match, "near": near or None,
            "tracker_counts": {k: int(getattr(pipe.tracker, k, 0)) for k in ("relinked", "app_reacquired", "reacquired",
                                                                              "reentered", "reentry_new", "reentry_deferred")}
            | {"app_mode": getattr(pipe.tracker.p, "app_mode", "cos")},
            "timing_ms": {k: {"mean": round(float(np.mean(v)), 3), "p50": round(float(np.median(v)), 3),
                              "p90": round(float(np.percentile(v, 90)), 3), "n": len(v)}
                          for k in sorted({k for t in tms for k in t})
                          for v in [[t[k] for t in tms if k in t]]}
            | {"reentry_per_step": round(1e3 * float(getattr(pipe.tracker, "reentry_s", 0.0)) / max(len(tms), 1), 4)}}
    for cls, r in results.items():
        print(table(cls, r))
        fmt = lambda v, d=3: "n/a" if v is None else f"{v:.{d}f}"
        print(f"{'':16s} IDSWR {fmt(r['IDSWR'], 4)} FNR {fmt(r['FNR'])} FPR {fmt(r['FPR_gt'])} (FP/Pred {fmt(r['FPR_pred'])})"
              + (f" | MOTP[m] n {r['MOTP_m']['n']} mean {r['MOTP_m']['mean']} median {r['MOTP_m']['median']} "
                 f"p90 {r['MOTP_m']['p90']} bias {r['MOTP_m']['bias']}" if "MOTP_m" in r else ""))
        if "MOTP_m" in r:
            print(f"{'':16s} MOTP[m] by range {json.dumps(r['MOTP_m']['by_range'])} by quality {json.dumps(r['MOTP_m']['by_quality'])}")
    print("by state:", json.dumps(st_fp))
    print("pred ids per GT id (worker):", results.get("worker", {}).get("pred_ids_per_gt"))
    print("worker recall by box height:", json.dumps(rh))
    if rr:
        print("worker recall by ref3d range:", json.dumps(rr))
    if ro:
        print("worker recall by occlusion:", json.dumps(ro))
    if by_ego:
        print("by ego state:", json.dumps(by_ego))
    for R, nz in near.items():
        w = nz["worker"]
        line = f"near {R:>4s} m worker HOTA {w['HOTA']:.3f} MOTA {w['MOTA']:.3f} IDF1 {w['IDF1']:.3f} IDsw {w['IDsw']} " \
               f"TP {w['TP']} FP {w['FP']} FN {w['FN']}"
        if "bev2m_worker" in nz:
            b = nz["bev2m_worker"]
            line += f" | BEV HOTA {b['HOTA']:.3f} MOTA {b['MOTA']:.3f} IDF1 {b['IDF1']:.3f} FN {b['FN']} FP {b['FP']}"
        print(line)
    for cls, r in bev.items():
        print(f"bev2m {cls:12s} HOTA {r['HOTA']:.3f} MOTA {r['MOTA']:.3f} IDF1 {r['IDF1']:.3f} IDsw {r['IDsw']} IDSWR {fmt(r['IDSWR'], 4)} "
              f"TP {r['TP']} FP {r['FP']} FN {r['FN']} | MOTP[m] n {r['MOTP_m']['n']} mean {r['MOTP_m']['mean']} "
              f"median {r['MOTP_m']['median']} p90 {r['MOTP_m']['p90']}")
    if mo:
        print("motion (GT static / walk):", json.dumps(mo))
    if missing:
        print(f"WARNING: {missing} GT frames had no matching pipeline frame (stamp mismatch)")
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(summ, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


if __name__ == "__main__":
    main()
