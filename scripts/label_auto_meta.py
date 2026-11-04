#!/usr/bin/env python3
"""Automatic schema-v2 drafts of a label segment (gt_design §C.3 / §D.1-3, docs/13 "평가 지표 v2"):

  (a) meta.ego_state = [[f0, f1, static | drive | swing | drive+swing]] from the IMU yaw rate (0.5 s mean > 5 deg/s
      -> swing) and the radar-Doppler ego speed of the pipeline's ego fit (> 0.1 m/s -> drive), merged into
      segments of >= 1 s; meta.ego_rate = {yaw_rate_max_dps, speed_max_mps} over the segment.
  (b) ids[ID].motion = [[f0, f1, static | walk | mixed]] from the 1 s displacement of the box ref3d.xy_level
      (< 0.3 m static, > 0.5 m walk, else mixed), measured in the pipeline's world frame so the ego motion does
      not count as walking; runs of >= 1 s. Needs ref3d (scripts/eval_fusion_depth.py --write-ref) first.

Written only where missing: ego_state / ego_rate when absent (--force-ego recomputes), motion only for IDs without
one (an existing motion is never overwritten). A provenance entry {stage: draft, who: label_auto_meta} is appended.

  python3 scripts/label_auto_meta.py data/labels/<seg> [--bag ...] [--config ...] [--calib ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.provenance import add_provenance, load_doc, ref3d_of, save_doc  # noqa: E402
from rcmot.pipeline import Pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

EGO_ORDER = ("static", "drive", "swing", "drive+swing")


class _NoDet:
    def __call__(self, img):
        return []


# ------------------------------------------------------------------------------------------------ run helpers
def runs_of(labels: list) -> list[list]:
    """[[f0, f1, label]] of the consecutive equal labels (None labels are kept as runs too)."""
    out = []
    for k, lab in enumerate(labels):
        if out and out[-1][2] == lab:
            out[-1][1] = k
        else:
            out.append([k, k, lab])
    return out


def merge_short(runs: list[list], stamps: np.ndarray, min_s: float) -> list[list]:
    """Absorb runs shorter than min_s seconds into the longer neighbour, until every run is long enough (or one
    run is left). The frame indices of `runs` index `stamps`."""
    runs = [list(r) for r in runs]

    def dur(r):
        period = float(np.median(np.diff(stamps))) if len(stamps) > 1 else 0.2
        return stamps[r[1]] - stamps[r[0]] + period

    while len(runs) > 1:
        short = [i for i, r in enumerate(runs) if dur(r) < min_s]
        if not short:
            break
        i = min(short, key=lambda i: dur(runs[i]))
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(nb, key=lambda j: dur(runs[j]))
        runs[i][2] = runs[j][2]
        merged = []
        for r in runs:
            if merged and merged[-1][2] == r[2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs = merged
    return runs


def fill_unknown(labels: list, default: str) -> list:
    """None labels take the nearest known label (forward, then backward fill); all unknown -> default."""
    out = list(labels)
    last = None
    for k, v in enumerate(out):
        if v is None:
            out[k] = last
        else:
            last = v
    last = None
    for k in range(len(out) - 1, -1, -1):
        if out[k] is None:
            out[k] = last
        else:
            last = out[k]
    return [default if v is None else v for v in out]


# ------------------------------------------------------------------------------------------------ ego state
def ego_samples(m: dict, cfg: dict, calib: RigCalib, bag: Path, pad: float = 1.0):
    """Per radar frame over the segment (with `pad` s before it): stamp, yaw rate [deg/s], ego speed [m/s],
    ego fit valid, world pose (WorldState or None)."""
    cfg = dict(cfg)
    cfg.setdefault("recovery", {})["enable"] = False
    pipe = Pipeline(_NoDet(), calib, cfg)
    rows, worlds = [], []
    tp = cfg["topics"]
    start = max(m["start"] - pad, 0.0)
    for img, rf, yaw in iterate_bag(bag, calib, tp["radar"], tp["image"], tp.get("imu"), start=start,
                                    duration=m["end"] - start + 0.5):
        res = pipe.step(img, rf, yaw)
        v = np.asarray(res.ego_vel, float).ravel()
        rows.append((rf.stamp, np.degrees(yaw), float(np.hypot(v[0], v[1])), bool(res.ego_valid)))
        worlds.append(getattr(res, "world", None))
    return np.array(rows, float).reshape(-1, 4), worlds


def smooth_abs(t: np.ndarray, x: np.ndarray, half: float = 0.25) -> np.ndarray:
    """Mean of |x| over +-half s around each sample (0.5 s window, bag_overview's turning rule)."""
    ax = np.abs(x)
    return np.array([ax[(t >= ti - half) & (t <= ti + half)].mean() for ti in t]) if len(t) else ax


def ego_state_draft(stamps: np.ndarray, samp: np.ndarray, yaw_thr: float, speed_thr: float, min_s: float):
    """([[f0, f1, state]], ego_rate, per-frame state) from the smoothed samples at the GT stamps."""
    t, yaw_s, spd = samp[:, 0], smooth_abs(samp[:, 0], samp[:, 1]), samp[:, 2]
    spd_s = np.array([spd[(t >= ti - 0.25) & (t <= ti + 0.25)].mean() for ti in t]) if len(t) else spd
    lab, ymax, smax = [], 0.0, 0.0
    for s in stamps:
        k = int(np.argmin(np.abs(t - s))) if len(t) else None
        if k is None or abs(t[k] - s) > 0.15:
            lab.append(None)
            continue
        swing, drive = yaw_s[k] > yaw_thr, spd_s[k] > speed_thr
        lab.append("drive+swing" if swing and drive else "swing" if swing else "drive" if drive else "static")
        ymax, smax = max(ymax, float(yaw_s[k])), max(smax, float(spd_s[k]))
    lab = fill_unknown(lab, "static")
    runs = merge_short(runs_of(lab), stamps, min_s)
    per_frame = [None] * len(stamps)
    for f0, f1, st in runs:
        for k in range(f0, f1 + 1):
            per_frame[k] = st
    return [[int(f0), int(f1), st] for f0, f1, st in runs], \
        {"yaw_rate_max_dps": round(ymax, 1), "speed_max_mps": round(smax, 2)}, per_frame


# ------------------------------------------------------------------------------------------------ motion
def motion_draft(frames: list, stamps: np.ndarray, tid: int, xy_world: dict, static_thr: float, walk_thr: float,
                 min_s: float, horizon: float = 1.0, tol: float = 0.3):
    """[[f0, f1, static|walk|mixed]] of one ID over its first..last GT frame from the `horizon` s displacement of
    xy_world[(frame, id)] (world frame, nan-free). Frames without a displacement (no reference, track end) take the
    nearest labelled frame; an ID without any reference gets a single mixed run."""
    kf = [k for k, f in enumerate(frames) if any(b["id"] == tid for b in f["boxes"])]
    if not kf:
        return None
    k0, k1 = kf[0], kf[-1]
    disp = {}
    for k in kf:
        p = xy_world.get((k, tid))
        if p is None:
            continue
        for sign in (1, -1):                                # forward displacement; backward near the track end
            cand = [j for j in kf if (k, tid) != (j, tid) and (j, tid) in xy_world and abs(stamps[j] - stamps[k] - sign * horizon) <= tol]
            if cand:
                j = min(cand, key=lambda j: abs(stamps[j] - stamps[k] - sign * horizon))
                disp[k] = float(np.linalg.norm(xy_world[(j, tid)] - p))
                break
    lab = []
    for k in range(k0, k1 + 1):
        d = [disp[j] for j in range(k - 2, k + 3) if j in disp]      # 5-frame median: reference jitter 0.2-0.3 m
        if not d:
            lab.append(None)
            continue
        dm = float(np.median(d))
        lab.append("static" if dm < static_thr else "walk" if dm > walk_thr else "mixed")
    n_known = sum(v is not None for v in lab)
    lab = fill_unknown(lab, "mixed")
    runs = merge_short(runs_of(lab), stamps[k0:k1 + 1], min_s)
    return [[int(k0 + f0), int(k0 + f1), st] for f0, f1, st in runs], n_known, len(lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--bag", help="override meta.bag")
    ap.add_argument("--config", help="override meta.config")
    ap.add_argument("--calib", help="override meta.calib")
    ap.add_argument("--yaw-thr", type=float, default=5.0, help="swing when the 0.5 s mean |yaw rate| exceeds this [deg/s]")
    ap.add_argument("--speed-thr", type=float, default=0.1, help="drive when the ego speed exceeds this [m/s]")
    ap.add_argument("--min-seg", type=float, default=1.0, help="minimum ego_state / motion segment length [s]")
    ap.add_argument("--static-thr", type=float, default=0.3, help="1 s displacement below this: static [m]")
    ap.add_argument("--walk-thr", type=float, default=0.5, help="1 s displacement above this: walk [m]")
    ap.add_argument("--force-ego", action="store_true", help="recompute ego_state / ego_rate even when present")
    ap.add_argument("--dry-run", action="store_true", help="print the drafts, do not write gt.json")
    a = ap.parse_args()
    gt_path = Path(a.seg) / "gt.json"
    doc = load_doc(gt_path)
    m, frames = doc["meta"], doc["frames"]
    stamps = np.array([f["stamp"] for f in frames], float)
    bag = Path(a.bag) if a.bag else ROOT / m["bag"]
    cfg = yaml.safe_load(open(a.config or ROOT / m["config"]))
    calib = RigCalib.load(str(a.calib or ROOT / m["calib"]))

    samp, worlds = ego_samples(m, cfg, calib, bag)
    if not len(samp):
        sys.exit(f"no radar frames read from {bag} for {m['start']}..{m['end']}")
    ego_segs, ego_rate, _ = ego_state_draft(stamps, samp, a.yaw_thr, a.speed_thr, a.min_seg)
    wrote = []
    if a.force_ego or not m.get("ego_state"):
        m["ego_state"], m["ego_rate"] = ego_segs, ego_rate
        wrote.append("ego_state")
    print("ego_state draft:", json.dumps(ego_segs), "ego_rate:", json.dumps(ego_rate),
          "(kept existing)" if "ego_state" not in wrote else "")

    # world xy of every referenced box: level xy -> world with the pose of the nearest radar frame
    t = samp[:, 0]
    xy_world, n_ref = {}, 0
    # ego_state static => hold the pose of the run's first frame (the pipeline heading jitters on the xpanner rigs)
    static_first = {}
    for f0, f1, st in (m.get("ego_state") or []):
        if st == "static":
            for k in range(f0, f1 + 1):
                static_first[k] = f0
    for k, f in enumerate(frames):
        kk = static_first.get(k, k)
        j = int(np.argmin(np.abs(t - stamps[kk])))
        w = worlds[j] if abs(t[j] - stamps[kk]) <= 0.15 else None
        for b in f["boxes"]:
            r = ref3d_of(b)
            if r is None or r.get("xy_level") is None:
                continue
            n_ref += 1
            p = np.asarray(r["xy_level"], float)
            xy_world[(k, b["id"])] = w.to_world(p)[0] if w is not None else p
    ids = doc.get("ids", {})
    done, kept = {}, []
    for sid, d in ids.items():
        tid = int(sid)
        if d.get("motion"):
            kept.append(tid)
            continue
        md = motion_draft(frames, stamps, tid, xy_world, a.static_thr, a.walk_thr, a.min_seg)
        if md is None:
            continue
        d["motion"] = md[0]
        done[tid] = md
    if n_ref == 0:
        print("WARNING: no box carries ref3d - run scripts/eval_fusion_depth.py --write-ref first; motion drafts are all 'mixed'")
    for tid, (segs, n_known, n_all) in sorted(done.items()):
        print(f"motion draft id {tid} ({ids[str(tid)].get('cls')}): {json.dumps(segs)}  frames with displacement {n_known}/{n_all}")
    if kept:
        print("motion kept (already present) for ids:", kept)
    if done:
        wrote.append(f"motion ids {sorted(done)}")
    if a.dry_run or not wrote:
        print("nothing written" if not wrote else "dry run: not written")
        return
    add_provenance(doc, "draft", "label_auto_meta", "scripts/label_auto_meta.py", ROOT,
                   note="; ".join(wrote) + f"; yaw_thr {a.yaw_thr} speed_thr {a.speed_thr} static<{a.static_thr} walk>{a.walk_thr} min_seg {a.min_seg}s")
    save_doc(doc, gt_path)
    print(f"written {gt_path}: {', '.join(wrote)}")


if __name__ == "__main__":
    main()
