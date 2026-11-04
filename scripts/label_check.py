#!/usr/bin/env python3
"""Consistency checks of a label segment (gt_design §D.5, docs/28 §4.5).

  python3 scripts/label_check.py data/labels/<seg> [--strict] [--json out.json] [--stamp-tol 0.02]

Checks (E = error, W = warning):
  id continuity   same id, adjacent frames: centre move > 1.5 x box height               W
  gaps            interpolated run > 15 frames                                             E
                  interpolated run of 7..15 frames not ignore(interp_gap)                  E
                  unlabelled hole of <= 15 frames (interpolation candidate)                W
  stamps          every GT frame stamp has a pipeline stamp (meta.stamps) within +-20 ms   E
  lidar coverage  non-ignore worker boxes with ref3d.quality != none >= 90 % (boryeong) / 80 % (arizona)  E
                  (skipped with a warning when the segment has no ref3d yet)
  range jumps     same id, adjacent frames, |d range_m| > 2 m: > 2 % of the pairs          E (pairs listed as W)
  ref3d wedge     ref3d.xy_level must project inside its own box (width +-50 %)               E
  ref3d vs mono   |ref3d.range_m - mono range (1.7 m worker)| > 5 m                        W (listed)
  motion          every frame of every id lies in a motion segment                         E
  ego_state       segments cover 0..N-1 without overlap, each >= 1 s                       E
  class           box cls == ids[id].cls, every box id is in ids                           E
  duplicates      two boxes of one id in a frame                                           E
                  IoU > 0.7 between boxes of different ids                                 W
  schema v2       enum values, occ <-> flags.occluded, ignore without reason               E / W
Exit code 1 when --strict and there are errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.draft import iou  # noqa: E402
from rcmot.labeling.gt import (EGO_STATES, GT, IGNORE_REASONS, INTERP_IGNORE_GAP, INTERP_MAX_GAP, LIGHTINGS, OCC,  # noqa: E402
                               RADARS, REF3D_QUALITY, ROLES, SITES, SPLITS)

COVERAGE_THR = {"boryeong_pg": 0.90, "arizona_xpanner": 0.80}
WORKER_H = 1.7


class Report:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.stats: dict = {}

    def E(self, s):
        self.errors.append(s)

    def W(self, s):
        self.warnings.append(s)

    def I(self, s):
        self.info.append(s)

    def to_dict(self):
        return {"errors": self.errors, "warnings": self.warnings, "info": self.info, "stats": self.stats}


def _mono_range(calib, box, H=WORKER_H):
    x1, y1, x2, y2 = box
    if y2 - y1 < 12 or y1 <= 2 or y2 >= calib.image_size[1] - 2:
        return None
    uc = 0.5 * (x1 + x2)
    dt, db = calib.pixel_ray(uc, y1), calib.pixel_ray(uc, y2)
    dtan = dt[2] / np.linalg.norm(dt[:2]) - db[2] / np.linalg.norm(db[:2])
    return None if dtan <= 1e-4 else float(H / dtan)


def check(gt: GT, stamp_tol: float = 0.02, root: Path = ROOT, max_list: int = 20) -> Report:
    r = Report()
    m, N = gt.meta, len(gt.frames)
    v2 = gt.schema_version >= 2
    rate = float(m.get("rate_hz") or 5.0)
    r.stats.update(gt.counts())
    r.stats["schema_version"] = gt.schema_version

    # ------------------------------------------------------------- per-frame: ids, class, duplicates, enums
    occ_mismatch = ign_noreason = 0
    for i, f in enumerate(gt.frames):
        ids = [b["id"] for b in f["boxes"]]
        for d in sorted({t for t in ids if ids.count(t) > 1}):
            r.E(f"f{i}: id {d} appears {ids.count(d)} times")
        for b in f["boxes"]:
            if b["id"] not in gt.ids:
                r.E(f"f{i}: box id {b['id']} not in ids")
            elif b["cls"] != gt.ids[b["id"]]["cls"]:
                r.E(f"f{i}: id {b['id']} box cls {b['cls']} != ids cls {gt.ids[b['id']]['cls']}")
            x1, y1, x2, y2 = b["xyxy"]
            if x2 <= x1 or y2 <= y1:
                r.E(f"f{i}: id {b['id']} degenerate box {b['xyxy']}")
            if v2:
                if b.get("occ", "none") not in OCC:
                    r.E(f"f{i}: id {b['id']} occ {b.get('occ')!r} invalid")
                if b.get("ignore_reason") not in IGNORE_REASONS + (None,):
                    r.E(f"f{i}: id {b['id']} ignore_reason {b.get('ignore_reason')!r} invalid")
                if (b.get("occ", "none") != "none") != ("occluded" in b["flags"]):
                    occ_mismatch += 1
                if "ignore" in b["flags"] and b.get("ignore_reason") is None:
                    ign_noreason += 1
                if b.get("ignore_reason") is not None and "ignore" not in b["flags"]:
                    r.E(f"f{i}: id {b['id']} has ignore_reason but no ignore flag")
                rf = b.get("ref3d")
                if rf is not None and rf.get("quality") not in REF3D_QUALITY:
                    r.E(f"f{i}: id {b['id']} ref3d.quality {rf.get('quality')!r} invalid")
        if len(f["boxes"]) > 1:
            B = np.array([b["xyxy"] for b in f["boxes"]], float)
            I = iou(B, B)
            for a in range(len(B)):
                for c in range(a + 1, len(B)):
                    if I[a, c] > 0.7 and f["boxes"][a]["id"] != f["boxes"][c]["id"]:
                        r.W(f"f{i}: ids {f['boxes'][a]['id']} and {f['boxes'][c]['id']} overlap IoU {I[a, c]:.2f}")
    if occ_mismatch:
        r.W(f"{occ_mismatch} boxes where occ != none does not match flags.occluded")
    if ign_noreason:
        r.W(f"{ign_noreason} ignore boxes without ignore_reason")

    # ------------------------------------------------------------- per-id: continuity, gaps, motion, ref3d
    jumps, pairs, jump_list, mono_list = 0, 0, [], []
    calib = None
    if m.get("calib") and (root / m["calib"]).exists():
        try:
            from rcmot.utils.calib import RigCalib
            calib = RigCalib.load(str(root / m["calib"]))
        except Exception as e:                                   # noqa: BLE001
            r.W(f"calib {m['calib']} not loadable ({e}); mono check skipped")
    else:
        r.W(f"calib {m.get('calib')} not found; mono check skipped")
    # ref3d must lie inside the box's own viewing wedge: its xy (at 1 m height) projects within the box width +-50 %
    if calib is not None:
        off, off_man = [], []
        for i, f in enumerate(gt.frames):
            for b in f["boxes"]:
                rf = b.get("ref3d") or {}
                if rf.get("xy_level") is None:
                    continue
                x1, y1, x2, y2 = b["xyxy"]
                uv, front = calib.project(np.array([[rf["xy_level"][0], rf["xy_level"][1], 1.0]]))
                w = x2 - x1
                if not front[0] or not np.isfinite(uv[0, 0]) or uv[0, 0] < x1 - 0.5 * w or uv[0, 0] > x2 + 0.5 * w:
                    (off_man if rf.get("src") == "manual" else off).append((i, b["id"], None if not front[0] else round(float(uv[0, 0]))))
        if off:
            r.E(f"{len(off)} ref3d outside the box's viewing wedge (u of the reference vs box): {off[:max_list]}")
        if off_man:
            r.W(f"{len(off_man)} hand-placed ref3d outside the box's viewing wedge (check in the GUI): {off_man[:max_list]}")
    n_cont = 0
    for tid in sorted(gt.ids):
        fr = gt.id_frames(tid)
        if not fr:
            r.W(f"id {tid}: in ids but has no boxes")
            continue
        # continuity + range jumps on adjacent frames
        for a, b in zip(fr[:-1], fr[1:]):
            ba, bb = gt.box_of(a, tid), gt.box_of(b, tid)
            if b - a == 1:
                xa, xb = np.array(ba["xyxy"]), np.array(bb["xyxy"])
                h = max(xa[3] - xa[1], 1.0)
                d = np.linalg.norm((xa[:2] + xa[2:]) / 2 - (xb[:2] + xb[2:]) / 2)
                if d > 1.5 * h:
                    n_cont += 1
                    if n_cont <= max_list:
                        r.W(f"id {tid} f{a}->f{b}: centre moves {d:.0f} px > 1.5 x h {h:.0f} px")
                ra, rb = ba.get("ref3d"), bb.get("ref3d")
                if ra and rb and ra.get("quality") != "none" and rb.get("quality") != "none" \
                        and ra.get("range_m") is not None and rb.get("range_m") is not None:
                    pairs += 1
                    if abs(ra["range_m"] - rb["range_m"]) > 2.0:
                        jumps += 1
                        jump_list.append(f"id {tid} f{a}->f{b}: range {ra['range_m']:.1f} -> {rb['range_m']:.1f} m")
        # unlabelled holes (interpolation candidates): short ones aggregated per id, 7..15 listed, > 15 info
        short = []
        for a, b in zip(fr[:-1], fr[1:]):
            hole = b - a - 1
            if 1 <= hole < INTERP_IGNORE_GAP:
                short.append((a + 1, hole))
            elif INTERP_IGNORE_GAP <= hole <= INTERP_MAX_GAP:
                r.W(f"id {tid}: {hole} empty frames f{a + 1}..f{b - 1} (interp + ignore(interp_gap) candidate)")
            elif hole > INTERP_MAX_GAP:
                r.I(f"id {tid}: {hole} empty frames f{a + 1}..f{b - 1} (left empty, > {INTERP_MAX_GAP})")
        if short:
            r.W(f"id {tid}: {len(short)} short holes (< {INTERP_IGNORE_GAP} frames, {sum(h for _, h in short)} frames "
                f"total, first f{short[0][0]}) -> interp as normal GT or leave if truly absent")
        r.stats["short_holes"] = r.stats.get("short_holes", 0) + len(short)
        # interpolated runs
        run: list[int] = []
        for i in fr + [None]:
            if i is not None and gt.box_of(i, tid)["src"] == "interp":
                run.append(i)
                continue
            if run:
                L = len(run)
                if L > INTERP_MAX_GAP:
                    r.E(f"id {tid}: interpolated run of {L} frames f{run[0]}..f{run[-1]} > {INTERP_MAX_GAP}")
                elif L >= INTERP_IGNORE_GAP:
                    bad = [k for k in run if "ignore" not in gt.box_of(k, tid)["flags"]
                           or (v2 and gt.box_of(k, tid).get("ignore_reason") != "interp_gap")]
                    if bad:
                        r.E(f"id {tid}: interpolated run of {L} frames f{run[0]}..f{run[-1]} needs ignore(interp_gap) "
                            f"({len(bad)} boxes missing it)")
                run = []
        # motion coverage
        mo = gt.ids[tid].get("motion") or []
        miss = [i for i in fr if not any(f0 <= i <= f1 for f0, f1, _ in mo)]
        if miss:
            r.E(f"id {tid}: {len(miss)} of {len(fr)} frames not in any motion segment"
                + ("" if mo else " (no motion)") + f" e.g. f{miss[0]}")
        for f0, f1, st in mo:
            if st not in ("static", "walk", "mixed") or f0 > f1:
                r.E(f"id {tid}: bad motion segment {[f0, f1, st]}")
        if v2:
            if gt.ids[tid].get("role") not in ROLES:
                r.E(f"id {tid}: role {gt.ids[tid].get('role')!r} invalid")
            if gt.ids[tid]["cls"] == "worker" and gt.ids[tid].get("role") == "background":
                r.W(f"id {tid}: worker with role background")
        # ref3d vs mono
        if calib is not None and gt.ids[tid]["cls"] == "worker":
            for i in fr:
                b = gt.box_of(i, tid)
                rf = b.get("ref3d")
                if rf and rf.get("quality") != "none" and rf.get("range_m") is not None:
                    mono = _mono_range(calib, b["xyxy"])
                    if mono is not None and abs(mono - rf["range_m"]) > 5.0:
                        mono_list.append(f"id {tid} f{i}: ref3d {rf['range_m']:.1f} m vs mono {mono:.1f} m")
    if n_cont > max_list:
        r.W(f"... {n_cont} continuity warnings in total")
    r.stats["continuity_warnings"] = n_cont

    # ------------------------------------------------------------- ref3d coverage / jumps
    workers = [b for f in gt.frames for b in f["boxes"] if b["cls"] == "worker" and "ignore" not in b["flags"]]
    with_ref = [b for b in workers if b.get("ref3d") is not None]
    any_ref = any(b.get("ref3d") is not None for f in gt.frames for b in f["boxes"])
    site = m.get("site")
    thr = COVERAGE_THR.get(site, 0.90)
    if not any_ref:
        r.W("no ref3d on any box yet: LiDAR coverage / range-jump / mono checks skipped")
        r.stats["ref3d_coverage"] = None
    else:
        good = sum(1 for b in with_ref if b["ref3d"].get("quality") != "none")
        cov = good / max(len(workers), 1)
        r.stats["ref3d_coverage"] = round(cov, 3)
        if cov < thr:
            r.E(f"LiDAR-box coverage {cov:.1%} < {thr:.0%} ({good}/{len(workers)} non-ignore worker boxes, site {site})")
        else:
            r.I(f"LiDAR-box coverage {cov:.1%} ({good}/{len(workers)})")
        frac = jumps / max(pairs, 1)
        r.stats["range_jump_frac"] = round(frac, 4)
        for s in jump_list[:max_list]:
            r.W(s)
        if frac > 0.02:
            r.E(f"range jumps > 2 m on {jumps}/{pairs} adjacent pairs = {frac:.1%} > 2 %")
        for s in mono_list[:max_list]:
            r.W(s)
        r.stats["mono_gap_boxes"] = len(mono_list)

    # ------------------------------------------------------------- stamps
    st = m.get("stamps")
    if not st:
        r.W("meta.stamps missing: stamp check skipped")
    else:
        st = np.asarray(st, float)
        errs = np.array([np.abs(st - f["stamp"]).min() for f in gt.frames])
        bad = np.flatnonzero(errs > stamp_tol)
        r.stats["stamp_err_max_ms"] = round(float(errs.max() * 1e3), 2) if len(errs) else 0.0
        if len(bad):
            r.E(f"{len(bad)} frames without a pipeline stamp within {stamp_tol * 1e3:.0f} ms (e.g. f{bad[0]} "
                f"{errs[bad[0]] * 1e3:.1f} ms)")
        if len(st) != N:
            r.W(f"meta.stamps has {len(st)} entries, frames {N}")
        d = np.diff([f["stamp"] for f in gt.frames])
        if len(d) and (d <= 0).any():
            r.E("frame stamps not strictly increasing")

    # ------------------------------------------------------------- meta / ego_state
    if v2:
        for k, allowed in (("split", SPLITS), ("site", SITES), ("radar", RADARS), ("lighting", LIGHTINGS)):
            if m.get(k) not in allowed:
                r.E(f"meta.{k} {m.get(k)!r} invalid")
        for k in ("mount", "imu_topic", "image_size"):
            if m.get(k) is None:
                r.W(f"meta.{k} missing")
        if not m.get("provenance"):
            r.W("meta.provenance empty")
    ego = m.get("ego_state")
    if not ego:
        (r.E if v2 else r.W)("meta.ego_state missing")
    else:
        segs = sorted([list(s) for s in ego])
        cov = np.zeros(N, int)
        for f0, f1, state in segs:
            if state not in EGO_STATES:
                r.E(f"ego_state {[f0, f1, state]}: state invalid")
            if f1 - f0 + 1 < rate * 1.0 - 1e-6:
                r.E(f"ego_state {[f0, f1, state]}: {f1 - f0 + 1} frames < 1 s ({rate:.2f} Hz)")
            cov[max(f0, 0):min(f1, N - 1) + 1] += 1
        if (cov == 0).any():
            r.E(f"ego_state leaves {(cov == 0).sum()} frames uncovered (first f{int(np.flatnonzero(cov == 0)[0])})")
        if (cov > 1).any():
            r.E(f"ego_state overlaps on {(cov > 1).sum()} frames")
        r.stats["ego_state"] = segs
    r.stats["notes_open"] = sum(1 for n in gt.notes if n.get("status", "open") == "open")
    r.stats["notes"] = len(gt.notes)
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("--strict", action="store_true", help="exit 1 on errors")
    ap.add_argument("--json", help="write the report as json")
    ap.add_argument("--stamp-tol", type=float, default=0.02)
    ap.add_argument("--max-list", type=int, default=20, help="max listed items per warning type")
    a = ap.parse_args()
    gt = GT(Path(a.seg) / "gt.json")
    rep = check(gt, stamp_tol=a.stamp_tol, max_list=a.max_list)
    name = Path(a.seg).name
    print(f"== {name}  v{gt.schema_version} split={gt.meta.get('split')} site={gt.meta.get('site')}  {rep.stats}")
    for s in rep.errors:
        print("  E", s)
    for s in rep.warnings:
        print("  W", s)
    for s in rep.info:
        print("  i", s)
    print(f"-- {len(rep.errors)} errors, {len(rep.warnings)} warnings")
    if a.json:
        json.dump({"seg": name, **rep.to_dict()}, open(a.json, "w"), indent=1)
    if a.strict and rep.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
