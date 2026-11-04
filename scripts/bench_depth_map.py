#!/usr/bin/env python3
"""Per-stage timing of the metric depth map (rcmot.fusion.depth_map.MetricDepth) on real frames of a labelled
segment: DA-V2 preprocessing / network / resize, static-point projection + disparity RANSAC, (s,t) KF + depth map,
and range_for_box() over the GT worker boxes of the frame.

  PYTHONPATH=src python3 scripts/bench_depth_map.py data/labels/cf_vib_100 [--frames 40] [--size 518x924] [--encoder vits]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.fusion.depth_map import DepthMapParams, MetricDepth  # noqa: E402
from rcmot.labeling.gt import GT  # noqa: E402
from rcmot.pipeline import Pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


class _NoDet:
    def __call__(self, img):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--size", default="518x924", help="network input HxW")
    ap.add_argument("--encoder", default="vits")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n-frames", type=int, default=3)
    a = ap.parse_args()
    gt = GT(Path(a.seg) / "gt.json")
    m = gt.meta
    cfg = yaml.safe_load(open(ROOT / m["config"]))
    cfg.setdefault("recovery", {})["enable"] = False
    cal = RigCalib.load(str(ROOT / m["calib"]))
    pipe = Pipeline(_NoDet(), cal, cfg)
    h, w = (int(v) for v in a.size.lower().split("x"))
    kw = dict(encoder=a.encoder, input_hw=(h, w), n_frames=a.n_frames)
    if a.weights:
        kw["weights"] = a.weights
    md = MetricDepth(cal, DepthMapParams(**kw), root=ROOT)
    want = {f"{f['stamp']:.6f}": f for f in gt.frames}
    rows, box_ms, n_boxes = [], [], 0
    import time
    for img, rf, yaw in iterate_bag(ROOT / m["bag"], cal, cfg["topics"]["radar"], cfg["topics"]["image"], cfg["topics"].get("imu"),
                                    start=max(m["start"] - 1.0, 0), duration=m["end"] - m["start"] + 1.5):
        res = pipe.step(img, rf, yaw)
        f = want.get(f"{rf.stamp:.6f}")
        if f is None:
            continue
        st = rf.stats
        stat = st.static_inlier & st.keep
        if st.below_ground is not None:
            stat &= ~st.below_ground
        r = md.update(img, rf.xyz[stat], res.world, stamp=rf.stamp)
        t0 = time.perf_counter()
        boxes = [b for b in f["boxes"] if b["cls"] == "worker" and "ignore" not in b["flags"]]
        for b in boxes:
            md.range_for_box(b["xyxy"])
        n_boxes += len(boxes)
        if boxes:
            box_ms.append((time.perf_counter() - t0) * 1e3 / len(boxes))
        rows.append(dict(r.ms, n_static=r.n_static, n_in=0 if r.fit is None else r.fit.n_in))
        if len(rows) >= a.frames + 1:
            break
    rows = rows[1:]                                               # first frame holds the model load / warm-up
    print(f"{a.encoder} {h}x{w} fp16, RTX: {len(rows)} frames of {Path(a.seg).name}, static points/frame median "
          f"{np.median([r['n_static'] for r in rows]):.0f} (inliers {np.median([r['n_in'] for r in rows]):.0f})")
    print(f"{'stage':<28}{'median ms':>10}{'p90 ms':>10}")
    for k, name in (("pre", "resize + normalise"), ("net", "DA-V2 network"), ("post", "to image resolution"),
                    ("fit", "project + sample + RANSAC"), ("map", "KF + metric map")):
        v = np.array([r[k] for r in rows])
        print(f"{name:<28}{np.median(v):>10.1f}{np.percentile(v, 90):>10.1f}")
    tot = np.array([sum(r[k] for k in ("pre", "net", "post", "fit", "map")) for r in rows])
    print(f"{'total per frame':<28}{np.median(tot):>10.1f}{np.percentile(tot, 90):>10.1f}")
    if box_ms:
        print(f"{'range_for_box (per box)':<28}{np.median(box_ms):>10.2f}{np.percentile(box_ms, 90):>10.2f}   ({n_boxes} boxes)")


if __name__ == "__main__":
    main()
