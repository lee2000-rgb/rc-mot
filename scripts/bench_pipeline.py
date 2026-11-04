#!/usr/bin/env python3
"""Headless pipeline benchmark on a bag (docs/18): per-stage ms and achieved FPS, no GUI.

  python3 scripts/bench_pipeline.py --dataset boryeong_halt-short --duration 60 [--weights models/yolo11s-seg.engine]
  python3 scripts/bench_pipeline.py --bag <dir> --config <yaml> --calib <json> --start 85 --duration 60
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.pipeline import build_pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--bag")
    ap.add_argument("--config")
    ap.add_argument("--calib")
    ap.add_argument("--weights")
    ap.add_argument("--start", type=float)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--set", action="append", default=[], help="cfg override a.b=c")
    a = ap.parse_args()
    if a.dataset:
        d = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))["datasets"][a.dataset]
        bag, config, calib = d["bag"], d["config"], d["calib"]
        start = float(d.get("start", 0.0)) if a.start is None else a.start
    else:
        bag, config, calib, start = a.bag, a.config, a.calib, a.start or 0.0
    rel = lambda p: p if Path(p).is_absolute() else str(ROOT / p)  # noqa: E731
    cfg = yaml.safe_load(open(rel(config)))
    for kv in a.set:
        k, v = kv.split("=", 1)
        dd = cfg
        for part in k.split(".")[:-1]:
            dd = dd.setdefault(part, {})
        dd[k.split(".")[-1]] = yaml.safe_load(v)
    cal = RigCalib.load(rel(calib))
    pipe = build_pipeline(cfg, cal, ROOT, a.weights)
    tp = cfg["topics"]
    tm, n, t_first, t_last = defaultdict(list), 0, None, None
    wall0 = time.perf_counter()
    for img, rf, yaw in iterate_bag(rel(bag), cal, tp["radar"], tp["image"], tp.get("imu"), start=start, duration=a.duration):
        t0 = time.perf_counter()
        res = pipe.step(img, rf, yaw)
        tm["total"].append((time.perf_counter() - t0) * 1e3)
        for k, v in res.timing_ms.items():
            tm[k].append(v)
        n += 1
        t_first = rf.stamp if t_first is None else t_first
        t_last = rf.stamp
    wall = time.perf_counter() - wall0
    print(f"frames {n} over {t_last - t_first:.1f} s of bag, wall {wall:.1f} s (incl. bag decode) -> {n / wall:.1f} FPS end-to-end")
    for k in ("radar", "detector", "fusion", "reid", "tracking", "total"):
        if k in tm and len(tm[k]) > 5:
            v = np.array(tm[k][5:])
            print(f"  {k:9s} median {np.median(v):6.1f} ms  p90 {np.percentile(v, 90):6.1f}  max {v.max():6.1f}")
    v = np.array(tm["total"][5:])
    print(f"  pipeline-only FPS (median) {1000 / np.median(v):.1f}, (p90) {1000 / np.percentile(v, 90):.1f}")
    print(f"  detector: {a.weights or pipe.cfg['detector']['weights']}  imgsz {pipe.cfg['detector']['imgsz']}  reid {cfg.get('reid', {}).get('enable')}")


if __name__ == "__main__":
    main()
