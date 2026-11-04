#!/usr/bin/env python3
"""Dump the ego-motion increments the tracker actually applies (dp, dyaw per radar scan) for a bag, so
scripts/conti_swing_drift.py --ext can score the real pipeline against the LiDAR reference (docs/16 §10).

  CUDA_VISIBLE_DEVICES= python3 scripts/conti_pose_dump.py --bag <dir> --config <yaml> --calib <json> --out chain.json
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
from rcmot.pipeline import Pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


class _NoDet:
    def __call__(self, img):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg.setdefault("recovery", {})["enable"] = False
    cfg.setdefault("reid", {})["enable"] = False
    cal = RigCalib.load(a.calib)
    pipe = Pipeline(_NoDet(), cal, cfg)
    tp = cfg["topics"]
    rows = []
    last = None
    for img, rf, yaw in iterate_bag(a.bag, cal, tp["radar"], tp["image"], tp.get("imu")):
        res = pipe.step(img, rf, yaw)
        dt = 0.066 if last is None else max(rf.stamp - last, 1e-3)
        last = rf.stamp
        v = res.ego_vel if pipe.ego_comp else np.zeros(3)
        rows.append({"t_bag": rf.bag_t, "t": rf.stamp, "dt": dt, "dyaw": float(yaw * dt), "dp": [float(x) for x in v * dt],
                     "ego_valid": bool(res.ego_valid)})
    json.dump(rows, open(a.out, "w"))
    print("scans", len(rows), "->", a.out)


if __name__ == "__main__":
    main()
