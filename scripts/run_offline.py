#!/usr/bin/env python3
"""Run the RC-MOT pipeline on a ROS 2 bag headless (--no-gui); without --no-gui it opens scripts/rcmot_app.py.

  PYTHONPATH=src python3 scripts/run_offline.py --bag ~/rosbag/04.callibration_test \
      --radar-topic /off_highway_premium_radar_sample_driver_left/locations
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcmot.pipeline import build_pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--calib", help="calib json overriding the config (e.g. another recording session)")
    ap.add_argument("--radar-topic")
    ap.add_argument("--image-topic")
    ap.add_argument("--weights")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()

    if not a.no_gui:                                       # GUI replay (seekable) lives in rcmot_app.py
        from rcmot_app import main as app_main

        sys.argv = [sys.argv[0], "--bag", a.bag, "--config", a.config, "--start", str(a.start)]
        for k in ("calib", "radar_topic", "image_topic", "weights"):
            if getattr(a, k):
                sys.argv += [f"--{k.replace('_', '-')}", getattr(a, k)]
        return app_main()

    cfg = yaml.safe_load(open(a.config))
    calib = RigCalib.load(a.calib or str((Path(a.config).parent / cfg["calib"]).resolve()))
    pipe = build_pipeline(cfg, calib, ROOT, a.weights)
    frames = iterate_bag(a.bag, calib, a.radar_topic or cfg["topics"]["radar"], a.image_topic or cfg["topics"]["image"],
                         cfg["topics"].get("imu"), start=a.start, duration=a.duration)
    n, t0 = 0, time.time()
    for img, radar, yaw in frames:
        res = pipe.step(img, radar, yaw)
        n += 1
        if n % 15 == 0:
            print(f"t={res.stamp:.2f} ego={res.ego_vel[:2].round(2)} dets={res.n_dets} "
                  f"tracks={[(t.tid, t.cls, round(float((t.vel[:2] ** 2).sum() ** 0.5), 2)) for t in res.tracks]}")
    print(f"{n} frames, {n / max(time.time() - t0, 1e-6):.1f} FPS")

if __name__ == "__main__":
    main()
