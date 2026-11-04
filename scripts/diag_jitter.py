#!/usr/bin/env python3
"""Why do standing people look like they move? (docs/16)

Runs the pipeline over a bag window and logs per frame: the global image motion between consecutive frames
(phase correlation on the upper image half = background / horizon, in px), the ego displacement / yaw the
tracker removed, the radar ego speed, the IMU yaw rate, and per confirmed track its displayed speed and
position. Summary: image shake (px, and as bearing at the focal length), track speed / position jitter of
camera-measured tracks, and how much of the track motion follows the ego terms.

  python3 scripts/diag_jitter.py --bag <dir> --calib <json> --config <yaml> --start 100 --duration 120 \
      --out <json> [--set ego_comp=false] [--no-imu]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_fusion_depth import set_cfg  # noqa: E402
from rcmot.pipeline import build_pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


def image_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[float, float, float]:
    """(dx, dy, response) of the global translation between two frames, upper half (far background)."""
    h = prev.shape[0] // 2
    a = np.float32(cv2.cvtColor(prev[:h], cv2.COLOR_BGR2GRAY))
    b = np.float32(cv2.cvtColor(cur[:h], cv2.COLOR_BGR2GRAY))
    win = cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--no-imu", action="store_true", help="yaw rate = 0 (no IMU)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        set_cfg(cfg, kv)
    cal = RigCalib.load(a.calib)
    pipe = build_pipeline(cfg, cal, ROOT)
    tp = cfg["topics"]
    frames, prev = [], None
    for img, rf, yaw in iterate_bag(a.bag, cal, tp["radar"], tp["image"], None if a.no_imu else tp.get("imu"),
                                    start=a.start, duration=a.duration):
        res = pipe.step(img, rf, yaw)
        sh = image_shift(prev, img) if prev is not None else (0.0, 0.0, 0.0)
        prev = img
        frames.append({"t": rf.stamp, "dx": sh[0], "dy": sh[1], "resp": sh[2], "yaw": yaw,
                       "ego": [float(x) for x in res.ego_vel[:2]], "ego_valid": bool(res.ego_valid),
                       "tracks": [{"id": int(t.tid), "cls": t.cls, "x": float(t.pos[0]), "y": float(t.pos[1]),
                                   "v": float(np.linalg.norm(t.vel[:2])), "vx": float(t.vel[0]), "vy": float(t.vel[1]),
                                   "u": None if t.box2d is None else float(0.5 * (t.box2d[0] + t.box2d[2])), "n_radar": int(t.n_radar), "static": bool(getattr(t, "static", False)),
                                   "vr": None if not hasattr(t, "vr") else t.vr, "meas": bool(t.time_since_update == 0),
                                   "src": t.source} for t in res.tracks]})
    fx = float(cal.K[0, 0])
    dx = np.array([f["dx"] for f in frames[1:]])
    dy = np.array([f["dy"] for f in frames[1:]])
    ego = np.array([np.linalg.norm(f["ego"]) for f in frames])
    yawr = np.array([f["yaw"] for f in frames])
    v = np.array([t["v"] for f in frames for t in f["tracks"] if t["cls"] == "worker" and t["meas"]])
    # per-track position jitter: std of the frame-to-frame lateral / radial step of camera-measured worker tracks
    steps = []
    last = {}
    for f in frames:
        for t in f["tracks"]:
            if t["cls"] != "worker" or not t["meas"]:
                continue
            if t["id"] in last:
                steps.append(np.hypot(t["x"] - last[t["id"]][0], t["y"] - last[t["id"]][1]))
            last[t["id"]] = (t["x"], t["y"])
    summ = {"frames": len(frames), "set": a.set, "no_imu": a.no_imu,
            "img_shift_px_p50_p90_p99": [round(float(np.percentile(np.hypot(dx, dy), q)), 2) for q in (50, 90, 99)],
            "img_dy_std_px": round(float(dy.std()), 2), "img_dx_std_px": round(float(dx.std()), 2),
            "img_shift_deg_p90": round(float(np.degrees(np.percentile(np.hypot(dx, dy), 90) / fx)), 3),
            "ego_speed_p50_p90": [round(float(np.percentile(ego, q)), 3) for q in (50, 90)],
            "moving_frac(ego>0.2)": round(float((ego > 0.2).mean()), 3),
            "yaw_rate_std_dps": round(float(np.degrees(yawr.std())), 3),
            "worker_speed_p50_p90": [round(float(np.percentile(v, q)), 3) for q in (50, 90)] if len(v) else None,
            "worker_shown_moving_frac(v>0.3)": round(float((v > 0.3).mean()), 3) if len(v) else None,
            "worker_step_m_p50_p90": [round(float(np.percentile(steps, q)), 3) for q in (50, 90)] if steps else None}
    print(json.dumps(summ, indent=1))
    json.dump({"summary": summ, "frames": frames}, open(a.out, "w"))


if __name__ == "__main__":
    main()
