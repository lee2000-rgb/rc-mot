#!/usr/bin/env python3
"""Validate tracked absolute velocities against a LiDAR reference (rough ground truth).

  python3 scripts/eval_velocity_lidar.py --bag data/bags/hdx_boryeong_pg/<bag> --calib configs/calib/<json> \
      [--start 0 --duration 30] [--set fusion.velocity_mode=lsq ...] [--tag name]

Reference: per frame, the LiDAR points (level frame, > 0.3 m above ground) around each confirmed track are
grouped (0.5 m connectivity) and the group nearest to the track gives a centroid. Centroids are moved into a
fixed frame by integrating the same ego motion the tracker uses (radar ego velocity + IMU yaw rate), and a
straight line over 1 s windows gives the LiDAR velocity. Compared with:
  * the track velocity at the window centre (absolute, rotated into the same frame)
The LiDAR centroid of a walking person wobbles with the gait; 1 s regression averages that out.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.calib.camera_lidar import header_clock_shift, lidar_clock_offset  # noqa: E402
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.detection.yolo2d import Yolo2D  # noqa: E402
from rcmot.pipeline import Pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag, pc2_to_struct  # noqa: E402
from rcmot.types import TrackState  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402
from rcmot.utils.rig_extrinsics import sensor_to_level  # noqa: E402

RADIUS = {"worker": 0.8, "heavy_equipment": 3.0}


class LidarStream:
    """Time-ordered LiDAR scans in the level frame, matched to a requested time (ROS clock)."""

    def __init__(self, bag, calib_json, topic="/ouster/points", radar_topic="/off_highway_premium_radar_sample_driver/locations"):
        """radar_topic: the radar topic of the profile config (cfg["topics"]["radar"]); its header-clock shift is
        part of the LiDAR -> radar-time offset (Continental bags: /radar/radar_detection_image, sensor clock)."""
        self.r = AnyReader([Path(bag)])
        self.r.open()
        cs = {c.topic: c for c in self.r.connections}
        if topic not in cs:
            raise KeyError(f"LiDAR topic {topic} not in bag {bag}: {sorted(cs)}")
        if radar_topic not in cs:
            raise KeyError(f"radar topic {radar_topic} not in bag {bag} (pass the config's topics.radar): {sorted(cs)}")
        self.conn = cs[topic]
        shift = header_clock_shift(self.r, cs[radar_topic])
        if abs(shift) > 1e5:        # the radar header is its own clock (Continental ARS548, ~3600 s since boot):
            shift = 0.0             # bag_source maps those stamps onto the bag timeline, so nothing to subtract
        self.off = lidar_clock_offset(self.r, self.conn) - shift + 0.05
        from rcmot.calib.camera_lidar import _tf_static
        self.tf = _tf_static(self.r, cs)
        self.R_lev, self.t_lev = sensor_to_level(calib_json)         # HDX (`extrinsics`) or xpanner (`T_base_sensor` + `R_level`) layout
        self.it = self.r.messages(connections=[self.conn])
        self.buf = []

    SWEEP = 0.1          # Ouster sweep [s]; the scan stamp used here (header + off) is the sweep middle

    def at(self, t: float, yaw_rate: float | None = None):
        """Level-frame points (> 0.3 m above ground) of the scan nearest to t and (scan mid time - t).
        yaw_rate [rad/s, level z]: de-skew for ego rotation - every point is rotated about z by
        yaw_rate * (t_point - t), t_point from the per-point Ouster field `t` (ns since sweep start), so the
        cloud is expressed in the level frame at the frame time t (swing at 56 deg/s smears a 100 ms sweep by
        5.6 deg otherwise; docs/13 v2). Without the field the whole scan is rotated by yaw_rate * (scan mid - t)."""
        while not self.buf or self.buf[-1][0] < t + 0.1:
            try:
                _, _, raw = next(self.it)
            except StopIteration:
                break
            m = self.r.deserialize(raw, self.conn.msgtype)
            self.buf.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 + self.off, m))
            self.buf = self.buf[-4:]
        if not self.buf:
            return None, np.inf
        ts, m = min(self.buf, key=lambda b: abs(b[0] - t))
        a = pc2_to_struct(m)
        xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
        keep = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1.0)
        xyz = xyz[keep]
        R, tt = lidar_to_sensor(self.tf, m.header.frame_id)
        lv = (xyz @ R.T + tt) @ self.R_lev.T + self.t_lev
        if yaw_rate is not None and abs(yaw_rate) > 1e-6:
            if "t" in a.dtype.names:
                tp = ts - 0.5 * self.SWEEP + a["t"][keep].astype(float) * 1e-9
            else:
                tp = np.full(len(lv), ts)
            ang = yaw_rate * (tp - t)
            c, s = np.cos(ang), np.sin(ang)
            x, y = lv[:, 0].copy(), lv[:, 1].copy()
            lv[:, 0], lv[:, 1] = c * x - s * y, s * x + c * y
        return lv[lv[:, 2] > 0.3], ts - t


def centroid_near(pts, tree, pos, radius):
    idx = tree.query_ball_point(pos[:2], radius)
    if len(idx) < 5:
        return None
    q = pts[idx]
    pairs = cKDTree(q[:, :2]).query_pairs(0.5, output_type="ndarray")
    n = len(q)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    lab = connected_components(g, directed=False)[1]
    best = min(np.unique(lab), key=lambda L: np.linalg.norm(q[lab == L, :2].mean(0) - pos[:2]))
    c = q[lab == best]
    return c[:, :2].mean(0) if len(c) >= 5 else None


def set_cfg(cfg, kv):
    k, v = kv.split("=", 1)
    d = cfg
    for part in k.split(".")[:-1]:
        d = d.setdefault(part, {})
    d[k.split(".")[-1]] = yaml.safe_load(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/hdx_boryeong.yaml"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", help="append a json line with the summary")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for kv in a.set:
        set_cfg(cfg, kv)
    calib_json = json.load(open(a.calib))
    cal = RigCalib.load(a.calib)
    dc = cfg["detector"]
    pipe = Pipeline(Yolo2D(str(ROOT / dc["weights"]), dc["class_map"], imgsz=dc["imgsz"], conf=dc["conf"]), cal, cfg)
    lid = LidarStream(a.bag, calib_json, radar_topic=cfg["topics"]["radar"])

    pose = np.zeros(3)                         # world x, y, yaw of the base frame
    last_t = None
    rec = defaultdict(list)                    # tid -> [(t, lidar_xy_world, track_v_world, cls)]
    for img, rf, yaw in iterate_bag(a.bag, cal, cfg["topics"]["radar"], cfg["topics"]["image"], cfg["topics"].get("imu"),
                                    start=a.start, duration=a.duration):
        res = pipe.step(img, rf, yaw)
        dt = 0.066 if last_t is None else rf.stamp - last_t
        last_t = rf.stamp
        c, s = np.cos(pose[2]), np.sin(pose[2])
        Rw = np.array([[c, -s], [s, c]])
        pose[:2] += Rw @ res.ego_vel[:2] * dt
        pose[2] += yaw * dt
        c, s = np.cos(pose[2]), np.sin(pose[2])
        Rw = np.array([[c, -s], [s, c]])
        pts, dts = lid.at(rf.stamp)
        if pts is None or abs(dts) > 0.06 or len(pts) == 0:
            continue
        tree = cKDTree(pts[:, :2])
        for t in res.tracks:
            if t.state != TrackState.CONFIRMED:
                continue
            cxy = centroid_near(pts, tree, t.pos, RADIUS.get(t.cls, 1.5))
            if cxy is not None:
                rec[t.tid].append((rf.stamp, Rw @ cxy + pose[:2], Rw @ t.vel[:2], t.cls))

    errs, sp_err, by_cls = [], [], defaultdict(list)
    for tid, rows in rec.items():
        ts = np.array([r[0] for r in rows])
        P = np.array([r[1] for r in rows])
        V = np.array([r[2] for r in rows])
        for i in range(len(rows)):
            m = np.abs(ts - ts[i]) <= a.window / 2
            if m.sum() < 6 or np.ptp(ts[m]) < 0.7 * a.window:
                continue
            A = np.c_[ts[m] - ts[i], np.ones(m.sum())]
            vl = np.linalg.lstsq(A, P[m], rcond=None)[0][0]
            e = V[i] - vl
            errs.append(e)
            sp_err.append(np.linalg.norm(V[i]) - np.linalg.norm(vl))
            by_cls[rows[i][3]].append((np.linalg.norm(e), np.linalg.norm(vl)))
    errs = np.array(errs)
    summ = {"tag": a.tag, "bag": Path(a.bag).name, "start": a.start, "duration": a.duration, "set": a.set,
            "n_windows": int(len(errs)), "n_tracks": len(rec)}
    if len(errs):
        en = np.linalg.norm(errs, axis=1)
        summ.update({"vel_err_rmse": float(np.sqrt(np.mean(en ** 2))), "vel_err_median": float(np.median(en)),
                     "vel_err_p90": float(np.percentile(en, 90)), "speed_bias": float(np.mean(sp_err)),
                     "per_class": {k: {"n": len(v), "median_err": float(np.median([x[0] for x in v])),
                                       "median_lidar_speed": float(np.median([x[1] for x in v]))} for k, v in by_cls.items()}})
        # error split by the LiDAR speed: standing (< 0.2), slow, walking (docs/16 standstill mode)
        by_speed = {}
        for cls, v in by_cls.items():
            v = np.array(v)
            for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 9.0)):
                m = (v[:, 1] >= lo) & (v[:, 1] < hi)
                if m.sum():
                    by_speed[f"{cls} lidar {lo}-{hi}"] = {"n": int(m.sum()), "err_median": round(float(np.median(v[m, 0])), 3),
                                                          "err_p90": round(float(np.percentile(v[m, 0], 90)), 3)}
        summ["by_speed"] = by_speed
    print(json.dumps(summ, indent=1))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(summ) + "\n")


if __name__ == "__main__":
    main()
