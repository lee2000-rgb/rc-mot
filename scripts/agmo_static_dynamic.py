#!/usr/bin/env python3
"""(c)+(e) for the agmo rig: re-read one radar topic (read-only, streaming) and, using the series from
agmo_extract.py, compare
  (e) dynamic-point labelling with the radar's own ego velocity (pipeline: ScanStats + EMA) vs. an ego velocity
      predicted from GNSS/odom + gyro (lever arm, mount yaw, time lag from agmo_analyze.py);
  (c) multi-scan static-map consistency: static points of scan k vs. the static points of the previous
      0.3-1.0 s, accumulated in a world frame with different pose sources (median 2D nearest-neighbour distance).

usage: PYTHONPATH=src python3 scripts/agmo_static_dynamic.py BAG NPZ --psi -1.69 --lever 1.94 --lag -0.125
"""
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.ndimage import uniform_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

from rcmot.radar.scan_stats import ScanStatistician, ScanStatsParams
from rcmot.sensors.bag_source import pc2_to_struct, radar_frame
from rcmot.utils.calib import RigCalib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("npz")
    ap.add_argument("--topic", default="/radar_mid/driver/locations")
    ap.add_argument("--psi", type=float, required=True, help="radar mount yaw [deg] (Doppler-fitted)")
    ap.add_argument("--lever", type=float, required=True, help="radar x offset from the GNSS/odom point [m]")
    ap.add_argument("--lag", type=float, required=True, help="gnss/imu time = radar stamp + lag [s]")
    ap.add_argument("--max-s", type=float, default=None)
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    od, im = d["/agmo/odom"], d["/agmo/imu"]
    th = od[:, 1]
    yaw = np.unwrap(Rot.from_quat(od[:, 5:9]).as_euler("xyz")[:, 2])
    wz5 = uniform_filter1d(im[:, 4], 5)
    wz1 = uniform_filter1d(im[:, 4], 50)
    st = d["str:/agmo/gnss/status"]
    ts = np.array([float(s.split("|")[0]) for s in st])
    rtk = np.array([int(s.split("rtk=")[1][0]) for s in st])
    rk = rtk[np.clip(np.searchsorted(ts, od[:, 0]), 0, len(ts) - 1)]
    psi = np.radians(a.psi)
    Rz = np.array([[np.cos(psi), -np.sin(psi)], [np.sin(psi), np.cos(psi)]])     # radar -> body (2D)

    ident = RigCalib(np.eye(3), np.zeros(5), (640, 400), np.eye(3), np.zeros(3), np.eye(3), np.zeros(3))
    stats = ScanStatistician(ScanStatsParams(), seed=0)
    p = stats.p
    ema = np.zeros(3)
    rows, scans = [], []
    with AnyReader([Path(a.bag)]) as r:
        conns = [c for c in r.connections if c.topic == a.topic]
        t_end = r.start_time + (a.max_s * 1e9 if a.max_s else 1e30)
        for c, t_ns, raw in r.messages(connections=conns):
            if t_ns > t_end:
                break
            m = r.deserialize(raw, c.msgtype)
            rf = radar_frame(pc2_to_struct(m), m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, ident)
            S = stats.compute(rf.stamp, rf.xyz_radar, rf.v_r, rf.snr, rf.rcs, rf.peer)
            if S.ego_valid:
                ema = 0.6 * ema + 0.4 * S.v_sensor
            tq = rf.stamp + a.lag
            if len(rf.v_r) == 0 or tq < th[0] or tq > th[-1]:
                continue
            vb = np.array([np.interp(tq, th, od[:, 9]), a.lever * np.interp(tq, th, wz5)])
            v_ref = np.zeros(3)
            v_ref[:2] = Rz.T @ vb                                                       # body -> radar
            D = rf.xyz_radar / np.linalg.norm(rf.xyz_radar, axis=1, keepdims=True)
            sig = S.sigma_v if np.isfinite(S.sigma_v) else p.sigma_v_floor

            def ndyn(vs):
                vc = rf.v_r + D @ vs
                return int(np.sum(S.keep & (np.abs(vc / sig) > p.dyn_z) & (np.abs(vc) > p.dyn_vmin)))

            spd = abs(np.interp(tq, th, od[:, 9]))
            rows.append([rf.stamp, spd, np.degrees(abs(np.interp(tq, th, wz1))), rk[min(np.searchsorted(th, tq), len(th) - 1)],
                         len(rf.v_r), S.keep.sum(), ndyn(S.v_sensor), ndyn(ema), ndyn(v_ref), ndyn(np.zeros(3)), sig,
                         float(S.ego_valid)])
            stat = S.static_inlier & (np.linalg.norm(rf.xyz_radar[:, :2], axis=1) < 25)
            scans.append((rf.stamp, rf.xyz_radar[stat, :2].astype(np.float32), S.v_sensor[:2].copy()))
    R_ = np.array(rows)
    t0 = th[0]
    mv = R_[:, 1] > 0.1
    names = ["radar ego (per scan)", "radar ego EMA0.4 (pipeline)", "GNSS/odom+gyro ego", "no ego compensation"]
    print(f"## (e) dynamic points per scan [{a.topic}], keep-filtered points p50 {np.median(R_[:, 5]):.0f}; scan sigma_v p50 {np.median(R_[mv, 10]):.3f} m/s (moving)")
    for lab, sel in [("moving all", mv), ("moving RTK FIXED", mv & (R_[:, 3] == 4)), ("moving DGNSS", mv & (R_[:, 3] == 2)),
                     ("moving straight |wz|<2", mv & (R_[:, 2] < 2)), ("turning |wz|>=5", mv & (R_[:, 2] >= 5)), ("static", ~mv)]:
        if not sel.any():
            continue
        cells = []
        for j, nm in enumerate(names):
            x = R_[sel, 6 + j] / np.maximum(R_[sel, 5], 1)
            cells.append(f"{nm}: {100 * np.mean(x):.1f}% (p95 {100 * np.percentile(x, 95):.0f}%)")
        print(f"  {lab:24s} n={sel.sum():5d} | " + " | ".join(cells))

    # (c) static map consistency
    print("\n## (c) static-map consistency: median 2D NN distance [m] of scan-k static points to the static points of the previous 0.3-1.0 s")
    tt = np.array([s[0] for s in scans])
    tq = tt + a.lag
    ok = (tq > th[0]) & (tq < th[-1])
    pose = {}
    x_o, y_o, ya = np.interp(tq, th, od[:, 2]), np.interp(tq, th, od[:, 3]), np.interp(tq, th, yaw)
    pose["odom (RTK pos + yaw_deg), lag-corrected"] = (x_o, y_o, ya)
    pose["odom, no lag correction"] = (np.interp(tt, th, od[:, 2]), np.interp(tt, th, od[:, 3]), np.interp(tt, th, yaw))
    # radar + gyro dead reckoning, re-anchored to odom every 10 s
    wz_t = np.interp(tq, th, im[:, 4])
    vbr = np.array([Rz @ s[2] for s in scans])
    vbr[:, 1] -= a.lever * np.interp(tq, th, wz5)
    xd, yd, yawd = np.zeros(len(tt)), np.zeros(len(tt)), np.zeros(len(tt))
    for i in range(len(tt)):
        if i == 0 or tt[i] - anchor >= 10.0:
            anchor = tt[i]
            xd[i], yd[i], yawd[i] = x_o[i], y_o[i], ya[i]
            continue
        dt = tt[i] - tt[i - 1]
        yawd[i] = yawd[i - 1] + wz_t[i] * dt
        c, s_ = np.cos(yawd[i - 1]), np.sin(yawd[i - 1])
        xd[i] = xd[i - 1] + (c * vbr[i, 0] - s_ * vbr[i, 1]) * dt
        yd[i] = yd[i - 1] + (s_ * vbr[i, 0] + c * vbr[i, 1]) * dt
    pose["radar ego + gyro dead reckoning (anchored every 10 s)"] = (xd, yd, yawd)
    pose["radar ego only, yaw from IMU, no GNSS"] = pose["radar ego + gyro dead reckoning (anchored every 10 s)"]
    del pose["radar ego only, yaw from IMU, no GNSS"]
    pose["translation from odom, NO yaw update (yaw frozen per 10 s)"] = (x_o, y_o, np.array([ya[np.searchsorted(tt, tt[i] - ((tt[i] - tt[0]) % 10.0))] for i in range(len(tt))]))
    pose["no ego motion (sensor frame)"] = (np.zeros(len(tt)), np.zeros(len(tt)), np.zeros(len(tt)))
    spd = np.abs(np.interp(tq, th, od[:, 9]))
    awz = np.degrees(np.abs(np.interp(tq, th, wz1)))
    rkk = rk[np.clip(np.searchsorted(th, tq), 0, len(th) - 1)]
    lever = np.array([a.lever, 0.0])
    res = {k: [] for k in pose}
    idx = np.flatnonzero(ok & (spd > 0.1))[::3]
    for k, (X, Y, Yw) in pose.items():
        def world(i):
            P = scans[i][1] @ Rz.T + lever
            c, s_ = np.cos(Yw[i]), np.sin(Yw[i])
            return P @ np.array([[c, s_], [-s_, c]]) + np.array([X[i], Y[i]])
        for i in idx:
            prev = np.flatnonzero((tt > tt[i] - 1.0) & (tt < tt[i] - 0.3))
            if len(prev) < 5 or len(scans[i][1]) < 10:
                continue
            M = np.concatenate([world(j) for j in prev])
            dist, _ = cKDTree(M).query(world(i))
            res[k].append((i, np.median(dist)))
    for lab, f in [("all moving", lambda i: True), ("straight |wz|<2", lambda i: awz[i] < 2), ("turning |wz|>=5", lambda i: awz[i] >= 5),
                   ("RTK FIXED", lambda i: rkk[i] == 4), ("DGNSS", lambda i: rkk[i] == 2)]:
        line = []
        for k in pose:
            v = [m_ for i, m_ in res[k] if f(i)]
            if v:
                line.append(f"{k}: {np.median(v):.3f}")
        if line:
            print(f"  {lab:16s} n={len([1 for i, _ in res[next(iter(pose))] if f(i)]):4d} | " + " | ".join(line))


if __name__ == "__main__":
    main()
