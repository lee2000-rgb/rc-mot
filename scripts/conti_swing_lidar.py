"""Pass 2 of the Continental swing analysis: LiDAR odometry (consecutive-scan ICP, IMU-deskewed, level frame)
over selected time windows; stores the per-scan pose chain and the downsampled non-ground points.

Usage: PYTHONPATH=src python scripts/conti_swing_lidar.py --bag <bag> --imu <bag>_imu.npz --windows t0,t1 [t0,t1 ...] --out <npz>
Times are seconds since bag start (bag receive clock). LiDAR scan times: Ouster header (sensor clock, sweep
start) + the IMU clock offset (median of IMU bag time - header; the IMU shares the Ouster clock) + per-point t.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.calib.camera_lidar import _tf_static                     # noqa: E402
from rcmot.calib.common import lidar_to_sensor                      # noqa: E402
from rcmot.sensors.bag_source import pc2_to_struct                   # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics                 # noqa: E402

LIDAR = "/ouster/points"


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def smooth_series(t, x, win):
    """Centred moving average (uniform samples assumed)."""
    n = max(1, int(round(win / np.median(np.diff(t)))))
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", default="configs/calib/hdx_boryeong_pg_conti1126.json")
    ap.add_argument("--imu", required=True)
    ap.add_argument("--windows", nargs="+", required=True, help="t0,t1 pairs [s since bag start]")
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--rmax", type=float, default=50.0)
    ap.add_argument("--zmin", type=float, default=0.3)
    ap.add_argument("--no-deskew", action="store_true")
    ap.add_argument("--p2l", action="store_true", help="point-to-plane ICP (normals from 1 m neighbourhoods)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wins = [tuple(map(float, w.split(","))) for w in a.windows]
    imu = np.load(a.imu)
    off = float(np.median(imu["t"] - imu["hdr"]))                 # sensor clock -> bag-relative seconds
    ext = RigExtrinsics.from_calib_json(json.load(open(a.calib)))
    R_lev, t_lev = ext.level()
    wz_lev = (np.stack([imu["wx"], imu["wy"], imu["wz"]], 1) @ R_lev.T)[:, 2]
    wz_s = smooth_series(imu["t"], wz_lev, 0.1)                   # 0.1 s average (one sweep)
    imu_t = imu["t"]

    scans = []        # (t_mid, pts level frame (deskewed), fitness, rmse, T 4x4 relative to the window start, win id)
    with AnyReader([Path(a.bag)]) as r:
        cs = {c.topic: c for c in r.connections}
        t0 = r.start_time * 1e-9
        tf = _tf_static(r, cs)
        conn = cs[LIDAR]
        for wi, (w0, w1) in enumerate(wins):
            prev, T, v_prev = None, np.eye(4), np.zeros(3)
            for _, t_ns, raw in r.messages(connections=[conn], start=int((t0 + w0 - a.margin - 0.5) * 1e9),
                                           stop=int((t0 + w1 + a.margin + 0.5) * 1e9)):
                m = r.deserialize(raw, conn.msgtype)
                hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                arr = pc2_to_struct(m)
                xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(float)
                tp = arr["t"].astype(float) * 1e-9
                ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1.5)
                xyz, tp = xyz[ok], tp[ok]
                Rl, tl = lidar_to_sensor(tf, m.header.frame_id)
                lv = (xyz @ Rl.T + tl) @ R_lev.T + t_lev
                t_start = hs + off
                t_mid = t_start + 0.05
                if t_mid < w0 - a.margin or t_mid > w1 + a.margin:
                    continue
                if not a.no_deskew:
                    w = float(np.interp(t_mid, imu_t, wz_s))
                    dt = tp - 0.05
                    c, s = np.cos(w * dt), np.sin(w * dt)
                    x, y = lv[:, 0].copy(), lv[:, 1].copy()
                    lv[:, 0] = c * x - s * y + v_prev[0] * dt
                    lv[:, 1] = s * x + c * y + v_prev[1] * dt
                keep = (lv[:, 2] > a.zmin) & (lv[:, 2] < 8.0) & (np.linalg.norm(lv[:, :2], axis=1) < a.rmax)
                pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(lv[keep])).voxel_down_sample(a.voxel)
                if a.p2l:
                    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
                est = o3d.pipelines.registration.TransformationEstimationPointToPlane() if a.p2l else o3d.pipelines.registration.TransformationEstimationPointToPoint()
                fit, rmse = 1.0, 0.0
                if prev is not None:
                    dtk = t_mid - prev[0]
                    dyaw = float(np.interp(t_mid - dtk / 2, imu_t, wz_s)) * dtk
                    init = np.eye(4)
                    init[:3, :3] = rotz(dyaw)
                    init[:3, 3] = v_prev * dtk
                    # pose of current scan in previous frame: p_prev = init @ p_cur
                    res = o3d.pipelines.registration.registration_icp(
                        pc, prev[1], 1.0, init,
                        est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                    res = o3d.pipelines.registration.registration_icp(
                        pc, prev[1], 0.4, res.transformation, est,
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
                    Trel = res.transformation.copy()
                    Trel[2, 3] = 0.0                                 # planar motion
                    yaw = np.arctan2(Trel[1, 0], Trel[0, 0])
                    Trel[:3, :3] = rotz(yaw)
                    T = T @ Trel
                    fit, rmse = res.fitness, res.inlier_rmse
                    v_prev = Trel[:3, 3] / max(dtk, 1e-3)            # velocity in current frame (approx)
                pts = np.asarray(pc.points).astype(np.float32)
                scans.append((t_mid, pts, fit, rmse, T.copy(), wi))
                prev = (t_mid, pc)
            print(f"window {wi} {w0:.1f}-{w1:.1f}: {sum(1 for s in scans if s[5] == wi)} scans", flush=True)

    np.savez_compressed(a.out, t=np.array([s[0] for s in scans]), fit=np.array([s[2] for s in scans]),
                        rmse=np.array([s[3] for s in scans]), T=np.stack([s[4] for s in scans]),
                        win=np.array([s[5] for s in scans]), n=np.array([len(s[1]) for s in scans]),
                        pts=np.concatenate([s[1] for s in scans]),
                        scan=np.concatenate([np.full(len(s[1]), i, np.int32) for i, s in enumerate(scans)]),
                        clock_off=off, windows=np.array(wins))
    print("saved", a.out, len(scans), "scans")


if __name__ == "__main__":
    main()
