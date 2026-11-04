"""Radar -> os_sensor extrinsic from stationary windows: joint point-to-plane ICP over many windows.

Each window (machine standing still for a few seconds) gives the accumulated LiDAR cloud (target, with
normals) and the Doppler-static radar points (source). One rigid transform (R, t) is shared by all windows
and found by robust (Cauchy) Gauss-Newton on the point-to-plane residuals n_q . (R p + t - q). Short night
stops (3-6 s each) are too sparse on their own; solved jointly they constrain the transform like one long
stop. Uncertainty: leave-one-window-out jackknife.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from rcmot.calib.common import DEFAULT_TOPICS, Topics, lidar_to_sensor
from rcmot.radar.scan_stats import local_contrast_polar, robust_z, combine_contrast
from rcmot.sensors.bag_source import pc2_to_struct



@dataclass
class Window:  # also carries .rcs (N,), .n_frames, .lidar_raw (accumulated LiDAR or None), set by load_window
    name: str
    q: np.ndarray                 # (M,3) LiDAR points, os_sensor, voxel-downsampled
    nq: np.ndarray                # (M,3) normals
    tree: cKDTree
    p: np.ndarray                 # (N,3) static radar points, radar frame


def load_window(bag: str, t0: float, t1: float, *, static_vr: float = 0.15, max_range: float = 60.0,
                voxel: float = 0.1, topics: Topics = DEFAULT_TOPICS, keep_raw: bool = False,
                max_scans: int = 40) -> Window:
    import open3d as o3d

    T_LIDAR, T_RADAR = topics.lidar, topics.radar
    L, R, RCS, CON, NFR = [], [], [], [], 0
    every = max(1, int(round((t1 - t0) * 10 / max_scans)))           # Ouster 10 Hz -> keep <= max_scans scans
    n_scan = 0
    with AnyReader([Path(bag)]) as r:
        cs = {c.topic: c for c in r.connections}
        tf = {}
        for c, _, raw in r.messages(connections=[cs["/tf_static"]]):
            for t in r.deserialize(raw, c.msgtype).transforms:
                q, tr = t.transform.rotation, t.transform.translation
                tf[(t.header.frame_id, t.child_frame_id)] = (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                                                             np.array([tr.x, tr.y, tr.z]))
        for c, _, raw in r.messages(connections=[cs[T_LIDAR], cs[T_RADAR]], start=r.start_time + int(t0 * 1e9),
                                    stop=r.start_time + int(t1 * 1e9)):
            if c.topic == T_LIDAR:
                n_scan += 1
                if (n_scan - 1) % every:
                    continue
            m = r.deserialize(raw, c.msgtype)
            a = pc2_to_struct(m)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
            n = np.linalg.norm(xyz, axis=1)
            k = np.isfinite(xyz).all(1) & (n > 1) & (n < max_range)
            if c.topic == T_LIDAR:
                Rm, t = lidar_to_sensor(tf, m.header.frame_id)
                L.append(xyz[k] @ Rm.T + t)
            else:
                ks = k & (np.abs(a["radial_velocity"]) < static_vr)
                R.append(xyz[ks])
                rcs = a["radar_cross_section"].astype(float) if "radar_cross_section" in a.dtype.names else np.zeros(len(a))
                snr = a["signal_noise_ratio"].astype(float) if "signal_noise_ratio" in a.dtype.names else rcs
                RCS.append(rcs[ks])
                # MCL2026 eq. (1): per-scan local contrast, min over RCS and SNR (polar k-NN neighbourhood)
                rr, az = np.linalg.norm(xyz[k], axis=1), np.arctan2(xyz[k, 1], xyz[k, 0])
                c_all = combine_contrast(local_contrast_polar(rr, az, rcs[k]), local_contrast_polar(rr, az, snr[k]), snr[k])
                CON.append(c_all[ks[k]])
                NFR += 1
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(L))).voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    q, nq = np.asarray(pc.points), np.asarray(pc.normals)
    w = Window(f"{Path(bag).name}:{t0:g}-{t1:g}", q, nq, cKDTree(q), np.concatenate(R))
    w.rcs, w.n_frames = np.concatenate(RCS), NFR
    w.contrast = np.concatenate(CON)
    w.lidar_raw = np.concatenate(L) if keep_raw else None
    return w


def _residuals(T, w: Window, max_d: float):
    R, t = T
    ps = w.p @ R.T + t
    d, i = w.tree.query(ps, distance_upper_bound=max_d)
    ok = np.isfinite(d)
    n = w.nq[i[ok]]
    r = np.einsum("ij,ij->i", n, ps[ok] - w.q[i[ok]])
    return r, n, ps[ok], ok


def joint_icp(windows: list[Window], R0=np.eye(3), t0=np.zeros(3), *, dists=(1.5, 0.8, 0.4), iters: int = 15,
              cauchy_c: float = 0.15, fix_t: bool = False, anchors=None, anchor_weight: float = 1.0,
              anchor_c: float = 0.3):
    """Robust point-to-plane Gauss-Newton shared over windows. Returns (R, t, stats).

    anchors: optional point correspondences [(p_radar (3,), q_lidar (3,))] (corner reflectors). Their
    horizontal (x, y) point-to-point residuals are added with total weight `anchor_weight` relative to all
    ICP terms (radar range / azimuth are accurate, elevation is not, so z is left to the ICP)."""
    R, t = R0.copy(), t0.copy()
    for max_d in dists:
        for _ in range(iters):
            H, g = np.zeros((6, 6)), np.zeros(6)
            for w in windows:
                r, n, ps, _ = _residuals((R, t), w, max_d)
                if len(r) < 10:
                    continue
                J = np.hstack([np.cross(ps, n), n])                  # d r / d [rotvec, t] (left perturbation)
                wt = 1.0 / (1.0 + (r / cauchy_c) ** 2) / len(windows) / max(len(r), 1) * 1e3
                H += (J * wt[:, None]).T @ J
                g += (J * wt[:, None]).T @ r
            if anchors:
                for p_r, q_l in anchors:
                    rp = R @ p_r
                    e = (rp + t - q_l)[:2]
                    J = np.hstack([-_skew(rp), np.eye(3)])[:2]
                    w_a = anchor_weight / len(anchors) * 1e3 / (1.0 + (np.linalg.norm(e) / anchor_c) ** 2)
                    H += w_a * J.T @ J
                    g += w_a * J.T @ e
            if fix_t:
                dx = np.zeros(6)
                dx[:3] = -np.linalg.solve(H[:3, :3] + 1e-6 * np.eye(3), g[:3])
            else:
                dx = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
            dR = Rotation.from_rotvec(dx[:3]).as_matrix()
            R, t = dR @ R, dR @ t + dx[3:]
            if np.linalg.norm(dx) < 1e-6:
                break
    stats = []
    for w in windows:
        r, *_ , ok = _residuals((R, t), w, dists[-1])
        stats.append({"window": w.name, "n_radar": len(w.p), "fitness": float(ok.mean()),
                      "rmse": float(np.sqrt(np.mean(r ** 2))) if len(r) else np.nan})
    return R, t, stats


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rpy_deg(R) -> np.ndarray:
    y, p, r = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return np.array([r, p, y])


def jackknife(windows: list[Window], R, t, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-window-out spread (std) of rpy [deg] and t [m]."""
    if len(windows) < 3:
        return np.full(3, np.nan), np.full(3, np.nan)
    rs, ts = [], []
    for k in range(len(windows)):
        Rk, tk, _ = joint_icp(windows[:k] + windows[k + 1:], R, t, **kw)
        rs.append(rpy_deg(Rk))
        ts.append(tk)
    n = len(windows)
    return np.sqrt((n - 1) / n * ((np.array(rs) - np.mean(rs, 0)) ** 2).sum(0)), \
        np.sqrt((n - 1) / n * ((np.array(ts) - np.mean(ts, 0)) ** 2).sum(0))
