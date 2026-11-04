#!/usr/bin/env python3
"""LiDAR-only WORLD reference of the standing workers of a swing segment and an audit of the GT ids against it
(docs/13 "LiDAR 월드 기준 감사").

The camera GT of an in-place swing links boxes by image position / world azimuth from the gyro heading, whose
error during fast swings smears a 22 m target by 1.5 m. The LiDAR sees 360 deg all the time, so a standing person
is one thin column in a world map once the scans are placed with a good pose:

  1. heading: scans in the level frame, de-skewed as a rotation about the swing axis with the IMU rate, machine
     body masked (voxels static in the sensor frame while swinging), then a SWING-CONSTRAINED scan-to-map
     registration: params (yaw, roll, pitch, z), xy on the lever-arm circle, robust point-to-plane cost; pass 1
     against a reference map of the still periods (sequential init = previous scan + gyro increment), pass 2
     against even/odd half maps. KISS-ICP (--kiss) and free 6-DoF ICP wander off the circle with this sparse
     32-beam LiDAR at 67 deg/s and are only kept as comparison rows. The pose at every GT frame is compared with
     cache/heading.json (pipeline gyro heading); all headings are scored by the sharpness of the accumulated
     static map (BEV nearest-point distance between scans >= 30 deg apart). The image/ref3d time gets --clock-dt
     (+20 ms: minimum of the world-azimuth spread of the static GT ids, the "camera referee").
  2. world map: de-skewed scans in the world frame (= level frame of the segment start, the pipeline's world),
     ground grid removed, per-scan clustering of the points > 0.6 m above ground, clusters linked over time;
     person-sized (height p90 1.2-2.3 m, footprint 0.3-1.2 m) and still >= 5 s -> static person P (plus isolated
     dense columns of the accumulated map); person-sized tracks that move > 1.5 m -> moving track M.
  3. audit: every P / M is projected into every GT frame (RigCalib, pose at the frame's image time): which GT
     ids cover it (ref3d world distance, else image overlap), P -> ids, id -> P, merge proposals, missed frames
     (in FOV, not occluded by closer LiDAR points, no box), GT boxes with no LiDAR object, two ids on one P in a
     frame, and boxes whose ref3d range contradicts the LiDAR person under them (sparse ref3d on the background).

Outputs <seg>/lidar_audit.md, lidar_audit.json, lidar_world_map.png (+ cache/lidar_poses.npz). gt.json is NOT modified.

  PYTHONPATH=src python3 scripts/label_lidar_world_audit.py data/labels/<seg> [--lever 1.17 -0.89] [--kiss] [--no-cache]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rcmot.calib.camera_lidar import lidar_clock_offset  # noqa: E402
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.labeling.gt import GT  # noqa: E402
from rcmot.sensors.bag_source import open_bag, pc2_to_struct  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402
from rcmot.utils.rig_extrinsics import sensor_to_level  # noqa: E402

H_MAX, FP_MIN, FP_MAX, HMED_MIN = 2.3, 0.3, 1.2, 0.8     # person: top height, footprint (posts < 0.3, groups > 1.2), median point height (cones ~0.5)


LEVER = {"excavator_front_1.6m": (1.17, -0.89), "excavator_cab_3.2m_pitch26": (0.96, 0.88)}
SWEEP = 0.1


# ---------------------------------------------------------------------------------------------------- small helpers
def T_of(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def yaw_of(T):
    return float(np.arctan2(T[1, 0], T[0, 0]))


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def voxel_first(P, size):
    _, i = np.unique(np.floor(P / size).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(i)]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------------------------------- stage A: bag
def read_bag(bag: Path, lidar_topic: str, imu_topic: str, image_topic: str, t0: float, t1: float, r_max: float):
    """LiDAR scans (os_sensor frame, per-point sensor-clock time), gyro (sensor clock) and image header stamps
    of [t0, t1] (bag seconds). off: ROS time = sensor clock + off (sweep start of the scan header)."""
    scans, gyro_t, gyro, img_t = [], [], [], []
    with open_bag(bag) as r:
        cs = {c.topic: c for c in r.connections}
        s0 = r.start_time * 1e-9
        off = lidar_clock_offset(r, cs[lidar_topic])
        tf = {}
        if "/tf_static" in cs:
            from rcmot.calib.camera_lidar import _tf_static
            tf = _tf_static(r, cs)
        want = [cs[lidar_topic], cs[imu_topic], cs[image_topic]]
        for conn, tn, raw in r.messages(connections=want, start=int((s0 + t0) * 1e9), stop=int((s0 + t1) * 1e9)):
            m = r.deserialize(raw, conn.msgtype)
            hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if conn.topic == imu_topic:
                gyro_t.append(hs)
                gyro.append((m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
            elif conn.topic == image_topic:
                img_t.append(hs)
            else:
                a = pc2_to_struct(m)
                xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
                rr = np.linalg.norm(xyz, axis=1)
                ok = np.isfinite(rr) & (rr > 0.8) & (rr < r_max)
                R, t = lidar_to_sensor(tf, m.header.frame_id)
                P = xyz[ok] @ R.T + t
                tp = hs + a["t"][ok].astype(np.float64) * 1e-9
                scans.append((hs, P.astype(np.float32), tp))
    return {"scans": scans, "gyro_t": np.array(gyro_t), "gyro": np.array(gyro), "img_t": np.array(img_t),
            "off": float(off), "bag_start": s0}


def self_mask(scans, swing_idx, cell=0.25, frac=0.6):
    """Voxels (sensor frame) occupied in >= frac of the swinging scans = the machine body (boom, cab, counterweight):
    static in the sensor frame while the world turns."""
    keys = []
    for k in swing_idx:
        keys.append(np.unique(_vkey(scans[k], cell)))
    u, n = np.unique(np.concatenate(keys), return_counts=True)
    return u[n >= frac * len(swing_idx)], cell


def _vkey(P, cell):
    v = np.floor(P / cell).astype(np.int64) + 1_000_000
    return (v[:, 0] * 2_000_003 + v[:, 1]) * 2_000_003 + v[:, 2]


def is_body(P, body, cell):
    if len(body) == 0:
        return np.zeros(len(P), bool)
    return np.isin(_vkey(P, cell), body)


def deskew_swing(P, tp, t_ref, gyro_t, gz, axis_xy):
    """De-skew a level-frame scan for the in-place swing: every point is turned about the swing axis (level xy
    axis_xy, z up) by the gyro-integrated yaw between its time and t_ref, so the cloud is expressed in the level frame
    at t_ref. The exact IMU rate is used (no constant-velocity assumption); the lever arm makes the sensor itself move
    along its circle within one sweep (0.14 m at 60 deg/s), which a rotation about the sensor z would miss."""
    w_mid = 0.5 * (np.interp(tp, gyro_t, gz) + np.interp(t_ref, gyro_t, gz))
    ang = w_mid * (tp - t_ref)
    c, s = np.cos(ang), np.sin(ang)
    x, y = P[:, 0] - axis_xy[0], P[:, 1] - axis_xy[1]
    Q = P.copy()
    Q[:, 0], Q[:, 1] = c * x - s * y + axis_xy[0], s * x + c * y + axis_xy[1]
    return Q


# ---------------------------------------------------------------------------------------------------- stage B: odometry
def kiss_odometry(scans_ds, voxel=0.4, r_max=60.0):
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP
    kc = KISSConfig()
    kc.data.max_range, kc.data.min_range, kc.data.deskew = r_max, 0.8, False     # gyro-deskewed already
    kc.mapping.voxel_size = voxel
    kc.adaptive_threshold.initial_threshold = 1.0
    kiss = KissICP(kc)
    poses = []
    for P in scans_ds:
        kiss.register_frame(P.astype(np.float64), np.full(len(P), 0.5))
        poses.append(kiss.last_pose.copy())
    return np.array(poses)


def gyro_heading(gyro_t, gz, still_dps=0.5, win_s=1.0):
    """Bias and the trapezoid-integrated yaw [rad] at gyro_t. Still samples: 1 s-smoothed rate within 2 deg/s of
    zero, then within still_dps of their median (the Ouster IMU bias is < 1 deg/s). The bias is the median of the
    1 s-smoothed rate over them: the engine vibration gives the raw rate an asymmetric +-1.7 deg/s noise whose raw
    median is 0.7 deg/s off the mean (142951); the integral follows the mean, and the median of the smoothed rate
    rejects slow nudges of the machine."""
    k = max(int(round(win_s / max(np.median(np.diff(gyro_t)), 1e-3))), 1)
    sm = np.convolve(gz, np.ones(k) / k, mode="same")
    low = np.abs(sm) < np.radians(2.0)
    b0 = float(np.median(sm[low])) if low.any() else 0.0
    still = np.abs(sm - b0) < np.radians(still_dps)
    bias = float(np.median(sm[still])) if still.any() else b0
    w = gz - bias
    yaw = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(gyro_t))])
    return yaw, bias, still


def model_pose(yaw, lever_level, T_lev):
    """Sensor pose (sensor(t) -> sensor(0)) of the swing model: the level frame turns by yaw about the swing axis
    at -lever_level (level frame of the start)."""
    R = rotz(yaw)
    o = -lever_level + R[:2, :2] @ lever_level
    return np.linalg.inv(T_lev) @ T_of(R, np.r_[o, 0.0]) @ T_lev


def _o3d_cloud(P, normals=False, radius=0.6):
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, np.float64)))
    if normals:
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=20))
    return pc


def icp_to_map(P, T0, tgt, max_dist, iters=40, kernel=0.3):
    """Point-to-plane ICP (open3d, Tukey kernel) of the scan points P (sensor frame) onto the map cloud `tgt`
    (world frame, normals present) from the initial pose T0. Returns (pose, fitness, inlier rmse)."""
    import open3d as o3d
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(
        o3d.pipelines.registration.TukeyLoss(k=kernel))
    res = o3d.pipelines.registration.registration_icp(
        _o3d_cloud(P), tgt, max_dist, T0, est,
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iters))
    return np.asarray(res.transformation), float(res.fitness), float(res.inlier_rmse)


def still_runs(T_mid, rate, still_dps=1.0, min_s=2.0, max_s=4.0):
    """[(i0, i1)] scan index runs where the 0.5 s-smoothed |rate| < still_dps for >= min_s (engine vibration puts
    +-1.7 deg/s on the raw rate); runs longer than max_s are cut into chunks so that a slow nudge or the gyro bias
    cannot smear a reference chunk (each chunk is registered on its own)."""
    k = 5
    sm = np.convolve(rate, np.ones(k) / k, mode="same") if len(rate) > k else rate
    still = np.abs(sm) < np.radians(still_dps)
    runs, i = [], 0
    while i < len(still):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(still) and still[j + 1]:
            j += 1
        if T_mid[j] - T_mid[i] >= min_s:
            a = i
            while a <= j:
                b = a
                while b + 1 <= j and T_mid[b + 1] - T_mid[a] < max_s:
                    b += 1
                if T_mid[b] - T_mid[a] >= min_s or (a == i and b == j):
                    runs.append((a, b))
                a = b + 1
        i = j + 1
    return runs


class SwingPose:
    """Pose of the level frame in the world (= level frame at the start) under the in-place swing model:
    params p = (yaw, roll, pitch, z): origin xy on the lever-arm circle about the swing axis at -lever, free roll /
    pitch (the machine rocks ~1.5 deg) and height. T_WL(p): level(t) -> world; with T_lev (sensor -> level) the
    sensor points go to the world as apply(T_WL @ T_lev, P)."""

    def __init__(self, lever_level, T_lev):
        self.lever, self.T_lev = np.asarray(lever_level, float), T_lev

    def T_WL(self, p):
        yaw, roll, pitch, z = p
        Rz = rotz(yaw)
        R = Rz @ Rotation.from_euler("yx", [pitch, roll]).as_matrix()
        o = -self.lever + Rz[:2, :2] @ self.lever
        return T_of(R, np.r_[o, z])

    def T_WS(self, p):
        return self.T_WL(p) @ self.T_lev


class NormalMap:
    """World-frame map with normals (open3d) and a KD-tree for point-to-plane residuals."""

    def __init__(self, pts, voxel=0.1, radius=0.6):
        self.pts = voxel_first(np.asarray(pts, np.float64), voxel)
        pc = _o3d_cloud(self.pts, normals=True, radius=radius)
        self.nrm = np.asarray(pc.normals)
        self.tree = cKDTree(self.pts)

    def residuals(self, Q, max_dist):
        d, j = self.tree.query(Q, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        r = np.einsum("ij,ij->i", Q[ok] - self.pts[j[ok]], self.nrm[j[ok]])
        return r, ok


def register_swing(P, p0, nmap: NormalMap, model: SwingPose, max_dist, iters=8, f_scale=0.1, yaw_lim=np.radians(8)):
    """Constrained ICP of the LEVEL-frame scan P onto the world map: least squares (soft-L1) of the point-to-plane
    residuals over (yaw, roll, pitch, z), correspondences re-found each outer iteration. Returns (p, fitness, rmse)."""
    from scipy.optimize import least_squares
    p = np.asarray(p0, float).copy()
    lo = [p0[0] - yaw_lim, -np.radians(6), -np.radians(6), -0.6]
    hi = [p0[0] + yaw_lim, np.radians(6), np.radians(6), 0.6]
    fit, rmse = 0.0, np.inf
    for _ in range(iters):
        Q = apply(model.T_WL(p), P)
        d, j = nmap.tree.query(Q, distance_upper_bound=max_dist)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            break
        src, m_pts, m_nrm = P[ok], nmap.pts[j[ok]], nmap.nrm[j[ok]]

        def fun(x):
            return np.einsum("ij,ij->i", apply(model.T_WL(x), src) - m_pts, m_nrm)

        res = least_squares(fun, p, loss="soft_l1", f_scale=f_scale, bounds=(lo, hi), max_nfev=40, xtol=1e-8, ftol=1e-8)
        dp = res.x - p
        p = res.x
        r = fun(p)
        fit, rmse = float(ok.mean()), float(np.sqrt(np.mean(np.minimum(r * r, max_dist ** 2))))
        if abs(dp[0]) < np.radians(0.003) and np.abs(dp[1:3]).max() < np.radians(0.01) and abs(dp[3]) < 1e-3:
            break
    return p, fit, rmse


def refine_poses(scans, yaw0, T_mid, rate, model: SwingPose, voxel=0.1, log=print):
    """Swing-constrained scan-to-map registration of every scan:
      pass 1: reference map = still runs (the first run at the gyro yaw, every later run's merged cloud registered to
              the map so far), then every scan in time order, initialised from the previous scan + gyro increment,
              registered to the reference map (0.6 m correspondences);
      pass 2: even / odd half maps of all scans (pass-1 poses), every scan registered to the OTHER half (0.3 m).
    scans: body-free LEVEL-frame points; yaw0: gyro yaw at the scan mid times (rad, 0 at the start).
    Returns (params (n,4), fitness, rmse)."""
    n = len(scans)
    P = [voxel_first(np.asarray(s, np.float64), voxel) for s in scans]
    runs = still_runs(T_mid, rate)
    log(f"  still runs (scan idx, s): {[(i0, i1, round(T_mid[i1] - T_mid[i0], 1)) for i0, i1 in runs]}")
    p = np.zeros((n, 4))
    p[:, 0] = yaw0
    fits, rmses = np.zeros(n), np.zeros(n)
    ref_pts = []
    for q, (i0, i1) in enumerate(runs):
        ks = range(i0, i1 + 1, max(1, (i1 - i0) // 40))
        if q == 0:
            ref_pts.append(np.concatenate([apply(model.T_WL(p[k]), P[k]) for k in ks]))
            continue
        nmap = NormalMap(np.concatenate(ref_pts), voxel)
        cloud = np.concatenate([apply(np.linalg.inv(model.T_WL(p[i0])) @ model.T_WL(p[k]), P[k]) for k in ks])  # run cloud in the run's first level frame
        pq, fit, rmse = register_swing(voxel_first(cloud, voxel), p[i0], nmap, model, 0.6)
        log(f"  still run {q} ({T_mid[i0] - T_mid[0]:.0f}-{T_mid[i1] - T_mid[0]:.0f} s) -> map: yaw {np.degrees(p[i0, 0]):.2f} -> {np.degrees(pq[0]):.2f} deg, "
            f"roll/pitch {np.degrees(pq[1]):.2f}/{np.degrees(pq[2]):.2f} deg, z {pq[3]:.2f} m, fitness {fit:.2f} rmse {rmse:.3f}")
        for k in range(i0, i1 + 1):
            p[k] = pq + np.r_[yaw0[k] - yaw0[i0], 0, 0, 0]
        ref_pts.append(np.concatenate([apply(model.T_WL(p[k]), P[k]) for k in ks]))
    if ref_pts:
        nmap = NormalMap(np.concatenate(ref_pts), voxel)
        tic = time.time()
        prev, n_guard = None, 0
        for k in range(n):
            init = p[k].copy() if prev is None else prev + np.r_[yaw0[k] - yaw0[k - 1], 0, 0, 0]
            q, fits[k], rmses[k] = register_swing(P[k], init, nmap, model, 0.6, yaw_lim=np.radians(4))
            if prev is not None and abs(wrap(q[0] - init[0])) > np.radians(3.0):   # divergence guard; the gyro increment is good to 0.2 deg/scan
                q, n_guard = init, n_guard + 1
            p[k] = q
            prev = p[k]
        log(f"  pass 1 (reference map {len(nmap.pts)} pts): fitness median {np.median(fits):.2f} p10 {np.percentile(fits, 10):.2f}, "
            f"rmse median {np.median(rmses):.3f} m, gyro-guarded scans {n_guard} ({time.time() - tic:.0f} s)")
        if np.median(fits) < 0.9 or n_guard > 0.2 * n:
            log("  pass 1 unreliable (open side of the site / swing-axis tilt): pass 2 starts from the gyro model instead")
            p = np.zeros((n, 4))
            p[:, 0] = yaw0
    for it, md in enumerate((0.3,)):
        maps = [NormalMap(np.concatenate([apply(model.T_WL(p[k]), P[k]) for k in range(h, n, 2)]), voxel) for h in (0, 1)]
        new = p.copy()
        tic = time.time()
        for k in range(n):
            new[k], fits[k], rmses[k] = register_swing(P[k], p[k], maps[(k + 1) % 2], model, md, yaw_lim=np.radians(3))
        p = new
        log(f"  pass {it + 2} (half maps, {md} m): fitness median {np.median(fits):.2f} p10 {np.percentile(fits, 10):.2f}, "
            f"rmse median {np.median(rmses):.3f} m ({time.time() - tic:.0f} s)")
    return p, fits, rmses


def pose_at(T_times, T_list, t):
    """Interpolated pose (slerp + linear translation) at time t."""
    k = np.searchsorted(T_times, t)
    if k <= 0:
        return T_list[0]
    if k >= len(T_times):
        return T_list[-1]
    a, b = T_list[k - 1], T_list[k]
    u = (t - T_times[k - 1]) / max(T_times[k] - T_times[k - 1], 1e-6)
    Rab = Rotation.from_matrix(np.stack([a[:3, :3], b[:3, :3]]))
    R = Slerp([0.0, 1.0], Rab)([u]).as_matrix()[0]
    return T_of(R, (1 - u) * a[:3, 3] + u * b[:3, 3])


# ---------------------------------------------------------------------------------------------------- sharpness
def map_sharpness(scans_lv, T_list, yaw, min_deg=30.0, z_min=0.5, r_rng=(3.0, 40.0), n_pairs=80, seed=0):
    """Sharpness of the accumulated map: for pairs of scans whose headings differ by >= min_deg, the BEV (xy)
    nearest-point distance of the above-ground points (z > z_min in the world frame, 3-40 m from the sensor) of one
    scan to the other. Returns (median of the pair medians [m], fraction of points within 0.15 m, n pairs). A sharp
    map puts a thin wall or a standing person at the same xy from both headings; a heading error e moves a point at
    range r by r*e."""
    rng = np.random.default_rng(seed)
    n = len(scans_lv)
    pairs = []
    for _ in range(n_pairs * 50):
        i, j = rng.integers(0, n, 2)
        if abs(wrap(yaw[i] - yaw[j])) >= np.radians(min_deg):
            pairs.append((i, j))
        if len(pairs) >= n_pairs:
            break
    med, thin = [], []
    for i, j in pairs:
        A, B = apply(T_list[i], scans_lv[i]), apply(T_list[j], scans_lv[j])
        ra, rb = np.linalg.norm(A[:, :2] - T_list[i][:2, 3], axis=1), np.linalg.norm(B[:, :2] - T_list[j][:2, 3], axis=1)
        A = A[(A[:, 2] > z_min) & (ra > r_rng[0]) & (ra < r_rng[1]), :2]
        B = B[(B[:, 2] > z_min) & (rb > r_rng[0]) & (rb < r_rng[1]), :2]
        if len(A) < 100 or len(B) < 100:
            continue
        d, _ = cKDTree(A).query(B, distance_upper_bound=2.0)
        d = d[np.isfinite(d)]
        if len(d) > 50:
            med.append(np.median(d))
            thin.append(np.mean(d < 0.15))
    return (float(np.median(med)) if med else float("nan"), float(np.median(thin)) if thin else float("nan"), len(med))


# ---------------------------------------------------------------------------------------------------- stage C: map / persons
class GroundGrid:
    """Ground height z_g(x, y) of the world map: p20 of z per cell (cell m), holes filled from the nearest cell, 5x5
    median (a single rut cell must not pull the ground down and turn the flat ground around it into "objects").
    The 32-beam rings leave gaps of several metres on the ground, hence the coarse cell."""

    def __init__(self, W, cell=2.0, half=70.0):
        self.cell, self.half = cell, half
        n = int(2 * half / cell)
        ij = np.floor((W[:, :2] + half) / cell).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        key, z = ij[ok, 0] * n + ij[ok, 1], W[ok, 2]
        o = np.lexsort((z, key))
        key, z = key[o], z[o]
        uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
        G = np.full(n * n, np.nan)
        for u, s, c in zip(uk, st, cnt):
            if c >= 3:
                G[u] = z[s + int(0.2 * (c - 1))]
        G = G.reshape(n, n)
        # fill holes from the nearest filled cell, then 3x3 median
        from scipy.ndimage import distance_transform_edt, median_filter
        miss = np.isnan(G)
        if miss.any() and (~miss).any():
            idx = distance_transform_edt(miss, return_distances=False, return_indices=True)
            G = G[idx[0], idx[1]]
        self.G = median_filter(G, size=5, mode="nearest")
        self.n = n

    def __call__(self, xy):
        ij = np.clip(np.floor((np.atleast_2d(xy) + self.half) / self.cell).astype(int), 0, self.n - 1)
        return self.G[ij[:, 0], ij[:, 1]]


def dbscan_xy(X, eps, min_pts):
    """Plain DBSCAN on (N,2) with a KD-tree (labels, -1 = noise)."""
    n = len(X)
    lab = -np.ones(n, int)
    if n == 0:
        return lab
    tree = cKDTree(X)
    nb = tree.query_ball_point(X, eps)
    k = 0
    for i in range(n):
        if lab[i] != -1 or len(nb[i]) < min_pts:
            continue
        lab[i] = k
        stack = list(nb[i])
        while stack:
            j = stack.pop()
            if lab[j] == -1:
                lab[j] = k
                if len(nb[j]) >= min_pts:
                    stack.extend(q for q in nb[j] if lab[q] == -1)
            elif lab[j] == -2:
                lab[j] = k
        k += 1
    return lab


def above_ground(W, zg: GroundGrid, sensor_xy, h_rng=(0.6, 2.6), r_rng=(2.5, 50.0)):
    """(xy, height above ground, range, world z) of the above-ground points of one world-frame scan. 0.6 m: the
    ground grid of a rough site is 0.3-0.4 m off in places and cones stay below; a person keeps the torso rings."""
    hgt = W[:, 2] - zg(W[:, :2])
    r = np.linalg.norm(W[:, :2] - sensor_xy, axis=1)
    sel = (hgt > h_rng[0]) & (hgt < h_rng[1]) & (r > r_rng[0]) & (r < r_rng[1])
    return W[sel, :2], hgt[sel], r[sel], W[sel, 2]


def scan_objects(W, zg: GroundGrid, sensor_xy, eps=0.35, min_pts=3):
    """Above-ground clusters of one world-frame scan: dicts (xy, h, hmed, zlow, fp, n, r)."""
    Q, hq, rq, zq = above_ground(W, zg, sensor_xy)
    lab = dbscan_xy(Q, eps, min_pts)
    out = []
    for k in range(lab.max() + 1):
        s = lab == k
        xy = Q[s, :2]
        c = xy.mean(0)
        ext = xy - c
        fp = float(np.sqrt(np.linalg.eigvalsh(np.cov(ext.T) if len(xy) > 2 else np.eye(2) * 1e-6).max()) * 4) if len(xy) > 2 else float(np.ptp(xy, axis=0).max())
        fp = max(fp, float(np.ptp(xy, axis=0).max()))
        out.append({"xy": c, "h": float(hq[s].max()), "hmed": float(np.median(hq[s])), "zlow": float(zq[s].min()), "fp": fp, "n": int(s.sum()), "r": float(rq[s].mean())})
    return out


def map_persons(W_list, zg: GroundGrid, TL_list, T, cell=0.2, min_scans=30, min_s=5.0, fp_max=FP_MAX, h_max=H_MAX, half=70.0, h_occ=0.5):
    """Static persons from the ACCUMULATED map: 8-connected components of the 0.2 m occupancy grid of all above-ground
    points (a standing person is a thin column that collects points from every scan, even 3-5 points per scan at
    40 m); person-sized = footprint <= fp_max, p95 height in h_rng, seen in >= min_scans scans over >= min_s.
    Returns (dicts like the tracker's (xy, h, fp, n_scans, n_pts, t0, t1, windows, spread, r), occupancy grid, cell, half)."""
    from scipy.ndimage import label
    xs, hs, rs, ss = [], [], [], []
    for k, (W, TL) in enumerate(zip(W_list, TL_list)):
        xy, hgt, r, _ = above_ground(W, zg, TL[:2, 3])
        xs.append(xy), hs.append(hgt), rs.append(r), ss.append(np.full(len(xy), k))
    XY, HG, RR, SC = np.concatenate(xs), np.concatenate(hs), np.concatenate(rs), np.concatenate(ss)
    keep = HG > h_occ                                          # terrain bumps / low clutter do not join the columns
    XY, HG, RR, SC = XY[keep], HG[keep], RR[keep], SC[keep]
    n = int(2 * half / cell)
    ij = np.floor((XY + half) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    ij, XY, HG, RR, SC = ij[ok], XY[ok], HG[ok], RR[ok], SC[ok]
    grid = np.zeros((n, n), np.int32)
    np.add.at(grid, (ij[:, 0], ij[:, 1]), 1)
    grid_hi = np.zeros((n, n), np.int32)                      # > 1 m above ground: terrain bumps / cones drop out
    hi = HG > 1.0
    np.add.at(grid_hi, (ij[hi, 0], ij[hi, 1]), 1)
    lab, ncomp = label(grid >= 3, structure=np.ones((3, 3)))
    pl = lab[ij[:, 0], ij[:, 1]]
    out = []
    order = np.argsort(pl, kind="stable")
    pl_s = pl[order]
    bounds = np.searchsorted(pl_s, np.arange(1, ncomp + 2))
    for c in range(1, ncomp + 1):
        idx = order[bounds[c - 1]:bounds[c]]
        if len(idx) < 40:
            continue
        cells = np.unique(ij[idx], axis=0)
        fp = float(np.linalg.norm(np.ptp(cells, axis=0) + 1) * cell)
        if fp > fp_max or fp < FP_MIN + cell:
            continue
        h = float(np.percentile(HG[idx], 95))
        if not (h_min_at(np.median(RR[idx])) <= h <= h_max) or float(np.median(HG[idx])) < HMED_MIN + 0.1:
            continue
        scans = np.unique(SC[idx])
        if len(scans) < min_scans or len(idx) < max(100, 2.5 * len(scans)):     # 1 point per scan = noise column
            continue
        ts = T[scans]
        if ts[-1] - ts[0] < min_s:
            continue
        wins, w0, w1 = [], ts[0], ts[0]            # time windows: gaps > 2 s split
        for t in ts[1:]:
            if t - w1 > 2.0:
                wins.append((w0, w1))
                w0 = t
            w1 = t
        wins.append((w0, w1))
        wins = [(a, b) for a, b in wins if b - a >= 1.0] or [(ts[0], ts[-1])]
        c_xy = XY[idx].mean(0)
        out.append({"xy": c_xy, "t0": float(ts[0]), "t1": float(ts[-1]), "n_scans": int(len(scans)), "h": h, "fp": fp,
                    "n_pts": int(len(idx)), "r": float(np.median(RR[idx])), "spread": float(np.linalg.norm(XY[idx] - c_xy, axis=1).std()),
                    "windows": wins, "src": "map", "tracks": {-1 - c}})
    return out, grid_hi >= 3, cell, half


class Structures:
    """8-connected components of the occupancy grid of the points > 1 m above ground (>= 3 points per 0.2 m cell):
    the component size at a position tells a lone person (<= ~20 cells) from a wall / container / vehicle fragment
    (hundreds); terrain bumps and cones do not reach 1 m."""

    def __init__(self, occ, cell, half):
        from scipy.ndimage import label
        self.lab, n = label(occ, structure=np.ones((3, 3)))
        self.size = np.bincount(self.lab.ravel())
        self.cell, self.half = cell, half

    def component_cells(self, xy, radius=0.6):
        i0, j0 = np.floor((np.asarray(xy) + self.half) / self.cell).astype(int)
        k = int(np.ceil(radius / self.cell))
        sub = self.lab[max(i0 - k, 0):i0 + k + 1, max(j0 - k, 0):j0 + k + 1]
        labs = np.unique(sub[sub > 0])
        return int(self.size[labs].max()) if len(labs) else 0


def track_objects(objs_per_scan, T, gate=1.0, lost_s=2.0):
    """Greedy nearest-neighbour linking of the per-scan clusters over time -> tracks [{t, xy, h, fp, n, r}]."""
    tracks, active = [], []
    for k, objs in enumerate(objs_per_scan):
        t = T[k]
        active = [a for a in active if t - a["t"][-1] <= lost_s]
        if objs and active:
            A = np.array([a["xy"][-1] for a in active])
            B = np.array([o["xy"] for o in objs])
            D = np.linalg.norm(A[:, None] - B[None], axis=2)
            used_a, used_b = set(), set()
            for idx in np.argsort(D, axis=None):
                i, j = divmod(int(idx), len(objs))
                if D[i, j] > gate:
                    break
                if i in used_a or j in used_b:
                    continue
                used_a.add(i), used_b.add(j)
                a, o = active[i], objs[j]
                for key in ("xy", "h", "hmed", "zlow", "fp", "n", "r"):
                    a[key].append(o[key])
                a["t"].append(t)
        else:
            used_b = set()
        for j, o in enumerate(objs):
            if j not in used_b:
                a = {"t": [t], "xy": [o["xy"]], "h": [o["h"]], "hmed": [o["hmed"]], "zlow": [o["zlow"]], "fp": [o["fp"]], "n": [o["n"]], "r": [o["r"]]}
                tracks.append(a), active.append(a)
    for a in tracks:
        for key in ("t", "h", "hmed", "zlow", "fp", "n", "r"):
            a[key] = np.array(a[key], float)
        a["xy"] = np.array(a["xy"], float).reshape(-1, 2)
    return tracks


def h_min_at(r):
    """Lowest acceptable measured top height (p90 over the scans) of a person at range r: 1.2 m near, relaxed to
    1.1 m beyond 30 m because the 32-beam rings (1.4 deg = 0.024 r apart) rarely hit the head of a far person;
    the machine's pitch rocking (+-1.2 deg) scans the body over time, so the p90 reaches the shoulders."""
    return np.maximum(1.1, 1.25 - 0.005 * np.asarray(r, float))


def person_like(a, frac=0.4):
    """A track whose observations are person-sized: top height p90 in [h_min(range), H_MAX], median footprint in
    [FP_MIN, FP_MAX] and footprint <= FP_MAX in >= frac of the scans (a fast-swing scan or a neighbour merges the
    cluster now and then), median point height >= HMED_MIN (a cone is all base)."""
    if len(a["t"]) < 5:
        return False
    h90 = float(np.percentile(a["h"], 90))
    ok = (a["fp"] <= FP_MAX) & (a["h"] >= 0.5)
    return bool(ok.mean() >= frac) and FP_MIN <= float(np.median(a["fp"])) <= FP_MAX and h_min_at(np.median(a["r"])) <= h90 <= H_MAX \
        and float(np.median(a["hmed"])) >= HMED_MIN


def still_phases(a, radius=0.5, min_s=5.0, per_m=0.015):
    """[(i0, i1, xy_mean)] stretches of a track within max(radius, per_m * range) of their running mean for >= min_s
    (the centroid of a sparse far cluster jitters ~0.5 m at 40 m)."""
    t, xy = a["t"], a["xy"]
    rad = max(radius, per_m * float(np.median(a["r"])))
    out, i = [], 0
    while i < len(t):
        j, c = i, xy[i].copy()
        while j + 1 < len(t):
            c2 = xy[i:j + 2].mean(0)
            if np.linalg.norm(xy[i:j + 2] - c2, axis=1).max() > rad:
                break
            j, c = j + 1, c2
        if t[j] - t[i] >= min_s:
            out.append((i, j, c))
        i = j + 1
    return out


def find_persons(objs_per_scan, T, log=print, min_s=5.0, min_scans=20, merge_r=0.6, gap_max=3.0):
    """Static persons P (still phases of person-like tracks, >= min_s, merged across re-appearances at the same
    spot) and moving person tracks M (person-like tracks whose path exceeds 1 m outside the still phases)."""
    tracks = track_objects(objs_per_scan, T)
    P, M = [], []
    for ti, a in enumerate(tracks):
        if not person_like(a):
            continue
        ph = still_phases(a, min_s=min_s)
        covered = np.zeros(len(a["t"]), bool)
        for i0, i1, c in ph:
            if i1 - i0 + 1 >= min_scans:
                P.append({"xy": c, "t0": float(a["t"][i0]), "t1": float(a["t"][i1]), "n_scans": int(i1 - i0 + 1),
                          "h": float(np.median(a["h"][i0:i1 + 1])), "fp": float(np.median(a["fp"][i0:i1 + 1])),
                          "n_pts": int(a["n"][i0:i1 + 1].sum()), "r": float(np.median(a["r"][i0:i1 + 1])),
                          "spread": float(np.linalg.norm(a["xy"][i0:i1 + 1] - c, axis=1).std()), "tracks": {ti},
                          "z0": float(np.median(a["zlow"][i0:i1 + 1]) - 0.15)})
                covered[i0:i1 + 1] = True
        rest = ~covered
        if rest.sum() >= 10:
            idx = np.where(rest)[0]
            sm = np.array([np.median(a["xy"][idx[max(0, q - 5):q + 6]], 0) for q in range(len(idx))])   # ~1 s median
            ext = float(np.linalg.norm(sm - sm.mean(0), axis=1).max() * 2)
            path = float(np.linalg.norm(np.diff(sm, axis=0), axis=1).sum())
            if ext > 1.5 and path > 1.5:
                M.append({"t": a["t"][idx], "xy": sm, "h": float(np.median(a["h"][idx])), "path_m": path,
                          "n_scans": int(len(idx)), "tracks": {ti}, "z0": float(np.median(a["zlow"][idx]) - 0.15)})
    # merge static persons at the same spot (re-appearances after occlusion by the machine body)
    P.sort(key=lambda q: q["t0"])
    merged = []
    for q in P:
        for u in merged:
            if np.linalg.norm(u["xy"] - q["xy"]) <= merge_r and (q["t0"] >= u["t1"] - 0.5 or u["t0"] >= q["t1"] - 0.5):
                n = u["n_scans"] + q["n_scans"]
                u["xy"] = (u["xy"] * u["n_scans"] + q["xy"] * q["n_scans"]) / n
                u["t0"], u["t1"] = min(u["t0"], q["t0"]), max(u["t1"], q["t1"])
                u["n_scans"], u["n_pts"] = n, u["n_pts"] + q["n_pts"]
                u["windows"].append((q["t0"], q["t1"]))
                u["h"] = max(u["h"], q["h"])
                u["tracks"] |= q["tracks"]
                break
        else:
            q["windows"] = [(q["t0"], q["t1"])]
            q["src"] = "track"
            merged.append(q)
    merged.sort(key=lambda q: (q["t0"], q["r"]))
    log(f"  tracks {len(tracks)}, person-like {sum(person_like(a) for a in tracks)}, static persons {len(merged)}, moving {len(M)}")
    return merged, M, tracks


# ---------------------------------------------------------------------------------------------------- stage D: audit
def interp_params(T_mid, p, t):
    """Swing params at time t (yaw unwrapped, linear)."""
    yaw = np.unwrap(p[:, 0])
    cols = [np.interp(t, T_mid, yaw)] + [np.interp(t, T_mid, p[:, c]) for c in (1, 2, 3)]
    return np.array(cols)


def person_box(cal: RigCalib, xy_level, h, half=0.35, z0=0.0):
    """Image box (xyxy) of an upright 0.7 m wide, h tall column standing at xy, ground z0 (level frame), None if
    behind the camera."""
    c = np.array([[xy_level[0] + dx, xy_level[1] + dy, z] for dx in (-half, half) for dy in (-half, half) for z in (z0 + 0.05, z0 + h)])
    uv, fr = cal.project(c)
    if not fr.all():
        return None
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def occlusion(Ww, hgt, sensor_xy, p_xy, h, half=0.35):
    """Fraction of the azimuth span of a person (world xy p_xy, height h) covered by LiDAR returns of the same scan
    that are closer to the sensor than the person (by > 1 m) and 0.5..h+0.3 m above the local ground (hgt)."""
    d = Ww[:, :2] - sensor_xy
    r = np.linalg.norm(d, axis=1)
    v = p_xy - sensor_xy
    rng_p = np.linalg.norm(v)
    span = np.arctan2(half, rng_p)
    da = wrap(np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(v[1], v[0]))
    sel = (np.abs(da) <= span + np.radians(0.2)) & (r < rng_p - 1.0) & (r > 1.0) & (hgt > 0.5) & (hgt < h + 0.3)
    if not sel.any():
        return 0.0
    hit = np.histogram(da[sel], bins=np.linspace(-span, span, 6))[0] > 0
    return float(hit.mean())


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("--lever", type=float, nargs=2, default=None, help="radar lever arm from the swing axis [m]")
    ap.add_argument("--image-topic", default="/oak/rgb/image_raw/compressed")
    ap.add_argument("--r-max", type=float, default=60.0)
    ap.add_argument("--no-cache", action="store_true", help="recompute the poses (cache/lidar_poses.npz)")
    ap.add_argument("--kiss", action="store_true", help="also run KISS-ICP odometry for the comparison table")
    ap.add_argument("--min-still-s", type=float, default=5.0, help="a static person stands >= this long")
    ap.add_argument("--gate", type=float, default=1.0, help="ref3d <-> LiDAR person world distance gate [m] (+ 5 %% of range)")
    ap.add_argument("--debug-xy", type=float, nargs=2, default=None, help="print the tracks near this world xy")
    ap.add_argument("--struct-cells", type=int, default=60, help="a tracker person inside a > 1 m occupancy component larger than this (0.2 m cells) is a structure fragment")
    ap.add_argument("--clock-dt", type=float, default=0.02, help="pose time = ROS time - LiDAR clock offset + this [s]; +0.02 minimises the world-azimuth "
                    "spread of the static GT ids on 142537 (camera referee: 1.3 -> 0.9 deg)")
    a = ap.parse_args()
    seg = Path(a.seg)
    seg = seg if seg.is_dir() else seg.parent
    g = GT(seg / "gt.json")
    m = g.meta
    cal = RigCalib.load(str(ROOT / m["calib"]))
    calib_json = json.load(open(ROOT / m["calib"]))
    R_lev, t_lev = sensor_to_level(calib_json)
    T_lev = T_of(R_lev, t_lev)
    lever_radar = np.array(a.lever if a.lever else LEVER.get(m.get("mount"), (0.0, 0.0)), float)
    lever = lever_radar - cal.t_lr[:2]                      # lever arm of the level-frame origin (rcmot.world)
    model = SwingPose(lever, T_lev)
    bag = Path(m["bag"]) if str(m["bag"]).startswith("/") else ROOT / m["bag"]
    log_lines = []

    def log(s):
        print(s, flush=True)
        log_lines.append(s)

    log(f"# LiDAR world audit: {seg.name}")
    log(f"bag {bag}  {m['start']}-{m['end']} s, lever (radar) {lever_radar.tolist()} -> level origin {np.round(lever, 3).tolist()}")

    # ---- A. bag
    tic = time.time()
    o = read_bag(bag, m["lidar_topic"], m["imu_topic"], a.image_topic, m["start"] - 1.5, m["end"] + 1.5, a.r_max)
    scans = o["scans"]
    gyro_t, gz = o["gyro_t"], o["gyro"][:, 2]
    off = o["off"]
    yaw_g, bias, _ = gyro_heading(gyro_t, gz)
    gzc = gz - bias
    T_mid = np.array([s[0] + 0.5 * SWEEP for s in scans])          # sensor clock
    axis_xy = -lever                                                # swing axis in the level frame
    ds = [deskew_swing(apply(T_lev, P.astype(np.float64)), tp, hs + 0.5 * SWEEP, gyro_t, gzc, axis_xy) for hs, P, tp in scans]
    rate = np.interp(T_mid, gyro_t, np.convolve(gzc, np.ones(10) / 10, mode="same"))   # 0.1 s mean rate at the scan mids
    swing_idx = np.where(np.abs(rate) > np.radians(10))[0]
    body, cell = self_mask(ds, swing_idx)
    body_mask = [is_body(P, body, cell) for P in ds]
    free = [P[~b] for P, b in zip(ds, body_mask)]
    free = [P[np.linalg.norm(P[:, :2], axis=1) > 2.0] for P in free]
    yaw0 = np.interp(T_mid, gyro_t, yaw_g)
    yaw0 -= yaw0[0]
    log(f"scans {len(scans)} ({T_mid[0] + off - o['bag_start']:.1f}-{T_mid[-1] + off - o['bag_start']:.1f} s bag), gyro bias {np.degrees(bias):.3f} deg/s, "
        f"swinging scans {len(swing_idx)}, machine-body voxels {len(body)} ({np.mean([b.mean() for b in body_mask]) * 100:.0f} % of the points), "
        f"read {time.time() - tic:.0f} s")

    # ---- B. poses (cached)
    cpath = seg / "cache" / "lidar_poses.npz"
    key = json.dumps([str(bag), m["start"], m["end"], lever_radar.tolist(), len(scans), a.r_max])
    p = None
    if cpath.exists() and not a.no_cache:
        z = np.load(cpath, allow_pickle=True)
        if str(z["key"]) == key:
            p, fits, rmses = z["p"], z["fits"], z["rmses"]
            log_lines.extend(str(z["log"]).split("\n"))
            log(f"poses from cache {cpath}")
    if p is None:
        sub = []
        p, fits, rmses = refine_poses(free, yaw0, T_mid, rate, model, log=lambda s: (log(s), sub.append(s)))
        cpath.parent.mkdir(exist_ok=True)
        np.savez(cpath, key=key, p=p, fits=fits, rmses=rmses, T_mid=T_mid, off=off, log="\n".join(sub))
    lv_free = free
    TL_gyro = [model.T_WL(np.r_[y, 0, 0, 0]) for y in yaw0]
    s_ref = map_sharpness(lv_free, [model.T_WL(q) for q in p], np.unwrap(p[:, 0]))
    s_gyr = map_sharpness(lv_free, TL_gyro, yaw0)
    pose_src = "swing-constrained scan-to-map"
    if s_gyr[1] > s_ref[1] * 1.1 and s_gyr[0] < s_ref[0]:            # registration did not beat the gyro model: keep the model
        log(f"refined map not sharper than the gyro+lever model (thin fraction {s_ref[1]:.3f} vs {s_gyr[1]:.3f}): the gyro+lever poses are used")
        p = np.zeros_like(p)
        p[:, 0] = yaw0
        pose_src = "gyro+lever model (registration rejected)"
    yaw_r = np.unwrap(p[:, 0])
    TL_ref = [model.T_WL(q) for q in p]
    # pipeline heading cache at the GT frames (theta at the frame stamp = radar scan time)
    hc = seg / "cache" / "heading.json"
    stamps = np.array(m["stamps"], float)
    st_sensor = stamps - off
    th = np.array([np.nan if x is None else x for x in json.load(open(hc))["theta"]], float) if hc.exists() else np.full(len(stamps), np.nan)
    have_th = np.isfinite(th)
    th_rel = th - (np.nanmedian(th[:5]) if have_th[:5].any() else 0.0)
    comp = {}
    if a.kiss:
        tic = time.time()
        try:
            Tk = kiss_odometry(free)                        # level-frame scans: poses are level(t) -> level(0)
            yk = np.unwrap([yaw_of(t) for t in Tk])
            TL_k = list(Tk)
            comp["kiss"] = (yk - yk[0], TL_k)
            log(f"KISS-ICP odometry {time.time() - tic:.0f} s")
        except Exception as e:  # noqa: BLE001
            log(f"KISS-ICP failed: {e}")
    th_mid = np.interp(T_mid, st_sensor[have_th], np.unwrap(th_rel[have_th])) if have_th.sum() > 2 else None
    comp["gyro+lever (this script)"] = (yaw0, TL_gyro)
    if th_mid is not None:
        comp["cache theta + lever (pipeline)"] = (th_mid, [model.T_WL(np.r_[y, 0, 0, 0]) for y in th_mid])
    comp[f"refined ({pose_src})"] = (yaw_r, TL_ref)
    yr_f = interp_params(T_mid, p, st_sensor)[0]
    yr_f -= np.median(yr_f[:5])
    log("")
    log("## 헤딩 비교 (GT 프레임 시각, 처음 5프레임 기준 0 deg)")
    log("")
    log("| 헤딩 | |Δ| vs refined 중앙값 [deg] | p90 | 최대 | 맵 선명도: xy 최근접 중앙값 [m] | 0.15 m 안 비율 | 쌍 |")
    log("|---|---|---|---|---|---|---|")
    heading_rows = {}
    for name, (y_mid, TL) in comp.items():
        yf = np.interp(st_sensor, T_mid, y_mid)
        yf -= np.median(yf[:5])
        d = np.degrees(wrap(yr_f - yf))
        s_med, s_thin, n_pairs = map_sharpness(lv_free, TL, yaw_r)
        heading_rows[name] = {"median_deg": float(np.median(np.abs(d))), "p90_deg": float(np.percentile(np.abs(d), 90)),
                              "max_deg": float(np.abs(d).max()), "sharp_xy_m": s_med, "thin_frac": s_thin, "pairs": n_pairs,
                              "signed_median_deg": float(np.median(d))}
        log(f"| {name} | {np.median(np.abs(d)):.2f} | {np.percentile(np.abs(d), 90):.2f} | {np.abs(d).max():.2f} | {s_med:.3f} | {s_thin:.3f} | {n_pairs} |")
    xy_o = np.array([t[:2, 3] for t in TL_ref])
    log("")
    log(f"refined: fitness 중앙값 {np.median(fits):.2f} (p10 {np.percentile(fits, 10):.2f}), 점-평면 rmse 중앙값 {np.median(rmses):.3f} m, "
        f"roll/pitch 표준편차 {np.degrees(p[:, 1].std()):.2f}/{np.degrees(p[:, 2].std()):.2f} deg, z {p[:, 3].min():.2f}..{p[:, 3].max():.2f} m, "
        f"선회축 (world) {np.round(-lever, 2).tolist()} m, 레벨 원점 궤적 반지름 {np.linalg.norm(lever):.2f} m")
    dthf = np.degrees(wrap(yr_f - (np.interp(st_sensor, T_mid, th_mid) - np.median(np.interp(st_sensor, T_mid, th_mid)[:5])))) if th_mid is not None else np.full(len(stamps), np.nan)

    # ---- C. world map / persons
    tic = time.time()
    W_free = [apply(TL, P) for TL, P in zip(TL_ref, lv_free)]
    allW = voxel_first(np.concatenate(W_free), 0.1)
    zg = GroundGrid(allW)
    objs = [scan_objects(W, zg, TL[:2, 3]) for W, TL in zip(W_free, TL_ref)]
    persons, movers, tracks = find_persons(objs, T_mid, log=log, min_s=a.min_still_s)
    mp, occ, occ_cell, occ_half = map_persons(W_free, zg, TL_ref, T_mid, min_s=a.min_still_s)
    structs = Structures(occ, occ_cell, occ_half)
    for q in persons:
        q["comp"] = structs.component_cells(q["xy"])
    def is_fragment(q):          # far wall / container fragments; a near person's own walking trail also makes a big component
        return (q["r"] > 35.0 and q["comp"] > a.struct_cells) or q["comp"] > 6 * a.struct_cells
    dropped = [q for q in persons if is_fragment(q)]
    persons = [q for q in persons if not is_fragment(q)]
    n_struct = len(dropped)
    if dropped:
        log("  dropped: " + ", ".join(f"({q['xy'][0]:.1f}, {q['xy'][1]:.1f}) r {q['r']:.0f} comp {q['comp']}" for q in dropped))
    n_new = 0
    for q in mp:                                              # map columns: add when no tracker person stands there
        q["comp"] = structs.component_cells(q["xy"])
        if is_fragment(q):
            continue
        near = [u for u in persons if np.linalg.norm(u["xy"] - q["xy"]) <= 0.7]
        if near:
            u = near[0]
            u["windows"] = sorted(set(u["windows"]) | set(q["windows"]))
            u["t0"], u["t1"] = min(u["t0"], q["t0"]), max(u["t1"], q["t1"])
            u["n_scans"] = max(u["n_scans"], q["n_scans"])
            u["src"] = "track+map"
            u["tracks"] |= q["tracks"]
        else:
            persons.append(q)
            n_new += 1
    zbias = float(np.median([q["z0"] - zg(q["xy"])[0] for q in persons if "z0" in q])) if any("z0" in q for q in persons) else 0.0
    for q in persons:
        q.setdefault("z0", float(zg(q["xy"])[0] + zbias))
        q.setdefault("comp", structs.component_cells(q["xy"]))
    persons.sort(key=lambda q: (q["t0"], q["r"]))
    log(f"  ground under the persons: column-foot minus ground grid median {zbias:+.2f} m (grid p10 of rough ground reads low)")
    log(f"  tracker persons dropped as part of a larger structure (> 1 m occupancy component > {a.struct_cells} cells of 0.2 m beyond 35 m, or > {6 * a.struct_cells}): {n_struct}")
    log(f"  accumulated-map columns (person-sized, >= 30 scans, >= {a.min_still_s:.0f} s): {len(mp)}, of which new {n_new}")
    log(f"map {len(allW)} voxels (0.1 m), clustering + tracking {time.time() - tic:.0f} s")
    if a.debug_xy:
        log(f"  tracks within 3 m of {a.debug_xy}:")
        for tr in tracks:
            d = np.linalg.norm(tr["xy"] - np.array(a.debug_xy), axis=1)
            if (d < 3).any() and len(tr["t"]) >= 3:
                log(f"    t {tr['t'][0] - (o['bag_start'] - off):.1f}-{tr['t'][-1] - (o['bag_start'] - off):.1f} s n {len(tr['t'])} xy {np.round(tr['xy'].mean(0), 1).tolist()} "
                    f"ext {np.ptp(tr['xy'], axis=0).max():.1f} h med {np.median(tr['h']):.2f} p90 {np.percentile(tr['h'], 90):.2f} (min {h_min_at(np.median(tr['r'])):.2f}) "
                    f"hmed {np.median(tr['hmed']):.2f} fp med {np.median(tr['fp']):.2f} frac {((tr['fp'] <= FP_MAX) & (tr['h'] >= 0.5)).mean():.2f} n/scan {np.median(tr['n']):.0f} person_like {person_like(tr)}")
    bag0 = o["bag_start"] - off                                  # sensor clock -> bag seconds: t - bag0
    log("")
    log("## LiDAR 정지 작업자 (world = 구간 시작 시점의 level 좌표, x 전방, y 좌측)")
    log("")
    log("| P | world x | world y | 거리 [m] | 높이 [m] | 폭 [m] | 처음 [s, bag] | 마지막 [s] | 스캔 수 | 점 수 | LiDAR xy σ [m] | 창 | 출처 | 주변 구조 셀 |")
    log("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, q in enumerate(persons):
        q["id"] = k
        log(f"| P{k} | {q['xy'][0]:.2f} | {q['xy'][1]:.2f} | {q['r']:.1f} | {q['h']:.2f} | {q['fp']:.2f} | {q['t0'] - bag0:.1f} | {q['t1'] - bag0:.1f} | "
            f"{q['n_scans']} | {q['n_pts']} | {q['spread']:.2f} | {len(q['windows'])} | {q['src']} | {q['comp']} |")
    log("")
    log("## LiDAR 이동 사람 트랙")
    log("")
    log("| M | 시작 [s] | 끝 [s] | 시작 xy | 끝 xy | 경로 [m] | 높이 [m] | 스캔 수 |")
    log("|---|---|---|---|---|---|---|---|")
    for k, q in enumerate(movers):
        q["id"] = k
        log(f"| M{k} | {q['t'][0] - bag0:.1f} | {q['t'][-1] - bag0:.1f} | ({q['xy'][0, 0]:.1f}, {q['xy'][0, 1]:.1f}) | ({q['xy'][-1, 0]:.1f}, {q['xy'][-1, 1]:.1f}) | "
            f"{q['path_m']:.1f} | {q['h']:.2f} | {q['n_scans']} |")

    # ---- D. audit
    img_t = o["img_t"]
    frame_rows = []                       # per frame: t_img (sensor clock), params
    for i, f in enumerate(g.frames):
        s = f["stamp"]
        k = int(np.argmin(np.abs(img_t - s))) if len(img_t) else -1
        t_img = (img_t[k] if k >= 0 and abs(img_t[k] - s) < 0.06 else s) - off + a.clock_dt
        t_ref = s - off + a.clock_dt
        frame_rows.append({"t_img": t_img, "t_ref": t_ref, "p_img": interp_params(T_mid, p, t_img), "p_ref": interp_params(T_mid, p, t_ref)})
    W, H = cal.image_size

    def usable(b):
        return b.get("cls") == "worker"

    P_ids = defaultdict(Counter)          # P -> id -> n frames
    id_P = defaultdict(Counter)           # id -> P -> n
    id_boxes = Counter()
    missed, dups, false_boxes, no_ref_match, per_frame, ref_bad = [], [], [], [], {}, []
    ref_w = defaultdict(lambda: {"refined": [], "cache": []})     # object -> GT ref3d world positions (refined / cache pose)
    objs_by_scan_t = T_mid
    for i, f in enumerate(g.frames):
        fr = frame_rows[i]
        T_img = model.T_WL(fr["p_img"])
        T_ref = model.T_WL(fr["p_ref"])
        Ti_inv, Tr_inv = np.linalg.inv(T_img), np.linalg.inv(T_ref)
        k_scan = int(np.argmin(np.abs(T_mid - fr["t_img"])))
        Ww_scan = apply(TL_ref[k_scan], ds[k_scan])                          # full scan incl. body, world frame
        hgt_scan = Ww_scan[:, 2] - zg(Ww_scan[:, :2])
        sensor_xy = T_img[:2, 3]
        # objects visible at this frame: static P in their windows, movers at this time
        cands = []
        for q in persons:
            if any(t0 - 1.0 <= fr["t_img"] <= t1 + 1.0 for t0, t1 in q["windows"]):
                cands.append(("P%d" % q["id"], q["xy"], q["h"], q["z0"]))
        for q in movers:
            if q["t"][0] - 0.3 <= fr["t_img"] <= q["t"][-1] + 0.3:
                xy = np.array([np.interp(fr["t_img"], q["t"], q["xy"][:, 0]), np.interp(fr["t_img"], q["t"], q["xy"][:, 1])])
                cands.append(("M%d" % q["id"], xy, q["h"], q["z0"]))
        boxes = [b for b in f["boxes"] if usable(b)]
        for b in boxes:
            id_boxes[b["id"]] += 1
        # world positions of the GT ref3d (level frame at the radar/frame stamp -> world)
        b_world = {}
        for b in boxes:
            r3 = b.get("ref3d") or {}
            if r3.get("xy_level") is not None:
                b_world[b["id"]] = apply(T_ref, np.array([[r3["xy_level"][0], r3["xy_level"][1], 0.0]]))[0, :2]
        rec = {"frame": i, "objects": {}}
        matched_ids = set()
        cand2d = defaultdict(list)                     # GT id -> [(|log h ratio|, object name, iou)] image-only matches
        proj = {}
        ref_hit = {b["id"] for b in boxes if b["id"] in b_world and any(
            np.linalg.norm(b_world[b["id"]] - xy_w) <= a.gate + 0.05 * np.linalg.norm(apply(Ti_inv, np.array([[xy_w[0], xy_w[1], 0.0]]))[0, :2])
            for _, xy_w, _, _ in cands)}                # boxes whose ref3d sits on some LiDAR person: no image fallback for them
        for name, xy_w, h, z0_w in cands:
            xl = apply(Ti_inv, np.array([[xy_w[0], xy_w[1], z0_w]]))[0]
            bb = person_box(cal, xl[:2], max(h, 1.65), z0=float(xl[2]))      # LiDAR top height is a lower bound (rings)
            rng_p = float(np.linalg.norm(xl[:2]))
            info = {"xy_level": [round(float(xl[0]), 2), round(float(xl[1]), 2)], "range": round(rng_p, 1)}
            in_fov = bb is not None and bb[2] > 0 and bb[0] < W and bb[3] > 0 and bb[1] < H
            if not in_fov:
                rec["objects"][name] = {**info, "in_fov": False}
                continue
            edge = bb[0] < 0 or bb[2] > W or bb[1] < 0 or bb[3] > H
            occ = occlusion(Ww_scan, hgt_scan, sensor_xy, np.asarray(xy_w, float), h)
            hpx = float(bb[3] - bb[1])
            ids = []
            for b in boxes:
                if b["id"] in b_world and np.linalg.norm(b_world[b["id"]] - xy_w) <= a.gate + 0.05 * rng_p:
                    ids.append((b["id"], round(float(np.linalg.norm(b_world[b["id"]] - xy_w)), 2), "ref3d"))
                    continue
                if b["id"] in ref_hit or edge:
                    continue
                gx1, gy1, gx2, gy2 = b["xyxy"]                 # image overlap, tolerant to ~1 box width / blur
                cx, bh, wp = 0.5 * (gx1 + gx2), gy2 - gy1, bb[2] - bb[0]
                vov = min(gy2, bb[3]) - max(gy1, bb[1])
                if abs(cx - 0.5 * (bb[0] + bb[2])) <= max(0.75 * wp, 25.0) and vov >= 0.3 * min(bh, hpx) and 0.6 <= bh / max(hpx, 1) <= 1.7:
                    cand2d[b["id"]].append((abs(np.log(bh / max(hpx, 1))), name, round(float(iou(bb, b["xyxy"])), 2)))
            proj[name] = (bb, hpx, edge, occ, rng_p, ids)
            rec["objects"][name] = {**info, "in_fov": True, "box": [round(float(v), 1) for v in bb], "h_px": round(hpx, 1), "edge": bool(edge),
                                    "occ": round(occ, 2), "ids": ids}
        for tid, cl in cand2d.items():                 # image-only match: the object whose projected height fits best
            _, name, ov = min(cl)
            bb, hpx, edge, occ, rng_p, ids = proj[name]
            if tid in b_world:                         # ref3d exists but is far from the LiDAR person -> ref3d wrong (background)
                r3 = next(b["ref3d"] for b in boxes if b["id"] == tid)
                ref_bad.append((i, tid, name, float(r3["range_m"] or 0), round(rng_p, 1)))
                ids.append((tid, ov, "2d (ref3d %.1f m != LiDAR %.1f m)" % (r3["range_m"] or 0, rng_p)))
            else:
                ids.append((tid, ov, "2d"))
            rec["objects"][name]["ids"] = ids
        for name in proj:
            bb, hpx, edge, occ, rng_p, ids = proj[name]
            for tid, _, kind in ids:
                P_ids[name][tid] += 1
                id_P[tid][name] += 1
                matched_ids.add(tid)
                if tid in b_world and kind == "ref3d":
                    r3 = next(b["ref3d"]["xy_level"] for b in boxes if b["id"] == tid)
                    ref_w[name]["refined"].append(b_world[tid])
                    if have_th[i]:
                        ref_w[name]["cache"].append(apply(model.T_WL(np.r_[th_rel[i], 0, 0, 0]), np.array([[r3[0], r3[1], 0.0]]))[0, :2])
            if len({t for t, *_ in ids}) >= 2:
                dups.append((i, name, sorted({t for t, *_ in ids})))
            if not ids and not edge and occ < 0.5 and hpx >= 25 and name.startswith("P"):
                missed.append((i, name, round(rng_p, 1), round(hpx, 0), round(float(np.degrees(np.interp(fr["t_img"], gyro_t, gzc))), 0)))
        # GT boxes with no LiDAR person: is there any above-ground LiDAR cluster at the ref3d?
        for b in boxes:
            if b["id"] in matched_ids:
                continue
            ign = "ignore" in b["flags"] or b.get("ignore_reason")
            if b["id"] in b_world:
                xw = b_world[b["id"]]
                near = [(float(np.linalg.norm(ob["xy"] - xw)), ob) for ob in objs[k_scan]]
                near = [x for x in near if x[0] <= a.gate + 0.05 * float((b.get("ref3d") or {}).get("range_m") or 0)]
                if near:
                    d, ob = min(near, key=lambda x: x[0])
                    false_boxes.append((i, b["id"], "lidar-object-not-person-sized" if not (1.2 <= ob["h"] <= 2.2 and ob["fp"] <= 1.2) else "unlinked-person-cluster",
                                        f"h {ob['h']:.1f} fp {ob['fp']:.1f} n {ob['n']} d {d:.1f}", bool(ign)))
                else:
                    false_boxes.append((i, b["id"], "no-lidar-object", f"ref3d {b['ref3d']['range_m']} m", bool(ign)))
            else:
                no_ref_match.append((i, b["id"], bool(ign)))
        per_frame[i] = rec

    # ---- merge proposals (same object; same LiDAR track = still phases + walks of one tracked person) / splits
    proposals, proposals_track, splits = [], [], []
    obj_names = ["P%d" % q["id"] for q in persons] + ["M%d" % q["id"] for q in movers]
    obj_of = {"P%d" % q["id"]: q for q in persons}
    obj_of.update({"M%d" % q["id"]: q for q in movers})
    obj_win = {n: (q["windows"] if "windows" in q else [(float(q["t"][0]), float(q["t"][-1]))]) for n, q in obj_of.items()}

    def overlap_s(wa, wb):
        return max((min(a1, b1) - max(a0, b0) for a0, a1 in wa for b0, b1 in wb), default=-1.0)

    id_main = {}
    for tid, cnt in id_P.items():
        name, n = cnt.most_common(1)[0]
        if n / id_boxes[tid] >= 0.5:
            id_main[tid] = name
        objs_t = [(nm, k) for nm, k in cnt.items() if k >= 3]
        for i1 in range(len(objs_t)):
            for i2 in range(i1 + 1, len(objs_t)):
                (n1, k1), (n2, k2) = objs_t[i1], objs_t[i2]
                if overlap_s(obj_win[n1], obj_win[n2]) > 2.0 and min(k1, k2) >= 5 and not (obj_of[n1]["tracks"] & obj_of[n2]["tracks"]):
                    splits.append((tid, n1, k1, n2, k2, id_boxes[tid]))
    for name in obj_names:
        ids = sorted(t for t, nm in id_main.items() if nm == name)
        if len(ids) >= 2:
            main_id = max(ids, key=lambda t: id_boxes[t])
            proposals.append({"object": name, "ids": ids, "into": main_id, "frames": {int(t): int(P_ids[name][t]) for t in ids}})
    # same LiDAR track (union-find over shared track ids)
    parent = {n: n for n in obj_names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    by_track = defaultdict(list)
    for n in obj_names:
        for tr in obj_of[n]["tracks"]:
            if tr >= 0:
                by_track[tr].append(n)
    for ns in by_track.values():
        for n in ns[1:]:
            parent[find(n)] = find(ns[0])
    groups = defaultdict(list)
    for n in obj_names:
        groups[find(n)].append(n)
    for root, ns in groups.items():
        ids = sorted({t for t, nm in id_main.items() if nm in ns})
        if len(ids) >= 2 and len(ns) >= 2 and not any(set(ids) == set(pr["ids"]) for pr in proposals):
            main_id = max(ids, key=lambda t: id_boxes[t])
            proposals_track.append({"objects": sorted(ns), "ids": ids, "into": main_id,
                                    "frames": {int(t): {nm: int(P_ids[nm][t]) for nm in ns if P_ids[nm][t]} for t in ids}})

    log("")
    log("## LiDAR 사람 -> GT id (프레임 수; 0.5 이상이면 그 id의 주 대상)")
    log("")
    log("| 객체 | GT id: 프레임 수 | FOV 안 프레임 | 누락(FOV, 가림 없음, 박스 없음) |")
    log("|---|---|---|---|")
    fov_count = Counter()
    for i, rec in per_frame.items():
        for name, ob in rec["objects"].items():
            if ob.get("in_fov"):
                fov_count[name] += 1
    miss_count = Counter(n for _, n, *_ in missed)
    for name in obj_names:
        ids_s = ", ".join(f"{t}:{n}" for t, n in sorted(P_ids[name].items(), key=lambda x: -x[1])) or "-"
        log(f"| {name} | {ids_s} | {fov_count[name]} | {miss_count[name]} |")
    log("")
    log("## 정지 작업자 위의 GT ref3d 월드 위치 퍼짐 (헤딩 효과): refined 자세 vs cache theta")
    log("")
    log("| 객체 | ref3d 박스 | σ_xy refined [m] | σ_xy cache [m] | 최대 반경 refined | 최대 반경 cache |")
    log("|---|---|---|---|---|---|")
    spread_rows = {}
    for name in obj_names:
        if not name.startswith("P") or len(ref_w[name]["refined"]) < 3:
            continue
        A = np.array(ref_w[name]["refined"]); B = np.array(ref_w[name]["cache"]) if ref_w[name]["cache"] else np.zeros((0, 2))
        sa = float(np.sqrt(((A - np.median(A, 0)) ** 2).sum(1).mean())); ra = float(np.linalg.norm(A - np.median(A, 0), axis=1).max())
        sb = float(np.sqrt(((B - np.median(B, 0)) ** 2).sum(1).mean())) if len(B) > 2 else float("nan")
        rb = float(np.linalg.norm(B - np.median(B, 0), axis=1).max()) if len(B) > 2 else float("nan")
        spread_rows[name] = {"n": len(A), "sigma_refined": sa, "sigma_cache": sb, "max_refined": ra, "max_cache": rb}
        log(f"| {name} | {len(A)} | {sa:.2f} | {sb:.2f} | {ra:.2f} | {rb:.2f} |")
    log("")
    log("## GT id -> LiDAR 사람")
    log("")
    log("| GT id | 박스 수 | LiDAR 객체: 프레임 수 | 주 대상 |")
    log("|---|---|---|---|")
    for tid in sorted(id_boxes):
        s = ", ".join(f"{nm}:{n}" for nm, n in sorted(id_P[tid].items(), key=lambda x: -x[1])) or "-"
        log(f"| {tid} | {id_boxes[tid]} | {s} | {id_main.get(tid, '-')} |")
    log("")
    log("## (a) 합칠 id")
    log("")
    log("같은 LiDAR 정지 작업자가 주 대상인 id (확실):")
    for pr in proposals:
        log(f"- {pr['object']}: ids {pr['ids']} -> {pr['into']} (프레임 수 {pr['frames']})")
    if not proposals:
        log("- 없음")
    log("")
    log("같은 LiDAR 트랙(한 사람의 정지 구간 + 걷는 구간이 이어진 것)이 주 대상인 id (트랙 연결은 1 m 게이트의 최근접 연결이라 영상으로 확인):")
    for pr in proposals_track:
        log(f"- {pr['objects']}: ids {pr['ids']} -> {pr['into']} (프레임 수 {pr['frames']})")
    if not proposals_track:
        log("- 없음")
    if splits:
        log("")
        log("한 id가 같은 시간에 존재한 두 LiDAR 객체에 걸침 (분리 검토):")
        for tid, n1, k1, n2, k2, nb in splits:
            log(f"- id {tid} ({nb} 박스): {n1}:{k1} vs {n2}:{k2}")
    log("")
    by_obj = defaultdict(list)
    for i, name, rng_p, hpx, yaw_dps in missed:
        by_obj[name].append((i, rng_p, hpx, yaw_dps))
    labelled = {n for n in obj_names if P_ids[n]}
    n_b1 = sum(len(v) for n, v in by_obj.items() if n in labelled)
    log(f"## (b) 누락 프레임: GT id가 있는 LiDAR 사람이 FOV 안, 가림 < 50 %, 박스 높이 >= 25 px인데 그 프레임에 박스 없음 ({n_b1}; 35 m 밖은 LiDAR 점이 스캔당 3-5개라 신뢰도 낮음)")
    log("")
    never = []
    for name, rows in sorted(by_obj.items(), key=lambda kv: np.median([r[1] for r in kv[1]])):
        if name not in labelled:
            never.append((name, rows))
            continue
        fr_list = [r[0] for r in rows]
        fast = sum(abs(r[3]) > 30 for r in rows)
        far = " (35 m 밖, 낮은 신뢰도)" if np.median([r[1] for r in rows]) > 35 else ""
        log(f"- {name}{far}: {len(rows)} 프레임 (선회 > 30 deg/s 중 {fast}), 거리 {min(r[1] for r in rows):.0f}-{max(r[1] for r in rows):.0f} m, 박스 높이 {min(r[2] for r in rows):.0f}-{max(r[2] for r in rows):.0f} px: 프레임 {fr_list}")
    log("")
    log(f"GT id가 한 번도 붙지 않은 LiDAR 사람 크기 기둥 (표지판·기둥·장비 일부일 수 있음, 영상으로 확인; {len(never)})")
    log("")
    for name, rows in never:
        q = next(u for u in persons if "P%d" % u["id"] == name) if name.startswith("P") else None
        log(f"- {name}: 보이는 프레임 {len(rows)} (예 {[r[0] for r in rows][:6]}), 거리 {np.median([r[1] for r in rows]):.0f} m, 박스 높이 {np.median([r[2] for r in rows]):.0f} px"
            + (f", LiDAR 높이 {q['h']:.2f} m 폭 {q['fp']:.2f} m 출처 {q['src']}" if q else ""))
    log("")
    n_none = sum(1 for r in false_boxes if r[2] == "no-lidar-object")
    log(f"## (c) LiDAR 사람(P/M)에 대응하지 않는 GT 박스 ({len(false_boxes)} 박스; 그중 LiDAR에 아무 물체도 없음 {n_none}, ref3d 없는 미대응 박스 {len(no_ref_match)})")
    log("")
    log("분류: no-lidar-object = ref3d 위치 반경 게이트 안에 지면 위 LiDAR 점군이 전혀 없음(거짓 GT 후보, 단 35 m 밖은 LiDAR 한계); "
        "not-person-sized = 무언가 있으나 사람 크기가 아님(옆 물체와 합쳐짐·두 사람·장비); unlinked-person-cluster = 사람 크기 군집이 있으나 5 s 이상 추적되지 않음(추적 끊김)")
    log("")
    by_id = defaultdict(list)
    for i, tid, why, det, ign in false_boxes:
        by_id[tid].append((i, why, det, ign))
    for tid, rows in sorted(by_id.items()):
        kinds = Counter(r[1] for r in rows)
        none_fr = [r[0] for r in rows if r[1] == "no-lidar-object"]
        log(f"- id {tid}: {len(rows)}/{id_boxes[tid]} 박스 {dict(kinds)}; ignore {sum(r[3] for r in rows)}"
            + (f"; LiDAR 물체 없음 프레임 {none_fr[:30]}{' ...' if len(none_fr) > 30 else ''} ({rows[[r[0] for r in rows].index(none_fr[0])][2]})" if none_fr else ""))
    if no_ref_match:
        byn = defaultdict(list)
        for i, tid, ign in no_ref_match:
            byn[tid].append(i)
        log("ref3d 없고 2D로도 LiDAR 사람과 안 겹친 박스: " + "; ".join(f"id {t}: {v}" for t, v in sorted(byn.items())))
    log("")
    log(f"## (d) 같은 프레임에 같은 LiDAR 객체 위의 GT id 2개 ({len(dups)})")
    log("")
    for i, name, ids in dups:
        log(f"- 프레임 {i} {name}: ids {ids}")
    log("")
    log(f"## (e) ref3d가 틀린 GT 박스: 영상에서는 LiDAR 사람과 겹치는데 ref3d 거리가 그 사람과 다름 ({len(ref_bad)}; 빠른 선회 프레임의 희소 ref3d가 뒤 배경을 잡은 것)")
    log("")
    by_id = defaultdict(list)
    for i, tid, name, r_gt, r_li in ref_bad:
        by_id[tid].append((i, name, r_gt, r_li))
    for tid, rows in sorted(by_id.items()):
        log(f"- id {tid}: {len(rows)} 박스: " + ", ".join(f"f{i} {name} ref3d {rg:.1f}->{rl:.1f} m" if rg else f"f{i} {name} ref3d none->{rl:.1f} m" for i, name, rg, rl in rows[:12]) + (" ..." if len(rows) > 12 else ""))

    # ---- outputs
    out = {"segment": seg.name, "lever_radar": lever_radar.tolist(), "heading": heading_rows, "ref3d_spread": spread_rows,
           "heading_refined_minus_cache_deg": [None if np.isnan(x) else round(float(x), 2) for x in dthf],
           "persons": [{"id": q["id"], "xy": [round(float(v), 2) for v in q["xy"]], "range": round(q["r"], 1), "h": round(q["h"], 2),
                        "fp": round(q["fp"], 2), "t0": round(q["t0"] - bag0, 1), "t1": round(q["t1"] - bag0, 1), "n_scans": q["n_scans"],
                        "n_pts": q["n_pts"], "windows": [[round(t0 - bag0, 1), round(t1 - bag0, 1)] for t0, t1 in q["windows"]], "src": q["src"],
                        "gt_ids": dict(sorted(P_ids["P%d" % q["id"]].items()))} for q in persons],
           "movers": [{"id": q["id"], "t0": round(q["t"][0] - bag0, 1), "t1": round(q["t"][-1] - bag0, 1), "path_m": round(q["path_m"], 1),
                       "xy_start": [round(float(v), 2) for v in q["xy"][0]], "xy_end": [round(float(v), 2) for v in q["xy"][-1]],
                       "gt_ids": dict(sorted(P_ids["M%d" % q["id"]].items()))} for q in movers],
           "id_to_object": {int(t): dict(sorted(c.items())) for t, c in id_P.items()},
           "merge_proposals": proposals, "merge_proposals_track": proposals_track,
           "splits": [{"id": int(t), "a": n1, "a_frames": int(k1), "b": n2, "b_frames": int(k2), "boxes": int(nb)} for t, n1, k1, n2, k2, nb in splits],
           "missed": [{"frame": i, "object": n, "range": r, "h_px": h, "yaw_dps": y} for i, n, r, h, y in missed if n in labelled],
           "never_labelled": [{"object": n, "frames": [r[0] for r in rows]} for n, rows in never],
           "false_boxes": [{"frame": i, "id": int(t), "why": w, "detail": d, "ignore": g_} for i, t, w, d, g_ in false_boxes],
           "no_ref_unmatched": [{"frame": i, "id": int(t), "ignore": g_} for i, t, g_ in no_ref_match],
           "duplicates": [{"frame": i, "object": n, "ids": ids} for i, n, ids in dups],
           "ref3d_wrong": [{"frame": i, "id": int(t), "object": n, "ref3d_range": round(rg, 2), "lidar_range": rl} for i, t, n, rg, rl in ref_bad],
           "per_frame": per_frame}
    (seg / "lidar_audit.json").write_text(json.dumps(out, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    (seg / "lidar_audit.md").write_text("\n".join(log_lines) + "\n")

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 11))
    hgt = allW[:, 2] - zg(allW[:, :2])
    up = allW[(hgt > 0.3) & (hgt < 3.0)]
    ax.scatter(up[:, 1], up[:, 0], s=0.3, c="#c8c8c8", rasterized=True, label="LiDAR map (0.3-3 m above ground)")
    ax.plot(xy_o[:, 1], xy_o[:, 0], "-", c="#888888", lw=0.8, label="level origin (refined)")
    ax.plot([-lever[1]], [-lever[0]], "k+", ms=10, label="swing axis")
    cmap = plt.get_cmap("tab20")
    ids_sorted = sorted(id_boxes)
    for n_, tid in enumerate(ids_sorted):
        pts = []
        for i, f in enumerate(g.frames):
            for b in f["boxes"]:
                if b["id"] == tid and (b.get("ref3d") or {}).get("xy_level") is not None:
                    xl = b["ref3d"]["xy_level"]
                    pts.append(apply(model.T_WL(frame_rows[i]["p_ref"]), np.array([[xl[0], xl[1], 0.0]]))[0, :2])
        if pts:
            pts = np.array(pts)
            ax.scatter(pts[:, 1], pts[:, 0], s=9, color=cmap(n_ % 20), label=f"id {tid} ({len(pts)})", alpha=0.8)
    for q in persons:
        ax.plot(q["xy"][1], q["xy"][0], "o", mfc="none", mec="red", ms=14, mew=1.5)
        ax.annotate(f"P{q['id']}", (q["xy"][1], q["xy"][0]), fontsize=9, color="red", xytext=(6, 6), textcoords="offset points")
    for q in movers:
        ax.plot(q["xy"][:, 1], q["xy"][:, 0], "-", c="blue", lw=1.2)
        ax.annotate(f"M{q['id']}", (q["xy"][0, 1], q["xy"][0, 0]), fontsize=8, color="blue")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("world y (left) [m]"), ax.set_ylabel("world x (forward) [m]")
    ax.set_title(f"{seg.name}: LiDAR static persons (red) / movers (blue) and GT ref3d by id")
    ax.legend(fontsize=6, loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(seg / "lidar_world_map.png", dpi=110, bbox_inches="tight")
    log(f"\nwrote {seg / 'lidar_audit.md'}, lidar_audit.json, lidar_world_map.png")


if __name__ == "__main__":
    sys.exit(main())
