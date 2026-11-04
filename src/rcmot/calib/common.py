"""Shared calibration helpers: LiDAR ground plane, stationary windows from radar Doppler, tf priors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

from rcmot.radar.ego_velocity import estimate_ego_velocity
from rcmot.sensors.bag_source import pc2_to_struct


@dataclass(frozen=True)
class Topics:
    """Sensor topics of a rig (defaults = HDX Boryeong rig)."""
    image: str = "/oak/rgb/image_raw"
    camera_info: str = "/oak/rgb/camera_info"
    lidar: str = "/ouster/points"
    radar: str = "/off_highway_premium_radar_sample_driver/locations"
    imu: str = "/ouster/imu"

    def all(self) -> list[str]:
        return [self.image, self.camera_info, self.lidar, self.radar, self.imu, "/tf_static"]


DEFAULT_TOPICS = Topics()

# Ouster OS1 lidar_to_sensor_transform (os_sensor -> os_lidar), used when /tf_static was not recorded
OUSTER_OS_LIDAR = (np.diag([-1.0, -1.0, 1.0]), np.array([0.0, 0.0, 0.03618]))


def lidar_to_sensor(tf: dict, frame: str):
    """(R, t) with p_os_sensor = R p_frame + t for a LiDAR cloud published in `frame`."""
    if frame == "os_sensor":
        return np.eye(3), np.zeros(3)
    if ("os_sensor", frame) in tf:
        return tf[("os_sensor", frame)]
    if frame == "os_lidar":
        return OUSTER_OS_LIDAR
    raise KeyError(f"no os_sensor -> {frame} transform (record /tf_static)")


def ground_plane(xyz: np.ndarray, iters: int = 400, thr: float = 0.08, seed: int = 0):
    """RANSAC plane n.p + d = 0 (n up) on near-field low points -> (n, d = sensor height, n_inliers)."""
    rng = np.random.default_rng(seed)
    cand = xyz[(np.linalg.norm(xyz[:, :2], axis=1) < 25) & (xyz[:, 2] < 0)]
    if len(cand) > 60000:
        cand = cand[rng.choice(len(cand), 60000, replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        p = cand[rng.choice(len(cand), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        n = n / np.linalg.norm(n) * np.sign(n[2] or 1)
        if n[2] < 0.9:
            continue
        inl = np.abs(cand @ n - n @ p[0]) < thr
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    pts = cand[best]
    c = pts.mean(0)
    n = np.linalg.svd(pts - c, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    return n, float(-n @ c), best_n


def ground_plane_near(xyz: np.ndarray, n_prior, **kw):
    """ground_plane in a frame levelled by the prior normal, so steep mounts work too (ground_plane alone accepts
    tilts up to ~26 deg; the HDX excavator B rig looks 26 deg down). -> (n, d, n_inliers) in xyz's frame."""
    n0 = np.asarray(n_prior, float) / np.linalg.norm(n_prior)
    x = np.array([1.0, 0, 0]) - n0[0] * n0
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(n0, x), n0])
    n, d, k = ground_plane(xyz @ R.T, **kw)
    return R.T @ n, d, k


def stationary_windows(bag: str, *, still_speed: float = 0.08, min_len: float = 3.0, max_len: float = 60.0,
                       margin: float = 0.5, topics: Topics = DEFAULT_TOPICS) -> list[tuple[float, float]]:
    """Seconds-from-start windows where the radar ego speed (Doppler RANSAC, per-second median) is ~0.
    Long stops are cut into <= max_len pieces (each piece is one ICP window)."""
    sp: dict[int, list] = {}
    with AnyReader([Path(bag)]) as r:
        c = [x for x in r.connections if x.topic == topics.radar][0]
        for _, tn, raw in r.messages(connections=[c]):
            a = pc2_to_struct(r.deserialize(raw, c.msgtype))
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
            vr = a["radial_velocity"].astype(float)
            k = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1)
            if k.sum() < 8:
                continue
            if np.percentile(np.abs(vr[k]), 70) < 0.05:
                s = 0.0
            else:
                e = estimate_ego_velocity(xyz[k], vr[k], inlier_thr=0.2, planar=True)
                s = float(np.linalg.norm(e.v_sensor[:2])) if e.valid else np.nan
            sp.setdefault(int((tn - r.start_time) / 1e9), []).append(s)
    secs = sorted(sp)
    still = {i: np.nanmedian(sp[i]) < still_speed for i in secs}
    out, i = [], 0
    while i < len(secs):
        if still[secs[i]]:
            j = i
            while j + 1 < len(secs) and still[secs[j + 1]] and secs[j + 1] == secs[j] + 1:
                j += 1
            t0, t1 = secs[i] + margin, secs[j] + 1 - margin
            while t1 - t0 >= min_len:
                out.append((t0, min(t1, t0 + max_len)))
                t0 += max_len
            i = j + 1
        else:
            i += 1
    return out


def tf_prior(bag: str) -> dict:
    """os_sensor -> camera / radar translations from a bag's /tf_static (e.g. the HDX detection bags:
    os_sensor->oak-d-base-frame, os_sensor->bosch_4d_frame). Rotations there are 0 (hand measured)."""
    out = {}
    with AnyReader([Path(bag)]) as r:
        cs = [c for c in r.connections if c.topic == "/tf_static"]
        for c, _, raw in r.messages(connections=cs):
            for t in r.deserialize(raw, c.msgtype).transforms:
                tr, q = t.transform.translation, t.transform.rotation
                out[(t.header.frame_id, t.child_frame_id)] = {
                    "t": [tr.x, tr.y, tr.z],
                    "rpy_deg": Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz", degrees=True).tolist()}
    return out
