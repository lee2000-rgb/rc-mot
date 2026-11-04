"""Corner-reflector correspondences between the 4D radar and the LiDAR (field calibration).

Radar side: Doppler-static points of a stationary capture that stand out from their local background (MCL2026
eq. (1): per-scan polar k-NN robust z, min over RCS and SNR) are clustered in x-y; a reflector is a compact
cluster seen in most frames (persistence).
LiDAR side: around the radar cluster (after the initial extrinsic) the LiDAR must show a small isolated
object above the ground (reflector on a tripod / pole); its centroid is the match.

Radar range / azimuth are accurate and elevation is not, so the correspondences constrain x, y and yaw
(used as horizontal anchors in radar_lidar.joint_icp); roll, pitch, z come from the point-to-plane ICP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass
class Reflector:
    p_radar: np.ndarray            # (3,) radar-frame centroid
    q_lidar: np.ndarray | None     # (3,) os_sensor centroid of the matched LiDAR object
    rcs: float
    persistence: float             # radar points per frame
    spread: float                  # xy std of the radar cluster [m]
    lidar_extent: float = np.nan   # xy extent of the LiDAR object [m]
    note: str = ""


def _clusters(xy: np.ndarray, radius: float) -> np.ndarray:
    tree = cKDTree(xy)
    pairs = tree.query_pairs(radius, output_type="ndarray")
    n = len(xy)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    return connected_components(g, directed=False)[1]


def detect(window, ex, ground_normal, sensor_height, *, radius: float = 0.6, min_persistence: float = 0.5,
           contrast_min: float = 3.0, max_spread: float = 0.5, rng: tuple = (2.5, 45.0), search: float = 2.0,
           max_extent: float = 1.2, min_lidar_pts: int = 5) -> list[Reflector]:
    """Reflectors of one stationary window (rcmot.calib.radar_lidar.Window loaded with keep_raw=True)."""
    p = window.p
    if len(p) == 0:
        return []
    ps = ex.radar_to_sensor(p)
    # cluster only the points that stand out: MCL2026 local contrast min(z_RCS, z_SNR) computed per scan
    # (load_window); without it, the robust z of RCS over the whole window (still relative, not absolute)
    con = getattr(window, "contrast", None)
    if con is None or len(con) != len(p):
        from rcmot.radar.scan_stats import robust_z
        con = robust_z(window.rcs)
    bright = np.flatnonzero(con >= contrast_min)
    if len(bright) == 0:
        return []
    lab = _clusters(ps[bright, :2], radius)
    out = []
    for c in np.unique(lab):
        idx = bright[lab == c]
        pers = len(idx) / max(window.n_frames, 1)
        if pers < min_persistence:
            continue
        xy = ps[idx, :2]
        spread = float(np.sqrt(xy.var(0).sum()))
        r = float(np.linalg.norm(xy.mean(0)))
        mean_rcs = float(np.median(window.rcs[idx]))
        if spread > max_spread or not (rng[0] <= r <= rng[1]):
            continue
        out.append(Reflector(np.median(p[idx], axis=0), None, mean_rcs, pers, spread))
    # LiDAR match: small isolated above-ground object near each candidate
    L = window.lidar_raw
    if L is None or not out:
        return out
    n = np.asarray(ground_normal, float)
    above = L[(L @ n + sensor_height) > 0.2]
    tree = cKDTree(above[:, :2])
    for ref in out:
        c = ex.radar_to_sensor(ref.p_radar[None])[0]
        near = tree.query_ball_point(c[:2], search)
        if len(near) < min_lidar_pts:
            ref.note = "no LiDAR object near the radar cluster"
            continue
        pts = above[near]
        lab = _clusters(pts[:, :2], 0.3)
        best = None
        for k in np.unique(lab):
            q = pts[lab == k]
            if len(q) < min_lidar_pts:
                continue
            ext = float(np.ptp(q[:, 0]) ** 2 + np.ptp(q[:, 1]) ** 2) ** 0.5
            d = np.linalg.norm(q[:, :2].mean(0) - c[:2])
            if ext <= max_extent and (best is None or d < best[0]):
                best = (d, q, ext)
        if best is None:
            ref.note = "LiDAR object near the cluster is too large (not an isolated reflector)"
            continue
        # the whole cluster must be isolated: no other above-ground LiDAR points just around it
        q = best[1]
        ring = tree.query_ball_point(q[:, :2].mean(0), best[2] / 2 + 0.6)
        if len(ring) > 1.5 * len(q):
            ref.note = "LiDAR object not isolated"
            continue
        ref.q_lidar, ref.lidar_extent = q.mean(0), best[2]
    return out


def anchors(refs: list[Reflector]) -> list[tuple[np.ndarray, np.ndarray]]:
    return [(r.p_radar, r.q_lidar) for r in refs if r.q_lidar is not None]
