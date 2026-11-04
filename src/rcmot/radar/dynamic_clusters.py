"""Moving radar objects not explained by the camera ("unclassified movers", docs/08 §3[B]-4, docs/09 §4.4).

Dynamic points (ScanStats.dynamic) are linked when their standardised distance is small:
    d_ij^2 = |p_i - p_j|^2 / (s_i^2 + s_j^2 + e^2) + (v_i - v_j)^2 / (2 sigma_v^2)
with s = per-point position sigma (range sigma along the line of sight, r * sigma_az across it, isotropic
max used for simplicity), e = object extent, sigma_v = the scan's Doppler noise. Connected components with
enough summed reliability are reported; points already used by a camera detection or a track are excluded.
No class, no track birth (v1): shown in the GUI as a warning layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from rcmot.radar.ego_velocity import estimate_object_velocity


@dataclass
class Mover:
    pos: np.ndarray                   # (3,) level frame centroid
    vel: np.ndarray | None            # (3,) absolute velocity estimate (level) or None
    v_radial: float                   # weighted median compensated Doppler [m/s]
    n: int
    weight: float                     # summed reliability
    idx: np.ndarray


def cluster_movers(radar, calib, *, exclude: np.ndarray | None = None, d_max: float = 3.0, extent: float = 0.5,
                   sigma_range: float = 0.15, sigma_az_deg: float = 1.0, min_weight: float = 1.5,
                   min_pts: int = 2, max_range: float = 35.0, ego_k: float = 0.12) -> list[Mover]:
    """max_range: movers beyond it are dropped (far 2-3-point returns, nothing the camera could confirm).
    ego_k: the dynamic speed floor is raised to ego_k * |v_sensor| while the rig moves. On a 67 deg/s swing the
    sensor moves 1.5 m/s and turns >3 deg within one scan, so static returns 50 m away keep 0.1-0.2 m/s of
    compensated Doppler (142537: 6.4 '?' per frame at 50-90 deg/s, 0.05 when still)."""
    st = radar.stats
    if st is None or len(radar.v_r) == 0:
        return []
    m = st.dynamic.copy()
    if exclude is not None and len(exclude):
        m[exclude] = False
    v_floor = ego_k * float(np.linalg.norm(st.v_sensor)) if np.isfinite(st.v_sensor).all() else 0.0
    m &= (np.abs(st.v_comp) > v_floor) & (np.linalg.norm(radar.xyz_radar, axis=1) < max_range)
    idx = np.flatnonzero(m)
    if len(idx) < min_pts:
        return []
    p = radar.xyz[idx]
    r = np.linalg.norm(radar.xyz_radar[idx], axis=1)
    s = np.maximum(sigma_range, r * np.radians(sigma_az_deg))
    v = st.v_comp[idx]
    sv = max(st.sigma_v, 0.02)
    radius = d_max * np.sqrt(2 * s.max() ** 2 + extent ** 2)        # largest distance that can pass the test
    pairs = cKDTree(p[:, :2]).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]
        d2 = ((p[i, :2] - p[j, :2]) ** 2).sum(1) / (s[i] ** 2 + s[j] ** 2 + extent ** 2) + (v[i] - v[j]) ** 2 / (2 * sv ** 2)
        pairs = pairs[d2 <= d_max ** 2]
    n = len(idx)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    lab = connected_components(g, directed=False)[1]
    out = []
    for c in np.unique(lab):
        k = np.flatnonzero(lab == c)
        w = st.alpha[idx[k]]
        if len(k) < min_pts or w.sum() < min_weight:
            continue
        pos = np.average(p[k], axis=0, weights=w + 1e-3)
        ov = estimate_object_velocity(radar.xyz_radar[idx[k]], v[k], sigma_vr=sv)
        vel = calib.R_lr @ ov.v if ov is not None and not ov.radial_only else None
        out.append(Mover(pos, vel, float(np.median(v[k])), len(k), float(w.sum()), idx[k]))
    return out
