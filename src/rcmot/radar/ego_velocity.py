"""Doppler-based ego-velocity and object-velocity estimation for 4D radar.

Model (radar frame, sensor at origin):
    a static point i with unit direction d_i measures   v_r,i = -d_i . v_s
    a point on a moving rigid object (translation v_o) v_r,i =  d_i . (v_o - v_s)
so after ego compensation   v_r,i + d_i . v_s = d_i . v_o .

References (papers/03_radar_ego_velocity_doppler): Kellner et al. 2013 (instantaneous ego-motion),
Doer & Trommer 2020 (REVE, RANSAC + LSQ), Kellner et al. 2014 (full-motion of objects from Doppler).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EgoVelocityResult:
    v_sensor: np.ndarray          # (3,) sensor velocity in the radar frame [m/s]
    inlier_mask: np.ndarray       # (N,) True = static point
    valid: bool
    residual_std: float


def unit_directions(xyz: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(r, 1e-6)


def estimate_ego_velocity(xyz: np.ndarray, v_r: np.ndarray, *, iters: int = 60, inlier_thr: float = 0.15,
                          min_points: int = 8, min_inlier_ratio: float = 0.35, planar: bool = False,
                          rng: np.random.Generator | None = None) -> EgoVelocityResult:
    """RANSAC + least squares on  v_r = -D v_s.

    planar=True estimates only (vx, vy) (use when elevation is noisy or the vehicle moves on a plane).
    """
    n = len(v_r)
    dim = 2 if planar else 3
    if n < min_points:
        return EgoVelocityResult(np.zeros(3), np.zeros(n, bool), False, np.inf)
    D = unit_directions(xyz)[:, :dim]
    rng = rng or np.random.default_rng(0)
    best = np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, dim, replace=False)
        try:
            v = np.linalg.lstsq(-D[idx], v_r[idx], rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        inl = np.abs(-D @ v - v_r) < inlier_thr
        if inl.sum() > best.sum():
            best = inl
    if best.sum() < max(min_points, min_inlier_ratio * n):
        return EgoVelocityResult(np.zeros(3), best, False, np.inf)
    v = np.linalg.lstsq(-D[best], v_r[best], rcond=None)[0]
    inl = np.abs(-D @ v - v_r) < inlier_thr                         # re-score with refined model
    v = np.linalg.lstsq(-D[inl], v_r[inl], rcond=None)[0]
    res = (-D[inl] @ v - v_r[inl]).std()
    v3 = np.zeros(3)
    v3[:dim] = v
    return EgoVelocityResult(v3, inl, True, float(res))


def compensate_doppler(xyz: np.ndarray, v_r: np.ndarray, v_sensor: np.ndarray) -> np.ndarray:
    """Ego-compensated radial velocity (projection of the absolute point velocity onto d_i)."""
    return v_r + unit_directions(xyz) @ v_sensor


@dataclass
class ObjectVelocity:
    v: np.ndarray                 # (3,) absolute velocity in radar frame (only observable part is reliable)
    cov: np.ndarray               # (3,3)
    radial_only: bool             # True when the cluster's angular spread is too small for a full solve


def estimate_object_velocity(xyz: np.ndarray, v_r_comp: np.ndarray, *, sigma_vr: float = 0.1,
                             min_spread_deg: float = 2.0, planar: bool = True) -> ObjectVelocity | None:
    """Least-squares object velocity from ego-compensated Doppler of one cluster.

    With a small azimuth spread only the radial component is observable; then the radial projection along
    the cluster centroid direction is returned (radial_only=True) and the tracker's KF supplies the
    tangential component.
    """
    if len(v_r_comp) == 0:
        return None
    D = unit_directions(xyz)
    dim = 2 if planar else 3
    az = np.degrees(np.arctan2(D[:, 1], D[:, 0]))
    if len(v_r_comp) >= dim + 1 and np.ptp(az) >= min_spread_deg:
        A = D[:, :dim]
        info = A.T @ A / sigma_vr ** 2
        if np.linalg.cond(info) < 1e4:
            v = np.linalg.solve(info, A.T @ v_r_comp / sigma_vr ** 2)
            cov = np.full((3, 3), 0.0)
            cov[:dim, :dim] = np.linalg.inv(info)
            if planar:
                cov[2, 2] = 1e3
            out = np.zeros(3)
            out[:dim] = v
            return ObjectVelocity(out, cov, False)
    c = unit_directions(xyz.mean(0, keepdims=True))[0]
    vr = float(np.median(v_r_comp))
    cov = np.eye(3) * 1e3                                           # unobservable directions
    cov -= np.outer(c, c) * (1e3 - sigma_vr ** 2 / len(v_r_comp))   # well-known along the line of sight
    return ObjectVelocity(vr * c, cov, True)
