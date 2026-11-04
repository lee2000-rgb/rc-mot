"""Ego pose in a world frame fixed at the start of the processed segment (docs/24): gyro yaw + radar-Doppler sensor
velocity, optionally with the swing (slew) axis model of an excavator (docs/23 §3):

  level-frame origin (ground below the LiDAR) = axis + R(theta) r,   r = its lever arm from the swing axis
  v_sensor = v_axis + w x r   ->   v_axis = v_sensor - w x r          (both in the current level frame)

With the lever arm the axis stays put during an in-place swing (only the yaw is integrated: no translation drift);
without it the sensor itself is dead-reckoned from its Doppler velocity (~1.2 m drift per 45 s of fast swinging).
The gyro bias is re-learnt while the radar says the machine stands still, and the yaw is held meanwhile
(Ouster IMU bias ~0.7 deg/s, docs/23 §3.5)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


@dataclass
class WorldParams:
    enable: bool = True
    lever_arm: list | None = None     # RADAR position from the swing axis [m], machine frame ~ level frame (x fwd, y left)
    still_speed: float = 0.05         # [m/s] sensor speed below which (with |w - bias| < still_rate) the machine stands still
    still_rate_dps: float = 2.0
    bias_tau_s: float = 10.0          # gyro-bias time constant while standing still
    axis_deadband: float = 0.15       # [m/s] axis speed below this = not driving (swing residuals, Doppler noise)
    trail_step: float = 0.1           # [m] spacing of the stored trail points


@dataclass
class WorldState:
    t: float
    theta: float                      # heading of the level frame in the world frame [rad]
    origin: np.ndarray                # (2,) world xy of the level-frame origin
    axis: np.ndarray | None           # (2,) world xy of the swing axis (lever-arm model only)
    lever: np.ndarray | None          # (2,) lever arm of the level-frame origin (level frame)
    still: bool
    bias: float                       # gyro bias [rad/s]
    w: float                          # bias-corrected yaw rate [rad/s]
    trail_ref: list                   # past origin positions (world), shared append-only list of the segment ...
    trail_n: int                      # ... of which the first trail_n existed at this frame
    axis_trail_ref: list | None
    axis_trail_n: int = 0

    def trail(self, k: int = 2000) -> np.ndarray:
        return np.array(self.trail_ref[max(self.trail_n - k, 0):self.trail_n]).reshape(-1, 2)

    def axis_trail(self, k: int = 2000) -> np.ndarray | None:
        if self.axis_trail_ref is None:
            return None
        return np.array(self.axis_trail_ref[max(self.axis_trail_n - k, 0):self.axis_trail_n]).reshape(-1, 2)

    def R(self) -> np.ndarray:
        return rot2(self.theta)

    def to_world(self, p) -> np.ndarray:
        """Level-frame points (N,>=2) -> world xy (N,2)."""
        p = np.atleast_2d(np.asarray(p, float))
        return p[:, :2] @ self.R().T + self.origin


class WorldPose:
    def __init__(self, params: WorldParams, radar_t_level=None):
        """radar_t_level: radar position in the level frame (calib t_lr); converts the configured radar lever arm into
        the lever arm of the level-frame origin."""
        self.p = params
        la = params.lever_arm
        rt = np.zeros(2) if radar_t_level is None else np.asarray(radar_t_level, float)[:2]
        self.lever = None if la is None else np.asarray(la, float)[:2] - rt
        self.reset()

    def reset(self):
        self.theta, self.bias, self.t = 0.0, 0.0, None
        self.axis = None if self.lever is None else -self.lever.copy()     # world = level frame at the start
        self.origin = np.zeros(2)
        self.trail = [self.origin.copy()]                  # new lists per segment: states of the old one keep theirs
        self.axis_trail = None if self.lever is None else [self.axis.copy()]

    def update(self, t: float, dt: float, yaw_rate: float, v_sensor, valid: bool = True) -> WorldState:
        p = self.p
        v = np.asarray(v_sensor, float)[:2]
        still = bool(valid and np.linalg.norm(v) < p.still_speed and abs(yaw_rate - self.bias) < np.radians(p.still_rate_dps))
        if still:
            self.bias += (yaw_rate - self.bias) * min(dt / p.bias_tau_s, 1.0)
            w = 0.0
        else:
            w = yaw_rate - self.bias
        th_mid = self.theta + 0.5 * w * dt
        if self.lever is not None:
            r = self.lever
            v_ax = v - w * np.array([-r[1], r[0]])
            if still or np.linalg.norm(v_ax) < p.axis_deadband:
                v_ax = np.zeros(2)
            self.axis = self.axis + rot2(th_mid) @ v_ax * dt
            self.theta += w * dt
            self.origin = self.axis + rot2(self.theta) @ r
        else:
            if not still:
                self.origin = self.origin + rot2(th_mid) @ v * dt
            self.theta += w * dt
        if np.linalg.norm(self.origin - self.trail[-1]) >= p.trail_step:
            self.trail.append(self.origin.copy())
            if self.axis_trail is not None and np.linalg.norm(self.axis - self.axis_trail[-1]) >= p.trail_step:
                self.axis_trail.append(self.axis.copy())
        self.t = t
        return WorldState(t, self.theta, self.origin.copy(), None if self.axis is None else self.axis.copy(),
                          None if self.lever is None else self.lever.copy(), still, self.bias, w,
                          self.trail, len(self.trail), self.axis_trail, 0 if self.axis_trail is None else len(self.axis_trail))
