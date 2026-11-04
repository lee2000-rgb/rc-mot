#!/usr/bin/env python3
"""EXPERIMENTAL (scripts only, not wired into src/): EgoMotionProvider for the rc-mot tracker on the agmo rig.

Wraps the delivered radar+IMU dead-reckoning backbone (CausalBackbone, radar_dr_delivery) and optionally a
tiny 2D EKF that absorbs RTK FIXED positions. Per tracker step it returns the relative motion of the
*tracker frame* (level frame attached to the sensor head, i.e. base_link + lever arm) between two scan times.

    prov = EgoMotionProvider(lever=2.09, radar_time_offset=-0.04)
    prov.add_imu(t, acc_xyz, gyro_xyz)               # /agmo/imu (header stamp), base_link axes
    prov.add_radar_ego(t_radar, v_sensor_xy, ok)     # ScanStats v_sensor (radar frame), ok = ego_valid
    prov.add_gnss(t, east, north, rtk_code, v_en)    # /agmo/gnss/fix (ENU) + status, optional
    step = prov.step(t_prev, t_cur)                  # -> EgoStep(dp (2,), dyaw, v_ego (2,), flags)

The tracker then calls  track.ego_compensate(dp3, dyaw)  exactly as today (pipeline.py), i.e. dp is the
displacement of the tracker frame expressed in the previous tracker frame, dyaw its rotation.
World pose (for maps / danger zones / static-object persistence): prov.world_pose(t) -> (E, N, yaw, valid).
"""
from __future__ import annotations

import bisect
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, "/home/islab/agmo/workspaces/radar-baseline/radar_dr_delivery/offline")
from radar_imu_odometry.backbone import CausalBackbone  # noqa: E402


def _R(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class EgoStep:
    dp: np.ndarray                    # (2,) tracker-frame displacement, expressed in the previous tracker frame [m]
    dyaw: float                       # [rad]
    v_ego: np.ndarray                 # (2,) tracker-frame velocity at t_cur (for Doppler compensation) [m/s]
    flags: dict = field(default_factory=dict)   # radar_ego_ok, ego_age_s, rtk_fixed, gnss_used, world_valid


class EgoMotionProvider:
    def __init__(self, lever: float = 2.09, radar_yaw_deg: float = 0.0, radar_time_offset: float = -0.04,
                 v_max: float = 3.0, ego_timeout_s: float = 0.5, use_gnss: bool = False,
                 rtk_sigma: float = 0.02, q_pos: float = 0.05, q_yaw_deg: float = 0.3, freeze_bias_after_s: float = 10.0):
        self.bb = CausalBackbone(lever=lever)
        self.L = lever
        self.Rm = _R(np.radians(radar_yaw_deg))     # radar -> body (2D)
        self.off = radar_time_offset
        self.v_max = v_max
        self.timeout = ego_timeout_s
        self.use_gnss = use_gnss
        self.t_first = None
        self.freeze_after = freeze_bias_after_s
        self.last_ego_t = -np.inf
        self.last_v = np.zeros(2)
        self.last_wz = 0.0
        # 2D EKF on base pose in ENU (x, y, yaw), driven by backbone increments; RTK FIXED position updates
        self.x = None
        self.P = np.diag([1e6, 1e6, 1e2])
        self.rtk_var = rtk_sigma ** 2
        self.q_pos, self.q_yaw = q_pos, np.radians(q_yaw_deg)
        self.t_ekf = None
        self.hist: list[tuple[float, np.ndarray]] = []      # (t, ekf state) for interpolation
        self.rtk_fixed = False
        self._n_rej = 0
        self._dt, self._dx, self._dy, self._dw = [], [], [], []       # own DR history (O(log n) lookup)

    # ---------------- inputs ----------------
    def add_imu(self, t, acc, gyro):
        if self.t_first is None:
            self.t_first = t
        self.bb.add_imu(t, acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2])
        self.last_wz = gyro[2]
        if not self.bb._bias_frozen and t - self.t_first > self.freeze_after:
            self.bb.freeze_bias()

    def add_radar_ego(self, t_radar, v_sensor_xy, ok: bool):
        """Radar-frame sensor velocity -> body forward speed (the backbone adds yaw_rate*lever itself)."""
        t = t_radar + self.off
        v = np.asarray(v_sensor_xy, float)
        vb = self.Rm @ v
        ok = bool(ok) and np.all(np.isfinite(vb)) and np.linalg.norm(vb) < self.v_max   # reject absurd fits
        if ok:
            self.last_ego_t, self.last_v = t, vb
            self.bb.add_ego(t, vb[0], 0.0)
            self._log(t)
        elif t - self.last_ego_t > self.timeout:
            # stale radar ego (sparse/blocked scene): hold zero-lateral, decay forward speed
            self.bb.add_ego(t, 0.5 * self.last_v[0], 0.0)
            self.last_v = 0.5 * self.last_v
            self._log(t)
        self._ekf_predict_to(t)

    def add_gnss(self, t, east, north, rtk_code: int, v_en=None):
        self.rtk_fixed = rtk_code == 4
        if not self.use_gnss:
            return
        self._ekf_predict_to(t)
        if self.x is None:
            if rtk_code == 4 and v_en is not None and np.hypot(*v_en) > 0.2:
                h0 = np.arctan2(v_en[1], v_en[0])                                    # init heading from track
                self.x = np.array([east + self.L * np.cos(h0), north + self.L * np.sin(h0), h0])
                self.P = np.diag([self.rtk_var, self.rtk_var, np.radians(10) ** 2])
            return
        if rtk_code != 4:
            return                                       # FIXED only (DGNSS jumps up to 2 m on agmoD)
        c, s = np.cos(self.x[2]), np.sin(self.x[2])
        H = np.array([[1, 0, self.L * s], [0, 1, -self.L * c]], float)
        S = H @ self.P @ H.T + np.eye(2) * self.rtk_var
        K = self.P @ H.T @ np.linalg.inv(S)
        r = np.array([east, north]) - (self.x[:2] - self.L * np.array([c, s]))
        if r @ np.linalg.solve(S, r) > 100 and self._n_rej < 20:     # gate gross outliers, re-accept if persistent
            self._n_rej += 1
            return
        self._n_rej = 0
        self.x = self.x + K @ r
        self.P = (np.eye(3) - K @ H) @ self.P
        self.hist.append((t, self.x.copy()))

    def _log(self, t):
        if self._dt and t <= self._dt[-1]:
            return
        self._dt.append(t); self._dx.append(self.bb._pos[0]); self._dy.append(self.bb._pos[1]); self._dw.append(self.bb._yaw)

    def dr_pose(self, t):
        """(x, y, yaw) of the DR backbone at t (linear interpolation, held at the ends)."""
        n = len(self._dt)
        if n == 0:
            return np.zeros(3)
        k = bisect.bisect_left(self._dt, t)
        if k <= 0:
            return np.array([self._dx[0], self._dy[0], self._dw[0]])
        if k >= n:
            return np.array([self._dx[-1], self._dy[-1], self._dw[-1]])
        t0, t1 = self._dt[k - 1], self._dt[k]
        f = (t - t0) / (t1 - t0)
        return np.array([self._dx[k - 1] + f * (self._dx[k] - self._dx[k - 1]), self._dy[k - 1] + f * (self._dy[k] - self._dy[k - 1]),
                         self._dw[k - 1] + f * (self._dw[k] - self._dw[k - 1])])

    def _ekf_predict_to(self, t):
        if self.x is None or not self.bb.ready():
            return
        if self.t_ekf is None:
            self.t_ekf = t
            return
        if t <= self.t_ekf:
            return
        pa, pb = self.dr_pose(self.t_ekf), self.dr_pose(t)
        dx, dy = _R(-pa[2]) @ (pb[:2] - pa[:2])
        dw = pb[2] - pa[2]
        c, s = np.cos(self.x[2]), np.sin(self.x[2])
        F = np.array([[1, 0, -s * dx - c * dy], [0, 1, c * dx - s * dy], [0, 0, 1]])
        self.x = self.x + np.array([c * dx - s * dy, s * dx + c * dy, dw])
        d = np.hypot(dx, dy)
        sp = (self.q_pos * d) ** 2 + 0.003 ** 2                      # 5 %/distance + 3 mm per radar step
        Q = np.diag([sp, sp, (0.05 * abs(dw)) ** 2 + (self.q_yaw * 0.066) ** 2])
        self.P = F @ self.P @ F.T + Q
        self.t_ekf = t
        self.hist.append((t, self.x.copy()))

    # ---------------- outputs ----------------
    def sensor_pose(self, t):
        """(x, y, yaw) of the sensor head (tracker frame): EKF in ENU (if GNSS used and initialised) else DR frame."""
        if self.use_gnss and len(self.hist) >= 2:
            k = bisect.bisect_left(self._ht, t) if hasattr(self, "_ht") and len(self._ht) == len(self.hist) else None
            if k is None:
                self._ht = [h[0] for h in self.hist]
                k = bisect.bisect_left(self._ht, t)
            k = min(max(k, 1), len(self.hist) - 1)
            (t0, x0), (t1, x1) = self.hist[k - 1], self.hist[k]
            f = np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1)
            return x0 + f * (x1 - x0)
        return self.dr_pose(t)

    def world_pose(self, t):
        if not (self.use_gnss and self.x is not None):
            return None
        return self.sensor_pose(t), self.rtk_fixed

    def step(self, t_prev, t_cur) -> EgoStep:
        a, b = self.sensor_pose(t_prev + self.off), self.sensor_pose(t_cur + self.off)
        # the backbone integrates R(yaw)[v_x, v_y + yaw_rate*lever], i.e. the radar (sensor-head) point -> tracker frame
        dp = _R(-a[2]) @ (b[:2] - a[:2])
        dyaw = _wrap(b[2] - a[2])
        v = np.array([self.last_v[0], self.last_wz * self.L])
        return EgoStep(dp, float(dyaw), v, dict(radar_ego_ok=(t_cur + self.off - self.last_ego_t) < self.timeout,
                                                ego_age_s=float(t_cur + self.off - self.last_ego_t),
                                                rtk_fixed=self.rtk_fixed, gnss_used=self.use_gnss and self.x is not None))
