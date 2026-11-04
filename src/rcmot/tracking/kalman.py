"""3D constant-velocity Kalman filter with optional Doppler-velocity measurement.

State x = [px, py, pz, vx, vy, vz] in a gravity-aligned world (odom) frame, so absolute velocity is
the state itself. Measurements:
    position  z_p = H_p x + n,  H_p = [I3 0]
    velocity  z_v = H_v x + n,  H_v = [0 I3]   (cov from radar LSQ; radial-only case has huge
                                                 cov in the unobservable directions)
"""
from __future__ import annotations

import numpy as np

H_POS = np.hstack([np.eye(3), np.zeros((3, 3))])
H_VEL = np.hstack([np.zeros((3, 3)), np.eye(3)])


class KalmanCV3D:
    def __init__(self, pos: np.ndarray, pos_cov: np.ndarray, *, accel_std: float = 1.5,
                 init_vel_std: float = 3.0):
        self.x = np.concatenate([pos, np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = pos_cov
        self.P[3:, 3:] = np.eye(3) * init_vel_std ** 2
        self.accel_std = accel_std

    def predict(self, dt: float) -> None:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = self.accel_std ** 2
        G = np.vstack([np.eye(3) * 0.5 * dt ** 2, np.eye(3) * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q

    def _update(self, z: np.ndarray, R: np.ndarray, H: np.ndarray) -> None:
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form

    def update_position(self, z: np.ndarray, R: np.ndarray) -> None:
        self._update(z, R, H_POS)

    def update_velocity(self, z: np.ndarray, R: np.ndarray) -> None:
        self._update(z, R, H_VEL)

    def mahalanobis_pos(self, z: np.ndarray, R: np.ndarray) -> float:
        y = z - H_POS @ self.x
        S = H_POS @ self.P @ H_POS.T + R
        return float(y @ np.linalg.solve(S, y))

    @property
    def pos(self) -> np.ndarray:
        return self.x[:3]

    @property
    def vel(self) -> np.ndarray:
        return self.x[3:]

    def update_radial(self, u: np.ndarray, z: float, var: float) -> None:
        """Scalar radial-velocity measurement z = u . v (u: unit line of sight, level frame)."""
        H = np.r_[np.zeros(3), u][None, :]
        self._update(np.array([z]), np.array([[var]]), H)

    def radial_d2(self, u: np.ndarray, z: float, var: float) -> float:
        """Squared Mahalanobis distance of a radial measurement (chi^2, 1 dof)."""
        s = u @ self.P[3:, 3:] @ u + var
        return float((z - u @ self.vel) ** 2 / s)

    def velocity_d2(self, v: np.ndarray, cov: np.ndarray) -> float:
        """Squared Mahalanobis distance of a full horizontal velocity measurement (chi^2, 2 dof)."""
        S = self.P[3:5, 3:5] + cov[:2, :2]
        e = v[:2] - self.vel[:2]
        return float(e @ np.linalg.solve(S, e))
