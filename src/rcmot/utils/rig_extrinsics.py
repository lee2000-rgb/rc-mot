"""Extrinsic model of the HDX rig (Ouster os_sensor as the hub) and its conversion to RigCalib json.

  radar  -> os_sensor : p_s = R_rs p_r + t_rs        (Bosch points are published in `os_sensor`; R_rs, t_rs
                                                      is the residual correction found by radar-LiDAR ICP)
  os_sensor -> camera : p_c = R_OPT (dR p_s - t_cam)  (23.send LRCI convention; dR = Rz(yaw)Ry(pitch)Rx(roll))
  os_sensor -> level  : p_l = R_lev p_s + [0,0,h]     (LiDAR ground plane, z up, origin on the ground)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

R_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)


def rot_rpy(rpy_deg) -> np.ndarray:
    r, p, y = rpy_deg
    return Rotation.from_euler("ZYX", [y, p, r], degrees=True).as_matrix()


@dataclass
class RigExtrinsics:
    cam_rpy_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])     # dR
    cam_t: list = field(default_factory=lambda: [0.0, 0.0, -0.04])         # camera origin in os_sensor [m]
    radar_rpy_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])   # radar -> os_sensor correction
    radar_t: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ground_normal: list = field(default_factory=lambda: [0.0, 0.0, 1.0])  # in os_sensor
    sensor_height: float = 1.7

    # --- point maps (row-vector arrays (N,3)) --------------------------------------------------------
    def radar_to_sensor(self, xyz):
        return xyz @ rot_rpy(self.radar_rpy_deg).T + np.asarray(self.radar_t)

    def sensor_to_cam(self, xyz):
        return (xyz @ rot_rpy(self.cam_rpy_deg).T - np.asarray(self.cam_t)) @ R_OPT.T

    def level(self) -> tuple[np.ndarray, np.ndarray]:
        n = np.asarray(self.ground_normal, float)
        n = n / np.linalg.norm(n)
        x = np.array([1.0, 0, 0]) - n[0] * n
        x /= np.linalg.norm(x)
        return np.stack([x, np.cross(n, x), n]), np.array([0, 0, self.sensor_height])

    def to_calib_json(self, K, D, image_size, *, name: str = "", distortion_model: str = "") -> dict:
        R_lev, t_lev = self.level()
        R_rs, t_rs = rot_rpy(self.radar_rpy_deg), np.asarray(self.radar_t, float)
        R_lr, t_lr = R_lev @ R_rs, R_lev @ t_rs + t_lev                          # radar -> level
        A = R_OPT @ rot_rpy(self.cam_rpy_deg)
        R_cl = A @ R_lev.T                                                       # level -> camera optical
        t_cl = -R_cl @ t_lev - R_OPT @ np.asarray(self.cam_t, float)
        return {
            "bag_key": name, "frame": "level (ground below os_sensor, z up)",
            "image_size": {"width": int(image_size[0]), "height": int(image_size[1])},
            "K": np.asarray(K, float).tolist(), "D": np.asarray(D, float).tolist(),
            "distortion_model": distortion_model, "camera_time_offset_s": 0.0,
            "T_base_level_radar": {"note": "p_level = R @ p_radar + t", "R": R_lr.tolist(), "t": t_lr.tolist()},
            "T_cam_level": {"note": "p_cam_opt = R @ p_level + t", "R": R_cl.tolist(), "t": t_cl.tolist()},
            "extrinsics": asdict(self),
        }

    @classmethod
    def from_calib_json(cls, d: dict) -> "RigExtrinsics":
        return cls(**d["extrinsics"])


def sensor_to_level(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) with p_level = R p_os_sensor + t for either calib json layout:
      * HDX field calib (`extrinsics` key): RigExtrinsics.level()
      * 23.send / xpanner layout (`T_base_sensor` = base_link <- os_sensor from the bag tf_static, `R_level` =
        base_link <- base_link_level): p_level = R_level^T (R_bs p_s + t_bs). This is the chain the json's own
        T_base_level_radar (= R_level^T T_base_radar) and T_cam_level (dR applied on the camera side) were built with."""
    if "extrinsics" in d:
        return RigExtrinsics.from_calib_json(d).level()
    if "T_base_sensor" in d and "R_level" in d:
        R_bs, t_bs = np.asarray(d["T_base_sensor"]["R"], float), np.asarray(d["T_base_sensor"]["t"], float)
        R_lev = np.asarray(d["R_level"]["R"], float)
        return R_lev.T @ R_bs, R_lev.T @ t_bs
    raise KeyError("calib json has neither 'extrinsics' nor 'T_base_sensor' + 'R_level'")
