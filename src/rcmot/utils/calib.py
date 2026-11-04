"""Rig calibration loader for the 23.send calib.json format (radar -> base_link_level -> camera optical)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RigCalib:
    K: np.ndarray
    D: np.ndarray
    image_size: tuple[int, int]        # (w, h)
    R_lr: np.ndarray                   # radar -> level
    t_lr: np.ndarray
    R_cl: np.ndarray                   # level -> camera optical
    t_cl: np.ndarray
    camera_time_offset_s: float = 0.0
    radar_time_offset_s: float = 0.0   # radar scan describes the state at (stamp + this); Continental: -0.067 s (docs/16 §10)
    swing_axis_level: np.ndarray | None = None   # excavator swing axis (unit, level frame): scripts/swing_axis_calib.py

    @classmethod
    def load(cls, path: str) -> "RigCalib":
        d = json.load(open(path))
        tr, tc = d["T_base_level_radar"], d["T_cam_level"]
        return cls(np.array(d["K"], float), np.array(d["D"], float),
                   (d["image_size"]["width"], d["image_size"]["height"]),
                   np.array(tr["R"]), np.array(tr["t"]), np.array(tc["R"]), np.array(tc["t"]),
                   float(d.get("camera_time_offset_s", 0.0)), float(d.get("radar_time_offset_s", 0.0)),
                   None if d.get("swing_axis_level") is None else np.asarray(d["swing_axis_level"], float))

    def radar_to_level(self, xyz: np.ndarray) -> np.ndarray:
        return xyz @ self.R_lr.T + self.t_lr

    def level_to_radar_dir(self, v: np.ndarray) -> np.ndarray:
        return v @ self.R_lr

    def project(self, xyz_level: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (uv (N,2), in_front mask)."""
        p = xyz_level @ self.R_cl.T + self.t_cl
        front = p[:, 2] > 0.3
        uv = np.full((len(p), 2), np.nan)
        if front.any():
            uv[front] = cv2.projectPoints(p[front], np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)
        return uv, front

    def camera_center(self) -> np.ndarray:
        """Camera optical centre in the level frame."""
        return -self.R_cl.T @ self.t_cl

    def pixel_ray(self, u: float, v: float) -> np.ndarray:
        """Unit direction (level frame) of the camera ray through pixel (u, v) (distortion removed)."""
        n = cv2.undistortPoints(np.array([[[u, v]]], float), self.K, self.D).reshape(2)
        d = self.R_cl.T @ np.array([n[0], n[1], 1.0])
        return d / np.linalg.norm(d)

    def pixel_ray_to_ground(self, u: float, v: float, ground_z: float = 0.0) -> np.ndarray | None:
        """Intersect the camera ray through pixel (u,v) with the plane z=ground_z (level frame)."""
        n = cv2.undistortPoints(np.array([[[u, v]]], float), self.K, self.D).reshape(2)
        ray_c = np.array([n[0], n[1], 1.0])
        R_lc = self.R_cl.T
        origin = -R_lc @ self.t_cl
        ray = R_lc @ ray_c
        if ray[2] >= -1e-6:
            return None
        s = (ground_z - origin[2]) / ray[2]
        return origin + s * ray


OBJ_SIZE = {"worker": (0.7, 1.8), "heavy_equipment": (5.0, 3.0)}   # (width, height) m


def object_box(calib: RigCalib, cls: str, pos: np.ndarray) -> np.ndarray | None:
    """Image box (xyxy) of an upright object of the class size standing at a 3D level-frame position."""
    w, h = OBJ_SIZE.get(cls, (1.0, 1.8))
    c = np.array([[pos[0] + dx, pos[1] + dy, z] for dx in (-w / 2, w / 2) for dy in (-w / 2, w / 2) for z in (0.0, h)])
    uv, fr = calib.project(c)
    if not fr.all():
        return None
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
