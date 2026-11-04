"""Data types shared by detection, fusion, tracking and GUI. Frame convention: base_link_level
(x forward, y left, z up, gravity aligned), as in 23.send calib.json."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


@dataclass
class Det2D:
    cls: str
    score: float
    box: np.ndarray                    # (4,) xyxy pixels
    mask: np.ndarray | None = None     # instance mask cropped to the box (bool, rows from int(y1), cols from int(x1))


@dataclass
class RadarFrame:
    stamp: float
    xyz: np.ndarray                    # (N,3) in base_link_level
    v_r: np.ndarray                    # (N,) raw radial velocity, + = away from sensor
    rcs: np.ndarray
    snr: np.ndarray
    xyz_radar: np.ndarray              # (N,3) in sensor frame (for Doppler geometry)
    v_r_comp: np.ndarray | None = None # ego-compensated radial velocity
    static_mask: np.ndarray | None = None
    peer: np.ndarray | None = None     # Bosch idx_azimuth_ambiguity_peer
    az_prob: np.ndarray | None = None  # Bosch azimuth_angle_probability (0..255)
    stats: object | None = None        # rcmot.radar.scan_stats.ScanStats of this scan
    bag_t: float | None = None         # bag receive time when replayed (timeline)
    clock_offset_s: float | None = None  # header -> bag clock offset applied to `stamp` (sensor-clock radars, bag replay)
    img_dt: float = 0.0                # stamp of the paired camera image - this scan's stamp [s] (docs/21 E2-c)
    bg_score: np.ndarray | None = None # per return: background score of the BEV cell memory (docs/41 §10), None = not used
    depth: tuple | None = None         # (stereo depth [m] (h, w), K) nearest to the paired image (work monitor, docs/24)


@dataclass
class Det3D:
    cls: str
    score: float
    box2d: np.ndarray                  # (4,) xyxy
    pos: np.ndarray                    # (3,) base_link_level
    pos_cov: np.ndarray                # (3,3)
    vel: np.ndarray | None = None      # (3,) absolute velocity, base_link_level (full LSQ)
    vel_cov: np.ndarray | None = None
    vr: float | None = None            # radial-only velocity measurement z = u . v (when no full velocity)
    vr_u: np.ndarray | None = None     # (3,) unit line of sight (level frame)
    vr_var: float | None = None
    radar_idx: np.ndarray | None = None  # radar points used (indices into the RadarFrame)
    n_radar: int = 0
    source: str = "radar"              # radar | ground_ray | roi_research | radar_only
    crop: np.ndarray | None = None     # BGR thumbnail for the GUI gallery
    reid_crop: np.ndarray | None = None  # background-masked crop for ReID (seg detector), else crop is used
    emb: np.ndarray | None = None      # L2-normalised appearance embedding (ReID)
    crop_mask: np.ndarray | None = None  # person pixels of `crop` (seg detector), for the part-colour histograms
    hist: np.ndarray | None = None     # (4, 144) sqrt colour histograms pants / arms / shoes / whole (docs/32)
    dbg: dict | None = None            # fusion diagnostics (cluster scoring / gate, docs/30), evaluation only
    rlik: object | None = None         # docs/38: rcmot.fusion.range_lik.RangeLik (range distribution), range_model "lik"


class TrackState(Enum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    RADAR_ONLY = 3                     # camera lost, kept alive by radar points (Radar-Gated Recovery)


@dataclass
class TrackOut:
    tid: int
    cls: str
    state: TrackState
    pos: np.ndarray
    vel: np.ndarray                    # absolute velocity (ego motion removed)
    box2d: np.ndarray | None
    score: float
    hits: int
    time_since_update: float
    thumbnail: np.ndarray | None = None
    history: list = field(default_factory=list)          # past positions (current base frame)
    history_t: list = field(default_factory=list)        # their times [s]
    vel_cov: np.ndarray | None = None
    pos_cov: np.ndarray | None = None
    age: float = 0.0
    source: str = ""
    n_radar: int = 0
    exist: float = 1.0                 # existence score (tracker S4)
    static: bool = False               # standstill mode: velocity held at zero (docs/16)
    time_since_camera: float = 0.0     # [s] since the last camera-backed update (radar-only / coasting tracks grow it)


@dataclass
class AlarmState:
    """Alarm decision of one processed frame (rcmot.alarm.Alarm.decide, after hysteresis; docs/18, docs/20)."""
    level: int                         # 0 none, 1 caution, 2 warning, 3 danger
    nearest_tid: int | None
    nearest_range: float | None        # horizontal distance [m] of the nearest counted track
    zones: tuple                       # (danger, warning, caution) [m]
    sound_enabled: bool
    confirmed_only: bool
