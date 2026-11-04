"""One-frame pipeline (docs/08 §2):
radar ScanStats (standardisation, ego velocity, dynamics) -> 2D detection -> radar-camera fusion
-> appearance embeddings (optional ReID) -> 3D tracking (+ Radar-Gated Recovery) -> unclassified movers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import numpy as np

from rcmot.fusion.depth_stage import make_depth_stage
from rcmot.fusion.radar_camera import FusionParams, RangeReliability, fuse
from rcmot.fusion.image_shake import ImageShake
from rcmot.radar.cell_memory import CellMemory, CellMemoryParams, learn_mask
from rcmot.radar.dynamic_clusters import Mover, cluster_movers
from rcmot.radar.ego_velocity import compensate_doppler
from rcmot.radar.scan_stats import ScanStatistician, ScanStatsParams, sigmoid
from rcmot.reid.appearance import AppearanceStage
from rcmot.tracking.recovery import RadarGatedRecovery, RGRParams
from rcmot.tracking.tracker import RadarCameraTracker, TrackerParams, iou_xyxy
from rcmot.types import Det2D, RadarFrame, TrackOut
from rcmot.utils.calib import RigCalib, object_box
from rcmot.work.monitor import WorkMonitor, WorkParams
from rcmot.world import WorldParams, WorldPose


def _cover(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of box a's area inside box b."""
    w = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    h = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    return w * h / max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)


@dataclass
class FrameResult:
    stamp: float
    tracks: list[TrackOut]
    ego_vel: np.ndarray               # base_link_level, m/s
    ego_valid: bool
    radar_xyz: np.ndarray             # (N,3) level frame
    radar_static: np.ndarray          # bool (ego-fit inliers)
    n_dets: int
    radar_vcomp: np.ndarray | None = None
    radar_dynamic: np.ndarray | None = None
    radar_alpha: np.ndarray | None = None
    sigma_v: float = float("nan")
    dets2d: list[Det2D] = field(default_factory=list)
    movers: list[Mover] = field(default_factory=list)
    timing_ms: dict = field(default_factory=dict)
    yaw_rate: float = 0.0
    alarm: object | None = None       # rcmot.types.AlarmState, set by the worker (Alarm.decide), None when no alarm
    world: object | None = None       # rcmot.world.WorldState: ego pose in the segment's world frame (docs/24)
    work: object | None = None        # rcmot.work.monitor.WorkState: soil-pile height map / volume / cycles (docs/24)
    recon: object | None = None       # rcmot.work.recon.ReconState: drive 3D reconstruction (docs/26 §6)


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotate_det(d, R: np.ndarray, dp: np.ndarray | None = None) -> None:
    """Rotate a Det3D's position / position covariance about z (image time -> scan time, docs/21 E2-c), then
    translate by dp (the base's motion between the two times, scan frame). The radar velocity terms (vel, vel_cov,
    vr_u) are built from the unrotated sensor-frame points, i.e. already in scan time."""
    d.pos = R @ d.pos + (0.0 if dp is None else dp)
    d.pos_cov = R @ d.pos_cov @ R.T
    if getattr(d, "rlik", None) is not None:                 # docs/38: the range distribution's ray / camera centre
        d.rlik.rotate(R, dp)


def _track_snapshot(t, r_prior: float, n_hit: int, n_hist: int = 15) -> dict:
    """docs/37 cue dumps: the state of the track whose range prior a box received, BEFORE this frame's update (level
    frame of the previous scan; hist = filtered positions of the last updates, t relative to the newest)."""
    ht = np.asarray(t.hist_t[-n_hist:], float)
    hp = np.asarray(t.hist_p[-n_hist:], float).reshape(-1, 3)
    return {"tid": int(t.tid), "hits": int(t.hits), "age": round(float(t.age), 3), "state": t.state.name,
            "static": int(bool(t.static)), "pos": [round(float(v), 3) for v in t.kf.pos[:2]],
            "vel": [round(float(v), 3) for v in t.kf.vel[:2]],
            "Pp": [round(float(t.kf.P[0, 0]), 4), round(float(t.kf.P[0, 1]), 4), round(float(t.kf.P[1, 1]), 4)],
            "Pv": [round(float(t.kf.P[3, 3]), 4), round(float(t.kf.P[4, 4]), 4)],
            "since_pos": round(float(t.since_pos), 3), "t_cam": round(float(t.t_cam), 3),
            "t_radar": round(float(min(t.t_radar, 1e3)), 3), "n_h": len(t.h_obs),
            "height": None if t.height() is None else round(float(t.height()), 3),
            "radar_pos": None if t.radar_pos is None else [round(float(v), 3) for v in t.radar_pos[:2]],
            "last_src": t.last_source, "r_prior": round(float(r_prior), 3), "n_hit": int(n_hit),
            "hist_t": [round(float(v - ht[-1]), 3) for v in ht] if len(ht) else [],
            "hist_xy": [[round(float(a), 3), round(float(b), 3)] for a, b in hp[:, :2]]}


def radar_type_of(cfg: dict) -> str:
    """'bosch' | 'conti' (key of fusion.radar_lag_s) from cfg["radar_type"] (the eval scripts set it from
    meta.radar, e.g. conti_ars548 / bosch_premium) or else the radar topic name."""
    t = str(cfg.get("radar_type") or cfg.get("topics", {}).get("radar", "")).lower()
    return "conti" if ("conti" in t or "ars548" in t or "radar_detection" in t) else "bosch"


class Pipeline:
    def __init__(self, detector, calib: RigCalib, cfg: dict, reid=None, depth_engine=None, recon=None):
        self.det, self.calib, self.cfg, self.reid = detector, calib, cfg, reid
        self.depth_engine = depth_engine                      # work monitor's VGGT-Omega engine (docs/24), kept across resets
        self.recon = recon                                    # rcmot.work.recon.DriveRecon (docs/26 §6), kept across resets
        only = cfg.get("detector", {}).get("only")           # e.g. [worker]: people only (heavy equipment detection off)
        self.only_classes = set(only) if only else None
        self._pending_only = None
        if hasattr(self.det, "set_project_classes"):
            self.det.set_project_classes(self.only_classes)
        self._pending_det = None                              # detector swap requested by the GUI (docs/18)
        self.cue_log_on = False                               # docs/37: fuse() cue records in last_cues (evaluation dumps)
        self.last_cues: list = []
        self.last_prior_tracks: list = []                     # per detection: snapshot of the prior's track (cue_log_on)
        self.reset()

    def request_detector(self, weights: str) -> None:
        """Replace the 2D detector (and the recovery's) before the next frame; built here, swapped in step()."""
        from rcmot.detection.yolo2d import Yolo2D

        dc = self.cfg["detector"]
        self._pending_det = (weights, Yolo2D(weights, dc["class_map"], imgsz=dc["imgsz"], conf=dc["conf"]))

    def set_only_classes(self, only: set | None) -> None:
        """GUI '사람만': applied before the next frame (the detector runs on the worker thread)."""
        self._pending_only = (set(only) if only else None,)

    def _swap_detector(self) -> None:
        if self._pending_only is not None:
            self.only_classes, self._pending_only = self._pending_only[0], None
            if hasattr(self.det, "set_project_classes"):
                self.det.set_project_classes(self.only_classes)
        if self._pending_det is None:
            return
        weights, det = self._pending_det
        self._pending_det = None
        if hasattr(det, "set_project_classes"):
            det.set_project_classes(self.only_classes)
        self.det = det
        if self.rgr is not None:
            self.rgr.det = det
        self.cfg["detector"]["weights"] = weights

    def reset(self) -> None:
        """Forget all temporal state (tracks, scan window, ego velocity), e.g. after seeking in a bag."""
        cfg, detector, calib = self.cfg, self.det, self.calib
        self.tracker = RadarCameraTracker(TrackerParams(**cfg.get("tracker", {})),
                                          project_box=lambda cls, pos: object_box(calib, cls, pos))
        rc = dict(cfg.get("recovery", {}))
        self.rgr = RadarGatedRecovery(detector, calib, RGRParams(**rc)) if rc.pop("enable", True) else None
        self.fusion_p = FusionParams(**cfg.get("fusion", {}))
        if self.rgr is not None:
            self.rgr.fusion_p = self.fusion_p                        # RGR fuses its candidates with the same settings
        if self.fusion_p.range_model == "lik":                       # docs/38: the tracker multiplies the boxes' range
            self.tracker.p.range_lik = True                          # likelihoods with its predictions
            self.tracker.p.lik_radar_backed = self.fusion_p.lik_radar_backed
        self.ego_comp = cfg.get("ego_comp", True)
        ev = cfg.get("ego_velocity", {})
        self.ema = ev.get("ema_alpha", 0.4)
        self.ego_max = ev.get("max_speed", 15.0)             # [m/s] sanity gate (docs/16: 41 m/s fits on near-empty scans)
        self.ego_max_jump = ev.get("max_jump", 1.0)          # [m/s] per scan, vs the EMA
        self.ego_age = 0.0                                   # [s] since the last accepted radar ego velocity
        self.stats = ScanStatistician(ScanStatsParams(**cfg.get("scan_stats", {})))
        self.last_dets3d: list = []                           # fused detections of the last step (evaluation hooks)
        self.last_ego_static: bool = True                     # ... and the ego state / base velocity of that step
        self.last_v_lvl = np.zeros(3)
        self.radar_type = radar_type_of(cfg)                  # key of fusion.radar_lag_s (impl2)
        self.shake = ImageShake() if self.fusion_p.shake_gain > 0 else None   # docs/21 E6-4 (off by default)
        self.range_rel = RangeReliability()               # camera-range reliability, learned from radar online
        self.dstage = make_depth_stage(cfg, calib, self.fusion_p)   # docs/33 wave 3 metric depth map (None: off)
        cm = dict(cfg.get("cell_memory") or {})
        self.cellmem = CellMemory(CellMemoryParams(**cm)) if cm.get("enable", False) else None   # docs/41 §10
        if self.cellmem is not None:
            self.fusion_p.lik_bg_slope, self.fusion_p.lik_bg_fmin = self.cellmem.p.bg_slope, self.cellmem.p.f_min
        mv = cfg.get("movers", {})
        self.movers_on = mv.get("enable", True)
        self.movers_kw = {k: mv[k] for k in ("max_range", "ego_k", "min_pts", "min_weight") if k in mv}
        self.reid_every = int(cfg.get("reid", {}).get("every", 1))   # embeddings every N frames (Orin: OSNet 33-44 ms, docs/18)
        self.app_stage = AppearanceStage(cfg.get("reid", {}), calib)   # crowd mean / part colour (docs/32)
        if self.app_stage.active:
            self.tracker.p.app_mode = self.app_stage.mode
            if "app_scales" not in cfg.get("tracker", {}):
                self.tracker.p.app_scales = self.app_stage.scales
            print(f"[rcmot] reid appearance mode {self.app_stage.mode} (crowd_mean {self.app_stage.p.crowd_mean}, "
                  f"part_colour {self.app_stage.p.part_colour}, camera pitch {self.app_stage.pitch_down:.1f} deg)",
                  flush=True)
        self._frame_i = 0
        self.ego_vel = np.zeros(3)
        self.last_t: float | None = None
        wc = cfg.get("world", {})
        self.world = WorldPose(WorldParams(**wc), calib.t_lr) if wc.get("enable", True) else None
        wk = cfg.get("work", {})
        self.work = WorkMonitor(WorkParams(**wk), calib, self.depth_engine) if wk.get("enable", False) and self.world is not None else None
        if self.recon is not None:
            self.recon.reset()

    def _radar(self, radar: RadarFrame) -> bool:
        st = self.stats.compute(radar.stamp, radar.xyz_radar, radar.v_r, radar.snr, radar.rcs, radar.peer, radar.xyz,
                                R_level=self.calib.R_lr)
        radar.stats = st
        ok = st.ego_valid and np.isfinite(st.v_sensor).all() and np.linalg.norm(st.v_sensor) <= self.ego_max \
            and (self.ego_age > 1.0 or np.linalg.norm(st.v_sensor - self.ego_vel) <= self.ego_max_jump * (1.0 + self.ego_age / 0.066))
        if ok:
            self.ego_vel = (1 - self.ema) * self.ego_vel + self.ema * st.v_sensor
            self.ego_age = 0.0
        st.ego_valid = bool(ok)
        radar.v_r_comp = compensate_doppler(radar.xyz_radar, radar.v_r, self.ego_vel)
        st.v_comp = radar.v_r_comp
        st.z_v = st.v_comp / st.sigma_v
        sp = self.stats.p                                     # soft motion probability with the final ego velocity
        st.p_dyn = sigmoid((np.abs(st.z_v) - sp.dyn_z) / sp.dyn_soft) * \
            sigmoid((np.abs(st.v_comp) - sp.dyn_vmin) / (0.25 * sp.dyn_vmin))
        radar.static_mask = st.static_inlier
        return st.ego_valid

    def _track_range_priors(self, dets2d: list[Det2D], dt: float, v_ego: np.ndarray, yaw_rate: float = 0.0,
                            img_dt: float = 0.0, inflate: bool = False) -> list:
        """Per detection: (range, sigma), learned height, (velocity, covariance) and (predicted BEV xy, 2x2
        covariance, seconds since the last position measurement) of the camera-anchored track predicted into that
        box -- the motion-predicted 3D ROI used to pick the box's radar points (the xy prior is the 2-D cluster gate
        of docs/30). When several tracks project into the box, the one agreeing best with the box's monocular range
        wins, else the nearest (the visible object of a box is the front one).
        img_dt != 0 (impl2, moving ego): the tracks (in the previous scan's base frame) are propagated to the IMAGE
        time dt + img_dt -- rotation yaw_rate x (dt + img_dt), translation (v_track - v_ego) x (dt + img_dt) -- for
        the projected box, the range and the xy prior alike; inflate: the xy covariance is widened across the line of
        sight by (horizon x sigma_w x r)^2 and along it by (horizon x sigma_v,ego)^2 (horizon = since the last
        position measurement until the image). With img_dt 0 / inflate False the static-ego prior is unchanged."""
        from rcmot.fusion.radar_camera import mono_range

        p, C = self.fusion_p, self.calib.camera_center()
        dt_p = dt + img_dt                                            # tracks -> image time (impl2) or scan time
        Rz = _rz(-yaw_rate * dt_p)[:2, :2] if self.ego_comp else np.eye(2)   # base turns by yaw_rate * dt until then
        R3 = _rz(-yaw_rate * dt_p) if (self.ego_comp and img_dt != 0.0) else None   # also the projected box (impl2)
        s_w, s_v = np.radians(p.prior_sigma_w_deg), p.prior_sigma_v_ego
        cands = []
        for t in self.tracker.tracks:
            anchored = p.track_prior_radar_anchor and t.radar_pos is not None and t.t_radar <= p.track_prior_max_tradar_s
            if t.state.name == "TENTATIVE" or (t.t_cam > p.track_prior_max_tcam_s and not anchored):
                continue                      # a radar-anchored track keeps its prior through a camera gap (docs/16 §9)
            pos = t.kf.pos + (t.kf.vel - v_ego) * dt_p if self.ego_comp else t.kf.pos + t.kf.vel * dt_p
            xy, P2 = t.predict_xy(dt_p, v_ego if self.ego_comp else None)
            xy, P2 = Rz @ xy, Rz @ P2 @ Rz.T                          # in this scan's base frame (as the tracker's
                                                                      # ego_compensate will do); the scalar range /
                                                                      # projected box stay unrotated unless R3 (docs/14)
            if p.track_prior_radar_anchor and t.radar_pos is not None and t.t_radar <= p.track_prior_max_tradar_s:
                # range from the last RADAR-backed position (+ motion since), not from the filter state that the
                # camera-only ranges have been dragging (docs/16 §9: mono+track feedback drift)
                pos_r = t.radar_pos + (t.kf.vel - v_ego) * (t.t_radar + dt_p) if self.ego_comp else t.radar_pos + t.kf.vel * (t.t_radar + dt_p)
                pos = pos_r
            if R3 is not None:
                pos = R3 @ pos
            box = object_box(self.calib, t.cls, pos)
            if box is not None:
                sig = max(float(np.sqrt(np.trace(t.kf.P[:2, :2]) / 2)), p.track_prior_sigma_floor)
                if p.track_prior_radar_anchor and t.radar_pos is not None and t.t_radar <= p.track_prior_max_tradar_s:
                    sig = float(np.hypot(sig, p.track_prior_anchor_rate * t.t_radar))
                    P2 = P2 + np.eye(2) * (p.track_prior_anchor_rate * t.t_radar) ** 2
                    xy = pos[:2].copy() if R3 is not None else Rz @ pos[:2]   # same radar-anchored position as the range
                if p.track_prior_ego_lag_s > 0:                       # moving ego: image / radar / IMU timing errors
                    lag, r_t = p.track_prior_ego_lag_s, max(float(np.linalg.norm(xy - C[:2])), 1.0)
                    u_t = (xy - C[:2]) / r_t
                    perp = np.array([-u_t[1], u_t[0]])
                    P2 = P2 + np.eye(2) * (lag * float(np.linalg.norm(v_ego[:2]))) ** 2 \
                        + np.outer(perp, perp) * (lag * abs(yaw_rate) * r_t) ** 2
                if inflate:                                           # impl2: gyro noise x range across, ego speed along
                    hz, r_t = t.since_pos + dt_p, max(float(np.linalg.norm(xy - C[:2])), 1.0)
                    u_t = (xy - C[:2]) / r_t
                    perp = np.array([-u_t[1], u_t[0]])
                    P2 = P2 + np.outer(perp, perp) * (hz * s_w * r_t) ** 2 + np.outer(u_t, u_t) * (hz * s_v) ** 2
                lo = float(np.linalg.eigvalsh(P2).min())
                if lo < p.track_prior_sigma_floor ** 2:
                    P2 = P2 + np.eye(2) * (p.track_prior_sigma_floor ** 2 - lo)
                cands.append((t.cls, box, float(np.linalg.norm(pos[:2] - C[:2])), sig, t.height(),
                              (t.kf.vel.copy(), t.kf.P[3:, 3:].copy()) if t.hits >= 3 else None,
                              (np.asarray(xy, float).copy(), P2, t.since_pos + dt_p), t))
        out, heights, vels, xys = [], [], [], []
        self.last_prior_tracks = []
        for d in dets2d:
            same = [c for c in cands if c[0] == d.cls]
            hit = [c for c in same if iou_xyxy(c[1][None], d.box[None])[0, 0] >= p.track_prior_iou]
            if not hit and p.track_prior_centre:              # narrow person boxes: IoU drops below 0.3 with a small
                cx, cy = 0.5 * (d.box[0] + d.box[2]), 0.5 * (d.box[1] + d.box[3])   # lateral offset -> centre test
                inside = [c for c in same if c[1][0] <= cx <= c[1][2] and c[1][1] <= cy <= c[1][3]
                          and not any(o is not d and o.cls == d.cls and iou_xyxy(c[1][None], o.box[None])[0, 0] >= p.track_prior_iou
                                      for o in dets2d)]
                hit = inside if len(inside) == 1 else []
            if not hit:                                       # an occluded box is SHORTER than the track's full box:
                h = d.box[3] - d.box[1]                       # contained in it -- only if unique (crossing people)
                inside = [c for c in same if _cover(d.box, c[1]) >= p.track_prior_cover and h < p.track_prior_short * (c[1][3] - c[1][1])
                          and not any(o is not d and o.cls == d.cls and iou_xyxy(c[1][None], o.box[None])[0, 0] >= p.track_prior_iou
                                      for o in dets2d)]               # not a track another box already explains
                hit = inside if len(inside) == 1 else []
            if not hit:
                out.append(None)
                heights.append(None)
                vels.append(None)
                xys.append(None)
                if self.cue_log_on:
                    self.last_prior_tracks.append(None)
                continue
            H = p.class_height.get(d.cls)
            mono = mono_range(d.box, H, self.calib, p) if H is not None else None
            best = min(hit, key=lambda c: abs(c[2] - mono[0]) / np.hypot(c[3], mono[1]) if mono else c[2])
            out.append((best[2], best[3]))
            if self.cue_log_on:
                self.last_prior_tracks.append(_track_snapshot(best[7], best[2], len(hit)))
            heights.append(best[4])
            vels.append(best[5])
            xys.append(best[6])
        return out, heights, vels, xys

    def _suppress_shadowed(self, tracks: list[TrackOut]) -> list[TrackOut]:
        """Output-level image NMS (docs/14): a track without a camera update this frame whose projected box lies
        mostly inside (>= shadow_cover of its area) a camera-updated box of the same class is a copy on the same
        image ray (lost / radar-only at another depth): it is not drawn (the track itself is kept)."""
        thr = self.cfg.get("output", {}).get("shadow_cover", 0.5)
        meas = [t for t in tracks if t.time_since_update == 0 and t.box2d is not None and t.source != "radar_only"]
        if not meas or thr > 1.0:
            return tracks
        keep = []
        for t in tracks:
            if t in meas:
                keep.append(t)
                continue
            box = object_box(self.calib, t.cls, t.pos)
            if box is not None and any(m.cls == t.cls and _cover(box, m.box2d) >= thr for m in meas):
                continue
            keep.append(t)
        return keep

    def _box_time_offset(self, radar: RadarFrame) -> float:
        """t_img - (time the scan describes) [s] (impl2). rf.img_dt = image stamp - rf.stamp, and rf.stamp already
        carries calib.radar_time_offset_s (bag_source / live source); when fusion.radar_lag_s names this radar type
        with a number, that latency replaces the calib's for the box-time rotation (only the difference matters)."""
        lag = (self.fusion_p.radar_lag_s or {}).get(self.radar_type)
        if lag is None or isinstance(lag, str):
            return float(radar.img_dt)
        return float(radar.img_dt + self.calib.radar_time_offset_s - float(lag))

    def ego_static_of(self, v_lvl: np.ndarray, yaw_rate: float) -> bool:
        return bool(np.linalg.norm(v_lvl[:2]) <= self.fusion_p.ego_static_speed
                    and abs(yaw_rate) <= np.radians(self.fusion_p.ego_static_yaw_deg))

    def _fuse(self, dets2d: list[Det2D], radar: RadarFrame, yaw_rate: float, v_lvl: np.ndarray, ego_static: bool,
              image: np.ndarray | None = None, shake=None, priors=(None, None, None, None), dmap=None) -> list:
        """fuse() in the image's time (docs/21 E2-c, impl2): while the base turns, the radar points are rotated by
        the yaw swept between the time the scan describes and the image (box_time / rewind_image_yaw), the fused
        positions are rotated back by yaw_rate x img_dt and moved by v_ego x img_dt into the scan frame the tracker
        lives in. Static ego (box_time_moving_only): no rotation, as before."""
        p = self.fusion_p
        range_priors, heights, vels, xys = priors
        moving = not ego_static
        on = p.rewind_image_yaw != 0 or (p.box_time and (moving or not p.box_time_moving_only))
        mult = float(p.rewind_image_yaw) if p.rewind_image_yaw != 0 else 1.0   # legacy: -1 = sign check
        a = mult * yaw_rate * -self._box_time_offset(radar) if on else 0.0
        r_in = replace(radar, xyz=radar.xyz @ _rz(a).T) if abs(a) > 1e-6 else radar
        self.last_cues = [] if self.cue_log_on else self.last_cues
        dets3d = fuse(dets2d, r_in, self.calib, image=image, params=p, range_priors=range_priors,
                      reliability=self.range_rel, heights=heights if p.track_height else None,
                      vel_priors=vels, shake_px=shake, xy_priors=xys, ego_static=ego_static, ego_yaw_rate=yaw_rate,
                      dmap=dmap, cue_log=self.last_cues if self.cue_log_on else None,
                      ego_speed=float(np.linalg.norm(np.asarray(v_lvl, float)[:2])))
        if r_in is not radar:
            Rb = _rz(mult * yaw_rate * radar.img_dt)
            dp = v_lvl * radar.img_dt if self.ego_comp else None
            for d in dets3d:
                _rotate_det(d, Rb, dp)
        return dets3d

    def fuse_boxes(self, dets2d: list[Det2D], radar: RadarFrame, yaw_rate: float) -> list:
        """Evaluation hook: fuse() on given boxes with this frame's ego state / box-time correction but WITHOUT track
        priors (eval_fusion_depth's no-prior protocol); call after step() on the same frame."""
        dmap = None
        if self.dstage is not None:                       # this frame's depth map, no track bias (no tracks here)
            dmap = self.dstage.boxes(dets2d, None, moving=not self.last_ego_static,
                                     a_box=self._box_time_angle(radar, yaw_rate, self.last_ego_static))
        return self._fuse(dets2d, radar, yaw_rate, self.last_v_lvl, self.last_ego_static, dmap=dmap)

    def _box_time_angle(self, radar: RadarFrame, yaw_rate: float, ego_static: bool) -> float:
        """The box-time rotation angle of _fuse (radar points -> image time), for the depth stage."""
        p = self.fusion_p
        on = p.rewind_image_yaw != 0 or (p.box_time and (not ego_static or not p.box_time_moving_only))
        mult = float(p.rewind_image_yaw) if p.rewind_image_yaw != 0 else 1.0
        return mult * yaw_rate * -self._box_time_offset(radar) if on else 0.0

    def _depth_stage(self, image, radar: RadarFrame, dets2d: list[Det2D], dt: float, v_lvl: np.ndarray, yaw_rate: float,
                     tm: dict) -> list:
        """docs/33 wave 3: depth-map stage before fuse() -- history / map update (every N-th frame a new map) and the
        per-box depth-map ranges (with the per-track bias of the track predicted into each box)."""
        t0 = time.perf_counter()
        ego_static = self.ego_static_of(v_lvl, yaw_rate)
        moving = not ego_static
        a = self._box_time_angle(radar, yaw_rate, ego_static)
        self.dstage.step(image, radar, dt, v_lvl if self.ego_comp else np.zeros(3), yaw_rate if self.ego_comp else 0.0,
                         moving, a_box=a)
        p = self.fusion_p
        img_dt_p = radar.img_dt if (p.prior_img_time and (moving or not p.prior_moving_only)) else 0.0
        out = self.dstage.boxes(dets2d, self.tracker.tracks, moving=moving, a_box=a, dt=dt + img_dt_p,
                                v_ego=v_lvl if self.ego_comp else np.zeros(3), yaw_rate=yaw_rate if self.ego_comp else 0.0,
                                t_now=radar.stamp)
        tm["depth_map"] = (time.perf_counter() - t0) * 1e3
        tm["depth_map_new"] = float(self.dstage.ms.get("computed", False))
        return out

    def step(self, image: np.ndarray, radar: RadarFrame, yaw_rate: float = 0.0) -> FrameResult:
        tm = {}
        self._swap_detector()
        dt = 0.066 if self.last_t is None else max(radar.stamp - self.last_t, 1e-3)
        self.last_t = radar.stamp
        self.ego_age += dt
        t0 = time.perf_counter()
        ok = self._radar(radar)
        tm["radar"] = (time.perf_counter() - t0) * 1e3
        v_lvl = self.calib.R_lr @ self.ego_vel                    # sensor velocity ~ base velocity (no yaw lever arm)
        t0 = time.perf_counter()
        dets2d = self.det(image)
        tm["detector"] = (time.perf_counter() - t0) * 1e3
        dmap = self._depth_stage(image, radar, dets2d, dt, v_lvl, yaw_rate, tm) if self.dstage is not None else None
        t0 = time.perf_counter()
        p = self.fusion_p
        ego_static = self.ego_static_of(v_lvl, yaw_rate)
        self.last_ego_static, self.last_v_lvl = ego_static, v_lvl
        if self.cellmem is not None:                             # docs/41 §10: kept only while the machine stands still
            self.cellmem.step(radar.stamp, ego_static)
            radar.bg_score = self.cellmem.score(radar.xyz) if self.cellmem.valid(radar.stamp) else None
        if self.rgr is not None:
            self.rgr.ego_static, self.rgr.ego_yaw_rate = ego_static, yaw_rate
            self.rgr.ego_speed = float(np.linalg.norm(v_lvl[:2]))
        moving = not ego_static
        img_dt_p = radar.img_dt if (p.prior_img_time and (moving or not p.prior_moving_only)) else 0.0   # impl2
        priors = self._track_range_priors(dets2d, dt, v_lvl, yaw_rate, img_dt=img_dt_p,
                                          inflate=moving or not p.prior_moving_only) if p.track_prior else (None,) * 4
        shake = self.shake.update(image) if self.shake is not None else None
        dets3d = self._fuse(dets2d, radar, yaw_rate, v_lvl, ego_static, image=image, shake=shake, priors=priors, dmap=dmap)
        tm["fusion"] = (time.perf_counter() - t0) * 1e3
        self.last_dets3d = dets3d
        self._frame_i += 1
        if self.reid is not None and dets3d and (self._frame_i % max(self.reid_every, 1) == 0):
            t0 = time.perf_counter()
            embs = self.reid([d.reid_crop if d.reid_crop is not None else d.crop for d in dets3d])
            for d, e in zip(dets3d, embs):
                d.emb = e
            self.app_stage(dets3d, radar.stamp)
            tm["reid"] = (time.perf_counter() - t0) * 1e3
        used = [d.radar_idx for d in dets3d if d.radar_idx is not None]
        used = np.concatenate(used) if used else np.zeros(0, int)
        if self.rgr is not None:
            self.rgr.set_frame(image, radar, [d.box2d for d in dets3d], used)
        st = radar.stats
        # world pose from THIS scan's Doppler velocity (the EMA lags ~0.1 s: on 67 deg/s swing reversals the lag
        # shows up as swing-axis motion); the held EMA only when the scan's fit is invalid. Computed before the
        # tracker (independent of it): the re-entry memory (tracker.reentry_memory) keeps its anchors in this frame
        v_scan = self.calib.R_lr @ st.v_sensor if ok and np.isfinite(st.v_sensor).all() else v_lvl
        world = self.world.update(radar.stamp, dt, yaw_rate, v_scan[:2], ok) if self.world is not None else None
        t0 = time.perf_counter()
        tracks = self.tracker.update(dets3d, dt, ego_dp=v_lvl * dt if self.ego_comp else None,
                                     ego_dyaw=yaw_rate * dt if self.ego_comp else 0.0, recover=self.rgr, world=world)
        tm["tracking"] = (time.perf_counter() - t0) * 1e3
        if self.dstage is not None:                               # per-track radar / depth-map ratio (B-a bias)
            self.dstage.observe(dets3d, self.tracker.tracks, radar.stamp)
        if self.cellmem is not None and ego_static:              # learn this scan's static returns away from people
            t0 = time.perf_counter()
            self.cellmem.learn(radar.stamp, radar.xyz, learn_mask(st, radar.xyz, self.calib.camera_center(),
                                                                  self.tracker.tracks, dets3d, self.cellmem.p, self.calib))
            tm["cell_memory"] = (time.perf_counter() - t0) * 1e3
        tracks = self._suppress_shadowed(tracks)
        movers = []
        if self.movers_on:
            movers = cluster_movers(radar, self.calib, exclude=used, **self.movers_kw)
            near_track = [t.pos[:2] for t in tracks]                 # also drop movers sitting on a track
            movers = [m for m in movers if not near_track or
                      min(np.linalg.norm(m.pos[:2] - q) for q in near_track) > 1.5]
        work = None
        if self.work is not None:
            t0 = time.perf_counter()
            work = self.work.update(radar.stamp, dt, world, radar.xyz, st.dynamic,
                                    [d.box for d in dets2d if d.cls == "worker"], radar.depth, image, st.keep, st.alpha,
                                    radar.img_dt)
            tm["work"] = (time.perf_counter() - t0) * 1e3
        recon = None
        if self.recon is not None and world is not None:
            t0 = time.perf_counter()
            stat = st.static_inlier & st.keep
            if st.below_ground is not None:
                stat &= ~st.below_ground
            recon = self.recon.update(radar.stamp + radar.img_dt, image, world, radar.xyz[stat], dets2d, img_dt=radar.img_dt)
            tm["recon"] = (time.perf_counter() - t0) * 1e3
        return FrameResult(radar.stamp, tracks, v_lvl, ok, radar.xyz, radar.static_mask, len(dets3d),
                           radar.v_r_comp, st.dynamic, st.alpha, st.sigma_v, dets2d, movers, tm, yaw_rate,
                           world=world, work=work, recon=recon)


def build_pipeline(cfg: dict, calib: RigCalib, root, weights: str | None = None) -> Pipeline:
    """Detector + optional ReID from the config (paths relative to the project root)."""
    from pathlib import Path

    from rcmot.detection.yolo2d import Yolo2D

    dc = cfg["detector"]
    det = Yolo2D(weights or str(Path(root) / dc["weights"]), dc["class_map"], imgsz=dc["imgsz"], conf=dc["conf"])
    reid = None
    rc = cfg.get("reid", {})
    if rc.get("enable", False):
        from rcmot.reid.extractor import ReID

        reid = ReID(str(Path(root) / rc["weights"]))
    engine = None
    wk = cfg.get("work", {})
    if wk.get("enable", False) and wk.get("depth", "stereo") == "vggt_omega":
        from rcmot.work.depth_engine import shared_engine
        from rcmot.work.monitor import WorkParams

        wp = WorkParams(**wk)
        if (Path(root) / wp.vggt_ckpt).exists():
            engine = shared_engine(wp.vggt_ckpt, Path(root), views=wp.vggt_views, keyframe_gap_s=wp.vggt_gap_s, sync=wp.vggt_sync)
        else:
            print(f"[rcmot] work.depth vggt_omega: {wp.vggt_ckpt} not found, using stereo", flush=True)
    recon = None
    rc = cfg.get("recon", {})
    if rc.get("enable", False):
        from rcmot.work.recon import DriveRecon, ReconParams

        rp = ReconParams(**rc)
        if (Path(root) / rp.vggt_ckpt).exists():
            recon = DriveRecon(rp, calib, Path(root))
        else:
            print(f"[rcmot] recon: {rp.vggt_ckpt} not found, drive reconstruction off", flush=True)
    return Pipeline(det, calib, cfg, reid=reid, depth_engine=engine, recon=recon)
