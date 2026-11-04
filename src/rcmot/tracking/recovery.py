"""Radar-Gated Recovery (RGR): MOTDD PGR (third_party/MOTDD, IW-FCV 2026) extended with 4D radar.

PGR re-runs the detector on a zoomed crop around the KF prediction of each track that lost its match,
removes candidates that duplicate detections already used this frame, gates the rest with a buffered IoU
and re-activates the track with the best `BIoU x score` candidate. RGR keeps that pipeline and adds radar:

  who   candidates = unmatched tracks with radar support (points near the 3D prediction whose
        ego-compensated Doppler agrees with the track velocity), most support first, then longest
        tracklet (PGR order); budget K
  where the crop is centred on the projection of the radar-refined 3D position, sized from the class's
        physical size at that range (PGR: max(crop_scale * sqrt(area), min_crop))
  what  one batched detector call on all crops (low confidence)
  check PGR dedup + buffered-IoU gate, plus class match and a 3D gate between the candidate's
        radar-fused position and the track
  else  if no image candidate survives but the radar support is strong enough, return a
        "radar_only" measurement (position + Doppler velocity) that keeps the track alive

Flags `use_radar_roi` / `use_radar_gate` / `radar_only` switch the radar parts off for the ablation
(all off = PGR applied to the 3D tracker).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rcmot.calib.reflectors import _clusters
from rcmot.fusion.radar_camera import FusionParams, _velocity, fuse
from rcmot.tracking.tracker import Track, iou_xyxy
from rcmot.types import Det2D, Det3D, RadarFrame
from rcmot.utils.calib import RigCalib


def dilate_xyxy(boxes: np.ndarray, ratio: float) -> np.ndarray:
    """MOTDD `_dilate_xyxy`: grow every box by ratio*w/2 and ratio*h/2 on each side."""
    if ratio <= 0 or boxes.size == 0:
        return boxes
    out = boxes.astype(np.float64).copy()
    w, h = out[:, 2] - out[:, 0], out[:, 3] - out[:, 1]
    out[:, [0, 2]] += np.outer(0.5 * ratio * w, [-1, 1])
    out[:, [1, 3]] += np.outer(0.5 * ratio * h, [-1, 1])
    return out


def buffered_iou(a: np.ndarray, b: np.ndarray, ratio: float) -> np.ndarray:
    """MOTDD `buffered_iou` (BIoU_r): IoU after dilating both box sets by `ratio`."""
    return iou_xyxy(dilate_xyxy(a, ratio), dilate_xyxy(b, ratio))


@dataclass
class RGRParams:
    K: int = 4                        # candidate budget per frame (PGR)
    crop_scale: float = 3.0           # crop side = max(crop_scale * sqrt(area), min_crop); PGR uses 8 for drones
    min_crop: int = 192
    crop_imgsz: int = 640
    conf_thresh: float = 0.1          # PGR
    dedup_iou: float = 0.5            # PGR tau_dedup
    double_count_iou: float = 0.3     # PGR
    gate_iou: float = 0.3             # PGR tau_gate
    gate_buffer_ratio: float = 0.2    # PGR r
    use_radar_roi: bool = True
    use_radar_gate: bool = True
    radar_only: bool = True
    # radar support of a track: statistical gates (docs/09 §4.8) instead of a fixed radius / Doppler tolerance
    extent: dict = field(default_factory=lambda: {"worker": 0.4, "default": 2.5})   # object half-size [m]
    gate_xy: float = 9.21             # chi^2_2 99 % on (track cov + radar point cov + extent)
    gate_dop: float = 6.63            # chi^2_1 99 % on (sigma_v(scan)^2 + u'P_v u)
    sigma_range: float = 0.15         # radar range sigma [m]
    sigma_az_deg: float = 1.0         # radar azimuth sigma at az_prob = 255 [deg]
    sigma_v_default: float = 0.05
    max_track_var: float = 1.0        # [m^2] cap of the track covariance used in the support gate
    min_support: int = 1              # points needed to be a radar-supported candidate
    min_radar_only: int = 2           # points needed for a radar-only measurement
    gate_chi2: float = 16.27          # chi2(3 dof, 99.9%) on the recovered position
    obj_size: dict = field(default_factory=lambda: {"worker": (0.8, 1.8), "default": (5.0, 3.0)})  # (width, height) m
    # radar-only must stay on returns that can belong to this object (docs/12 S1). Occupied / span checks are
    # off by default: no gain on the GT segments, -0.04 day IDF1 with the seg detector (docs/14)
    occupied_radius: dict = field(default_factory=lambda: {"worker": 0.0, "default": 0.0})  # points owned by measured tracks
    shadow_iou: float = 0.5           # image overlap with a closer measured same-class box -> shadowed
    shadow_gap: float = 1.0           # [m] "closer" margin
    max_cluster_span: dict = field(default_factory=lambda: {"worker": 1e3, "default": 1e3})  # [m] larger -> not this class
    fuse_track_prior: bool = True     # docs/30: fuse the recovered box with the track's range / BEV prior and learned
                                      # height (as the main fusion does), so the cluster gate applies here too


@dataclass
class _Cand:
    track: Track
    support: np.ndarray               # radar point indices
    pos: np.ndarray                   # 3D centre used for the ROI
    pred_box: np.ndarray              # xyxy predicted box in the image


class RadarGatedRecovery:
    def __init__(self, detector, calib: RigCalib, params: RGRParams | None = None,
                 fusion_params: FusionParams | None = None):
        self.det, self.calib = detector, calib
        self.p = params or RGRParams()
        self.fusion_p = fusion_params or FusionParams()               # settings of the main fusion (Pipeline sets it)
        self.ego_static: bool | None = None                           # this frame's ego state (Pipeline sets it)
        self.ego_yaw_rate: float = 0.0                                # ... and its yaw rate [rad/s] (gate widening, impl2)
        self.ego_speed: float = 0.0                                   # ... and |base velocity| [m/s] (docs/38 ground cue)
        self.image: np.ndarray | None = None
        self.radar: RadarFrame | None = None
        self.used_boxes = np.zeros((0, 4))
        self.used_radar = np.zeros(0, int)
        self.stats = {"candidates": 0, "image": 0, "radar_only": 0}

    def set_frame(self, image: np.ndarray, radar: RadarFrame, det_boxes: list[np.ndarray],
                  used_radar: np.ndarray | None = None) -> None:
        self.image, self.radar = image, radar
        self.used_boxes = np.array(det_boxes, float).reshape(-1, 4)
        self.used_radar = np.zeros(0, int) if used_radar is None else np.asarray(used_radar, int)
        self.occupied: list = []

    # ---------------------------------------------------------------- helpers
    def _radar_support(self, t: Track) -> np.ndarray:
        """Radar points statistically compatible with the track in position (x, y) and Doppler."""
        rf, p = self.radar, self.p
        if rf is None or len(rf.xyz) == 0:
            return np.zeros(0, int)
        st = rf.stats
        keep = (st.keep if st is not None else np.ones(len(rf.xyz), bool)).copy()
        keep[self.used_radar] = False                                 # returns already explained by a detection
        for pos, cls, *_ in self.occupied:                               # ... or lying on a track measured this frame
            r_occ = p.occupied_radius.get(cls, p.occupied_radius["default"])
            keep &= np.linalg.norm(rf.xyz[:, :2] - pos[:2], axis=1) > r_occ
        d = rf.xyz[:, :2] - t.kf.pos[:2]
        near = np.flatnonzero(keep & (np.linalg.norm(d, axis=1) < 15.0))
        if len(near) == 0:
            return near
        r = np.linalg.norm(rf.xyz_radar[near], axis=1)
        azp = np.clip(rf.az_prob[near] / 255.0, 0.1, 1.0) if rf.az_prob is not None else np.ones(len(near))
        s_tan = r * np.radians(p.sigma_az_deg) / np.sqrt(azp)
        u = rf.xyz[near, :2] / np.maximum(np.linalg.norm(rf.xyz[near, :2], axis=1, keepdims=True), 1e-6)
        ext = p.extent.get(t.cls, p.extent["default"]) ** 2
        w_, V_ = np.linalg.eigh(t.kf.P[:2, :2])                       # cap the track's own spread: a long-lost
        P = V_ @ np.diag(np.minimum(w_, p.max_track_var)) @ V_.T      # track must not pull in the background
        ok = np.zeros(len(near), bool)
        for k, i in enumerate(near):                                 # per-point anisotropic covariance
            R = np.outer(u[k], u[k]) * p.sigma_range ** 2 + (np.eye(2) - np.outer(u[k], u[k])) * s_tan[k] ** 2
            S = P + R + ext * np.eye(2)
            ok[k] = d[i] @ np.linalg.solve(S, d[i]) <= p.gate_xy
        near = near[ok]
        v_comp = rf.v_r_comp if rf.v_r_comp is not None else (st.v_comp if st is not None else None)
        if len(near) == 0 or v_comp is None or t.hits < 2:
            return near
        sig_v = st.sigma_v if st is not None and np.isfinite(st.sigma_v) else p.sigma_v_default
        u3 = rf.xyz_radar[near] / np.linalg.norm(rf.xyz_radar[near], axis=1, keepdims=True) @ self.calib.R_lr.T
        dop = np.array([t.kf.radial_d2(u3[k], v_comp[i], sig_v ** 2) for k, i in enumerate(near)])
        return near[dop <= p.gate_dop]

    def _shadowed(self, t: Track, box: np.ndarray) -> bool:
        """The track's image region is covered by a closer, camera-measured object of the same class: the
        radar returns behind that object are background (or an invisible occluded person) and must not keep
        a second box on the same image ray alive (docs/14)."""
        C = self.calib.camera_center()
        r_t = np.linalg.norm(t.kf.pos[:2] - C[:2])
        for occ in self.occupied:
            if len(occ) < 3 or occ[2] is None or occ[1] != t.cls:
                continue
            if np.linalg.norm(occ[0][:2] - C[:2]) + self.p.shadow_gap < r_t and \
                    iou_xyxy(box[None], np.asarray(occ[2], float)[None])[0, 0] >= self.p.shadow_iou:
                return True
        return False

    def _project_box(self, cls: str, pos: np.ndarray) -> np.ndarray | None:
        w, h = self.p.obj_size.get(cls, self.p.obj_size["default"])
        x, y = pos[0], pos[1]
        z0 = min(pos[2] - h / 2, 0.0)
        corners = np.array([[x + dx, y + dy, z] for dx in (-w / 2, w / 2) for dy in (-w / 2, w / 2)
                            for z in (z0, z0 + h)])
        uv, front = self.calib.project(corners)
        if not front.all():
            return None
        W, H = self.calib.image_size
        box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
        if box[2] < 0 or box[3] < 0 or box[0] > W or box[1] > H:
            return None
        return box

    def _crop(self, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        H, W = self.image.shape[:2]
        cx, cy = 0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
        s = np.sqrt(max((box[2] - box[0]) * (box[3] - box[1]), 1.0))
        side = int(min(max(self.p.crop_scale * s, self.p.min_crop), max(W, H)))
        x1, y1 = int(np.clip(cx - side / 2, 0, max(W - side, 0))), int(np.clip(cy - side / 2, 0, max(H - side, 0)))
        x2, y2 = min(x1 + side, W), min(y1 + side, H)
        return self.image[y1:y2, x1:x2], np.array([x1, y1, x1, y1], float)

    def _radar_only_det(self, c: _Cand) -> Det3D | None:
        """Position from the support cluster closest to the prediction (not all gated points)."""
        rf, idx = self.radar, c.support
        st = rf.stats
        ext = self.p.extent.get(c.track.cls, self.p.extent["default"])
        xy = rf.xyz[idx, :2]
        lab = _clusters(xy, max(ext, 0.5))
        best = min(np.unique(lab), key=lambda L: np.linalg.norm(xy[lab == L].mean(0) - c.track.kf.pos[:2]))
        idx = idx[lab == best]
        span = np.ptp(rf.xyz[idx, :2], axis=0).max() if len(idx) > 1 else 0.0
        if span > self.p.max_cluster_span.get(c.track.cls, self.p.max_cluster_span["default"]) \
                or len(idx) < self.p.min_radar_only:
            return None                                              # e.g. a worker track sliding onto a machine
        w = (st.alpha[idx] if st is not None else np.ones(len(idx))) + 1e-3
        pos = np.r_[np.average(rf.xyz[idx, :2], axis=0, weights=w), c.track.kf.pos[2]]
        spread = np.sqrt(np.average((rf.xyz[idx, :2] - pos[:2]) ** 2, axis=0, weights=w))
        sig = np.sqrt(spread ** 2 + (0.5 * ext) ** 2 + 0.3 ** 2)
        d = Det3D(c.track.cls, c.track.score, c.pred_box, pos, np.diag(np.r_[sig ** 2, 1.0]), n_radar=len(idx),
                  source="radar_only", radar_idx=idx)
        v_comp = rf.v_r_comp if rf.v_r_comp is not None else (st.v_comp if st is not None else None)
        if v_comp is not None:
            sig_v = st.sigma_v if st is not None and np.isfinite(st.sigma_v) else self.p.sigma_v_default
            _velocity(d, rf, idx, w, v_comp, sig_v, self.calib, FusionParams())
        return d

    def _fuse_priors(self, t: Track) -> dict:
        """The track's range prior, BEV prior (its predicted state at this frame, covariance capped as in the support
        gate) and learned height, in the form fuse() takes (one candidate box)."""
        fp = self.fusion_p
        C = self.calib.camera_center()
        w_, V_ = np.linalg.eigh(t.kf.P[:2, :2])
        P2 = V_ @ np.diag(np.minimum(w_, self.p.max_track_var)) @ V_.T
        sig = max(float(np.sqrt(np.trace(P2) / 2)), fp.track_prior_sigma_floor)
        return {"range_priors": [(float(np.linalg.norm(t.kf.pos[:2] - C[:2])), sig)],
                "xy_priors": [(t.kf.pos[:2].copy(), P2, float(t.since_pos))],
                "heights": [t.height()] if fp.track_height else None}

    # ------------------------------------------------------------------- main
    def __call__(self, tracks: list[Track], occupied: list | None = None) -> dict[int, Det3D]:
        """occupied: (position, class) of tracks measured this frame; their radar returns are not reused."""
        self.occupied = occupied or []
        if self.image is None or not tracks:
            return {}
        p = self.p
        cands: list[_Cand] = []
        for t in tracks:
            sup = self._radar_support(t) if (p.use_radar_roi or p.use_radar_gate or p.radar_only) else np.zeros(0, int)
            if p.use_radar_roi and len(sup) < p.min_support:
                continue                                             # radar sees nothing there: skip
            pos = t.kf.pos.copy()
            if p.use_radar_roi and len(sup):
                pos[:2] = np.median(self.radar.xyz[sup, :2], axis=0)
            box = self._project_box(t.cls, pos)                      # PGR: projection of the KF prediction
            if box is None or self._shadowed(t, box):
                continue
            cands.append(_Cand(t, sup, pos, box))
        cands.sort(key=lambda c: (-len(c.support), -c.track.tracklet_len))
        cands = cands[:p.K]
        self.stats["candidates"] += len(cands)
        if not cands:
            return {}

        crops, offs = zip(*(self._crop(c.pred_box) for c in cands))
        results = self.det.detect_batch(list(crops), imgsz=p.crop_imgsz, conf=p.conf_thresh)

        out: dict[int, Det3D] = {}
        used = self.used_boxes
        for c, off, dets in zip(cands, offs, results):
            dets = [Det2D(d.cls, d.score, d.box + off) for d in dets if d.cls == c.track.cls]
            best, best_score = None, -np.inf
            if dets:
                boxes = np.array([d.box for d in dets])
                max_iou = iou_xyxy(boxes, used).max(1) if len(used) else np.zeros(len(dets))
                keep = (max_iou < p.dedup_iou) & (max_iou < p.double_count_iou)
                biou = buffered_iou(c.pred_box[None], boxes, p.gate_buffer_ratio)[0]
                priors = self._fuse_priors(c.track) if p.fuse_track_prior else {}
                for k in np.flatnonzero(keep & (biou >= p.gate_iou)):
                    d3 = fuse([dets[k]], self.radar, self.calib, image=self.image, params=self.fusion_p,
                              ego_static=self.ego_static, ego_yaw_rate=self.ego_yaw_rate, ego_speed=self.ego_speed,
                              **priors)
                    if not d3:
                        continue
                    d3 = d3[0]
                    if p.use_radar_gate and getattr(d3, "rlik", None) is not None:
                        # docs/38: the box's range likelihood x the track's prediction; radar support at that range
                        from rcmot.tracking.range_select import select
                        sel = select(d3, c.track.kf.pos[:2], c.track.kf.P[:2, :2], 0.5, eps=0.05,
                                     radar_ratio=self.fusion_p.lik_radar_backed)
                        if sel is None or not sel.radar or sel.d2 > p.gate_chi2:
                            continue
                    elif p.use_radar_gate:
                        if not d3.source.startswith("radar"):
                            continue                                 # recovered box has no radar evidence
                        R = d3.pos_cov + np.eye(3) * 0.25
                        if c.track.kf.mahalanobis_pos(d3.pos, R) > p.gate_chi2:
                            continue
                    score = biou[k] * dets[k].score
                    if score > best_score:
                        best, best_score = d3, score
            if best is not None:
                best.source = "roi_research"
                out[c.track.tid] = best
                used = np.vstack([used, best.box2d[None]])
                self.stats["image"] += 1
            elif p.radar_only and len(c.support) >= p.min_radar_only:
                d = self._radar_only_det(c)
                if d is not None:
                    out[c.track.tid] = d
                    self.stats["radar_only"] += 1
        return out
