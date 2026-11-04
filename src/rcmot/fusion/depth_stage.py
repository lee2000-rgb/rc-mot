"""Online metric depth-map stage for the fusion (docs/33 wave 3, Track B; research/fusion_wave/trackB/report.md §10).

Per frame, before fuse() (Pipeline):
  history   the static radar points of the last n_frames scans (ScanStats ego-fit inliers, kept, not below ground,
            p_dyn below static_pdyn_max), kept in the current base frame with the ego motion (v dt, yaw_rate dt)
            while the ego moves (a static ego is not compensated: the xpanner gyro bias at rest would rotate them)
  map       every `every`-th frame: DA-V2 Small (rcmot.fusion.depth_map.MetricDepth, input 518x924 or 392x700) on
            the image, disparity-domain RANSAC fit against the accumulated static points (rotated into the image time
            with the box-time angle on a turning base, as fuse() rotates its points) and the (s, t) Kalman filter.
            The frames in between reuse that map ("async-equivalent": the map is ready on its own frame).
  quality   the map is offered only when the frame's fit has an inlier median relative error <= quality_rel_err, its s
            lies in the physical box, and the KF has not rejected / failed the last max_rejects fits
  per box   r_dmap = bottom-third median of the newest map, back-projected through the region centre (horizontal range
            from the camera; also from the level origin for the B-b feature). A map made 1-2 frames earlier on a moving
            ego is read at the box's position in the map's own image (box centre point moved by the ego motion since).
            ok = quality and r_dmap <= max_range (25 m).
  bias      (B-a) per track: median of radar range / r_dmap over the track's radar-backed frames of the last
            bias_window_s (>= bias_min_n); fuse() reports the ratio on its radar-backed detections (Det3D.dmap_ratio),
            observe() files it under the track that took the detection (identity of the 2-D box). The box -> track
            lookup for the next frame uses the track's projected box at the predicted position (IoU / unique centre).
Not implemented (docs/33 §8.4): per-point depth consistency (B-c), the mask-median sample, ranges beyond max_range,
the weak measurement while coasting (B-d; needs the tracker).
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from rcmot.fusion.depth_map import ROOT, DepthMapParams, MetricDepth
from rcmot.utils.calib import RigCalib, object_box

MAD_K = 1.4826
_NET: dict = {}                       # (weights, encoder, device, fp16) -> (model, torch): kept across Pipeline.reset()


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(min(a[2], b[2]) - max(a[0], b[0]), 0.0)
    h = max(min(a[3], b[3]) - max(a[1], b[1]), 0.0)
    i = w * h
    return float(i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-9))


@dataclass
class DepthStageParams:
    enable: bool = False              # run the stage although no fusion use (fusion.cluster_dmap / dmap_fallback) is on
    every: int = 3                    # network + fit every N-th frame, the frames in between reuse the last map
    phase: int = 0                    # first map on frame `phase` (evaluation: GT frames every 3rd scan are phase-locked
                                      # to the maps; phase 1 / 2 score the 2- / 1-frame-old maps)
    input_hw: tuple = (518, 924)      # DA-V2 input H x W: 518x924 (Track B default) or 392x700 (0.64x cost)
    n_frames: int = 3                 # static scans accumulated for the fit
    weights: str = "models/depth_anything_v2_vits.pth"
    encoder: str = "vits"
    fp16: bool = True
    device: str = "cuda"
    max_range: float = 25.0           # [m] r_dmap above this is not offered (Track B: 25-30 m 33 %, 30+ m 4 % within 1 m)
    quality_rel_err: float = 0.10     # fit inlier median |Z_fit - Z_radar| / Z_radar above this: map not offered
    max_rejects: int = 3              # this many KF-rejected / failed fits in a row: map not offered
    static_pdyn_max: float = 0.5      # static fit points: p_dyn below this (and ego-fit inlier, kept, not below ground)
    box_time: bool = True             # moving ego: fit points rotated into the image's time (Pipeline box-time angle)
    stale_shift: bool = True          # moving ego, map 1-2 frames old: read it at the box's position in the map's image
    stale_ref_m: float = 20.0         # [m] range of the box-centre point used for that shift
    bias_window_s: float = 10.0       # B-a per-track bias window
    bias_min_n: int = 3
    sigma_rel_floor: float = 0.06     # relative sigma floor of a dmap range (Track B §5: learned MAD over-confident)
    sigma_window: int = 150           # learned spread: 1.4826 MAD of ln(radar / dmap) over this many radar-backed boxes
    sigma_min_n: int = 20
    track_iou: float = 0.3            # box -> track association for the bias (one-to-one over all live tracks)
    bias_max_tcam_s: float = 1.0      # [s] tracks without a camera update for longer do not take a box for the bias
    bias_range: tuple = (0.67, 1.5)   # a track bias outside this is not used (a track locked on a wrong radar cluster)
    classes: tuple = ("worker",)      # boxes that get a depth-map range
    ransac_vectorized: bool = True    # depth_map.fit_disparity: all RANSAC hypotheses in one numpy pass

    @classmethod
    def from_cfg(cls, d: dict | None) -> "DepthStageParams":
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        p = cls(**{k: v for k, v in d.items() if k in names})
        p.input_hw = tuple(int(v) for v in p.input_hw)
        p.classes = tuple(p.classes)
        p.bias_range = tuple(float(v) for v in p.bias_range)
        return p


class DepthStage:
    def __init__(self, calib: RigCalib, params: DepthStageParams | None = None, root: Path = ROOT):
        self.calib, self.p, self.root = calib, params or DepthStageParams(), Path(root)
        p = self.p
        self.md = MetricDepth(calib, DepthMapParams(weights=p.weights, encoder=p.encoder, input_hw=p.input_hw, fp16=p.fp16,
                                                    device=p.device, n_frames=p.n_frames,
                                                    ransac_vectorized=p.ransac_vectorized), root=self.root)
        key = (p.weights, p.encoder, p.device, p.fp16)
        if key in _NET:
            self.md.model, self.md.torch = _NET[key]
        self._key = key
        self.C = calib.camera_center()
        self.reset()

    def reset(self) -> None:
        self.since: int | None = None     # frames since the newest map (None: no map yet)
        self.A, self.b = np.eye(3), np.zeros(3)    # x_mapscan = A x_scan + b (base motion since the map's scan)
        self.a_map = 0.0                  # box-time angle of the map's frame
        self.quality = False
        self.n_rej = 0
        self.ratios: dict = {}            # Track -> deque[(t, radar / dmap range)]
        self.lr_win: deque = deque(maxlen=self.p.sigma_window)
        self.ms: dict = {}
        self.n_maps = 0
        self.last_fit = None
        self.frame = 0

    # ------------------------------------------------------------------ per frame
    def static_xyz(self, radar) -> np.ndarray:
        st = radar.stats
        if st is None:
            return np.zeros((0, 3))
        m = st.static_inlier & st.keep
        if getattr(st, "below_ground", None) is not None:
            m = m & ~st.below_ground
        if getattr(st, "p_dyn", None) is not None:
            m = m & (st.p_dyn < self.p.static_pdyn_max)
        return radar.xyz[m]

    def step(self, image: np.ndarray | None, radar, dt: float, v_lvl: np.ndarray, yaw_rate: float, moving: bool,
             a_box: float = 0.0) -> bool:
        """History / map transform update with this frame's ego motion, then either a new map (every N-th frame) or
        this scan's static points into the history. a_box: the box-time angle of this frame (0 on a static ego).
        Returns True when a new map was made."""
        t0 = time.perf_counter()
        p = self.p
        if moving:                                         # base moved by v dt and turned yaw_rate dt since the last scan
            dp, dyaw = np.asarray(v_lvl, float) * dt, float(yaw_rate) * dt
            self.md.ego_compensate(dp, dyaw)
            if self.since is not None:
                self.b = self.A @ dp + self.b              # x_prev = R x_cur + dp -> x_map = A R x_cur + (A dp + b)
                self.A = self.A @ _rz(dyaw)
        xyz = self.static_xyz(radar)
        compute = image is not None and self.frame >= p.phase and (self.since is None or self.since + 1 >= max(int(p.every), 1))
        self.ms = {}
        if compute:
            if self.md.model is None:
                self.md._load()
                _NET[self._key] = (self.md.model, self.md.torch)
            fit_R = _rz(a_box) if (p.box_time and abs(a_box) > 1e-9) else None
            res = self.md.update(image, xyz, stamp=radar.stamp, fit_R=fit_R)
            self.ms.update({k: round(v, 3) for k, v in res.ms.items()})
            fit = res.fit
            upd = fit is not None and bool(res.updated)
            self.n_rej = 0 if upd else self.n_rej + 1
            s_lo, s_hi = self.md.p.s_range
            # offered only when THIS frame's fit was accepted: a KF-rejected fit leaves the map on the filter's older
            # scale, which this frame's radar points contradict (review 11-02 #4)
            self.quality = bool(res.depth is not None and upd and fit.rel_err <= p.quality_rel_err
                                and s_lo <= fit.s <= s_hi)
            if self.n_rej >= p.max_rejects:               # filter stuck on an old scale: restart it with the next fit
                self.md.kf.x, self.md.kf.P = None, None
                self.n_rej = 0
            self.last_fit = fit
            self.since, self.A, self.b = 0, np.eye(3), np.zeros(3)
            self.a_map = float(a_box) if p.box_time else 0.0   # the rotation the fit actually used
            self.n_maps += 1
        else:
            self.md.push_static(xyz)
            if self.since is not None:
                self.since += 1
        self.frame += 1
        self.ms["stage"] = round((time.perf_counter() - t0) * 1e3, 3)
        self.ms["computed"] = bool(compute)
        return compute

    # ------------------------------------------------------------------ per box
    def boxes(self, dets2d: list, tracks: list | None = None, *, moving: bool = False, a_box: float = 0.0,
              dt: float = 0.066, v_ego: np.ndarray | None = None, yaw_rate: float = 0.0, t_now: float = 0.0) -> list:
        """Per detection: {r, r_org, sig, ok, bias, sig_rel, age} from the newest map, or None (no map / class / sample)."""
        t0 = time.perf_counter()
        res = self.md.last
        out = [None] * len(dets2d)
        if res is None or res.depth is None or self.since is None or not dets2d:
            self.ms["boxes"] = round((time.perf_counter() - t0) * 1e3, 3)
            return out
        p = self.p
        a_cur = float(a_box) if p.box_time else 0.0
        # the map's image differs from this frame's whenever the base moved since the map or either frame carries a
        # box-time angle -- also when this frame already counts as static again (swing reversal / stop; review #3)
        moved = (np.abs(self.A - np.eye(3)).max() > 1e-9 or np.abs(self.b).max() > 1e-9
                 or abs(self.a_map) > 1e-9 or abs(a_cur) > 1e-9)
        shift = p.stale_shift and self.since > 0 and moved
        if shift:                                          # current image frame -> the map's image frame
            Am = _rz(self.a_map) @ self.A @ _rz(-a_cur)
            bm = _rz(self.a_map) @ self.b
        bias = self._track_bias(dets2d, tracks, dt, v_ego, yaw_rate, moving, t_now) if tracks else [None] * len(dets2d)
        sig_rel = p.sigma_rel_floor
        if len(self.lr_win) >= p.sigma_min_n:
            a = np.asarray(self.lr_win)
            sig_rel = max(sig_rel, MAD_K * float(np.median(np.abs(a - np.median(a)))))
        for k, d in enumerate(dets2d):
            if d.cls not in p.classes:
                continue
            box = np.asarray(d.box, float)
            if shift:
                ray = self.calib.pixel_ray(0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
                P = self.C + p.stale_ref_m / max(float(np.linalg.norm(ray[:2])), 1e-6) * ray
                uv, fr = self.calib.project(np.vstack([P, Am @ P + bm]))
                if not fr.all():
                    continue
                du = uv[1] - uv[0]
                box = box + np.array([du[0], du[1], du[0], du[1]])
            rb = self.md.range_for_box(box, sample="bottom")
            if rb is None:
                continue
            pos = np.asarray(rb[2]["pos"], float)
            if shift:
                pos = Am.T @ (pos - bm)                    # back into the current (image-time) base frame
            r_cam = float(np.linalg.norm(pos[:2] - self.C[:2]))
            out[k] = {"r": r_cam, "r_org": float(np.linalg.norm(pos[:2])), "sig": float(rb[1]),
                      "ok": bool(self.quality and r_cam <= p.max_range), "bias": bias[k], "sig_rel": sig_rel,
                      "age": int(self.since)}
        self.ms["boxes"] = round((time.perf_counter() - t0) * 1e3, 3)
        return out

    def _track_bias(self, dets2d, tracks, dt, v_ego, yaw_rate, moving, t_now) -> list:
        """Per detection the median radar / dmap ratio of the track predicted into its box (>= bias_min_n samples in
        the last bias_window_s, inside bias_range), or None. Boxes and tracks are associated ONE-TO-ONE over all live
        tracks of the class (not only those with a bias), so a biased track cannot lend its bias to a neighbour's or a
        new person's box (review 11-02 #2): greedy by IoU >= track_iou, then a centre-inside test that must be unique
        both ways. Tracks without a camera update for more than bias_max_tcam_s take no box."""
        p = self.p
        v_ego = np.zeros(3) if v_ego is None else np.asarray(v_ego, float)
        live = set(tracks)
        for t in [t for t in self.ratios if t not in live]:
            del self.ratios[t]
        cands = []                                         # (cls, predicted box, bias or None)
        for t in tracks:
            if getattr(t, "t_cam", 0.0) > p.bias_max_tcam_s:
                continue
            pos = t.kf.pos + (t.kf.vel - v_ego) * dt
            if moving:
                pos = _rz(-yaw_rate * dt) @ pos
            box = object_box(self.calib, t.cls, pos)
            if box is None:
                continue
            b = None
            dq = self.ratios.get(t)
            if dq is not None:
                while dq and dq[0][0] < t_now - p.bias_window_s:
                    dq.popleft()
                if len(dq) >= p.bias_min_n:
                    b = float(np.median([r for _, r in dq]))
                    if not (p.bias_range[0] <= b <= p.bias_range[1]):
                        b = None
            cands.append((t.cls, box, b))
        pairs = sorted(((_iou(c[1], d.box), i, j) for i, d in enumerate(dets2d) for j, c in enumerate(cands)
                        if c[0] == d.cls), key=lambda x: -x[0])
        d2c, used = {}, set()
        for iou, i, j in pairs:
            if iou < p.track_iou:
                break
            if i not in d2c and j not in used:
                d2c[i] = j
                used.add(j)
        ctr = {i: (0.5 * (d.box[0] + d.box[2]), 0.5 * (d.box[1] + d.box[3])) for i, d in enumerate(dets2d) if i not in d2c}

        def _in(c, xy):
            return c[1][0] <= xy[0] <= c[1][2] and c[1][1] <= xy[1] <= c[1][3]
        for i, xy in ctr.items():
            inside = [j for j, c in enumerate(cands) if j not in used and c[0] == dets2d[i].cls and _in(c, xy)]
            if len(inside) == 1 and sum(_in(cands[inside[0]], o) for k, o in ctr.items()
                                        if dets2d[k].cls == dets2d[i].cls) == 1:
                d2c[i] = inside[0]
                used.add(inside[0])
        return [cands[d2c[i]][2] if i in d2c else None for i in range(len(dets2d))]

    def observe(self, dets3d: list, tracks: list, t_now: float) -> None:
        """After the tracker update: file each radar-backed detection's radar / dmap ratio (Det3D.dmap_ratio, set by
        fuse()) under the track that took it (the track's box2d is the detection's box object)."""
        by_box = {id(t.box2d): t for t in tracks if getattr(t, "tsu", 1.0) == 0.0 and t.box2d is not None}
        for d in dets3d:
            ratio = getattr(d, "dmap_ratio", None)
            if ratio is None or not np.isfinite(ratio) or ratio <= 0:
                continue
            self.lr_win.append(float(np.log(ratio)))
            t = by_box.get(id(d.box2d))
            if t is not None:
                self.ratios.setdefault(t, deque(maxlen=200)).append((float(t_now), float(ratio)))


def make_depth_stage(cfg: dict, calib: RigCalib, fusion_p, root: Path = ROOT) -> DepthStage | None:
    """The stage when depth_map.enable or a fusion use of it (cluster_dmap, dmap_fallback) is on, else None."""
    dcfg = cfg.get("depth_map") or {}
    if not (dcfg.get("enable", False) or getattr(fusion_p, "cluster_dmap", False) or getattr(fusion_p, "dmap_fallback", False)):
        return None
    return DepthStage(calib, DepthStageParams.from_cfg(dcfg), root)
