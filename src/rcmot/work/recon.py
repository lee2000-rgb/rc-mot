"""Online wide-area 3D reconstruction while the machine drives (docs/26 §6): the offline scripts/drive_recon.py
pipeline run incrementally on the frames the GUI / live worker processes.

Per frame the pipeline hands over the image, the world pose (rcmot.world: gyro yaw + radar-Doppler dead reckoning)
and the scan's static radar points. A keyframe is taken every kf_dist m / kf_deg deg (or kf_max_dt s); every
`chunk - overlap` new keyframes a VGGT-Omega chunk of `chunk` views runs on a background thread (RTX 3080, 1280x720
-> 384x688: 16 views 1.5 s, 4.2 GB with the model) and is
  scaled    radar-Doppler travel of the chunk (driving) / radar depth ratio (swing, standing) / shared baselines,
  joined    to the previous chunk on the shared frames: cameras -> heading / xy, depth points -> tilt / height,
  refined   each keyframe's roll / pitch / height against the map built so far (3 DoF point-to-plane, <= 0.5 deg /
            0.1 m per keyframe, correction chained to all later keyframes),
  filtered  confidence (top conf_q, >= conf_min), sky, people (the detector's masks), own machine (bottom crop),
            points seen from >= min_views keyframes (0.3 m voxels), 0.1 m voxels, de-duplicated against the map.
The keyframe chain is anchored to the world frame of rcmot.world at the first keyframe, so the map, the BEV world
view and the trail share one frame (the VGGT heading replaces the gyro from the second chunk on; they drift apart by
a few degrees over 100 s). Same maths as the offline tool: scripts/recon_online_eval.py feeds the cached bag reads
through this class and scores the result against the LiDAR map (docs/26 §6).
"""
from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------------------------------- geometry
def se3(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T, P) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def voxel_keys(P, size) -> np.ndarray:
    key = np.floor(P / size).astype(np.int64)
    return (key[:, 0] + (1 << 20)) * (1 << 42) + (key[:, 1] + (1 << 20)) * (1 << 21) + (key[:, 2] + (1 << 20))


def voxel_first(P, size, *extra):
    """First point per voxel (and the matching rows of extra arrays)."""
    key = np.floor(P / size).astype(np.int64)
    _, idx = np.unique(key, axis=0, return_index=True)
    idx.sort()
    return (P[idx],) + tuple(e[idx] for e in extra)


def rot_between(a, b) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def mean_pose(Ts) -> np.ndarray:
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return se3(R, np.mean([T[:3, 3] for T in Ts], 0))


def align_rigid(P, Q, w, T0, delta=0.3, iters=8) -> np.ndarray:
    """T (P-frame <- Q-frame) minimising the Huber-weighted sum of w |P - T Q|^2 (weighted Kabsch, IRLS), from T0."""
    T = T0
    for _ in range(iters):
        r = np.linalg.norm(P - apply(T, Q), axis=1)
        wi = w * np.minimum(1.0, delta / np.maximum(r, 1e-9))
        wi = wi / wi.sum()
        pc, qc = wi @ P, wi @ Q
        H = (Q - qc).T @ ((P - pc) * wi[:, None])
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        T = se3(R, pc - R @ qc)
    return T


def model_shape(h: int, w: int, res: int = 512, patch: int = 16) -> tuple[int, int]:
    """VGGT-Omega 'balanced' input size (vggt_omega.utils.load_fn._balanced_target_shape): 1280x720 -> 384x688."""
    n = (res // patch) ** 2
    wp = np.sqrt(n / (h / w))
    hp = n / wp
    return max(1, int(np.round(hp))) * patch, max(1, int(np.round(wp))) * patch


def pixel_rays(calib, h, w) -> np.ndarray:
    """Undistorted camera rays (z = 1) of the model-resolution pixel centres."""
    W, H = calib.image_size
    u, v = np.meshgrid((np.arange(w) + 0.5) * W / w - 0.5, (np.arange(h) + 0.5) * H / h - 0.5)
    n = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64), calib.K, calib.D).reshape(h, w, 2)
    return np.dstack([n, np.ones((h, w))])


def sky_mask(rgb, above) -> np.ndarray:
    """Sky: blue or bright grey (cloud) pixels above the horizon, connected to the image top (VGGT puts part of the sky
    at 10-50 m with confidence > 1.2)."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    skyish = (((b > r + 15) & (b > 110)) | ((np.minimum(np.minimum(r, g), b) > 190) & (np.ptp(rgb, axis=2) < 30))) & above
    _, lab = cv2.connectedComponents(skyish.astype(np.uint8), connectivity=4)
    top = np.unique(lab[0][skyish[0]])
    return np.isin(lab, top[top > 0])


# ---------------------------------------------------------------------------------------------------- parameters
@dataclass
class ReconParams:
    enable: bool = False
    vggt_ckpt: str = "models/vggt/vggt_omega_1b_512.pt"
    res: int = 512                     # VGGT-Omega input resolution (long side)
    chunk: int = 16                    # views per inference
    overlap: int = 4                   # keyframes shared by consecutive chunks (pose chaining)
    kf_dist: float = 1.0               # new keyframe after this much travel [m] ...
    kf_deg: float = 8.0                # ... or this much turn [deg] ...
    kf_max_dt: float = 8.0             # ... or this long [s]
    kf_min_dt: float = 0.15
    conf_q: float = 0.5                # drop the least confident share of each depth map ...
    conf_min: float = 1.2              # ... and pixels below this confidence (sky ~1.0)
    crop_bottom: float = 0.07          # image share at the bottom not used (own machine)
    min_travel: float = 3.0            # [m] chunk travel below this: radar depth ratio / shared baselines
    max_range: float = 35.0            # camera points up to this depth [m]
    stride: int = 2                    # pixel stride at the model resolution
    voxel: float = 0.1
    ground_refine: bool = True
    gr_range: float = 20.0
    gr_dist: float = 1.0
    gr_max_deg: float = 0.5
    gr_max_dz: float = 0.1
    min_views: int = 2                 # keep points seen from >= this many keyframes (0.3 m voxels)
    mask_people: bool = True           # the detector's worker masks (dynamic, and VGGT depth on people is 3x worse)
    mask_dilate: int = 9               # [px, full resolution]
    kf_still_max: int = 2              # standstill (kf_max_dt) keyframes in a row at most: a chunk of identical views
                                       # has no baseline, its poses are noise and the chain breaks (140718 start: 35 %)
    flush_s: float = 12.0              # standing still this long with >= min_chunk views waiting that travelled
    min_chunk: int = 10                # >= min_travel: run them as a short chunk (shorter chunks chain worse)
    max_points: int = 3_000_000        # map budget (oldest chunks dropped beyond it)
    sync: bool = False                 # run chunks inline (offline evaluation)


@dataclass
class Keyframe:
    t: float
    rgb: np.ndarray                    # (h, w, 3) uint8, model resolution
    pmask: np.ndarray | None           # (h, w) bool: people
    radar: np.ndarray                  # (N, 3) static radar points, level frame
    T_W_lvl: np.ndarray                # world <- level frame at this keyframe
    c_W: np.ndarray                    # camera centre in the world


@dataclass
class ReconSnapshot:
    """Map published after each chunk (new arrays each time: the GUI keeps references across cached frames)."""
    t: float
    xyz: np.ndarray                    # (N, 3) float32, world frame
    rgb: np.ndarray                    # (N, 3) uint8
    kf_xyz: np.ndarray                 # (K, 3) refined camera centres of the keyframes mapped so far
    kf_t: np.ndarray
    n_chunks: int


@dataclass
class ReconState:
    t: float
    n_keyframes: int
    n_mapped: int                      # keyframes whose points are in the map
    n_chunks: int
    queued: int                        # chunks waiting / running
    points: int
    last_ms: float
    note: str                          # last chunk: scale source, travel, refine stats
    error: str | None
    snapshot: ReconSnapshot | None


def _pose_at(world, dt: float):
    """World pose dt seconds later (constant yaw rate; with the swing-axis model the sensor turns about the axis)."""
    from dataclasses import replace

    from rcmot.world import rot2

    th = world.theta + world.w * dt
    org = world.axis + rot2(th) @ world.lever if world.axis is not None and world.lever is not None else world.origin
    return replace(world, theta=th, origin=org)


class _GroundRefiner:
    """Sequential scan-to-map refinement of each keyframe's roll, pitch and height (3 DoF, point-to-plane against the
    map of the earlier keyframes); the correction of keyframe j is carried to all later keyframes (chain)."""

    def __init__(self, p: ReconParams):
        self.p = p
        self.corr = np.eye(4)
        self.M = np.zeros((0, 3))
        self.tree = self.normals = None
        self.tilt: list[float] = []
        self.dz: list[float] = []

    def step(self, j: int, local: np.ndarray, T_pre: np.ndarray) -> np.ndarray:
        p = self.p
        T = self.corr @ T_pre
        P0 = apply(T, local)
        if self.tree is not None and len(P0) > 200:
            c0 = T[:3, 3]
            for _ in range(4):
                P = apply(T, local)
                dist, idx = self.tree.query(P, distance_upper_bound=p.gr_dist)
                ok = np.isfinite(dist)
                if ok.sum() < 100:
                    break
                q, nn = self.M[idx[ok]], self.normals[idx[ok]]
                Pk = P[ok]
                r = np.einsum("ij,ij->i", nn, Pk - q)
                lever = Pk - c0
                J = np.stack([np.einsum("ij,ij->i", nn, np.cross(np.repeat([[1.0, 0, 0]], len(Pk), 0), lever)),
                              np.einsum("ij,ij->i", nn, np.cross(np.repeat([[0, 1.0, 0]], len(Pk), 0), lever)),
                              nn[:, 2]], 1)
                w = np.minimum(1.0, 0.15 / np.maximum(np.abs(r), 1e-6))
                A_ = J * w[:, None]
                x = np.linalg.solve(A_.T @ J + 1e-6 * np.eye(3), -(A_.T @ r))
                x[:2] = np.clip(x[:2], -np.radians(1.5), np.radians(1.5))
                R = Rotation.from_rotvec(np.r_[x[:2], 0.0]).as_matrix()
                T = se3(R, c0 - R @ c0 + [0, 0, x[2]]) @ T
            T0 = self.corr @ T_pre
            dR = Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T)
            if dR.magnitude() > np.radians(p.gr_max_deg):
                dR = Rotation.from_rotvec(dR.as_rotvec() * np.radians(p.gr_max_deg) / dR.magnitude())
                R = dR.as_matrix()
                T = se3(R, c0 - R @ c0 + [0, 0, T[2, 3] - T0[2, 3]]) @ T0
            if abs(T[2, 3] - T0[2, 3]) > p.gr_max_dz:
                T[2, 3] = T0[2, 3] + np.sign(T[2, 3] - T0[2, 3]) * p.gr_max_dz
            self.dz.append(float(T[2, 3] - T0[2, 3]))
            self.tilt.append(float(np.degrees(Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T).magnitude())))
        self.corr = T @ np.linalg.inv(T_pre)
        self.M = np.concatenate([self.M, apply(T, local)])
        if j % 2 == 0:
            self.M = voxel_first(self.M[np.linalg.norm(self.M[:, :2] - T[:2, 3], axis=1) < 60.0], 0.2)[0]
            self.tree = cKDTree(self.M)
            _, nb = self.tree.query(self.M, k=min(12, len(self.M)))
            Q = self.M[nb] - self.M[nb].mean(1, keepdims=True)
            self.normals = np.linalg.svd(Q, full_matrices=False)[2][:, 2, :]
            self.normals *= np.sign(self.normals[:, 2:3] + 1e-9)
        return T


class _Session:
    """All per-segment state; reset() replaces it, a chunk running on the old one finishes harmlessly."""

    def __init__(self, p: ReconParams):
        self.kf: list[Keyframe] = []
        self.next_start = 0                 # first keyframe of the next regular chunk
        self.flushed_at = -1
        self.t_motion = -1e9                # time of the last keyframe taken for travel / turn (flush timer)
        self.n_still = 0                    # standstill keyframes in a row
        self.emitted = 0                    # keyframes [0, emitted) are in the map
        self.n_chunks = 0
        self.queued = 0
        self.prev: dict | None = None       # previous chunk (depth, conf, scale, poses) for the join
        self.gr = _GroundRefiner(p)
        self.scales: list[float] = []
        self.log: list[dict] = []                       # per chunk: frames, scale source, travel, refine stats
        self.pairs = np.zeros((0, 2), np.int64)         # (0.3 m voxel key, keyframe) pairs of all points so far
        self.pending: list = []                         # single-view points waiting for a second keyframe
        self.map_keys = np.zeros(0, np.int64)           # 0.1 m voxels already in the map
        self.P = np.zeros((0, 3), np.float32)
        self.C = np.zeros((0, 3), np.uint8)
        self.kf_T: dict[int, np.ndarray] = {}
        self.snapshot: ReconSnapshot | None = None
        self.last_ms = 0.0
        self.note = ""
        self.error: str | None = None


class DriveRecon:
    def __init__(self, params: ReconParams, calib, root: Path):
        self.p, self.calib, self.root = params, calib, Path(root)
        self.engine = None                  # rcmot.work.depth_engine.VGGTOmegaEngine (model loaded on the first chunk)
        self.rays = self.above = None
        self.shape: tuple[int, int] | None = None
        self.T_lvl_cam = np.linalg.inv(se3(calib.R_cl, calib.t_cl))
        self.cc = calib.camera_center()
        self._q: deque = deque()
        self._wake = threading.Event()
        self._lock = threading.Lock()
        self.s = _Session(params)
        self.busy = False
        if not params.sync:
            threading.Thread(target=self._loop, daemon=True, name="drive-recon").start()

    # ------------------------------------------------------------------ frame side (pipeline thread)
    def reset(self):
        with self._lock:
            self._q.clear()
            self.s = _Session(self.p)

    def update(self, t: float, image: np.ndarray, world, radar_static: np.ndarray, dets2d=None, img_dt: float = 0.0) -> ReconState:
        """t: the image's time; world: the scan's pose (rcmot.world.WorldState), img_dt = t_image - t_scan."""
        p, s = self.p, self.s
        if world is not None:
            if abs(img_dt) > 1e-3:                                       # the image's own pose (turning: 1 deg / 30 ms)
                world = _pose_at(world, img_dt)
            new = not s.kf
            if not new:
                k = s.kf[-1]
                dt = t - k.t
                th_prev = np.arctan2(k.T_W_lvl[1, 0], k.T_W_lvl[0, 0])
                dth = (world.theta - th_prev + np.pi) % (2 * np.pi) - np.pi
                moved = np.linalg.norm(world.origin - k.T_W_lvl[:2, 3]) >= p.kf_dist or abs(dth) >= np.radians(p.kf_deg)
                new = dt >= p.kf_min_dt and (moved or (dt >= p.kf_max_dt and s.n_still < p.kf_still_max))
            if new:
                s.kf.append(self._keyframe(t, image, world, radar_static, dets2d))
                if len(s.kf) == 1 or moved:
                    s.t_motion, s.n_still = t, 0
                else:
                    s.n_still += 1
            while len(s.kf) >= s.next_start + p.chunk:
                self._enqueue(s, s.next_start, s.next_start + p.chunk, False)
                s.next_start += p.chunk - p.overlap
            waiting = len(s.kf) - s.next_start
            if waiting >= p.min_chunk and t - s.t_motion >= p.flush_s and s.flushed_at != len(s.kf) \
                    and self._travel(s.kf[s.next_start:]) >= p.min_travel:
                self._enqueue(s, s.next_start, len(s.kf), True)        # standing still: map what is waiting
                s.flushed_at = len(s.kf)
                s.next_start = len(s.kf) - p.overlap
        return self.state(t)

    @staticmethod
    def _travel(kfs) -> float:
        c = np.array([k.c_W[:2] for k in kfs])
        return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum()) if len(c) > 1 else 0.0

    def flush(self, min_views: int = 2):
        """Run the waiting keyframes as a last (short) chunk (end of a bag / offline evaluation)."""
        s = self.s
        if len(s.kf) - s.next_start >= max(min_views, 1) and s.flushed_at != len(s.kf):
            self._enqueue(s, s.next_start, len(s.kf), True)
            s.flushed_at = len(s.kf)
            s.next_start = len(s.kf) - self.p.overlap

    def state(self, t: float) -> ReconState:
        s = self.s
        return ReconState(t, len(s.kf), s.emitted, s.n_chunks, s.queued, len(s.P), s.last_ms, s.note, s.error, s.snapshot)

    def _keyframe(self, t, image, world, radar_static, dets2d) -> Keyframe:
        p = self.p
        if self.shape is None:
            self.shape = model_shape(image.shape[0], image.shape[1], p.res)
        h, w = self.shape
        rgb = cv2.cvtColor(cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        pm = None
        if p.mask_people and dets2d:
            m = np.zeros(image.shape[:2], np.uint8)
            n = 0
            for d in dets2d:
                if d.cls != "worker":
                    continue
                x1, y1 = max(int(d.box[0]), 0), max(int(d.box[1]), 0)
                if d.mask is not None:
                    hh, ww = d.mask.shape
                    m[y1:y1 + hh, x1:x1 + ww] |= d.mask.astype(np.uint8)
                else:
                    m[y1:int(d.box[3]), x1:int(d.box[2])] = 1
                n += 1
            if n:
                k = np.ones((p.mask_dilate, p.mask_dilate), np.uint8)
                pm = cv2.resize(cv2.dilate(m, k), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        c, s_ = np.cos(world.theta), np.sin(world.theta)
        T_W_lvl = se3(np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]), np.r_[world.origin, 0.0])
        rad = np.asarray(radar_static, np.float32).reshape(-1, 3) if radar_static is not None else np.zeros((0, 3), np.float32)
        return Keyframe(t, rgb, pm, rad, T_W_lvl, apply(T_W_lvl, self.cc[None])[0])

    def _enqueue(self, s: _Session, i0: int, i1: int, final: bool):
        s.queued += 1
        if self.p.sync:
            self._run(s, i0, i1, final)
            return
        with self._lock:
            self._q.append((s, i0, i1, final))
        self._wake.set()

    # ------------------------------------------------------------------ chunk side (background thread)
    def _loop(self):
        while True:
            self._wake.wait()
            with self._lock:
                job = self._q.popleft() if self._q else None
                if not self._q:
                    self._wake.clear()
            if job is None:
                continue
            s = job[0]
            if s is not self.s:                                   # reset meanwhile
                continue
            self.busy = True
            try:
                self._run(*job)
            except Exception as e:                                # noqa: BLE001  the GUI shows the error, tracking goes on
                s.error = repr(e)
                s.queued = max(s.queued - 1, 0)
                traceback.print_exc()
            self.busy = False

    def _model(self):
        if self.engine is None:
            from rcmot.work.depth_engine import shared_engine

            self.engine = shared_engine(self.p.vggt_ckpt, self.root, sync=True, res=self.p.res)
        if self.engine.model is None:
            with self.engine.lock:
                self.engine._load()
        return self.engine

    def _infer(self, kfs: list[Keyframe]) -> dict:
        import torch

        eng = self._model()
        from vggt_omega.utils.pose_enc import encoding_to_camera

        X = torch.stack([torch.from_numpy(k.rgb).permute(2, 0, 1).float().div_(255.0) for k in kfs]).to(eng.device)
        tic = time.perf_counter()
        with eng.lock, torch.inference_mode():
            out = eng.model(X)
        h, w = X.shape[-2:]
        E, Kp = encoding_to_camera(out["pose_enc"], (h, w))
        ck = dict(depth=out["depth"][0, ..., 0].float().cpu().numpy(), conf=out["depth_conf"][0].float().cpu().numpy(),
                  E=E[0].float().cpu().numpy(), fx=float(Kp[0, :, 0, 0].mean()), ms=(time.perf_counter() - tic) * 1e3)
        del X, out
        torch.cuda.empty_cache()
        return ck

    def _conf_mask(self, cf, d, lo, hi):
        return (cf >= max(np.quantile(cf, self.p.conf_q), self.p.conf_min)) & (d > lo) & (d < hi)

    def _radar_scale(self, ck, kfs) -> tuple[float, int]:
        """Median (radar camera depth / VGGT depth) at the keyframes' static radar points."""
        cal = self.calib
        W, H = cal.image_size
        h, w = ck["depth"].shape[1:]
        ratios = []
        for s, k in enumerate(kfs):
            if not len(k.radar):
                continue
            pc = k.radar @ cal.R_cl.T + cal.t_cl
            pc = pc[(pc[:, 2] > 2.0) & (pc[:, 2] < 40.0)]
            if not len(pc):
                continue
            uv = cv2.projectPoints(pc.astype(np.float64), np.zeros(3), np.zeros(3), cal.K, cal.D)[0].reshape(-1, 2)
            x = np.round((uv[:, 0] + 0.5) * w / W - 0.5).astype(int)
            y = np.round((uv[:, 1] + 0.5) * h / H - 0.5).astype(int)
            m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            d, cf = ck["depth"][s], ck["conf"][s]
            good = m.copy()
            good[m] = cf[y[m], x[m]] >= np.quantile(cf, 0.3)
            ratios.append(pc[good, 2] / np.maximum(d[y[good], x[good]], 1e-3))
        r = np.concatenate(ratios) if ratios else np.zeros(0)
        return (float(np.median(r)) if len(r) >= 15 else float("nan")), len(r)

    def _overlap_points(self, prev, sa, ck, sb, E_b, T_W_cam, sb_scale):
        """Shared frame seen in both chunks: world points of the previous chunk (P), this-chunk points (Q), weights."""
        p = self.p
        da, db = prev["depth"][sa] * prev["sv"], ck["depth"][sb] * sb_scale
        ca, cb = prev["conf"][sa], ck["conf"][sb]
        m = self._conf_mask(ca, da, 1.0, 25.0) & (cb >= max(np.quantile(cb, p.conf_q), p.conf_min)) & (np.abs(db - da) < 0.2 * da)
        m[int(round((1.0 - p.crop_bottom) * m.shape[0])):] = False
        sub = np.zeros_like(m)
        sub[::4, ::4] = True
        m &= sub
        P = apply(T_W_cam, self.rays[m] * da[m, None])
        Q = apply(np.linalg.inv(E_b), self.rays[m] * db[m, None])
        return P, Q, 1.0 / da[m] ** 2

    def _points(self, ck, s, scale, T, k: Keyframe):
        p = self.p
        d = ck["depth"][s] * scale
        gy, gx = np.gradient(np.log(np.maximum(d, 1e-3)))
        m = self._conf_mask(ck["conf"][s], d, 1.0, p.max_range) & (np.hypot(gx, gy) < 0.08)
        m[int(round((1.0 - p.crop_bottom) * m.shape[0])):] = False
        m &= ~sky_mask(k.rgb, self.above)
        if k.pmask is not None:
            m &= ~k.pmask
        sub = np.zeros_like(m)
        sub[::p.stride, ::p.stride] = True
        m &= sub
        return apply(T, self.rays[m] * d[m, None]).astype(np.float32), k.rgb[m]

    def _near_points(self, ck, s, scale, k: Keyframe):
        """Camera-frame points for the ground refinement (near, confident, no sky / people)."""
        p = self.p
        d = ck["depth"][s] * scale
        m = self._conf_mask(ck["conf"][s], d, 1.5, p.gr_range) & ~sky_mask(k.rgb, self.above)
        if k.pmask is not None:
            m &= ~k.pmask
        m[int(round((1.0 - p.crop_bottom) * m.shape[0])):] = False
        sub = np.zeros_like(m)
        sub[::4, ::4] = True
        m &= sub
        return self.rays[m] * d[m, None]

    def _run(self, s: _Session, i0: int, i1: int, final: bool):
        p = self.p
        kfs = s.kf[i0:i1]
        ck = self._infer(kfs)
        if s is not self.s:
            return
        h, w = ck["depth"].shape[1:]
        if self.rays is None or self.rays.shape[:2] != (h, w):
            self.rays = pixel_rays(self.calib, h, w)
            self.above = (self.rays @ self.calib.R_cl)[..., 2] > 0       # ray points up in the level frame: sky candidates
        # ---- scale: Doppler travel while driving, radar depth ratio otherwise, shared baselines for short travel
        C = np.array([-E[:, :3].T @ E[:, 3] for E in ck["E"]])
        tv = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
        cw = np.array([k.c_W for k in kfs])
        travel = float(np.linalg.norm(np.diff(cw[:, :2], axis=0), axis=1).sum())
        s_t = float(travel / tv) if tv > 1e-6 else float("nan")
        s_d, n_r = self._radar_scale(ck, kfs)
        use_t = travel >= p.min_travel and np.isfinite(s_t)
        sv, how = (s_t, "travel") if use_t else (s_d, "depth")
        prev = s.prev
        if prev is not None and travel < p.min_travel:
            n_sh = prev["i1"] - i0
            Cp = prev["C"][len(prev["C"]) - n_sh:] * prev["sv"]
            Cc = C[:n_sh]
            dp, dc = np.linalg.norm(np.diff(Cp, axis=0), axis=1), np.linalg.norm(np.diff(Cc, axis=0), axis=1)
            if dc.sum() > 1e-3:
                sv, how = float(dp.sum() / dc.sum()), "shared"
        if not np.isfinite(sv):
            sv, how = (float(np.median(s.scales)), "median") if s.scales else (1.0, "none")
        s.scales.append(float(sv))
        # ---- join to the previous chunk (hybrid: cameras -> heading / xy, shared depth points -> tilt / height)
        Es = [se3(E[:, :3], E[:, 3] * sv) for E in ck["E"]]
        if prev is None:
            T_W_C = kfs[0].T_W_lvl @ self.T_lvl_cam @ Es[0]
        else:
            shared = range(i0, prev["i1"])
            T_W_C = mean_pose([prev["poses"][j] @ Es[j - i0] for j in shared])
            PQW = [self._overlap_points(prev, j - prev["i0"], ck, j - i0, Es[j - i0], prev["poses"][j], sv) for j in shared]
            P_, Q_, w_ = (np.concatenate(x) for x in zip(*PQW))
            if len(P_) >= 200:
                T_pts = align_rigid(P_, Q_, w_, T_W_C)
                e3 = np.array([0.0, 0.0, 1.0])
                Rt = rot_between(T_W_C[:3, :3] @ (T_pts[:3, :3].T @ e3), e3)
                q0 = (T_W_C @ np.linalg.inv(Es[0]))[:3, 3]
                T_W_C = se3(Rt, q0 - Rt @ q0) @ T_W_C
                T_W_C[2, 3] += (T_pts @ np.linalg.inv(Es[0]))[2, 3] - q0[2]
        poses = {i0 + k: T_W_C @ np.linalg.inv(E) for k, E in enumerate(Es)}
        s.prev = dict(i0=i0, i1=i1, depth=ck["depth"], conf=ck["conf"], sv=sv, poses=poses, C=C)
        # ---- keyframes this chunk owns: ground refinement, points
        half = p.overlap // 2
        lo = max(i0 if s.n_chunks == 0 else i0 + half, s.emitted)
        hi = i1 if final else i1 - (p.overlap - half)
        P, Cc_, K = [], [], []
        for j in range(lo, hi):
            k = kfs[j - i0]
            T = poses[j]
            if p.ground_refine:
                T = s.gr.step(j, self._near_points(ck, j - i0, sv, k), T)
            s.kf_T[j] = T
            pts, col = self._points(ck, j - i0, sv, T, k)
            P.append(pts), Cc_.append(col), K.append(np.full(len(pts), j, np.int64))
        if hi > lo:
            s.emitted = hi
            self._accumulate(s, np.concatenate(P), np.concatenate(Cc_), np.concatenate(K), kfs[-1].t)
        s.n_chunks += 1
        s.queued = max(s.queued - 1, 0)
        s.last_ms = ck["ms"]
        tilt = s.gr.tilt[-max(hi - lo, 1):] if s.gr.tilt else []
        s.log.append(dict(i0=i0, i1=i1, final=final, lo=lo, hi=hi, scale=how, sv=round(float(sv), 3), travel_m=round(travel, 1),
                          radar_pts=n_r, tilt_p50=round(float(np.median(tilt)), 2) if len(tilt) else None,
                          tilt_max=round(float(np.max(tilt)), 2) if len(tilt) else None, ms=round(ck["ms"])))
        s.note = (f"묶음 {s.n_chunks}: {i1 - i0}장, {ck['ms'] / 1e3:.1f} s, 스케일 {how} {sv:.3f} (이동 {travel:.1f} m, 레이더 {n_r}점)"
                  + (f", 지면맞춤 기울기 중앙 {np.median(tilt):.2f}°" if len(tilt) else ""))

    def _accumulate(self, s: _Session, P, C, K, t):
        """Multi-view check (>= min_views keyframes per 0.3 m voxel, earlier single-view points re-checked), 0.1 m
        voxels, de-duplication against the map, snapshot."""
        p = self.p
        key3 = voxel_keys(P, 0.3)
        pairs = np.unique(np.c_[key3, K], axis=0)
        s.pairs = np.concatenate([s.pairs, pairs])
        if p.min_views > 1:
            uk, cnt = np.unique(s.pairs[:, 0], return_counts=True)
            good = uk[cnt >= p.min_views]
            s.pending.append((P, C, key3))
            keep_P, keep_C, rest = [], [], []
            for Pp, Cp, kp in s.pending:
                ok = np.isin(kp, good, assume_unique=False)
                keep_P.append(Pp[ok]), keep_C.append(Cp[ok])
                if (~ok).sum():
                    rest.append((Pp[~ok], Cp[~ok], kp[~ok]))
            s.pending = rest
            P, C = np.concatenate(keep_P), np.concatenate(keep_C)
        if not len(P):
            return
        P, C = voxel_first(P, p.voxel, C)
        keys = voxel_keys(P, p.voxel)
        new = ~np.isin(keys, s.map_keys)
        P, C, keys = P[new], C[new], keys[new]
        s.map_keys = np.sort(np.concatenate([s.map_keys, keys]))
        s.P, s.C = np.concatenate([s.P, P]), np.concatenate([s.C, C])
        if len(s.P) > p.max_points:
            s.P, s.C = s.P[-p.max_points:], s.C[-p.max_points:]
        ks = sorted(s.kf_T)
        s.snapshot = ReconSnapshot(t, s.P, s.C, np.array([s.kf_T[j][:3, 3] for j in ks]).reshape(-1, 3),
                                   np.array([s.kf[j].t for j in ks]), s.n_chunks + 1)
