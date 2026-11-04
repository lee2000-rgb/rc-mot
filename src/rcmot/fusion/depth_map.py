"""Per-frame metric depth map: monocular relative disparity (Depth Anything V2) scaled by the static radar points
of the scene, after Xie et al., "Robust 3D MOT via 4D mmWave Radar-Camera Fusion and Disparity-Domain Depth
Recovery", Sensors 26(7):2096, 2026, §3.3 (pres/framework_deck/ref/metric_depth_study.md §1).

  1. D_net = DA-V2 (small, fp16) on the image resized to `input_hw`, raw output (relative disparity, larger = nearer)
     resized back to the image; no normalisation.
  2. Static radar points (ScanStats static_inlier & keep) of the last `n_frames` scans, moved into the current level
     frame with the WorldPose (p_level = R(-theta)(p_world - origin)), projected with the rig calib, D_net sampled
     bilinearly at the pixel, reference depth = camera Z_c.
  3. Disparity-domain model 1/Z = s D_net + t: LSQ -> residual threshold tau = median|r| + k MAD|r| (k = 2.5)
     -> 2-point RANSAC with the physical box s in [0.001, 0.1], |t| <= 0.5 -> LSQ on the inliers,
     R = sigma_hat^2 (X^T X)^-1 with sigma_hat = 1.4826 MAD of the inlier residuals.
  4. (s, t) Kalman filter: F = I, Q = diag(q_s^2, q_t^2), measurement (s*, t*) with R, NIS gate chi2(2, 0.99) = 9.21
     (gated-out measurements only predict).
  5. Z(u, v) = 1 / (|s D_net + t| + eps); range_for_box(): median Z over the bottom third of the box (paper) or over
     the instance mask, back-projected through the region centre pixel -> horizontal range from the camera (the
     range convention of radar_camera.fuse) with a sigma from the fit's inlier relative error and the region spread.

torch only (no TensorRT); the network code is the official DA-V2 package vendored in third_party/depth_anything_v2
(Apache-2.0), weights models/depth_anything_v2_vits.pth (24.8 M params, 99 MB). Online use (docs/33 wave 3): the
Pipeline's depth stage, rcmot.fusion.depth_stage (every N-th frame, per-box range for fuse()).
"""
from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from rcmot.utils.calib import RigCalib

ROOT = Path(__file__).resolve().parents[3]
MAD_K = 1.4826
_MEAN, _STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
_ENCODERS = {"vits": dict(encoder="vits", features=64, out_channels=[48, 96, 192, 384]),
             "vitb": dict(encoder="vitb", features=128, out_channels=[96, 192, 384, 768]),
             "vitl": dict(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])}


@dataclass
class DepthMapParams:
    weights: str = "models/depth_anything_v2_vits.pth"
    encoder: str = "vits"
    input_hw: tuple = (518, 924)      # network input (multiples of 14); 518x924 keeps the 1280x720 aspect, 518x518 = paper
    fp16: bool = True
    device: str = "cuda"
    n_frames: int = 3                 # static-point accumulation (paper N = 3)
    min_points: int = 20              # projected static points needed for a fit
    min_depth: float = 1.0            # [m] Z_c of a radar point to take part
    mad_k: float = 2.5                # tau = median|r| + k MAD|r|
    ransac_iters: int = 300
    ransac_vectorized: bool = False   # all hypotheses at once (numpy, ~10x faster; other random draws than the loop)
    s_range: tuple = (0.001, 0.1)     # physical box of the disparity scale (paper, ViT-B raw output scale)
    t_max: float = 0.5
    kalman: bool = True
    q_s: float = 0.0005               # process noise of s per frame (paper: unspecified)
    q_t: float = 0.002
    nis_gate: float = 9.21            # chi2(2, 0.99)
    eps: float = 1e-3
    sample: str = "bottom"            # "bottom": bottom third of the box (paper) | "mask": instance mask, else bottom
    valid_depth: tuple = (0.1, 200.0)
    sigma_rel_floor: float = 0.03     # relative depth sigma floor
    seed: int = 0


@dataclass
class DepthFit:
    s: float
    t: float
    n: int                            # points used
    n_in: int                         # RANSAC inliers
    tau: float
    sigma: float                      # robust residual sigma in disparity units
    R: np.ndarray                     # (2,2) measurement covariance of (s, t)
    rel_err: float                    # median |Z_fit - Z_ref| / Z_ref over the inliers


@dataclass
class DepthMapResult:
    stamp: float
    disparity: np.ndarray             # (H, W) raw network output at image resolution
    depth: np.ndarray | None          # (H, W) metric Z_c [m] (None: no fit yet)
    fit: DepthFit | None              # this frame's measurement
    s: float | None                   # filtered (s, t) used for `depth`
    t: float | None
    nis: float | None
    updated: bool                     # KF took this frame's measurement
    n_static: int                     # static points offered (after accumulation)
    ms: dict = field(default_factory=dict)


def bilinear(D: np.ndarray, uv: np.ndarray) -> np.ndarray:
    h, w = D.shape
    u, v = np.clip(uv[:, 0], 0, w - 1.001), np.clip(uv[:, 1], 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - u0, v - v0
    return (D[v0, u0] * (1 - fu) * (1 - fv) + D[v0, u0 + 1] * fu * (1 - fv)
            + D[v0 + 1, u0] * (1 - fu) * fv + D[v0 + 1, u0 + 1] * fu * fv)


def fit_disparity(dn: np.ndarray, z_ref: np.ndarray, p: DepthMapParams, rng: np.random.Generator) -> DepthFit | None:
    """1/z_ref = s dn + t: global LSQ -> MAD threshold -> 2-point RANSAC -> inlier LSQ (paper eqs. 22-25)."""
    y = 1.0 / z_ref
    X = np.c_[dn, np.ones_like(dn)]
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    r = np.abs(y - X @ beta0)
    tau = float(np.median(r) + p.mad_k * np.median(np.abs(r - np.median(r))))
    tau = max(tau, 1e-6)
    n = len(dn)
    best_in, best_cnt = None, 0
    s_lo, s_hi = p.s_range
    if p.ransac_vectorized:
        ij = rng.integers(0, n, size=(p.ransac_iters, 2))
        ij = ij[ij[:, 0] != ij[:, 1]]
        dd = dn[ij[:, 0]] - dn[ij[:, 1]]
        ij, dd = ij[np.abs(dd) >= 1e-6], dd[np.abs(dd) >= 1e-6]
        s_h = (y[ij[:, 0]] - y[ij[:, 1]]) / dd
        t_h = y[ij[:, 0]] - s_h * dn[ij[:, 0]]
        ok = (s_h >= s_lo) & (s_h <= s_hi) & (np.abs(t_h) <= p.t_max)
        s_h, t_h = s_h[ok], t_h[ok]
        if len(s_h):
            inl = np.abs(y[None, :] - (s_h[:, None] * dn[None, :] + t_h[:, None])) <= tau
            cnt = inl.sum(1)
            j = int(np.argmax(cnt))
            if cnt[j] > 0:
                best_in, best_cnt = inl[j], int(cnt[j])
    for _ in range(0 if p.ransac_vectorized else p.ransac_iters):
        i, j = rng.choice(n, 2, replace=False)
        if abs(dn[i] - dn[j]) < 1e-6:
            continue
        s = (y[i] - y[j]) / (dn[i] - dn[j])
        t = y[i] - s * dn[i]
        if not (s_lo <= s <= s_hi and abs(t) <= p.t_max):
            continue
        inl = np.abs(y - (s * dn + t)) <= tau
        c = int(inl.sum())
        if c > best_cnt:
            best_in, best_cnt = inl, c
    if best_in is None or best_cnt < 4:
        return None
    Xi, yi = X[best_in], y[best_in]
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]
    res = yi - Xi @ beta
    sig = MAD_K * float(np.median(np.abs(res - np.median(res))))
    sig = max(sig, 1e-9)
    R = sig ** 2 * np.linalg.inv(Xi.T @ Xi)
    z_fit = 1.0 / (np.abs(Xi @ beta) + p.eps)
    rel = float(np.median(np.abs(z_fit - z_ref[best_in]) / z_ref[best_in]))
    return DepthFit(float(beta[0]), float(beta[1]), n, best_cnt, tau, sig, R, rel)


class ScaleKalman:
    """(s, t) random-walk Kalman filter with a NIS gate (paper eqs. 26-35)."""

    def __init__(self, q_s: float, q_t: float, gate: float):
        self.Q, self.gate = np.diag([q_s ** 2, q_t ** 2]), gate
        self.x, self.P = None, None

    def step(self, z: np.ndarray, R: np.ndarray) -> tuple[float, bool]:
        """Returns (NIS, accepted)."""
        if self.x is None:
            self.x, self.P = z.astype(float).copy(), R.copy()
            return 0.0, True
        self.P = self.P + self.Q
        y = z - self.x
        S = self.P + R
        nis = float(y @ np.linalg.solve(S, y))
        if nis > self.gate:
            return nis, False
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K) @ self.P
        return nis, True

    def predict_only(self):
        if self.P is not None:
            self.P = self.P + self.Q


class MetricDepth:
    def __init__(self, calib: RigCalib, params: DepthMapParams | None = None, root: Path = ROOT):
        self.calib, self.p, self.root = calib, params or DepthMapParams(), Path(root)
        self.model = None
        self.rng = np.random.default_rng(self.p.seed)
        self.kf = ScaleKalman(self.p.q_s, self.p.q_t, self.p.nis_gate)
        self.hist: deque = deque(maxlen=max(self.p.n_frames, 1))     # (xyz in world or level frame, had_world)
        self.last: DepthMapResult | None = None
        self.C = calib.camera_center()

    # ---------------------------------------------------------------- network
    def _load(self):
        import torch
        sys.path.insert(0, str(self.root / "third_party" / "depth_anything_v2"))
        from depth_anything_v2.dpt import DepthAnythingV2

        m = DepthAnythingV2(**_ENCODERS[self.p.encoder])
        sd = torch.load(str(self.root / self.p.weights), map_location="cpu")
        m.load_state_dict(sd, strict=False)
        m = m.to(self.p.device).eval()
        if self.p.fp16 and self.p.device != "cpu":
            m = m.half()
        self.model, self.torch = m, torch

    def disparity(self, img: np.ndarray, ms: dict | None = None) -> np.ndarray:
        """Raw relative disparity (float32, image resolution)."""
        if self.model is None:
            self._load()
        torch = self.torch
        h_in, w_in = self.p.input_hw
        t0 = time.perf_counter()
        x = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (w_in, h_in), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
        x = torch.from_numpy((x - _MEAN) / _STD).permute(2, 0, 1)[None].to(self.p.device)
        if self.p.fp16 and self.p.device != "cpu":
            x = x.half()
        if ms is not None:
            torch.cuda.synchronize() if self.p.device != "cpu" else None
            ms["pre"] = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
        with torch.inference_mode():
            f = self.model.pretrained.get_intermediate_layers(x, self.model.intermediate_layer_idx[self.p.encoder], return_class_token=True)
            d = torch.relu(self.model.depth_head(f, h_in // 14, w_in // 14))
            if self.p.device != "cpu":
                torch.cuda.synchronize()
            if ms is not None:
                ms["net"] = (time.perf_counter() - t0) * 1e3
                t0 = time.perf_counter()
            d = d.float().cpu().numpy()[0, 0]
        out = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        if ms is not None:
            ms["post"] = (time.perf_counter() - t0) * 1e3
        return out

    # ---------------------------------------------------------------- static points
    def push_static(self, xyz_level: np.ndarray, world=None) -> None:
        """Keep this scan's static points (level frame) for the next n_frames scans; `world` (rcmot.world.WorldState)
        stores them in the world frame so ego motion is compensated when they are reused (paper eq. 21)."""
        xyz = np.asarray(xyz_level, float).reshape(-1, 3)
        if world is not None and len(xyz):
            xyz = np.c_[world.to_world(xyz), xyz[:, 2]]
        self.hist.append((xyz, world is not None))

    def ego_compensate(self, dp: np.ndarray, dyaw: float) -> None:
        """Express the stored level-frame static points in the new base frame after the base moved by dp (old frame)
        and turned by dyaw (as Track.ego_compensate); entries stored in the world frame are left as they are."""
        c, s = np.cos(dyaw), np.sin(dyaw)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        dp = np.asarray(dp, float).reshape(3)
        self.hist = deque([(xyz if in_world or not len(xyz) else (xyz - dp) @ R, in_world) for xyz, in_world in self.hist],
                          maxlen=self.hist.maxlen)

    def static_points(self, world=None) -> np.ndarray:
        """Accumulated static points in the current level frame."""
        out = []
        for xyz, in_world in self.hist:
            if in_world and world is not None and len(xyz):
                xy = (xyz[:, :2] - world.origin) @ world.R()
                xyz = np.c_[xy, xyz[:, 2]]
            out.append(xyz)
        return np.concatenate(out) if out else np.zeros((0, 3))

    # ---------------------------------------------------------------- fit
    def fit(self, disp: np.ndarray, xyz_level: np.ndarray) -> DepthFit | None:
        """Disparity-domain fit of the scale against radar points in the level frame (reference depth = Z_c)."""
        if len(xyz_level) < self.p.min_points:
            return None
        uv, front = self.calib.project(xyz_level)
        pc = xyz_level @ self.calib.R_cl.T + self.calib.t_cl
        h, w = disp.shape
        ok = front & np.isfinite(uv[:, 0]) & (uv[:, 0] >= 1) & (uv[:, 0] < w - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 2) & (pc[:, 2] > self.p.min_depth)
        if ok.sum() < self.p.min_points:
            return None
        dn, z = bilinear(disp, uv[ok]), pc[ok, 2]
        good = dn > 1e-3                                             # zero disparity = infinity: excluded
        if good.sum() < self.p.min_points:
            return None
        return fit_disparity(dn[good], z[good], self.p, self.rng)

    def update(self, img: np.ndarray, static_xyz_level: np.ndarray, world=None, stamp: float = 0.0,
               disparity: np.ndarray | None = None, fit_R: np.ndarray | None = None) -> DepthMapResult:
        """One frame: push the scan's static points, infer (or take `disparity`), fit, filter, build the depth map.
        fit_R (3x3): the accumulated points are rotated by it before the fit (scan -> image time on a turning base)."""
        ms = {}
        self.push_static(static_xyz_level, world)
        disp = self.disparity(img, ms) if disparity is None else disparity
        t0 = time.perf_counter()
        pts = self.static_points(world)
        fit = self.fit(disp, pts if fit_R is None else pts @ np.asarray(fit_R, float).T)
        ms["fit"] = (time.perf_counter() - t0) * 1e3
        t0 = time.perf_counter()
        nis, upd = None, False
        if fit is not None:
            if self.p.kalman:
                nis, upd = self.kf.step(np.array([fit.s, fit.t]), fit.R)
                s, t = float(self.kf.x[0]), float(self.kf.x[1])
            else:
                s, t, upd = fit.s, fit.t, True
                self.kf.x = np.array([s, t])
        else:
            self.kf.predict_only()
            s, t = (None, None) if self.kf.x is None else (float(self.kf.x[0]), float(self.kf.x[1]))
        depth = None if s is None else (1.0 / (np.abs(s * disp + t) + self.p.eps)).astype(np.float32)
        ms["map"] = (time.perf_counter() - t0) * 1e3
        self.last = DepthMapResult(stamp, disp, depth, fit, s, t, nis, upd, len(pts), ms)
        self._rel = fit.rel_err if fit is not None else getattr(self, "_rel", 0.05)
        return self.last

    # ---------------------------------------------------------------- box range
    def range_for_box(self, box, mask: np.ndarray | None = None, sample: str | None = None):
        """(horizontal range from the camera [m], sigma [m], info) of a 2D box from the last depth map, or None.
        sample "bottom": median Z over the bottom third of the box (paper eq. 37); "mask": median over the
        instance mask (cropped to the box, Det2D.mask convention), falling back to the bottom third."""
        res = self.last
        if res is None or res.depth is None:
            return None
        sample = sample or self.p.sample
        H, W = res.depth.shape
        x1, y1, x2, y2 = [float(v) for v in box]
        xa, xb = max(int(x1), 0), min(int(np.ceil(x2)), W)
        ya, yb = max(int(y1), 0), min(int(np.ceil(y2)), H)
        if xb <= xa or yb <= ya:
            return None
        region = None
        how = "bottom"
        if sample == "mask" and mask is not None:
            m = np.zeros((yb - ya, xb - xa), bool)
            oy, ox = ya - max(int(y1), 0), xa - max(int(x1), 0)
            sub = mask[oy:oy + m.shape[0], ox:ox + m.shape[1]]
            m[:sub.shape[0], :sub.shape[1]] = sub
            if m.sum() >= 4:
                region, how = m, "mask"
        if region is None:
            region = np.zeros((yb - ya, xb - xa), bool)
            region[max(int(round(2 * (yb - ya) / 3)), 0):, :] = True
        crop = res.depth[ya:yb, xa:xb]
        vals = crop[region]
        lo, hi = self.p.valid_depth
        v = vals[np.isfinite(vals) & (vals > lo) & (vals < hi)]
        if v.size == 0:
            return None
        d = float(np.median(v))
        spread = MAD_K * float(np.median(np.abs(v - d)))
        ys, xs = np.nonzero(region)
        uc, vc = xa + float(xs.mean()), ya + float(ys.mean())
        n = cv2.undistortPoints(np.array([[[uc, vc]]], float), self.calib.K, self.calib.D).reshape(2)
        p_c = d * np.array([n[0], n[1], 1.0])
        p_l = self.calib.R_cl.T @ (p_c - self.calib.t_cl)
        rho = float(np.linalg.norm(p_l[:2] - self.C[:2]))
        rel = max(getattr(self, "_rel", 0.05), self.p.sigma_rel_floor)
        sig = float(np.hypot(rel * rho, 0.5 * spread))
        return rho, sig, {"z": d, "spread": spread, "n_px": int(v.size), "how": how, "pos": p_l}
