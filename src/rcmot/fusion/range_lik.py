"""Per-box range likelihood (docs/37 §5, online version: docs/38).

The fusion's range of a person box is not one number but a distribution over the horizontal range r from the camera
centre, built ONLY from this frame's measurements of this box (no track input, docs/36 §12: track feedback inside the
fusion locks wrong ranges in):

    p(r | box) ∝ (1/r) · L_mono(r) · L_ground(r) · L_width(r) · L_dm(r) · L_radar(r)          on GRID = 1 .. 90 m, 0.05 m

Camera cues (residual e = ln r_cue - ln r) share one robust model, fitted on the 19 GT segments (docs/37 §5.2):

    p(e | r) = w_core · t_ν(e; μ, s) + w_floor / W + w_post · N(e; μ_o, s_o)
      μ = Σ β_j Bm_j,  s² = Σ (exp φ_j · Bs_j)²        (designs below)
      floor: a gross outlier (the cue says nothing; constant in r, so a wrong cue does not drag)
      post:  height prior only -- posture / partial occlusion makes the box short and the range look farther (μ_o > 0);
             weight grows with the box aspect w / h
      (w_core, w_floor, w_post) = softmax(0, a_floor, c0 + c1 (aspect - 0.4))

      cue        r_cue                         mean design   scale design            used when
      mono       1.8 m / (tan θ_top - tan θ_bot) 1            1, 1/h_px               h_px >= 12; aspect > 0.75 -> "mono_wide"
                                                                                      fit; box cut (image top / bottom edge
                                                                                      or radar occluder) -> one-sided
                                                                                      P(e_core <= e): a cut box is shorter
                                                                                      than the person and looks FARTHER, so
                                                                                      it only says "not farther than r_cue"
      ground     box-bottom ground range,       1, q          1, q, q|ω|, q|v|        bottom visible, no occluder
                 RangeReliability slope-corrected            (q = r / h_cam, ω yaw rate [rad/s], v ego speed [m/s])
      width      0.5 m / Δaz                    1             1, 1/w_px               no side cut, w_px >= 4
      dm         depth-map range                1             1, r                    depth stage on and box ok

Radar (PDA form, docs/37 §5.1):
    L_radar(r) = 1 + κ Σ_c (r_c / 20)^α ℓ_c^γ t_ν(r; r_c + μ_R, σ_c),   σ_c² = σ_R0² + (σ_R1 r_c)²
      "1"  none of the clusters is the person; ℓ_c layout likelihood ratio (cluster shape only, NOT the height-prior
      agreement -- that would count the height cue twice)

Summary (summarize): the posterior is split into modes at the minima of a 0.1 m-smoothed copy; the dominant mode (most
mass) gives r (its median), sig (its std); sig_full = std of the whole posterior; p_alt = mass farther than
max(1 m, 3 sig) from r; modes = every mode with mass >= 0.05 as (median, std, mass).

Parameters: JSON from research/range_lik/export_params.py (configs/range_lik/full.json, or a leave-one-segment-out fold
configs/range_lik/loso/<seg>.json for evaluation). The formulas are a line-by-line port of research/range_unc/model.py
(tests/test_range_lik.py checks equality on logged boxes).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.special import gammaln
from scipy.stats import t as student_t

GRID = np.arange(1.0, 90.0, 0.05)
LG = np.log(GRID)
DR = 0.05
LAYOUT = ["logn", "u_mean", "u_std", "u_dev", "v_mean", "v_std", "spread_rel", "rv_abs"]
H_CLASS = 1.8               # height-prior height [m] (FusionParams.class_height["worker"])
W_PERSON = 0.5              # width-cue nominal width [m] (its bias is fitted)
WIDE_ASPECT = 0.75          # aspect above this: the "mono_wide" fit
_KS = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
_KS = _KS / _KS.sum()       # smoothing kernel of summarize (σ 2 cells = 0.1 m)


def t_logpdf(z, nu):
    return gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi) - (nu + 1) / 2 * np.log1p(z * z / nu)


@dataclass
class BoxCues:
    """Everything the likelihood needs from one box (NaN / False = cue not available). Built by fuse()."""
    r_mono: float = np.nan          # 1.8 / (tan θ_top - tan θ_bot) at the box-centre column [m]
    h_px: float = np.nan
    w_px: float = np.nan
    aspect: float = np.nan          # w_px / h_px
    cut: bool = False               # box touches the image top / bottom, or a radar occluder hides the lower body
    r_g: float = np.nan             # ground-contact range, RangeReliability-corrected [m]
    g_ok: bool = False
    h_cam: float = 1.6              # camera height above the level ground [m]
    yaw: float = 0.0                # |yaw rate| [rad/s]
    speed: float = 0.0              # |ego velocity| [m/s]
    r_w: float = np.nan             # 0.5 / Δaz [m]
    w_ok: bool = False
    r_dm: float = np.nan            # depth-map range [m]
    dm_ok: bool = False
    clusters: list = field(default_factory=list)   # dict(r = camera-centre median range [m], f = layout features or
                                                   # None (unscored), idx = member point indices)

    @property
    def m_ok(self) -> bool:
        return bool(np.isfinite(self.r_mono) and self.h_px >= 12)


class RangeLikModel:
    def __init__(self, P: dict):
        self.P = P
        lr = P["lr"]
        self.lr = {k: (np.asarray(v, float) if isinstance(v, list) else float(v)) for k, v in lr.items()}
        self.radar = P["radar"]
        self.kappa, self.gamma, self.alpha = float(P["kappa"]), float(P["gamma"]), float(P.get("alpha", 0.0))
        # docs/41: BEV azimuth-wedge clusters for boxes without any in-box cluster (fusion.lik_wedge_fallback)
        w = P.get("wedge")
        self.wedge = None if w is None else dict(
            w, lr={k: (np.asarray(v, float) if isinstance(v, list) else float(v)) for k, v in w["lr"].items()})

    # ------------------------------------------------------------------ cue terms
    def _robust(self, th: dict, e: np.ndarray, Bm, Bs, Bc, one_sided: bool) -> np.ndarray:
        mu = sum(b * x for b, x in zip(th["beta"], Bm))
        s = np.sqrt(sum((np.exp(f) * x) ** 2 for f, x in zip(th["phi"], Bs))) + 1e-9
        z = (e - mu) / s
        core = student_t.logcdf(z, th["nu"]) if one_sided else t_logpdf(z, th["nu"]) - np.log(s)
        lg = [np.zeros_like(z), np.full_like(z, th["a_floor"])]
        terms = [core, np.full_like(z, -np.log(th["W"]))]
        if Bc is not None and "c" in th and not one_sided:
            lg.append(th["c"][0] * Bc[0] + th["c"][1] * Bc[1])
            terms.append(-0.5 * ((e - th["mu_o"]) / th["s_o"]) ** 2 - np.log(th["s_o"] * np.sqrt(2 * np.pi)))
        L = np.stack(lg)
        lw = L - np.logaddexp.reduce(L, axis=0)
        if one_sided:
            return np.logaddexp(lw[0] + core, lw[1] + np.log(0.5))
        return np.logaddexp.reduce(lw + np.stack(terms), axis=0)

    def camera_terms(self, c: BoxCues) -> dict:
        """Per-cue log-likelihood over GRID of the camera cues of one box."""
        out, one, P = {}, np.ones_like(GRID), self.P
        if c.m_ok:
            wide = c.aspect > WIDE_ASPECT
            name = "mono" if (c.cut or not wide) else "mono_wide"
            if name in P:
                e = np.log(c.r_mono) - LG
                out["mono"] = self._robust(P[name], e, [one], [one, one / c.h_px], [one, one * (c.aspect - 0.4)],
                                           one_sided=c.cut)
        if c.g_ok and "ground" in P:
            q = GRID / c.h_cam
            e = np.log(c.r_g) - LG
            out["ground"] = self._robust(P["ground"], e, [one, q], [one, q, q * c.yaw, q * c.speed], None, False)
        if c.w_ok and "width" in P:
            e = np.log(c.r_w) - LG
            out["width"] = self._robust(P["width"], e, [one], [one, one / c.w_px], None, False)
        if c.dm_ok and "dm" in P:
            e = np.log(c.r_dm) - LG
            out["dm"] = self._robust(P["dm"], e, [one], [one, GRID], None, False)
        return out

    def layout_l(self, f: dict | None) -> float:
        """Layout likelihood ratio target / clutter of one cluster."""
        lr = self.lr
        if f is None:
            return float(lr["unscored_l"])
        F = np.array([f.get(k) if f.get(k) is not None else lr["mu"][j] for j, k in enumerate(LAYOUT)], float)
        z = float(((F - lr["mu"]) / lr["sd"]) @ lr["coef"] + lr["b"])
        p = 1.0 / (1.0 + np.exp(-z))
        pr = lr["prior"]
        return (p / max(1 - p, 1e-6)) / (pr / (1 - pr))

    def radar_parts(self, clusters: list) -> list:
        """Per cluster the term (r_c/20)^α ℓ^γ t(r; r_c + μ_R, σ_c) over GRID (dense array) and its ℓ."""
        th = self.radar
        out = []
        for c in clusters:
            l = self.layout_l(c.get("f")) * c.get("bgf", 1.0) * c.get("gwf", 1.0)   # bgf: cell memory (docs/41 §10), gwf: below-ground weight
            s = float(np.sqrt(np.exp(2 * th["phi"][0]) + (np.exp(th["phi"][1]) * c["r"]) ** 2))
            lo, hi = np.searchsorted(GRID, [c["r"] - 8 * s - 1, c["r"] + 8 * s + 1])
            z = (c["r"] - GRID[lo:hi] - th["beta"][0]) / s
            S = np.zeros_like(GRID)
            S[lo:hi] = (c["r"] / 20.0) ** self.alpha * (l ** self.gamma) * np.exp(t_logpdf(z, th["nu"])) / s
            out.append((S, l))
        return out

    def wedge_l(self, f: dict) -> float:
        """Likelihood ratio person / clutter of one wedge cluster (docs/41 §5, features self.wedge["feats"])."""
        lr = self.wedge["lr"]
        F = np.array([f[k] for k in self.wedge["feats"]], float)
        z = float(((F - lr["mu"]) / lr["sd"]) @ lr["coef"] + lr["b"])
        p = 1.0 / (1.0 + np.exp(-z))
        pr = lr["prior"]
        return (p / max(1 - p, 1e-6)) / (pr / (1 - pr))

    def wedge_term(self, clusters: list) -> tuple[np.ndarray | None, list]:
        """Radar log term of wedge clusters [{r, f}]: log(1 + kappa_w sum_c (r_c/20)^alpha_w l_c^gamma_w t(r; r_c + mu_R,
        sigma_c)) with the same residual model as the in-box term; also the per-cluster (S, l) parts."""
        th, w = self.radar, self.wedge
        parts = []
        for c in clusters:
            l = self.wedge_l(c["f"]) * c.get("bgf", 1.0) * c.get("gwf", 1.0)
            s = float(np.sqrt(np.exp(2 * th["phi"][0]) + (np.exp(th["phi"][1]) * c["r"]) ** 2))
            lo, hi = np.searchsorted(GRID, [c["r"] - 8 * s - 1, c["r"] + 8 * s + 1])
            z = (c["r"] - GRID[lo:hi] - th["beta"][0]) / s
            S = np.zeros_like(GRID)
            S[lo:hi] = (c["r"] / 20.0) ** w["alpha"] * (l ** w["gamma"]) * np.exp(t_logpdf(z, th["nu"])) / s
            parts.append((S, l))
        if not parts:
            return None, parts
        return np.log1p(w["kappa"] * sum(S for S, _ in parts)), parts

    def radar_term(self, parts: list) -> np.ndarray | None:
        if not parts:
            return None
        return np.log1p(self.kappa * sum(S for S, _ in parts))


def posterior(log_terms: list) -> np.ndarray:
    """Normalised posterior density over GRID: prior 1/r times the given log-likelihood arrays."""
    tot = -LG + (sum(log_terms) if log_terms else 0.0)
    tot = tot - tot.max()
    p = np.exp(tot)
    p /= p.sum() * DR
    return p


def segments(p: np.ndarray) -> tuple[np.ndarray, list]:
    """Peaks of a 0.1 m-smoothed copy of p (>= 1 % of the highest) and the cut indices between them (the minima): mode
    j spans GRID[cuts[j]:cuts[j + 1]]."""
    ps = np.convolve(p, _KS, mode="same")
    pk = np.flatnonzero((ps[1:-1] >= ps[:-2]) & (ps[1:-1] > ps[2:]) & (ps[1:-1] >= 0.01 * ps.max())) + 1
    if len(pk) == 0:
        pk = np.array([int(np.argmax(ps))])
    cuts = [0]
    for a, b in zip(pk[:-1], pk[1:]):
        cuts.append(a + int(np.argmin(ps[a:b + 1])))
    cuts.append(len(p))
    return pk, cuts


def summarize(p: np.ndarray, alt_min_sep: float = 1.0, mode_min_mass: float = 0.05) -> dict:
    """Dominant-mode median / std, full std, alternative mass, number of modes, and the modes with mass >= mode_min_mass
    as (median, std, mass) sorted by mass (research/range_unc/model.py summarize + the mode list)."""
    w = p * DR
    pk, cuts = segments(p)
    tot = max(w.sum(), 1e-300)
    modes = []
    for j in range(len(cuts) - 1):
        lo, hi = cuts[j], cuts[j + 1]
        mj = w[lo:hi].sum()
        if mj <= 0:
            continue
        wm = w[lo:hi] / mj
        mean = float(wm @ GRID[lo:hi])
        sd = float(np.sqrt(max(wm @ (GRID[lo:hi] - mean) ** 2, 1e-6)))
        med = float(np.interp(0.5, np.cumsum(wm), GRID[lo:hi]))
        modes.append((med, sd, float(mj / tot), mean))
    j = int(np.argmax([m[2] for m in modes]))
    mu, sl, _, mean = modes[j]
    mf = float(w @ GRID / tot)
    sf = float(np.sqrt(max(w @ (GRID - mf) ** 2 / tot, 1e-6)))
    far = np.abs(GRID - mu) > max(alt_min_sep, 3 * sl)
    p_alt = float(w[far].sum() / tot)
    big = sorted([m[:3] for m in modes if m[2] >= mode_min_mass], key=lambda m: -m[2])
    masses = [float(w[cuts[i]:cuts[i + 1]].sum() / tot) for i in range(len(cuts) - 1)]
    return dict(r=mu, sig=sl, r_mean=mean, sig_full=sf, p_alt=p_alt, n_modes=len(pk), modes=big, cuts=list(map(int, cuts)),
                masses=masses, dom=j)


def mode_moments(L: np.ndarray, r: float) -> tuple[float, float] | None:
    """Mean and std of the (unnormalised) density L over its own mode (segments()) that contains range r: the
    measurement a track takes from the box once the product with its prediction has picked the mode -- the box's own
    likelihood, not the product, so the prediction is not counted twice."""
    _, cuts = segments(L)
    j = int(np.clip(np.searchsorted(GRID, r), 0, len(GRID) - 1))
    k = int(np.clip(np.searchsorted(cuts, j, side="right") - 1, 0, len(cuts) - 2))
    lo, hi = cuts[k], cuts[k + 1]
    wl = L[lo:hi] * DR
    s = wl.sum()
    if s <= 0:
        return None
    wl = wl / s
    m = float(wl @ GRID[lo:hi])
    return m, float(np.sqrt(max(wl @ (GRID[lo:hi] - m) ** 2, 0.05 ** 2)))


def peak_gauss(L: np.ndarray, r: float, drop: float = 0.5) -> tuple[float, float] | None:
    """Gaussian (Laplace) approximation of the likelihood peak a track takes: climb ln L from range r to the local
    maximum, fit a parabola to ln L over the contiguous cells within `drop` of that maximum (the +-1 sigma core of a
    Gaussian), return its vertex and sigma = sqrt(-1 / 2a). A flat top (one-sided cue only) gives the width of the
    plateau / sqrt(12). The tails (floors, other modes) are not in it: they matter for the association, which uses the
    whole likelihood; the Kalman update takes the core of the chosen peak."""
    lL = np.log(np.maximum(L, 1e-300))
    n = len(lL)
    j = int(np.clip(np.searchsorted(GRID, r), 0, n - 1))
    while True:
        k = j
        if j > 0 and lL[j - 1] > lL[k]:
            k = j - 1
        if j < n - 1 and lL[j + 1] > lL[k]:
            k = j + 1
        if k == j:
            break
        j = k
    top = lL[j]
    lo = j
    while lo > 0 and lL[lo - 1] >= top - drop:
        lo -= 1
    hi = j
    while hi < n - 1 and lL[hi + 1] >= top - drop:
        hi += 1
    if hi - lo >= 2:
        x = GRID[lo:hi + 1] - GRID[j]
        a, b, _ = np.polyfit(x, lL[lo:hi + 1], 2)
        if a < 0:
            v = float(np.clip(-b / (2 * a), x[0], x[-1]))
            return float(GRID[j] + v), float(max(np.sqrt(-1.0 / (2 * a)), DR))
        return float(0.5 * (GRID[lo] + GRID[hi])), float(max((GRID[hi] - GRID[lo]) / np.sqrt(12), DR))
    return float(GRID[j]), DR


def window_moments(L: np.ndarray, center: float, half: float) -> tuple[float, float] | None:
    """Mean and std of the (unnormalised) density L inside |r - center| <= half (research temporal._moments)."""
    win = np.abs(GRID - center) <= half
    wl = L[win] * DR
    s = wl.sum()
    if s <= 0:
        return None
    wl = wl / s
    m = float(wl @ GRID[win])
    return m, float(np.sqrt(max(wl @ (GRID[win] - m) ** 2, 0.05 ** 2)))


@dataclass
class RangeLik:
    """What a fused person box carries for the tracker (Det3D.rlik): the range likelihood split into its camera part
    and its radar part (the tracker shifts the camera part by the track's camera bias), its summary, the geometry
    of the box ray, and the radar clusters with their own velocity measurements."""
    C: np.ndarray                   # (3,) camera centre, level frame of the scan
    ray: np.ndarray                 # (3,) box-centre pixel ray (not normalised), level frame
    log_cam: np.ndarray | None      # Σ camera-cue log-likelihoods over GRID (None: no camera cue)
    log_rad: np.ndarray | None      # radar log-likelihood log(1 + κ S) over GRID (None: no cluster)
    summ: dict                      # summarize() of the posterior
    radar_backed: bool              # radar term at the dominant mode >= ln 3 (docs/37 §5.3)
    b_sig: float                    # camera bearing sigma [rad]
    sig_up: float                   # vertical position sigma of the 3-D point [m]
    cut: bool = False               # the box is cut (image edge / radar occluder): its camera cues are one-sided
    cues: BoxCues | None = None     # the inputs (diagnosis / figures: per-cue terms via RangeLikModel.camera_terms)
    clusters: list = field(default_factory=list)   # dict(r, idx, n, l, vel fields of that cluster: vel, vel_cov, vr,
                                                   # vr_u, vr_var)
    log_bev: np.ndarray | None = None   # range-only BEV wedge term (docs/38 §13.6, fusion.lik_wedge_mode): shapes the
                                        # distribution, never radar support / velocity (log_rad, clusters: in-box only)

    @property
    def hor(self) -> float:
        return float(max(np.linalg.norm(self.ray[:2]), 1e-6))

    @property
    def u(self) -> np.ndarray:
        return self.ray[:2] / self.hor

    def point(self, r: float) -> np.ndarray:
        """3-D point on the box ray at horizontal range r."""
        return self.C + (r / self.hor) * self.ray

    def lik(self, cam_shift: float = 0.0) -> np.ndarray:
        """Likelihood over GRID (max 1), the camera part evaluated at r e^cam_shift (track camera bias b: the camera
        says r e^b when the truth is r)."""
        tot = np.zeros_like(GRID)
        if self.log_cam is not None:
            tot = tot + (np.interp(GRID * np.exp(cam_shift), GRID, self.log_cam) if cam_shift else self.log_cam)
        if self.log_rad is not None:
            tot = tot + self.log_rad
        if self.log_bev is not None:
            tot = tot + self.log_bev
        return np.exp(tot - tot.max())

    def rotate(self, R: np.ndarray, dp: np.ndarray | None) -> None:
        """Rigid motion of the geometry (image time -> scan time, Pipeline._rotate_det); ranges along the ray keep."""
        self.C = R @ self.C + (0.0 if dp is None else dp)
        self.ray = R @ self.ray                     # the clusters' velocity terms are built from the unrotated sensor
                                                    # points, i.e. already in scan time (as Det3D.vel / vr_u)


_MODELS: dict = {}


def load_range_lik(path: str) -> RangeLikModel:
    if path not in _MODELS:
        pth = Path(path)
        if not pth.is_absolute():
            pth = Path(__file__).resolve().parents[3] / pth
        _MODELS[path] = RangeLikModel(json.load(open(pth)))
    return _MODELS[path]
