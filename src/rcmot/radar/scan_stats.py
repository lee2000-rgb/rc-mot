"""Per-scan statistical standardisation of 4D radar returns (docs/09_statistical_normalization_reid.html).

Instead of absolute thresholds, every feature is expressed relative to the statistics of the scan it belongs
to (MCL2026 eq. (1), agmo radar_localization/weighting.py):

  robust z            z = (a - median) / max(1.4826 MAD, floor)
  range-conditioned   SNR / RCS z inside equal-count range bins of the scan (SNR and RCS fall with range;
                      near-range SNR saturates at ~31 dB -> MAD floor 0.5 dB)
  local contrast      z against the k=12 nearest neighbours in normalised polar space (range/dr, az/daz);
                      contrast c = min(z_RCS, z_SNR) (both channels must stand out)
  reliability         alpha = sigmoid((z_SNR_range - z0) / tau)   (soft, no cut)
  Doppler noise       sigma_v(scan) = 1.4826 MAD of the static residuals of the ego-velocity fit
  dynamic             |z_v| > dyn_z (statistical: above this scan's noise) AND |v_c| > dyn_vmin (physical floor)

Ego velocity: LMedS initial fit -> adaptive inlier threshold max(k sigma, floor) -> alpha-weighted LSQ,
iterated (MSAC-like), so the threshold follows the scan's Doppler noise (standing 0.012 m/s vs driving
0.02-0.06 m/s on the HDX bags) instead of a fixed 0.2 m/s.
Scans with few points borrow the bin statistics of the last `window_s` seconds.

Below-ground returns (docs/16): multipath images of real targets appear BELOW the ground plane (and farther than
the target), e.g. 5 % of the returns at 20-30 m on construction_field lie 1.5-9 m under the ground. Per scan
(pooled over `ground_window_s`), in equal-count range bins of the level-frame returns, the ground level is a
Theil-Sen line through the bins' lower quartiles of z and its spread the bin's robust sigma (elevation noise grows with range); a return
more than max(ground_k sigma, ground_margin) below the range-interpolated level is dropped (keep = False).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

MAD_K = 1.4826


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def robust_z(a: np.ndarray, ref: np.ndarray | None = None, floor: float = 1e-3) -> np.ndarray:
    ref = a if ref is None else ref
    med = np.median(ref)
    return (a - med) / max(MAD_K * np.median(np.abs(ref - med)), floor)


def local_contrast_polar(r: np.ndarray, az: np.ndarray, val: np.ndarray, *, k: int = 12, dr: float = 3.0,
                         daz: float = 0.05, floor: float = 0.5) -> np.ndarray:
    """agmo local_prominence_polar: robust z of each value against its k nearest neighbours in (r/dr, az/daz)."""
    n = len(val)
    if n < k + 2:
        return np.zeros(n)
    coords = np.c_[r / dr, az / daz]
    _, idx = cKDTree(coords).query(coords, k=k + 1)
    nv = val[idx[:, 1:]]
    bg = np.median(nv, axis=1)
    mad = np.median(np.abs(nv - bg[:, None]), axis=1)
    return (val - bg) / np.maximum(MAD_K * mad, floor)


def scan_z(v: np.ndarray, ref: np.ndarray, kind: str, floor: float) -> np.ndarray:
    """z of the returns v against one distribution `ref` (the scan, or the recent window of a small scan), no range
    bins (docs/38 §12): "scan" median / 1.4826 MAD; "shape" mean / std if ref looks Gaussian (|skewness| < 1 and
    |excess kurtosis| < 1, Xie et al. 2026 §3.2) else median / MAD; "rank" normal scores of the mid-rank."""
    ref = np.asarray(ref, float)
    if len(ref) == 0:
        return np.zeros(len(v))
    if kind == "rank":
        srt = np.sort(ref)
        lo, hi = np.searchsorted(srt, v, "left"), np.searchsorted(srt, v, "right")
        from scipy.stats import norm
        return norm.ppf((0.5 * (lo + hi) + 0.5) / (len(ref) + 1))
    if kind == "shape" and len(ref) >= 8:
        from scipy.stats import kurtosis, skew
        if abs(skew(ref)) < 1.0 and abs(kurtosis(ref)) < 1.0:
            return (v - ref.mean()) / max(ref.std(), floor)
    med = np.median(ref)
    return (v - med) / max(MAD_K * np.median(np.abs(ref - med)), floor)


def snr_saturated(snr: np.ndarray, min_frac: float = 0.02) -> np.ndarray:
    """Points whose SNR sits at the sensor's reporting cap (Bosch off-highway radar: 31 dB; every strong target
    shares that value, so its SNR local contrast is ~0). Returns a mask; all False when nothing is capped."""
    n = len(snr)
    if n == 0:
        return np.zeros(0, bool)
    sat = snr >= np.max(snr) - 1e-3
    return sat if sat.sum() >= 2 and sat.mean() >= min_frac else np.zeros(n, bool)   # a cap = tied maxima


def combine_contrast(c_rcs: np.ndarray, c_snr: np.ndarray, snr: np.ndarray) -> np.ndarray:
    """MCL2026 eq. (1): min over the RCS and SNR local contrasts, except at SNR-saturated points, where the SNR
    branch carries no information and the RCS contrast is used alone (a +12 dBsm corner reflector at 4 m was
    contrast 0 under the plain min: docs/19, indoor capture 2026-10-24)."""
    c = np.minimum(c_rcs, c_snr)
    sat = snr_saturated(snr)
    if sat.any():
        c = c.copy()
        c[sat] = c_rcs[sat]
    return c


@dataclass
class ScanStatsParams:
    # range-conditioned z
    min_bin: int = 40                 # points per range bin
    max_bins: int = 6
    mad_floor_db: float = 0.5
    # local contrast
    k_local: int = 12
    polar_dr: float = 3.0
    polar_daz: float = 0.05
    # reliability weight
    alpha_z0: float = -1.0
    alpha_tau: float = 0.5
    alpha_source: str = "none"        # default "none" since 11-03 (docs/38 §12, server r40: every alternative beat the
                                      # range-binned SNR alpha, alpha = 1 best: BEV HOTA +0.017 [+0.004, +0.034]).
                                      # Reliability alpha from the range-conditioned z of: "snr" (agmo alpha_snr), "rcs",
                                      # "both" (min of the two, RCS alone at SNR-capped points, as combine_contrast),
                                      # "none" (alpha = 1). Bosch SNR caps at 31 dB: near-range bins then have MAD at the
                                      # 0.5 dB floor and a return 2 dB below the cap gets z = -4 (docs/38 §12).
                                      # Whole-scan RCS statistics (no range bins): "scan_rcs" median / MAD, "shape_rcs"
                                      # mean / std when the scan's RCS looks Gaussian (|skew|, |excess kurtosis| < 1, Xie
                                      # et al. 2026 §3.2) else median / MAD, "rank_rcs" normal scores of the mid-rank
    # ego velocity
    planar: bool = True
    ego_frame: str = "level"         # planar fit in the gravity-levelled frame (R_level from the pipeline; the machine moves
                                      # horizontally) -- docs/23 §3.3/§5.2 (10-28): the radar-frame planar fit read 8 % low on
                                      # the 26 deg tilted excavator-B rig; implemented 11-04 (audit). "radar" = the old fit
    lmeds_iters: int = 80
    inlier_k: float = 3.0             # inliers: |residual| < k sigma
    inlier_floor: float = 0.03        # m/s, lower bound of the inlier threshold
    sigma_v_floor: float = 0.01       # m/s
    min_inlier_ratio: float = 0.35
    standstill_p70: float = 0.05      # m/s, fast path: |v_r| 70th percentile below -> ego = 0
    # dynamic labelling
    dyn_z: float = 3.0
    dyn_soft: float = 0.5
    dyn_vmin: float = 0.1             # m/s, physical floor (micro-motion / vibration)
    # small scans
    min_pts: int = 80
    window_s: float = 1.0
    # below-ground multipath (needs level-frame z)
    ground_enable: bool = True
    ground_k: float = 3.0
    ground_margin: float = 1.0        # [m] never closer to the ground level than this
    ground_bins: int = 8
    ground_min_bin: int = 60
    ground_window_s: float = 1.0
    ground_q: float = 25.0            # percentile of z per range bin taken as the ground level
    # below-ground rule (research/ground, user question 11-04): the per-bin MAD of z mixes trees / objects / elevation
    # noise and grows far away, so the drop line g - max(3 sigma, 1 m) fell to -9 m beyond ~35 m (no check at all)
    ground_sigma: str = "elev"        # "elev" (default since 11-04): s(r) = hypot(ground_s0, r * ground_el_sigma), fitted on
                                      # 373k ground-clutter returns vs LiDAR (<= 45 m, docs/38 §13.3) | "mad": per-bin MAD of
                                      # all z (until 11-04; the drop line fell to -8 m beyond 35 m)
    ground_s0: float = 0.03           # [m] height noise at zero range (elev model)
    ground_el_sigma: float = 0.0075   # [rad] elevation noise, 0.43 deg (elev model)
    ground_mode: str = "drop"         # "drop": keep = False below the line | "soft": no drop, weight
                                      # w_g = sigmoid((z - line) / (ground_soft_tau * s)) into alpha and the lik cluster score
    ground_soft_tau: float = 0.5
    # clutter
    drop_z: float = -2.0              # drop only if this weak AND isolated
    isolated_m: float = 1.5
    drop_source: str = "rcs_scan"     # which z is "weak": "rcs_scan" whole-scan RCS z (ScanStats.z_rcs_scan; default since
                                      # 11-04, docs/38 §12.7), "snr" range-binned SNR z (until 11-04; Bosch SNR caps at 31 dB,
                                      # docs/38 §12.1), "none" (no weak drop; ambiguity pairs / below-ground still dropped)
    ambig_source: str = "rcs"         # azimuth-ambiguity pair: drop the member with the lower "rcs" (ties -> lower SNR;
                                      # default since 11-04) or the lower range-binned "snr" z (until 11-04: both members
                                      # of a near pair often sit at the 31 dB SNR cap -> tie -> both kept)


@dataclass
class ScanStats:
    v_sensor: np.ndarray              # (3,) sensor velocity in the radar frame
    ego_valid: bool
    inlier_ratio: float
    sigma_v: float                    # Doppler noise of this scan [m/s]
    v_comp: np.ndarray                # ego-compensated radial velocity
    z_v: np.ndarray                   # v_comp / sigma_v
    p_dyn: np.ndarray                 # soft dynamic probability
    dynamic: np.ndarray               # bool
    static_inlier: np.ndarray         # bool, inlier of the ego fit
    z_snr: np.ndarray                 # range-conditioned
    z_rcs: np.ndarray
    contrast: np.ndarray              # local contrast min(z_rcs, z_snr) (polar k-NN)
    alpha: np.ndarray                 # reliability weight (0,1)
    keep: np.ndarray                  # clutter filter result
    n: int = 0
    used_window: bool = False
    below_ground: np.ndarray | None = None     # bool, dropped as below-ground multipath
    ground_z: np.ndarray | None = None         # per-point ground level [m, level frame]
    ground_w: np.ndarray | None = None         # soft below-ground weight (ground_mode "soft"), else None
    keep_pre_ground: np.ndarray | None = None  # keep before the below-ground drop (mirror returns: BEV evidence, docs/38 §13.6)
    z_rcs_scan: np.ndarray | None = None       # whole-scan RCS z (median / 1.4826 MAD over the scan, no range bins):
                                               # the point score with the best LiDAR real-vs-ghost separation (docs/38
                                               # §12.3), kept for algorithms that need one (alpha is off by default)
    extra: dict = field(default_factory=dict)


def _dirs(xyz):
    return xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-6)


def fit_ego_velocity(xyz: np.ndarray, v_r: np.ndarray, w: np.ndarray, p: ScanStatsParams,
                     rng: np.random.Generator):
    """LMedS seed + adaptive-threshold weighted LSQ on v_r = -d . v_s. Returns (v3, inliers, sigma, ratio, ok)."""
    n = len(v_r)
    dim = 2 if p.planar else 3
    D = _dirs(xyz)[:, :dim]
    if n < max(8, dim + 2):
        return np.zeros(3), np.zeros(n, bool), np.nan, 0.0, False
    if np.percentile(np.abs(v_r), 70) < p.standstill_p70:          # standing still: v = 0
        v = np.zeros(dim)
    else:
        best, best_med = None, np.inf
        for _ in range(p.lmeds_iters):
            idx = rng.choice(n, dim, replace=False)
            try:
                v = np.linalg.solve(-D[idx], v_r[idx])
            except np.linalg.LinAlgError:
                continue
            med = np.median((-D @ v - v_r) ** 2)
            if med < best_med:
                best, best_med = v, med
        if best is None:
            return np.zeros(3), np.zeros(n, bool), np.nan, 0.0, False
        v = best
    res = -D @ v - v_r
    sigma = max(MAD_K * np.median(np.abs(res - np.median(res))), p.sigma_v_floor)
    inl = np.zeros(n, bool)
    for _ in range(3):
        inl = np.abs(res) < max(p.inlier_k * sigma, p.inlier_floor)
        if inl.sum() < dim + 2:
            break
        if np.any(v):                                               # keep exact zero at standstill
            sw = np.sqrt(w[inl])
            v = np.linalg.lstsq(-D[inl] * sw[:, None], v_r[inl] * sw, rcond=None)[0]
        res = -D @ v - v_r
        r_in = res[inl]
        sigma = max(MAD_K * np.median(np.abs(r_in - np.median(r_in))), p.sigma_v_floor)
    ratio = inl.mean()
    v3 = np.zeros(3)
    v3[:dim] = v
    return v3, inl, sigma, ratio, ratio >= p.min_inlier_ratio


class ScanStatistician:
    """Computes ScanStats per radar scan; keeps a short window of past scans for small ones."""

    def __init__(self, params: ScanStatsParams | None = None, seed: int = 0):
        self.p = params or ScanStatsParams()
        self.rng = np.random.default_rng(seed)
        self.hist: deque = deque()          # (stamp, r, snr, rcs)
        self.ground_hist: deque = deque()   # (stamp, horizontal range, level z)

    def ground_level(self, stamp: float, rh: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-point ground level and robust z sigma, from range bins of the pooled recent returns."""
        p = self.p
        self.ground_hist.append((stamp, rh, z))
        while self.ground_hist and self.ground_hist[0][0] < stamp - p.ground_window_s:
            self.ground_hist.popleft()
        R = np.concatenate([h[1] for h in self.ground_hist])
        Z = np.concatenate([h[2] for h in self.ground_hist])
        nb = int(np.clip(len(R) // p.ground_min_bin, 1, p.ground_bins))
        edges = np.quantile(R, np.linspace(0, 1, nb + 1))
        rc, zc, sc = [], [], []
        for k in range(nb):
            m = (R >= edges[k]) & (R <= edges[k + 1])
            if m.sum() < 5:
                continue
            lo = np.percentile(Z[m], p.ground_q)               # ground = lower surface; objects pull the median up
            rc.append(np.median(R[m]))
            zc.append(lo)
            sc.append(MAD_K * np.median(np.abs(Z[m] - np.median(Z[m]))))
        if not rc:
            return np.full(len(z), -np.inf), np.zeros(len(z))
        rc, zc = np.asarray(rc), np.asarray(zc)
        if len(rc) >= 3:                                        # ground is smooth in range: Theil-Sen line
            sl = [(zc[j] - zc[i]) / (rc[j] - rc[i]) for i in range(len(rc)) for j in range(i + 1, len(rc)) if rc[j] > rc[i]]
            b = float(np.median(sl)) if sl else 0.0
            a = float(np.median(zc - b * rc))
            return a + b * rh, np.interp(rh, rc, sc)
        return np.interp(rh, rc, zc), np.interp(rh, rc, sc)

    def _range_z(self, r, vals, ref_r, ref_vals):
        """z of `vals` (dict channel->array) inside equal-count range bins built from the reference points."""
        p = self.p
        nb = int(np.clip(len(ref_r) // p.min_bin, 1, p.max_bins))
        edges = np.quantile(ref_r, np.linspace(0, 1, nb + 1)[1:-1]) if nb > 1 else np.array([])
        b_ref, b = np.searchsorted(edges, ref_r), np.searchsorted(edges, r)
        out = {}
        for ch, v in vals.items():
            z = np.zeros(len(r))
            for k in range(nb):
                ref = ref_vals[ch][b_ref == k]
                m = b == k
                if not m.any() or len(ref) == 0:
                    continue
                med = np.median(ref)
                z[m] = (v[m] - med) / max(MAD_K * np.median(np.abs(ref - med)), p.mad_floor_db)
            out[ch] = z
        return out

    def compute(self, stamp: float, xyz: np.ndarray, v_r: np.ndarray, snr: np.ndarray, rcs: np.ndarray,
                peer: np.ndarray | None = None, xyz_level: np.ndarray | None = None,
                R_level: np.ndarray | None = None) -> ScanStats:
        """xyz in the radar frame (Doppler geometry); xyz_level (same points, gravity-levelled) enables the
        below-ground filter."""
        p = self.p
        n = len(v_r)
        r = np.linalg.norm(xyz, axis=1)
        az = np.arctan2(xyz[:, 1], xyz[:, 0])
        # window statistics for small scans
        self.hist.append((stamp, r, snr, rcs))
        while self.hist and self.hist[0][0] < stamp - p.window_s:
            self.hist.popleft()
        used_window = n < p.min_pts and len(self.hist) > 1
        if used_window:
            ref_r = np.concatenate([h[1] for h in self.hist])
            ref = {"snr": np.concatenate([h[2] for h in self.hist]), "rcs": np.concatenate([h[3] for h in self.hist])}
        else:
            ref_r, ref = r, {"snr": snr, "rcs": rcs}
        if n == 0:
            e = np.zeros(0)
            return ScanStats(np.zeros(3), False, 0.0, np.nan, e, e, e, e.astype(bool), e.astype(bool), e, e, e, e,
                             e.astype(bool), 0, used_window)
        zr = self._range_z(r, {"snr": snr, "rcs": rcs}, ref_r, ref)
        z_snr, z_rcs = zr["snr"], zr["rcs"]
        z_rcs_scan = scan_z(rcs, ref["rcs"], "scan", p.mad_floor_db)
        if p.alpha_source == "none":
            alpha = np.ones(n)
        else:
            if p.alpha_source in ("scan_rcs", "shape_rcs", "rank_rcs"):
                za = scan_z(rcs, ref["rcs"], p.alpha_source.split("_")[0], p.mad_floor_db)
            elif p.alpha_source == "rcs":
                za = z_rcs
            elif p.alpha_source == "both":
                za = np.minimum(z_snr, z_rcs)
                sat = snr_saturated(snr)
                za[sat] = z_rcs[sat]
            else:
                za = z_snr
            alpha = sigmoid((za - p.alpha_z0) / p.alpha_tau)
        c_rcs = local_contrast_polar(r, az, rcs, k=p.k_local, dr=p.polar_dr, daz=p.polar_daz, floor=p.mad_floor_db)
        c_snr = local_contrast_polar(r, az, snr, k=p.k_local, dr=p.polar_dr, daz=p.polar_daz, floor=p.mad_floor_db)
        contrast = combine_contrast(c_rcs, c_snr, snr)

        # clutter: weak AND isolated, plus the weaker member of azimuth-ambiguity pairs (drop_source, ambig_source)
        keep = np.ones(n, bool)
        if p.drop_source == "none":
            weak = np.zeros(n, bool)
        else:
            weak = (z_rcs_scan if p.drop_source == "rcs_scan" else z_snr) < p.drop_z
        if weak.any() and n > 1:
            d, _ = cKDTree(xyz).query(xyz[weak], k=2)
            keep[np.flatnonzero(weak)[d[:, 1] > p.isolated_m]] = False
        if peer is not None:
            pi = np.asarray(peer).astype(int)
            for i in np.flatnonzero((pi >= 0) & (pi < n) & (pi != np.arange(n))):
                j = pi[i]
                if pi[j] != i:
                    continue
                if p.ambig_source == "snr":
                    weaker = z_snr[i] < z_snr[j]
                else:
                    weaker = rcs[i] < rcs[j] or (rcs[i] == rcs[j] and snr[i] < snr[j])
                if weaker:
                    keep[i] = False
        below, g_z, g_w, keep_pg = None, None, None, None
        if p.ground_enable and xyz_level is not None and n:
            rh = np.linalg.norm(xyz_level[:, :2], axis=1)
            g_z, g_s = self.ground_level(stamp, rh, xyz_level[:, 2])
            if p.ground_sigma == "elev":
                g_s = np.hypot(p.ground_s0, rh * p.ground_el_sigma)
            line = g_z - np.maximum(p.ground_k * g_s, p.ground_margin)
            below = xyz_level[:, 2] < line
            if p.ground_mode == "soft":
                g_w = sigmoid((xyz_level[:, 2] - line) / np.maximum(p.ground_soft_tau * g_s, 1e-3))
                alpha = alpha * g_w                                 # point weights (ego fit, movers, RGR, fusion)
            else:
                keep_pg = keep.copy()
                keep &= ~below

        if p.ego_frame == "level" and R_level is not None:     # directions rotated into the level frame (rotation only)
            R = np.asarray(R_level, float)
            v_l, inl, sigma, ratio, ok = fit_ego_velocity(xyz[keep] @ R.T, v_r[keep], alpha[keep], p, self.rng)
            v_s = R.T @ v_l                                     # back to the radar frame (v_r = -d_radar . v_s)
        else:
            v_s, inl, sigma, ratio, ok = fit_ego_velocity(xyz[keep], v_r[keep], alpha[keep], p, self.rng)
        static_inlier = np.zeros(n, bool)
        static_inlier[np.flatnonzero(keep)[inl]] = True
        v_comp = v_r + _dirs(xyz) @ v_s
        sig = sigma if np.isfinite(sigma) else p.sigma_v_floor
        z_v = v_comp / sig
        p_dyn = sigmoid((np.abs(z_v) - p.dyn_z) / p.dyn_soft) * sigmoid((np.abs(v_comp) - p.dyn_vmin) / (0.25 * p.dyn_vmin))
        dynamic = keep & (np.abs(z_v) > p.dyn_z) & (np.abs(v_comp) > p.dyn_vmin)
        return ScanStats(v_s, ok, float(ratio), float(sig), v_comp, z_v, p_dyn, dynamic, static_inlier, z_snr, z_rcs,
                         contrast, alpha, keep, n, used_window, below, g_z, ground_w=g_w, keep_pre_ground=keep_pg,
                         z_rcs_scan=z_rcs_scan)
