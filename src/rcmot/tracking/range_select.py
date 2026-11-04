"""docs/38: how the tracker uses a box's range distribution (Det3D.rlik, rcmot.fusion.range_lik).

For one (track, box) pair, with the track predicted to this scan:

  geometry   C camera centre, u horizontal unit direction of the box-centre ray, u⊥ across it (level frame)
             r_p = (xy_pred - C)·u   predicted range along the ray,   l_p = (xy_pred - C)·u⊥  lateral offset
             v_al = uᵀ P u + e²,  v_lat = u⊥ᵀ P u⊥ + e² + (r_p σ_bearing)² + 0.05²   (P track xy covariance,
             e object extent sigma)
  likelihood L(r) of the box over GRID: camera part shifted by the track's camera bias b (the camera says r e^b),
             radar part unchanged (RangeLik.lik)
  prior      g(r) = N(r; r_p, v_al + 0.05² [+ r_p² var(b) when the box is not radar-backed])
  product    post(r) ∝ L(r) [(1 - ε) g(r) + ε / 89 m]      ε: the track itself may be wrong (robust)
  mask       only basins of L x 1/r holding >= lik_min_mass (and the dominant one) may be taken (basin_mask; "mode" =
             the older per-summary-mode mask, significant())
  mode       the dominant mode of post picks a peak of L; the MEASUREMENT is the Gaussian core of that peak of L alone
             (range_lik.peak_gauss: parabola on ln L within 0.5 of the peak; not post -- the prediction must not leak
             into its own measurement -- and not the moments of the whole mode, which wide camera shoulders drag):
             range r_m, sigma s_m (+ r_m² var(b) for a camera-only box)
  cost       d²_along = -2 ln( m √(2π (s_m² + v_al)) ),  m = ∫ L̂(r) g(r) dr (L̂ = L normalised over r)
               = the 1-D Mahalanobis distance for a single Gaussian mode; a minor mode adds -2 ln(its mass)
             d²_lat   = l_p² / v_lat
             d² = d²_along + d²_lat   (gated like the old 3-D Mahalanobis, tracker.gate_pos)
  radar      the measurement is radar-backed when the radar term at r_m is >= ln(lik_radar_backed); then the radar
             fields (points, velocity) are those of the cluster nearest r_m, else the box is camera-only for this track
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rcmot.fusion.range_lik import _KS, DR, GRID, peak_gauss, posterior, summarize
from rcmot.types import Det3D

UNIF = 1.0 / (GRID[-1] - GRID[0])
LN3 = float(np.log(3.0))


def _ray_basis(d: np.ndarray):
    a = d / np.linalg.norm(d)
    lat = np.cross([0.0, 0.0, 1.0], a)
    lat = lat / max(np.linalg.norm(lat), 1e-9)
    up = np.cross(a, lat)
    return np.stack([a, lat, up], 1)


def significant(rl, min_mass: float) -> np.ndarray:
    """Grid mask of the box posterior's modes holding >= min_mass (summarize cuts / masses): the only ranges a track may
    take from this box. A shoulder of a few % next to a sharp radar mode is not an alternative a track prediction may
    pick -- the track's own covariance is not honest enough for that (docs/38 §5: val_day_haltmid)."""
    sm = rl.summ
    m = np.zeros(len(GRID), bool)
    cuts, ms = sm.get("cuts"), sm.get("masses")
    if not cuts:
        return np.ones(len(GRID), bool)
    for i, mi in enumerate(ms):
        if mi >= min_mass or i == sm.get("dom"):
            m[cuts[i]:cuts[i + 1]] = True
    return m


@dataclass
class Sel:
    r: float                 # measurement range along the box ray [m]
    s: float                 # its sigma [m]
    d2: float                # d²_along + d²_lat
    d2_along: float
    d2_lat: float
    det: Det3D               # the measurement as a Det3D (point on the ray at r, covariance along s / across bearing)
    radar: bool              # radar-backed at r
    lm: float                # ln m (marginal likelihood of the box under this track's prediction, for hypotheses)
    mass: float              # posterior mass of the box mode the measurement came from
    dominant: bool           # it is the box's dominant mode


def lateral_d2(det: Det3D, xy: np.ndarray, P2: np.ndarray, ext: float) -> tuple[float, float]:
    """(d²_lat, r_p): cheap pre-gate before the along-ray product."""
    rl = det.rlik
    u = rl.u
    perp = np.array([-u[1], u[0]])
    dxy = np.asarray(xy, float)[:2] - rl.C[:2]
    r_p, l_p = float(dxy @ u), float(dxy @ perp)
    v_lat = float(perp @ P2 @ perp) + ext ** 2 + (max(r_p, 1.0) * rl.b_sig) ** 2 + 0.05 ** 2
    return l_p * l_p / v_lat, r_p


def basin_mask(L: np.ndarray, min_mass: float) -> np.ndarray:
    """Grid mask of the basins of L x 1/r (the box posterior with this track's camera shift) holding >= min_mass, plus the
    dominant one. Basins are split at EVERY valley of the 0.1 m-smoothed posterior, not only between the summary modes
    (whose peaks must reach 1 % of the highest): a 0.5 % bump inside a mode -- e.g. the 'standing' reading of a
    crouching person's box, 1.68 x farther -- is its own basin and is dropped (docs/38 §7.4, conti_side_drive_b)."""
    p = L / GRID
    ps = np.convolve(p, _KS, mode="same")
    pk = np.flatnonzero((ps[1:-1] >= ps[:-2]) & (ps[1:-1] > ps[2:])) + 1
    if len(pk) <= 1:
        return np.ones(len(GRID), bool)
    cuts = [0] + [a + int(np.argmin(ps[a:b + 1])) for a, b in zip(pk[:-1], pk[1:])] + [len(p)]
    w = p / max(p.sum(), 1e-300)
    dom = int(np.argmax(ps[pk]))
    m = np.zeros(len(GRID), bool)
    for i in range(len(pk)):
        if i == dom or w[cuts[i]:cuts[i + 1]].sum() >= min_mass:
            m[cuts[i]:cuts[i + 1]] = True
    return m


def select(det: Det3D, xy: np.ndarray, P2: np.ndarray, ext: float, bias: tuple[float, float] = (0.0, 0.0),
           eps: float = 0.05, radar_ratio: float = 3.0, lat_gate: float | None = None, min_mass: float = 0.05,
           mask: str = "basin") -> Sel | None:
    """The measurement this box gives a track predicted at xy (level, 2) with covariance P2 (2x2). None when the
    lateral offset alone exceeds lat_gate or the likelihood vanishes."""
    rl = det.rlik
    d2_lat, r_p = lateral_d2(det, xy, P2, ext)
    if lat_gate is not None and d2_lat > lat_gate:
        return None
    u = rl.u
    v_al = float(u @ P2 @ u) + ext ** 2
    b, vb = bias
    cam = not rl.radar_backed
    L = rl.lik(cam_shift=b)
    L = L * (basin_mask(L, min_mass) if mask == "basin" else significant(rl, min_mass))
    Z = L.sum() * DR
    if Z <= 0:
        return None
    sd = np.sqrt(v_al + 0.05 ** 2 + ((r_p * r_p * vb) if cam else 0.0))
    g = np.exp(-0.5 * ((GRID - r_p) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    post = L * ((1 - eps) * g + eps * UNIF)
    if post.sum() <= 0:
        return None
    sp = summarize(post / (post.sum() * DR))
    mm = peak_gauss(L, sp["r"])
    if mm is None:
        return None
    r_m, s_m = mm
    if cam and vb > 0:
        s_m = float(np.sqrt(s_m * s_m + r_m * r_m * vb))
    m = float((L / Z) @ g * DR)
    S = s_m * s_m + v_al
    d2_al = max(-2.0 * np.log(max(m, 1e-300) * np.sqrt(2 * np.pi * S)), 0.0)
    j = int(np.clip(np.searchsorted(GRID, r_m), 0, len(GRID) - 1))
    rb = rl.log_rad is not None and rl.log_rad[j] >= np.log(radar_ratio)
    hor = rl.hor
    s_ = r_m / hor
    B = _ray_basis(rl.ray)
    cov = B @ np.diag([max(s_m, 0.05) ** 2, max(s_ * rl.b_sig, 0.05) ** 2, rl.sig_up ** 2]) @ B.T
    kw = dict(pos=rl.point(r_m), pos_cov=cov, n_radar=0, radar_idx=None, vel=None, vel_cov=None, vr=None, vr_u=None,
              vr_var=None, source="lik")
    if rb and rl.clusters:
        c = min(rl.clusters, key=lambda c: abs(c["r"] - r_m))
        if abs(c["r"] - r_m) <= max(1.5, 3 * s_m):
            kw.update(n_radar=c["n"], radar_idx=c["idx"], vel=c["vel"], vel_cov=c["vel_cov"], vr=c["vr"], vr_u=c["vr_u"],
                      vr_var=c["vr_var"], source="lik+radar")
        else:
            rb = False
    sm = rl.summ
    cuts, ms = sm.get("cuts"), sm.get("masses")
    mass, dom = 1.0, True
    if cuts:
        k = int(np.clip(np.searchsorted(cuts, j, side="right") - 1, 0, len(ms) - 1))
        mass, dom = float(ms[k]), k == sm.get("dom")
    dm = replace(det, **kw)
    dm._src_oid = getattr(det, "_src_oid", id(det))   # the fused box this measurement came from (evaluation logs)
    return Sel(r=float(r_m), s=float(s_m), d2=float(d2_al + d2_lat), d2_along=float(d2_al), d2_lat=float(d2_lat),
               det=dm, radar=bool(rb), lm=float(np.log(max(m, 1e-300))), mass=mass, dominant=bool(dom))


def camera_only_range(det: Det3D) -> tuple[float, float, float] | None:
    """(r, sigma, p_alt) of the box's camera-only posterior (prior 1/r), cached on the RangeLik; None without a camera
    cue. Used to learn the track's camera bias b from radar-backed updates."""
    rl = det.rlik
    if rl.log_cam is None:
        return None
    c = getattr(rl, "_cam_summ", None)
    if c is None:
        s = summarize(posterior([rl.log_cam]))
        c = (s["r"], s["sig"], s["p_alt"])
        rl._cam_summ = c
    return c
