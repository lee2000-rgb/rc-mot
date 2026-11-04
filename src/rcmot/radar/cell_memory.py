"""BEV cell memory of static radar returns (docs/41 §8-9): where the radar keeps seeing static returns while the machine
stands still is background (posts, machine body, ground clutter); a person arrives, moves, or is excluded from learning.

v1 = the stationary mode of docs/41 §9.3 (the offline test research/bev_wedge/cellmap_eval.py):
  * 0.5 m cells in the level frame of the vehicle; the memory is kept only while the machine stands still (ego speed and
    yaw rate below the pipeline's static thresholds) and cleared as soon as it moves or swings (no ego compensation yet);
  * every scan LEARNS its kept, Doppler-static returns (|v_comp| < max(3 sigma_v, 0.1 m/s)) except those near a person:
    within exclude_m of a worker track, or inside a person box's azimuth wedge (box left / right rays +- margin) and its
    range window (range distribution median +- max(exclude_m, 3 sigma_full));
  * background score of a return at time t = share of the scans of [t - window_s, t) in which its 3x3 cell block held a
    learned return (capped at 1) -- the same score as the offline test;
  * the score is used only after min_learn_s of continuous stationary learning (before that it is no evidence).
The fusion turns a radar cluster's mean score bg into a likelihood-ratio factor on its layout score (range_lik.radar_parts):
  f(bg) = odds(sigmoid(a + b bg)) / prior_odds, clipped to [f_min, f_max]   (a, b, prior fitted offline,
  research/bev_wedge/fit_bg_lr.py on the wedge clusters of 11 stationary segments, track-based exclusion)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class CellMemoryParams:
    enable: bool = False
    cell_m: float = 0.5
    window_s: float = 5.0
    min_learn_s: float = 5.0          # continuous stationary learning before the score is used
    exclude_m: float = 1.0            # returns this close to a worker track are not learned
    margin_deg: float = 1.5           # person-box wedge margin for the learning exclusion
    bg_slope: float = -4.14           # fit_bg_lr.py (docs/41 §10)
    f_min: float = 0.05


class CellMemory:
    def __init__(self, p: CellMemoryParams):
        self.p = p
        self.hist: deque = deque()     # (t, int64 keys of the cells learned in that scan)
        self.count: dict = {}          # key -> number of scans in the window that learned it (running)
        self.t_still: float | None = None

    def reset(self) -> None:
        self.hist.clear()
        self.count.clear()
        self.t_still = None

    def step(self, t: float, stationary: bool) -> None:
        """Start of a scan: clear on motion, drop learned scans older than the window."""
        if not stationary:
            self.reset()
            return
        if self.t_still is None:
            self.t_still = t
        while self.hist and self.hist[0][0] < t - self.p.window_s:
            for k in self.hist.popleft()[1]:
                c = self.count[k] - 1
                if c:
                    self.count[k] = c
                else:
                    del self.count[k]

    def valid(self, t: float) -> bool:
        return self.t_still is not None and t - self.t_still >= self.p.min_learn_s and len(self.hist) > 0

    def cells(self, xy: np.ndarray) -> np.ndarray:
        return np.floor(np.asarray(xy, float)[:, :2] / self.p.cell_m).astype(int)

    @staticmethod
    def _key(c: np.ndarray) -> np.ndarray:
        return (c[:, 0].astype(np.int64) + 2 ** 20) * 2 ** 21 + (c[:, 1].astype(np.int64) + 2 ** 20)

    def score(self, xy: np.ndarray) -> np.ndarray:
        """Background score in [0, 1] of every point (level-frame xy) from the scans in the window (this scan excluded)."""
        n = len(xy)
        if n == 0 or not self.count:
            return np.zeros(n)
        keys = np.fromiter(self.count.keys(), np.int64, len(self.count))
        cnt = np.fromiter(self.count.values(), np.int64, len(self.count))
        o = np.argsort(keys)
        keys, cnt = keys[o], cnt[o]
        c = self.cells(xy)
        out = np.zeros(n)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                q = self._key(c + np.array([dx, dy]))
                j = np.clip(np.searchsorted(keys, q), 0, len(keys) - 1)
                out += np.where(keys[j] == q, cnt[j], 0)
        return np.minimum(out / len(self.hist), 1.0)

    def learn(self, t: float, xy: np.ndarray, mask: np.ndarray) -> None:
        if self.t_still is None:
            return
        ks = np.unique(self._key(self.cells(np.asarray(xy)[np.asarray(mask, bool)]))).tolist()
        self.hist.append((t, ks))
        for k in ks:
            self.count[k] = self.count.get(k, 0) + 1

    def factor(self, bg: float) -> float:
        return float(np.clip(np.exp(self.p.bg_slope * float(bg)), self.p.f_min, 1.0))


def learn_mask(stats, xyz: np.ndarray, C: np.ndarray, tracks, dets3d, p: CellMemoryParams, calib) -> np.ndarray:
    """Returns to learn this scan: kept, Doppler-static, not near a worker track, not inside a person box's wedge x range
    window."""
    n = len(xyz)
    if stats is None or n == 0:
        return np.zeros(n, bool)
    sv = stats.sigma_v if np.isfinite(stats.sigma_v) else 0.1
    m = np.asarray(stats.keep, bool) & (np.abs(stats.v_comp) < max(3 * sv, 0.1))
    xy = xyz[:, :2]
    for t in tracks:
        if t.cls == "worker" and t.state.name in ("CONFIRMED", "LOST", "RADAR_ONLY", "TENTATIVE"):
            pos = t.kf.pos if hasattr(t, "kf") else t.pos          # tracker.Track or TrackOut
            m &= np.hypot(*(xy - np.asarray(pos, float)[:2]).T) > p.exclude_m
    rel = xy - np.asarray(C, float)[:2]
    rng = np.hypot(rel[:, 0], rel[:, 1])
    az = np.arctan2(rel[:, 1], rel[:, 0])
    mg = np.radians(p.margin_deg)
    for d in dets3d:
        if d.cls != "worker":
            continue
        m &= np.hypot(*(xy - np.asarray(d.pos, float)[:2]).T) > p.exclude_m
        rl = getattr(d, "rlik", None)
        x1, y1, x2, y2 = [float(v) for v in d.box2d]
        vm = 0.5 * (y1 + y2)
        al = np.arctan2(*calib.pixel_ray(x1, vm)[[1, 0]])
        ar = np.arctan2(*calib.pixel_ray(x2, vm)[[1, 0]])
        lo, hi = min(al, ar) - mg, max(al, ar) + mg
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        inw = np.abs(np.angle(np.exp(1j * (az - mid)))) <= half
        if rl is not None:
            r0 = float(rl.summ["r"])
            w = max(p.exclude_m, 3.0 * float(rl.summ.get("sig_full", rl.summ["sig"])))
        else:
            r0 = float(np.hypot(*(np.asarray(d.pos, float)[:2] - np.asarray(C, float)[:2])))
            w = max(p.exclude_m, 0.3 * r0)
        m &= ~(inw & (np.abs(rng - r0) <= w))
    return m
