"""Crowd-relative appearance for same-uniform crews (docs/32, training-free).

Two optional additions to the OSNet embedding, both label-free:

* crowd mean (``reid.crowd_mean``): subtract the mean embedding of the people present in the current frame and
  renormalise. Whatever everybody wears (same helmet / vest) is removed; what differs between people remains.
* part colour (``reid.part_colour``): mask-weighted soft Lab histograms (4 x 6 x 6) of the pants, sleeve and shoe
  bands and of the whole person, the whole-person histogram also minus the crowd-mean histogram. In uniform crews the
  identity lives in the pants and sleeves; the helmet and vest carry none (docs/32 §3.2).

The tracker compares descriptors with ``fused_distance``: per-cue distance (OSNet cosine, histogram Hellinger),
scaled by a label-free global median, saturated by 3 tanh(z / 3), weighted mean. Histograms are stored as
sqrt-histograms (unit L2 norm), so Hellinger = sqrt(1 - a.b) and the gallery EMA is the same as for the embedding.

Bank check (research/reid_uniform/E_tracker): day / uniform pair AUC 0.944 / 0.931 (masked OSNet) ->
0.969 / 0.956 (crowd mean + part colour), re-entry rank-1 0.79 -> 0.91.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

SZ = (64, 128)                                       # (w, h) of the histogram crop
NB = (4, 6, 6)                                       # L, a, b bins
LO = np.array([0.0, -60.0, -60.0], np.float32)
HI = np.array([100.0, 60.0, 80.0], np.float32)
# band: (rows as a fraction of the box height, column ranges as a fraction of the width)
BANDS = (("pants", (0.55, 0.88), ((0.0, 1.0),)),
         ("arms", (0.22, 0.60), ((0.0, 0.25), (0.75, 1.0))),
         ("shoes", (0.88, 1.00), ((0.0, 1.0),)),
         ("whole", (0.0, 1.0), ((0.0, 1.0),)))
PARTS = tuple(b[0] for b in BANDS)
NBINS = int(np.prod(NB))


def _band_masks() -> list[np.ndarray]:
    out = []
    for _, (r0, r1), cols in BANDS:
        m = np.zeros((SZ[1], SZ[0]), bool)
        for c0, c1 in cols:
            m[int(r0 * SZ[1]):int(r1 * SZ[1]), int(c0 * SZ[0]):int(c1 * SZ[0])] = True
        out.append(m)
    return out


_BAND_IDX = [np.flatnonzero(m.reshape(-1)) for m in _band_masks()]
_NB = np.array(NB)
_SCALE = (_NB / (HI - LO)).astype(np.float32)
_BINS = [np.arange(n, dtype=np.float32) for n in NB]


def part_hists(crop_bgr: np.ndarray, mask: np.ndarray | None) -> np.ndarray | None:
    """(4, 144) sqrt-histograms (pants, arms, shoes, whole) of a tight person crop; mask = person pixels of the crop
    (None or < 30 px after resizing -> the whole box). Soft trilinear binning, weight = mask + 1e-3."""
    if crop_bgr is None or crop_bgr.size == 0 or min(crop_bgr.shape[:2]) < 4:
        return None
    im = cv2.resize(crop_bgr, SZ, interpolation=cv2.INTER_AREA)
    if mask is not None and mask.size:
        m = cv2.resize(mask.astype(np.uint8), SZ, interpolation=cv2.INTER_NEAREST).astype(bool)
        if m.sum() < 30:
            m = np.ones_like(m)
    else:
        m = np.ones((SZ[1], SZ[0]), bool)
    lab = cv2.cvtColor(im.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    x = (np.clip(lab, LO, HI - np.float32(1e-4)) - LO) * _SCALE - np.float32(0.5)                  # continuous bin coordinate per axis
    # trilinear binning = product of per-axis tent weights max(0, 1 - |x - bin|) (bins outside the grid drop out)
    wl = np.maximum(np.float32(0), np.float32(1) - np.abs(x[:, 0, None] - _BINS[0]))            # (N, 4)
    wa = np.maximum(np.float32(0), np.float32(1) - np.abs(x[:, 1, None] - _BINS[1]))            # (N, 6)
    wb = np.maximum(np.float32(0), np.float32(1) - np.abs(x[:, 2, None] - _BINS[2]))            # (N, 6)
    wl *= (m.reshape(-1).astype(np.float32) + np.float32(1e-3))[:, None]
    la = (wl[:, :, None] * wa[:, None, :]).reshape(-1, NB[0] * NB[1])        # (N, 24)
    out = np.zeros((len(BANDS), NBINS), np.float32)
    for k, sel in enumerate(_BAND_IDX):
        h = (la[sel].T @ wb[sel]).ravel()                                      # (L*a, b) -> l*36 + a*6 + b
        s = h.sum()
        out[k] = np.sqrt(h / s) if s > 1e-9 else np.sqrt(np.full(NBINS, 1.0 / NBINS))
    return out


def _nrm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def camera_pitch_down_deg(calib) -> float:
    """Angle of the optical axis below the horizontal (level frame) [deg]."""
    ax = calib.R_cl.T @ np.array([0.0, 0.0, 1.0])
    return float(np.degrees(-np.arcsin(np.clip(ax[2] / (np.linalg.norm(ax) + 1e-12), -1.0, 1.0))))


@dataclass
class AppearanceParams:
    crowd_mean: bool = True           # subtract the mean embedding of the people present (>= crowd_min_n); on since 88063b6 (docs/32 §5.5)
    part_colour: bool = True          # pants / sleeve / shoe + whole-minus-crowd colour histograms; on since 88063b6
    crowd_min_n: int = 2
    crowd_min_score: float = 0.5      # detections that form the crowd (false positives pollute the mean)
    crowd_classes: tuple = ("worker",)
    crowd_hold_s: float = 3.0         # alone in the frame: keep using the last crowd mean this long, else raw
    part_max_pitch_deg: float = 20.0  # steeper camera (excB ~26 deg): fixed body bands miss the legs -> no part colour
    extra: dict = field(default_factory=dict)


class AppearanceStage:
    """Per frame, after the OSNet embeddings: crowd-mean subtraction and part-colour histograms on the detections
    (in place: det.emb replaced, det.hist set). Stateless apart from the last crowd means (hold)."""

    def __init__(self, cfg: dict, calib=None):
        keys = AppearanceParams.__dataclass_fields__
        self.p = AppearanceParams(**{k: v for k, v in cfg.items() if k in keys})
        if isinstance(self.p.crowd_classes, list):
            self.p.crowd_classes = tuple(self.p.crowd_classes)
        self.pitch_down = camera_pitch_down_deg(calib) if calib is not None else 0.0
        self.parts_on = self.p.part_colour and self.pitch_down <= self.p.part_max_pitch_deg
        self._last = {}                                  # cls -> (t, mu_emb, mu_whole)

    @property
    def mode(self) -> str:
        """Distance mode for the tracker: 'fused' (parts), 'crowd' (crowd-mean embedding only) or 'cos' (raw)."""
        if self.parts_on:
            return "fused"
        return "crowd" if self.p.crowd_mean else "cos"

    @property
    def scales(self) -> dict:
        return {**DEFAULT_SCALES, **({} if self.p.crowd_mean else {"emb": RAW_EMB_SCALE})}

    @property
    def active(self) -> bool:
        return self.p.crowd_mean or self.parts_on

    def __call__(self, dets: list, t: float) -> None:
        if not self.active:
            return
        if self.parts_on:
            for d in dets:
                if getattr(d, "emb", None) is None:
                    continue
                d.hist = part_hists(d.crop, getattr(d, "crop_mask", None))
        self.crowd_step(dets, t)

    def crowd_step(self, dets: list, t: float) -> None:
        """Subtract the crowd mean from det.emb (crowd_mean) and from the whole-person histogram (det.hist[3])."""
        p = self.p
        for cls in {d.cls for d in dets}:
            group = [d for d in dets if d.cls == cls and getattr(d, "emb", None) is not None]
            if not group:
                continue
            crowd = [d for d in group if d.score >= p.crowd_min_score] if cls in p.crowd_classes else []
            mu_e = mu_w = None
            if len(crowd) >= p.crowd_min_n:
                mu_e = np.mean([d.emb for d in crowd], axis=0)
                hw = [d.hist[3] ** 2 for d in crowd if getattr(d, "hist", None) is not None]
                mu_w = np.mean(hw, axis=0) if len(hw) >= p.crowd_min_n else None
                self._last[cls] = (t, mu_e, mu_w)
            elif cls in self._last and t - self._last[cls][0] <= p.crowd_hold_s:
                _, mu_e, mu_w = self._last[cls]
            for d in group:
                if p.crowd_mean and mu_e is not None:
                    d.emb = _nrm(d.emb - mu_e).astype(np.float32)
                h = getattr(d, "hist", None)
                if h is not None and mu_w is not None:
                    w = np.maximum(h[3] ** 2 - mu_w, 0.0)
                    if w.sum() > 1e-9:
                        h = h.copy()
                        h[3] = np.sqrt(w / w.sum())
                        d.hist = h


# ---------------------------------------------------------------------------------------------- distances
# label-free global scales: median per-cue distance over the bank's protocol pairs (research/reid_uniform/E_tracker,
# 13,584 crops): emb = crowd-mean embedding (raw masked embedding 0.288), whole = whole-person minus crowd mean
DEFAULT_SCALES = {"emb": 0.73, "pants": 0.234, "arms": 0.273, "shoes": 0.235, "whole": 0.565}
RAW_EMB_SCALE = 0.288


def fused_distance(ea: np.ndarray, ha: np.ndarray | None, eb: np.ndarray, hb: np.ndarray | None,
                   scales: dict, w_emb: float = 1.0, cap: float = 3.0) -> float:
    """sum_c w_c * cap*tanh((d_c / s_c) / cap) / sum_c w_c; emb: cosine distance, parts: Hellinger."""
    z = (1.0 - float(ea @ eb)) / scales["emb"]
    num, den = w_emb * cap * np.tanh(z / cap), w_emb
    if ha is not None and hb is not None:
        bc = np.einsum("pk,pk->p", ha, hb)
        hel = np.sqrt(np.clip(1.0 - bc, 0.0, None))
        for k, name in enumerate(PARTS):
            num += cap * np.tanh(hel[k] / scales[name] / cap)
            den += 1.0
    return float(num / den)


def ema_hist(old: np.ndarray | None, new: np.ndarray, momentum: float) -> np.ndarray:
    if old is None:
        return new.copy()
    h = momentum * old + (1.0 - momentum) * new
    return (h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)).astype(np.float32)
