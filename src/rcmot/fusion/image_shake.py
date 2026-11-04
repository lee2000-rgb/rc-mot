"""High-frequency global image motion between consecutive frames (docs/21 E6, appendix 16a recommendations 3/4).

Phase correlation on the upper half of the image (background / horizon; people and machines are mostly in the
lower half), downscaled. The slow part (steady swing, driving) is removed by subtracting the median of the last
few frame-to-frame shifts, so what is left is shake (engine vibration, bumps) in full-resolution pixels. The
fusion adds it to the camera bearing sigma (horizontal) and, as pitch, to the ground-contact range sigma.
No IMU: the earlier IMU-based shake correction (23.send) made static targets worse (docs/16 §2).
"""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class ImageShake:
    def __init__(self, scale: float = 0.25, hist: int = 5):
        self.scale = scale
        self.prev: np.ndarray | None = None
        self.win: np.ndarray | None = None
        self.hist: deque = deque(maxlen=hist)

    def reset(self) -> None:
        self.prev = None
        self.hist.clear()

    def update(self, bgr: np.ndarray) -> tuple[float, float] | None:
        """(|dx|, |dy|) high-frequency shift of this frame vs the previous one [px], None on the first frame."""
        top = bgr[: bgr.shape[0] // 2]
        g = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top
        g = cv2.resize(g, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.win is None or self.win.shape != g.shape:
            self.win = cv2.createHanningWindow((g.shape[1], g.shape[0]), cv2.CV_32F)
        out = None
        if self.prev is not None and self.prev.shape == g.shape:
            (dx, dy), _ = cv2.phaseCorrelate(self.prev, g, self.win)
            d = np.array([dx, dy]) / self.scale
            self.hist.append(d)
            med = np.median(np.asarray(self.hist), axis=0)
            out = (float(abs(d[0] - med[0])), float(abs(d[1] - med[1])))
        self.prev = g
        return out
