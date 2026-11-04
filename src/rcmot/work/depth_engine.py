"""VGGT-Omega depth engine for the work monitor (docs/24 §4.5): keyframes -> dense relative depth on a background
thread; the monitor scales each depth map with the radar (median ratio at the radar points) and back-projects it
with the calibrated K. On 155530 (6 static frames vs LiDAR): height MAE 0.09-0.14 m (1 view) / 0.05-0.09 m (6 views
jointly), pile volume -8..+7 % / -5..+1 %, against 0.20-0.32 m for the radar-scaled OAK stereo.

VGGT-Omega (facebookresearch/vggt-omega, arXiv 2605.15195, non-commercial weights): third_party/vggt-omega,
models/vggt/vggt_omega_1b_512.pt. Heads in fp32, aggregator weights in bf16 (it runs under bf16 autocast anyway):
RTX 3080, 384x688: 1 view 0.1 s / 2.8 GB, 4 views 0.34 s / 3.3 GB."""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DepthJob:
    t: float
    image: np.ndarray                  # BGR, the camera's full resolution
    world: object                      # rcmot.world.WorldState of the frame
    radar_level: np.ndarray            # (N, 3) static radar points (level frame) of the frame: the metric scale
    boxes: list                        # person boxes (xyxy, image pixels)
    gen: int = 0
    forced: bool = False               # bucket just left the pile / dump spot: run next, fuse at once (docs/25 §8.1)
    event: dict | None = None
    depth: np.ndarray | None = None    # relative depth, image resolution (filled by the engine)
    ms: float = 0.0


_shared: dict = {}


def shared_engine(ckpt: str, root: Path, **kw) -> "VGGTOmegaEngine":
    """One engine (model, thread) per process and settings: GUI profile switches reuse it (each session's monitor
    reset() bumps its generation, so the previous session's results are dropped)."""
    key = (ckpt, str(root), tuple(sorted(kw.items())))
    e = _shared.get(key)
    if e is None or e.error is not None:
        e = _shared[key] = VGGTOmegaEngine(ckpt, root, **kw)
    return e


class VGGTOmegaEngine:
    def __init__(self, ckpt: str, root: Path, *, res: int = 512, views: int = 4, keyframe_gap_s: float = 2.0,
                 sync: bool = False, device: str = "cuda"):
        """views: frames per inference (the current keyframe + up to views-1 earlier keyframes at least keyframe_gap_s
        apart; joint inference makes the depth more consistent). sync: run inline (reproducible evaluations)."""
        self.ckpt, self.root, self.res, self.views, self.gap, self.sync, self.device = ckpt, Path(root), res, views, keyframe_gap_s, sync, device
        self.model = None
        self.error: str | None = None
        self.gen = 0
        self.busy = False
        self.history: deque = deque(maxlen=max(views - 1, 0))      # (t, preprocessed tensor) of earlier keyframes
        self._job: DepthJob | None = None
        self._next: DepthJob | None = None                         # forced job waiting for the running one
        self._done: list[DepthJob] = []
        self._lock = threading.Lock()
        self.lock = threading.Lock()                               # the model: one inference at a time (drive recon shares it)
        self._wake = threading.Event()
        self.n_done, self.last_ms = 0, 0.0
        if not sync:
            threading.Thread(target=self._loop, daemon=True).start()

    # ------------------------------------------------------------------ model
    def _load(self):
        import torch

        sys.path.insert(0, str(self.root / "third_party" / "vggt-omega"))
        from vggt_omega.models import VGGTOmega

        with torch.device("meta"):                                  # no CPU copy of the 4.6 GB weights: built empty,
            m = VGGTOmega()                                         # then pointed at the memory-mapped checkpoint (a
        sd = torch.load(str(self.root / self.ckpt), map_location="cpu", weights_only=True, mmap=True)
        m.load_state_dict(sd, assign=True)                          # plain load left ~7 GB of fragmented RAM per load
        del sd                                                      # in the GUI; 3 profile switches -> OOM kill)
        m = m.to(self.device).eval()
        m.aggregator.to(torch.bfloat16)
        self.model = m

    def _prep(self, bgr: np.ndarray):
        import torch
        from vggt_omega.utils.load_fn import _balanced_target_shape

        h, w = bgr.shape[:2]
        th, tw = _balanced_target_shape(h / w, self.res, 16)
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (tw, th), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)

    def _infer(self, job: DepthJob) -> DepthJob:
        import torch

        if self.model is None:
            self._load()
        x = self._prep(job.image)
        frames = [xh for t_, xh in self.history if job.t - t_ >= 0] + [x]
        t0 = time.perf_counter()
        with self.lock, torch.inference_mode():
            p = self.model(torch.stack(frames).to(self.device))
        d = p["depth"][0, -1, ..., 0].float().cpu().numpy()
        h, w = job.image.shape[:2]
        job.depth = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        job.ms = (time.perf_counter() - t0) * 1e3
        self.last_ms = job.ms
        if not self.history or job.t - self.history[-1][0] >= self.gap:
            self.history.append((job.t, x))
        self.n_done += 1
        return job

    def _loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            try:
                job = self._infer(job)
            except Exception as e:                       # noqa: BLE001  the monitor falls back to radar / stereo
                self.error = repr(e)
                import traceback
                traceback.print_exc()
            with self._lock:
                if job.gen == self.gen and job.depth is not None:
                    self._done.append(job)
                if self._next is not None:                       # a forced keyframe arrived meanwhile: run it now
                    self._job, self._next = self._next, None
                    self._wake.set()
                else:
                    self.busy = False

    # ------------------------------------------------------------------ API
    def reset(self):
        """New segment (seek): drop queued / running results and the keyframe history (the model stays loaded)."""
        with self._lock:
            self.gen += 1
            self._job, self._next, self._done = None, None, []
            self.history.clear()

    def submit(self, job: DepthJob) -> bool:
        """Accepted when idle (async) or always (sync: inferred right here)."""
        job.gen = self.gen
        if self.error is not None:
            return False
        if self.sync:
            try:
                self._done.append(self._infer(job))
            except Exception as e:                       # noqa: BLE001
                self.error = repr(e)
                return False
            return True
        with self._lock:
            if self.busy:
                if job.forced:                                   # replaces any waiting forced job
                    self._next = job
                    return True
                return False
            self.busy, self._job = True, job
        self._wake.set()
        return True

    def poll(self) -> list[DepthJob]:
        with self._lock:
            out, self._done = self._done, []
        return out
