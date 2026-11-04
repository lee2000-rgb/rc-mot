"""2D detector wrapper (Ultralytics YOLO .pt / TensorRT .engine).

COCO classes are remapped to the project classes; see configs/default.yaml `detector.class_map`
(person -> worker, truck/bus/car -> heavy_equipment by default). For Orin export with
scripts/export_trt.sh (FP16, imgsz 640 or 960).
"""
from __future__ import annotations

import numpy as np

from rcmot.types import Det2D


class Yolo2D:
    def __init__(self, weights: str, class_map: dict[str, str], *, imgsz: int = 960, conf: float = 0.1,
                 iou: float = 0.6, device: str | int | None = None, half: bool = True):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(weights)                                  # "-seg" weights -> instance masks too
        self.seg = getattr(self.model, "task", "detect") == "segment"
        self.class_map = class_map
        half = half and torch.cuda.is_available()                   # FP16 on CPU is extremely slow
        self.imgsz, self.conf, self.iou, self.device, self.half = imgsz, conf, iou, device, half
        try:                                                        # ultralytics >= 8.4.1xx renamed half -> quantize
            from ultralytics.cfg import DEFAULT_CFG_DICT
            self._prec = {"quantize": "fp16" if half else None} if "quantize" in DEFAULT_CFG_DICT else {"half": half}
        except Exception:
            self._prec = {"half": half}
        names = self.model.names
        self._names = names
        self.set_project_classes(None)

    def set_project_classes(self, only: set | None) -> None:
        """Detect only these project classes (e.g. {"worker"}: people only, heavy equipment off); None = all mapped."""
        self.only = None if only is None else set(only)
        self._keep = [i for i, n in self._names.items() if n in self.class_map and (self.only is None or self.class_map[n] in self.only)]

    def _to_dets(self, r) -> list[Det2D]:
        b = r.boxes
        if b is None or len(b) == 0:
            return []
        xyxy, sc, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        dets = [Det2D(self.class_map[self._names[c]], float(s), x) for x, s, c in zip(xyxy, sc, cl)]
        if getattr(r, "masks", None) is not None:
            M = r.masks.data.cpu().numpy() > 0.5                   # full image size (retina masks)
            for d, m in zip(dets, M):
                x1, y1 = max(int(d.box[0]), 0), max(int(d.box[1]), 0)
                # copy: a view would keep the whole (n_dets, H, W) mask stack alive in every cached GUI frame
                # (OAK 2016x1512: 3 MB per detection per frame -> 25 GB RSS after a few minutes of replay)
                d.mask = m[y1:int(np.ceil(d.box[3])), x1:int(np.ceil(d.box[2]))].copy()
        return dets

    def __call__(self, bgr: np.ndarray) -> list[Det2D]:
        r = self.model.predict(bgr, imgsz=self.imgsz, conf=self.conf, iou=self.iou, classes=self._keep,
                               device=self.device, verbose=False, retina_masks=self.seg, **self._prec)[0]
        return self._to_dets(r)

    def detect_batch(self, crops: list[np.ndarray], *, imgsz: int = 640, conf: float = 0.1) -> list[list[Det2D]]:
        """One batched call on several crops (Radar-Gated Recovery); boxes are in crop pixels."""
        if not crops:
            return []
        rs = self.model.predict(crops, imgsz=imgsz, conf=conf, iou=self.iou, classes=self._keep,
                                device=self.device, verbose=False, **self._prec)
        return [self._to_dets(r) for r in rs]
