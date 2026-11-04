"""Appearance embeddings for ReID (docs/09 §5): OSNet trained on MSMT17, L2-normalised features."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from rcmot.calib.camera_lidar import enhance_dark

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
MODELS = {"osnet_x0_25": ("osnet", "osnet_x0_25"), "osnet_ain_x1_0": ("osnet_ain", "osnet_ain_x1_0")}


class ReID:
    def __init__(self, weights: str, arch: str | None = None, device: str | None = None, size=(256, 128),
                 brighten: bool = True):
        self.size, self.brighten = size, brighten
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(weights).endswith(".engine"):                 # TensorRT engine of the same network (docs/18, Orin)
            from rcmot.reid.trt_runner import TrtRunner

            self.model = TrtRunner(str(weights))
            self.device = "cuda"
            return
        name = arch or next(k for k in MODELS if Path(weights).name.startswith(k))
        mod, fn = MODELS[name]
        module = __import__(f"rcmot.reid.{mod}", fromlist=[fn])
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        n_cls = sd["classifier.weight"].shape[0] if "classifier.weight" in sd else 1000
        self.model = getattr(module, fn)(num_classes=n_cls, pretrained=False, loss="softmax")
        self.model.load_state_dict(sd, strict=False)
        self.model.eval().to(self.device)

    def _prep(self, crop: np.ndarray) -> np.ndarray:
        if self.brighten:
            crop = enhance_dark(crop)
        img = cv2.resize(crop, (self.size[1], self.size[0]))[:, :, ::-1] / 255.0
        return ((img - MEAN) / STD).transpose(2, 0, 1).astype(np.float32)

    @torch.no_grad()
    def __call__(self, crops: list) -> list:
        """One embedding (or None for an empty crop) per crop."""
        ok = [i for i, c in enumerate(crops) if c is not None and c.size and min(c.shape[:2]) >= 8]
        out = [None] * len(crops)
        if not ok:
            return out
        x = torch.from_numpy(np.stack([self._prep(crops[i]) for i in ok])).to(self.device)
        f = self.model(x).cpu().numpy()
        f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-9
        for k, i in enumerate(ok):
            out[i] = f[k]
        return out
