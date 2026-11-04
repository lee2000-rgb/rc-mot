"""Draft labels: strong-detector boxes linked into tracklets offline (IoU + centre distance, Hungarian, gaps)."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None] - inter, 1e-9)


def link(frames: list[list[dict]], max_gap: int = 10, min_iou: float = 0.1, max_center: float = 1.5) -> list[list[dict]]:
    """frames[k] = [{cls, xyxy, conf}] -> same structure with 'id' set. Track ends are matched to new boxes by
    a cost of (1 - IoU) + normalised centre distance (in box heights, scaled by the gap length)."""
    tracks: list[dict] = []           # {id, cls, box, last}
    nid = 1
    for k, boxes in enumerate(frames):
        live = [t for t in tracks if k - t["last"] <= max_gap]
        C = np.full((len(live), len(boxes)), 1e6)
        if live and boxes:
            I = iou([t["box"] for t in live], [b["xyxy"] for b in boxes])
            for i, t in enumerate(live):
                tb = np.array(t["box"])
                h = max(tb[3] - tb[1], 10.0)
                gap = k - t["last"]
                for j, b in enumerate(boxes):
                    if b["cls"] != t["cls"]:
                        continue
                    bb = np.array(b["xyxy"])
                    d = np.linalg.norm((tb[:2] + tb[2:]) / 2 - (bb[:2] + bb[2:]) / 2) / h
                    lim = max_center * (1 + 0.3 * (gap - 1))
                    if I[i, j] < min_iou and d > lim:
                        continue
                    C[i, j] = (1 - I[i, j]) + d / lim + 0.05 * (gap - 1)
        r, c = linear_sum_assignment(C) if C.size else ([], [])
        used = set()
        for i, j in zip(r, c):
            if C[i, j] < 1e5:
                boxes[j]["id"] = live[i]["id"]
                live[i]["box"], live[i]["last"] = boxes[j]["xyxy"], k
                used.add(j)
        for j, b in enumerate(boxes):
            if j not in used:
                b["id"] = nid
                tracks.append({"id": nid, "cls": b["cls"], "box": b["xyxy"], "last": k})
                nid += 1
    return frames
