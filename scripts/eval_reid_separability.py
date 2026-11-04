#!/usr/bin/env python3
"""ReID separability on our own data, no labels needed (docs/09 §5.4).

Runs the tracker (without ReID) over bag windows, keeps CLEAN person crops of confirmed tracks (score >= 0.6,
box overlapping no other box, every `--every` frames), embeds them with each ReID model and reports the
AUC of "same track" vs "different tracks (same frame window)" cosine distances, and the median distances.
Track-ID errors of the tracker make this a slightly pessimistic estimate.

  python3 scripts/eval_reid_separability.py --bag <bag> --calib <json> --start 60 --duration 40 --tag day
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.detection.yolo2d import Yolo2D  # noqa: E402
from rcmot.pipeline import Pipeline  # noqa: E402
from rcmot.reid.extractor import ReID  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.tracking.tracker import iou_xyxy  # noqa: E402
from rcmot.types import TrackState  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

MODELS = ["models/reid/osnet_x0_25_msmt17.pth", "models/reid/osnet_ain_x1_0_msmt17.pth"]


def auc(pos, neg):
    """P(d_same < d_diff) (Mann-Whitney)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.r_[pos, neg]
    ranks = allv.argsort().argsort() + 1
    return 1.0 - (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/hdx_boryeong.yaml"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg.setdefault("recovery", {})["enable"] = False
    cal = RigCalib.load(a.calib)
    dc = cfg["detector"]
    pipe = Pipeline(Yolo2D(str(ROOT / dc["weights"]), dc["class_map"], imgsz=dc["imgsz"], conf=dc["conf"]), cal, cfg)
    crops = []                                     # (frame_idx, tid, crop)
    for k, (img, rf, yaw) in enumerate(iterate_bag(a.bag, cal, cfg["topics"]["radar"], cfg["topics"]["image"], None,
                                                   start=a.start, duration=a.duration)):
        res = pipe.step(img, rf, yaw)
        if k % a.every:
            continue
        boxes = np.array([t.box2d for t in res.tracks if t.box2d is not None]).reshape(-1, 4)
        for t in res.tracks:
            if t.cls != "worker" or t.state != TrackState.CONFIRMED or t.time_since_update > 0 or t.score < 0.6:
                continue
            others = boxes[[not np.array_equal(b, t.box2d) for b in boxes]]
            if len(others) and iou_xyxy(t.box2d[None], others).max() > 0.05:
                continue
            x1, y1, x2, y2 = t.box2d.astype(int)
            c = img[max(y1, 0):y2, max(x1, 0):x2]
            if c.size and c.shape[0] >= 48:
                crops.append((k, t.tid, c.copy()))
    print(f"{len(crops)} clean crops of {len({c[1] for c in crops})} tracks", file=sys.stderr)
    res_all = {"tag": a.tag, "bag": Path(a.bag).name, "n_crops": len(crops), "n_tracks": len({c[1] for c in crops})}
    for w in MODELS:
        reid = ReID(str(ROOT / w))
        E = np.array([e for b in range(0, len(crops), 32) for e in reid([c[2] for c in crops[b:b + 32]])])
        same, diff = [], []
        for i in range(len(crops)):
            for j in range(i + 1, len(crops)):
                if abs(crops[i][0] - crops[j][0]) > 150:          # compare within ~10 s
                    continue
                d = 1.0 - float(E[i] @ E[j])
                (same if crops[i][1] == crops[j][1] else diff).append(d)
        name = Path(w).stem
        res_all[name] = {"auc": float(auc(same, diff)) if same and diff else None, "n_same": len(same), "n_diff": len(diff),
                         "median_same": float(np.median(same)) if same else None,
                         "median_diff": float(np.median(diff)) if diff else None}
    print(json.dumps(res_all, indent=1))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res_all) + "\n")


if __name__ == "__main__":
    main()
