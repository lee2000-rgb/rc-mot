#!/usr/bin/env python3
"""Extract a labelling segment (docs/13): radar-paired frames at ~5 Hz (every 3rd), strong-detector draft boxes,
offline-linked draft ids -> data/labels/<name>/{frames/*.jpg, gt.json}.

  python3 scripts/label_extract.py --name day_halt_short_085 --bag data/bags/hdx_boryeong_pg/20251125_4R_bosch_halt-short_LC \
      --calib configs/calib/hdx_boryeong_pg_day1125.json --start 85 --duration 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.draft import link  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

CLS = {"person": "worker", "truck": "heavy_equipment", "car": "heavy_equipment", "bus": "heavy_equipment"}


def _rel(p) -> str:
    """Project-relative path when inside the project, else absolute (bags on the external drive)."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/hdx_boryeong.yaml"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--weights", default=str(ROOT / "models/yolo11x.pt"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.15)
    a = ap.parse_args()
    from ultralytics import YOLO

    out = ROOT / "data/labels" / a.name
    (out / "frames").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    cal = RigCalib.load(a.calib)
    tp = cfg["topics"]
    model = YOLO(a.weights)
    names = model.names
    frames, dets = [], []
    clock_off = None                                               # header -> bag clock offset used (Continental)
    for k, (img, rf, _) in enumerate(iterate_bag(a.bag, cal, tp["radar"], tp["image"], None, start=a.start,
                                                 duration=a.duration)):
        if k % a.every:
            continue
        clock_off = rf.clock_offset_s
        fn = f"frames/{len(frames):04d}.jpg"
        cv2.imwrite(str(out / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        r = model(img, imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        boxes = [{"cls": CLS[names[int(c)]], "xyxy": [round(float(v), 1) for v in b], "conf": round(float(s), 3)}
                 for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())
                 if names[int(c)] in CLS]
        frames.append({"stamp": round(rf.stamp, 6), "bag_t": round(float(rf.bag_t), 6), "file": fn, "boxes": []})
        dets.append(boxes)
    link(dets)
    ids = {}
    for f, bs in zip(frames, dets):
        for b in bs:
            f["boxes"].append({"id": b["id"], "cls": b["cls"], "xyxy": b["xyxy"], "flags": [], "src": "det",
                               "conf": b["conf"]})
            ids.setdefault(b["id"], {"cls": b["cls"], "desc": ""})
    gt = {"meta": {"bag": _rel(a.bag), "calib": str(Path(a.calib)),
                   "config": _rel(a.config), "start": a.start,
                   "end": a.start + a.duration, "rate_hz": round(len(frames) / a.duration, 2), "every": a.every,
                   "detector": Path(a.weights).name, "imgsz": a.imgsz, "conf": a.conf,
                   "stamps": [f["stamp"] for f in frames], "stage": "draft",
                   "radar_clock_offset_s": clock_off},           # eval tools pass it back to iterate_bag (same image pairing)
          "ids": {str(k): v for k, v in sorted(ids.items())}, "frames": frames,
          "review": {"ids": {str(k): "draft" for k in ids}, "frames": {}}, "notes": []}
    json.dump(gt, open(out / "gt.json", "w"), indent=1)
    print(f"{out}: {len(frames)} frames, {len(ids)} draft ids, {sum(len(f['boxes']) for f in frames)} boxes")


if __name__ == "__main__":
    main()
