#!/usr/bin/env python3
"""Sample the camera of a bag at ~N Hz (read-only, streaming) and count YOLO detections per frame
(person / vehicle-like classes) -> CSV (bag-relative seconds). Used to find 'ego moving + people/equipment' windows.

usage: PYTHONPATH=src python3 scripts/agmo_yolo_scan.py BAG OUT.csv [--topic /oak/rgb/image_raw/compressed] [--hz 2]
"""
import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from ultralytics import YOLO

from rcmot.sensors.bag_source import image_to_bgr

VEH = {"car", "truck", "bus", "motorcycle", "bicycle", "tractor"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("out")
    ap.add_argument("--topic", default="/oak/rgb/image_raw/compressed")
    ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--weights", default="models/yolo11s.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    a = ap.parse_args()
    model = YOLO(a.weights)
    names = model.names
    rows = []
    with AnyReader([Path(a.bag)]) as r:
        conns = [c for c in r.connections if c.topic == a.topic]
        t0 = r.start_time
        nxt = 0.0
        for c, t_ns, raw in r.messages(connections=conns):
            ts = (t_ns - t0) * 1e-9
            if ts < nxt:
                continue
            nxt = ts + 1.0 / a.hz
            img = image_to_bgr(r.deserialize(raw, c.msgtype))
            res = model.predict(img, conf=a.conf, verbose=False, imgsz=640)[0]
            cls = [names[int(k)] for k in res.boxes.cls.cpu().numpy()]
            conf = res.boxes.conf.cpu().numpy()
            hgt = (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1]).cpu().numpy() if len(cls) else np.zeros(0)
            pp = [i for i, k in enumerate(cls) if k == "person"]
            vv = [i for i, k in enumerate(cls) if k in VEH]
            rows.append((ts, len(pp), len(vv), max([conf[i] for i in pp], default=0), max([hgt[i] for i in pp], default=0),
                         max([hgt[i] for i in vv], default=0)))
            if len(rows) % 200 == 0:
                print(f"{ts:.0f}s", flush=True)
    np.savetxt(a.out, np.array(rows), delimiter=",", fmt="%.3f",
               header="t_rel_s,n_person,n_vehicle,max_person_conf,max_person_h_px,max_vehicle_h_px")
    print("saved", a.out, len(rows))


if __name__ == "__main__":
    main()
