#!/usr/bin/env python3
"""Quick content survey of long bags (no GT): every --every seconds take one radar+image pair, run the 2D
detector and the radar scan statistics, and record object counts, ego speed and radar point count.
Writes <out>/<bag>.json and a contact sheet <out>/<bag>.jpg (one tile per sample, boxes drawn), so segments
with workers / machines / ego motion can be picked for full pipeline runs.

  python3 scripts/survey_bags.py --bag <dir> --calib configs/calib/xpanner03.json --config configs/default.yaml \
      --every 30 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.detection.yolo2d import Yolo2D  # noqa: E402
from rcmot.radar.scan_stats import ScanStatistician  # noqa: E402
from rosbags.highlevel import AnyReader  # noqa: E402

from rcmot.sensors.bag_source import _stamp, image_to_bgr, pc2_to_struct, radar_frame  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

COL = {"worker": (0, 255, 255), "heavy_equipment": (255, 128, 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--every", type=float, default=30.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--radar-topic")
    ap.add_argument("--image-topic")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tp = cfg["topics"]
    rt, it = a.radar_topic or tp["radar"], a.image_topic or tp["image"]
    cal = RigCalib.load(a.calib)
    dc = cfg["detector"]
    det = Yolo2D(str(ROOT / "models/yolo11s.pt"), dc["class_map"], imgsz=dc["imgsz"], conf=0.3)
    reader = AnyReader([Path(a.bag)])
    reader.open()
    conns = {c.topic: c for c in reader.connections}
    want = [conns[rt], conns[it]]
    t0, t1 = reader.start_time * 1e-9, reader.end_time * 1e-9
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.bag).name
    rows, tiles = [], []
    for s in np.arange(0.0, t1 - t0 - 1.0, a.every):
        st = ScanStatistician()
        got, scans = None, []
        for conn, t_ns, raw in reader.messages(connections=want, start=int((t0 + s) * 1e9)):   # one open reader
            if t_ns * 1e-9 > t0 + s + 4.0:
                break
            msg = reader.deserialize(raw, conn.msgtype)
            if conn.topic == rt:
                rf = radar_frame(pc2_to_struct(msg), _stamp(msg), cal)
                scans.append((rf, st.compute(rf.stamp, rf.xyz_radar, rf.v_r, rf.snr, rf.rcs, rf.peer)))
                continue
            si = _stamp(msg) - cal.camera_time_offset_s           # image header lags: pair by stamp, not arrival
            m = [(abs(si - rf.stamp), rf, stt) for rf, stt in scans[2:]]   # >= 3 scans: window statistics
            m = [x for x in m if x[0] <= 0.06]
            if m:
                dt_img, rf, stats = min(m, key=lambda x: x[0])
                got = (image_to_bgr(msg), rf, stats, dt_img)
                break
        if got is None:
            rows.append({"t": round(float(s), 1), "paired": False})
            continue
        img, rf, stats, dt_img = got
        ds = det(img)
        cnt = Counter(d.cls for d in ds)
        rows.append({"t": round(float(s), 1), "paired": True, "n_radar": int(len(rf.v_r)),
                     "ego_speed": round(float(np.linalg.norm(stats.v_sensor[:2])), 2), "ego_valid": bool(stats.ego_valid),
                     "sigma_v": round(float(stats.sigma_v), 3), "dt_img": round(float(dt_img), 3), "n_dynamic": int(stats.dynamic.sum()),
                     **{c: cnt.get(c, 0) for c in COL},
                     "worker_h_px": sorted(int(d.box[3] - d.box[1]) for d in ds if d.cls == "worker")})
        vis = img.copy()
        for d in ds:
            x1, y1, x2, y2 = d.box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), COL.get(d.cls, (200, 200, 200)), 3)
        uv, front = cal.project(rf.xyz)
        for (u, v), f, dyn in zip(uv, front, stats.dynamic):
            if f and 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                cv2.circle(vis, (int(u), int(v)), 4, (0, 0, 255) if dyn else (0, 200, 0), -1)
        vis = cv2.resize(vis, (480, int(480 * vis.shape[0] / vis.shape[1])))
        r = rows[-1]
        cv2.putText(vis, f"{s:.0f}s W{r['worker']} H{r['heavy_equipment']} v{r['ego_speed']}", (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(vis)
        print(name, r, flush=True)
    json.dump({"bag": a.bag, "calib": a.calib, "rows": rows}, open(out / f"{name}.json", "w"), indent=1)
    if tiles:
        h, w = tiles[0].shape[:2]
        tiles = [cv2.resize(t, (w, h)) for t in tiles]
        per = 6
        sheet = [np.hstack(tiles[i:i + per] + [np.zeros_like(tiles[0])] * (per - len(tiles[i:i + per])))
                 for i in range(0, len(tiles), per)]
        cv2.imwrite(str(out / f"{name}.jpg"), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
