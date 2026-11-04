#!/usr/bin/env python3
"""Drop every ref3d that does not project inside its own box (width +-50 %, 1 m height) - a reference the camera box
cannot be looking at is wrong, and no reference is better than a wrong one.  Hand-placed (src manual) references are
kept and listed.  PYTHONPATH=src python3 scripts/label_ref_wedge.py data/labels/<seg> [--apply]"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.gt import GT
from rcmot.utils.calib import RigCalib

ap = argparse.ArgumentParser(); ap.add_argument("seg"); ap.add_argument("--apply", action="store_true"); a = ap.parse_args()
seg = Path(a.seg); seg = seg if seg.is_dir() else seg.parent
g = GT(seg / "gt.json"); cal = RigCalib.load(str(ROOT / g.meta["calib"]))
dropped, manual = [], []
for i, f in enumerate(g.frames):
    for b in f["boxes"]:
        rf = b.get("ref3d") or {}
        if rf.get("xy_level") is None:
            continue
        x1, y1, x2, y2 = b["xyxy"]; w = x2 - x1
        uv, fr = cal.project(np.array([[rf["xy_level"][0], rf["xy_level"][1], 1.0]]))
        if fr[0] and np.isfinite(uv[0, 0]) and x1 - 0.5 * w <= uv[0, 0] <= x2 + 0.5 * w:
            continue
        if rf.get("src") == "manual":
            manual.append((i, b["id"])); continue
        dropped.append((i, b["id"], rf.get("src")))
        if a.apply:
            g.set_ref3d(i, b["id"], {"range_m": None, "xy_level": None, "n_lidar": 0, "quality": "none", "src": "wedge_dropped"})
print(f"{seg.name}: outside own box: {len(dropped)} dropped {dropped[:12]}{'...' if len(dropped) > 12 else ''}; manual kept {manual}")
if a.apply and dropped:
    g.add_provenance("audit", "agent", "scripts/label_ref_wedge.py", note=f"{len(dropped)} ref3d outside their box set to none: {dropped[:20]}")
    g.save(backup=True)
