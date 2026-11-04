"""Rig-to-ground per bag: LiDAR RANSAC ground plane on one frame every ~step s -> sensor height, roll, pitch (os_sensor).
The plane is fitted in a frame levelled by the first scan (ground_plane_near), as field_calib.py reground does."""
import json, sys
from pathlib import Path
import numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.calib.common import ground_plane_near
ts = get_typestore(Stores.ROS2_HUMBLE); step = 10.0
for bag in sys.argv[2:]:
    rows, prior = [], None
    with Reader(bag) as r:
        t0 = r.start_time; nxt = 0.0
        cons = [c for c in r.connections if c.topic == "/ouster/points"]
        for c, t, raw in r.messages(connections=cons):
            s = (t - t0) * 1e-9
            if s < nxt: continue
            nxt = s + step; m = ts.deserialize_cdr(raw, c.msgtype)
            off = {f.name: f.offset for f in m.fields}; a = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
            P = np.c_[[a[:, off[k]:off[k] + 4].copy().view(np.float32).ravel() for k in "xyz"]].T
            P = P[np.isfinite(P).all(1) & (np.linalg.norm(P, axis=1) > 1.5)]
            if m.header.frame_id == "os_lidar": P = P * [-1, -1, 1] + [0, 0, 0.03618]
            if prior is None:                              # first scan: twice from level (steep mounts)
                prior = ground_plane_near(P, ground_plane_near(P, [0, 0, 1])[0])[0]
            n, h, k = ground_plane_near(P, prior)
            rows.append((s, h, np.degrees(np.arctan2(n[1], n[2])), np.degrees(np.arctan2(-n[0], n[2])), k))
    A = np.array(rows)
    out = {"bag": bag.rstrip("/").split("/")[-1], "n": len(A), "height_m": round(float(np.median(A[:, 1])), 3),
           "height_p10_p90": [round(float(x), 3) for x in np.percentile(A[:, 1], [10, 90])],
           "roll_deg": round(float(np.median(A[:, 2])), 2), "pitch_deg": round(float(np.median(A[:, 3])), 2),
           "pitch_p10_p90": [round(float(x), 2) for x in np.percentile(A[:, 3], [10, 90])], "inliers_p50": int(np.median(A[:, 4])),
           "series": [[round(float(v), 3) for v in r_] for r_ in A]}
    json.dump(out, open(f"{sys.argv[1]}/{out['bag']}.ground.json", "w"))
    print(out["bag"], {k: v for k, v in out.items() if k not in ("series", "bag")}, flush=True)
