"""Reflector height check: radar and LiDAR z of the reflector at each spot, in the calib's level frame (ground = 0).

  python3 scripts/reflector_height_check.py <cal bag> <calib.json> ["0.5-22:8.5,5.0;31-50:20.0,0.5;56-80.5:7.5,-5.0"]
spots = still window (s from the bag start) : approximate level-frame x,y of the reflector (docs/22 §4)."""
import json, sys, numpy as np
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation
ts = get_typestore(Stores.ROS2_HUMBLE)
B = sys.argv[1]; C = json.load(open(sys.argv[2]))
SPEC = sys.argv[3] if len(sys.argv) > 3 else "0.5-22:8.5,5.0;31-50:20.0,0.5;56-80.5:7.5,-5.0"   # hdx_cal_20261026_104647
SPOTS = [(float(w.split("-")[0]), float(w.split("-")[1]), tuple(float(v) for v in xy.split(",")))
         for w, xy in (s.split(":") for s in SPEC.split(";"))]
R = np.array(C["T_base_level_radar"]["R"]); t = np.array(C["T_base_level_radar"]["t"])
n = np.array(C["extrinsics"]["ground_normal"]); h = C["extrinsics"]["sensor_height"]
Rg = Rotation.align_vectors([[0, 0, 1]], [n])[0].as_matrix()          # lidar (os_sensor) -> level rotation
def pc(m, names):
    off = {f.name: f.offset for f in m.fields}; a = np.frombuffer(m.data, np.uint8).reshape(-1, m.point_step)
    return np.c_[[a[:, off[k]:off[k] + 4].copy().view(np.float32).ravel() for k in names]].T
rad = {i: [] for i in range(3)}; lid = {i: [] for i in range(3)}
with Reader(B) as r:
    t0 = r.start_time
    cons = [c for c in r.connections if c.topic in ("/radar_mid/driver/locations", "/ouster/points")]
    for c, tt, raw in r.messages(connections=cons):
        s = (tt - t0) * 1e-9; m = ts.deserialize_cdr(raw, c.msgtype)
        for i, (a, b, xy) in enumerate(SPOTS):
            if not a + 2 <= s <= b - 2: continue
            if c.topic.startswith("/radar"):
                P = pc(m, ["x", "y", "z", "radar_cross_section", "radial_velocity", "elevation_angle", "radial_distance"])
                L = P[:, :3] @ R.T + t
                k = (np.hypot(L[:, 0] - xy[0], L[:, 1] - xy[1]) < 1.0) & (P[:, 3] > 3) & (np.abs(P[:, 4]) < 0.3)
                if k.any():
                    j = np.argmax(P[k, 3]); rad[i].append(np.r_[L[k][j], P[k][j, 3], np.degrees(P[k][j, 5]), P[k][j, 6]])
            else:
                P = pc(m, ["x", "y", "z"]); P = P[np.isfinite(P).all(1) & (np.linalg.norm(P, axis=1) > 1)]
                if m.header.frame_id == "os_lidar": P = P * [-1, -1, 1] + [0, 0, 0.03618]      # os_lidar -> os_sensor
                L = P @ Rg.T; L[:, 2] += h
                lid[i].append(L[np.hypot(L[:, 0] - xy[0], L[:, 1] - xy[1]) < 2.5])
print("spot         range | radar: n  z_level p50 [p10,p90]  elev_deg  rcs | LiDAR: object z [min,max], local ground, obj xy")
for i, (a, b, xy) in enumerate(SPOTS):
    A = np.array(rad[i]); L = np.vstack(lid[i])
    g = np.percentile(L[:, 2], 5)                                       # local ground near the spot
    o = L[L[:, 2] > g + 0.15]                                             # above-ground points
    if len(A): c0 = np.median(A[:, :2], axis=0)
    o = o[np.hypot(o[:, 0] - c0[0], o[:, 1] - c0[1]) < 0.6]
    print(f"{xy} {np.hypot(*c0):5.1f} m | n{len(A):4d} z {np.median(A[:,2]):+.3f} [{np.percentile(A[:,2],10):+.2f},{np.percentile(A[:,2],90):+.2f}]  "
          f"elev {np.median(A[:,4]):+.2f}  rcs {np.median(A[:,3]):4.1f} | LiDAR obj z [{np.percentile(o[:,2],2):+.2f},{np.percentile(o[:,2],98):+.2f}] "
          f"top-cluster p90 {np.percentile(o[:,2],90):+.2f}, ground {g:+.3f}, xy ({np.median(o[:,0]):.2f},{np.median(o[:,1]):+.2f}) n{len(o)}")
