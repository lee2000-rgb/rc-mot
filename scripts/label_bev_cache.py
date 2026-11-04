#!/usr/bin/env python3
"""Per-frame LiDAR top-down (BEV) point cache for the review GUI's 3D panel (docs/13).

For every GT frame the LiDAR points within +-dt of the image time (one sweep; three sweeps while the ego is static) are expressed in the level frame of that moment
(the ref3d.xy_level frame): scans in the os_sensor frame -> level frame (calib), de-skewed about the swing axis with
the IMU rate for the excavator mounts (same model as label_lidar_world_audit.py), ground height = 10th percentile of
z within 25 m.  Points are split into ground (height < 0.35 m, 0.4 m cells) and above-ground (0.35-3.5 m, 0.1 m
cells) and stored as int16 centimetres + uint8 height (0.05 m steps):

  cache/bev_points.npz: key, xy (N,2) int16 cm, h (N,) uint8, off (F+1,) int64, n_above (F,) int32, zg (F,) float32,
                        rxy (M,2) int16 cm, rz (M,) float16 m, rv (M,) float16 raw radial velocity (+ away), roff (F+1,) int64

Frame f = xy[off[f]:off[f+1]], of which the first n_above[f] points are above ground; its radar scan (the one the GT
frame was extracted with, same stamp; level frame, not ego-compensated) = rxy[roff[f]:roff[f+1]].  gt.json is not touched.

  PYTHONPATH=src python3 scripts/label_bev_cache.py data/labels/<seg> [--dt 0.05] [--r-max 60] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rcmot.calib.camera_lidar import lidar_clock_offset  # noqa: E402
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.labeling.gt import GT  # noqa: E402
from rcmot.sensors.bag_source import open_bag, pc2_to_struct  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402
from rcmot.utils.rig_extrinsics import sensor_to_level  # noqa: E402

from rcmot.sensors.bag_source import iterate_bag  # noqa: E402

from label_lidar_world_audit import LEVER, SWEEP, deskew_swing, gyro_heading  # noqa: E402

CLOCK_DT = 0.02          # image time -> LiDAR sensor clock (+20 ms, label_lidar_world_audit --clock-dt)
H_GROUND, H_MAX = 0.35, 3.5
CELL_GROUND, CELL_ABOVE = 0.4, 0.1
MAX_ABOVE, MAX_GROUND = 12000, 4000


def read_lidar_imu(bag: Path, lidar_topic: str, imu_topic: str | None, t0: float, t1: float, r_max: float):
    scans, gyro_t, gyro = [], [], []
    with open_bag(bag) as r:
        cs = {c.topic: c for c in r.connections}
        s0 = r.start_time * 1e-9
        off = lidar_clock_offset(r, cs[lidar_topic])
        tf = {}
        if "/tf_static" in cs:
            from rcmot.calib.camera_lidar import _tf_static
            tf = _tf_static(r, cs)
        want = [cs[lidar_topic]] + ([cs[imu_topic]] if imu_topic in cs else [])
        for conn, tn, raw in r.messages(connections=want, start=int((s0 + t0) * 1e9), stop=int((s0 + t1) * 1e9)):
            m = r.deserialize(raw, conn.msgtype)
            hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if conn.topic == imu_topic:
                gyro_t.append(hs)
                gyro.append(m.angular_velocity.z)
            else:
                a = pc2_to_struct(m)
                xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)
                rr = np.linalg.norm(xyz, axis=1)
                ok = np.isfinite(rr) & (rr > 1.0) & (rr < r_max)
                R, t = lidar_to_sensor(tf, m.header.frame_id)
                P = xyz[ok] @ R.T + t
                tp = hs + a["t"][ok].astype(np.float64) * 1e-9
                scans.append((hs, P.astype(np.float32), tp))
    return scans, np.array(gyro_t), np.array(gyro), float(off)


def voxel_first(P, size):
    _, i = np.unique(np.floor(P[:, :2] / size).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(i)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("--dt", type=float, default=0.05, help="points within +-dt s of the image time [s] (one 10 Hz sweep)")
    ap.add_argument("--dt-static", type=float, default=0.15, help="window while the ego is static (3 sweeps: more points on a standing person)")
    ap.add_argument("--r-max", type=float, default=60.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-radar", action="store_true", help="skip the radar scans")
    a = ap.parse_args()
    seg = Path(a.seg)
    seg = seg if seg.is_dir() else seg.parent
    g = GT(seg / "gt.json")
    m = g.meta
    if not m.get("lidar_topic"):
        print(f"{seg.name}: no LiDAR topic, nothing to cache")
        return
    out = seg / "cache" / "bev_points.npz"
    bag = Path(m["bag"]) if str(m["bag"]).startswith("/") else ROOT / m["bag"]
    key = json.dumps([str(bag), m["start"], m["end"], len(g.frames), a.dt, a.dt_static, a.r_max, 5, not a.no_radar])
    if out.exists() and not a.force and str(np.load(out, allow_pickle=True)["key"]) == key:
        print(f"{seg.name}: cache up to date ({out})")
        return
    cal = RigCalib.load(str(ROOT / m["calib"]))
    R_lev, t_lev = sensor_to_level(json.load(open(ROOT / m["calib"])))
    tic = time.time()
    scans, gyro_t, gz, off = read_lidar_imu(bag, m["lidar_topic"], m.get("imu_topic"), m["start"] - 1.0, m["end"] + 1.0, a.r_max)
    print(f"{seg.name}: {len(scans)} scans, {len(gyro_t)} gyro samples, read {time.time() - tic:.0f} s", flush=True)
    axis_xy = None
    if m.get("mount") in LEVER and len(gyro_t) > 10:
        _, bias, _ = gyro_heading(gyro_t, gz)
        gz = gz - bias
        axis_xy = -(np.array(LEVER[m["mount"]], float) - cal.t_lr[:2])     # swing axis in the level frame
    hs_all = np.array([s[0] for s in scans])
    xy_all, h_all, offs, n_above, zgs = [], [], [0], [], []
    for i, f in enumerate(g.frames):
        t_ref = f["stamp"] - off + CLOCK_DT
        dt = a.dt_static if g.ego_state_at(i) in ("static", None) else a.dt
        near = np.where(np.abs(hs_all + 0.5 * SWEEP - t_ref) < dt + 0.2)[0]
        P, T = [], []
        for k in near:
            hs, Pk, tp = scans[k]
            sel = np.abs(tp - t_ref) <= dt
            P.append(Pk[sel])
            T.append(tp[sel])
        P = np.concatenate(P) if P else np.zeros((0, 3), np.float32)
        T = np.concatenate(T) if T else np.zeros(0)
        P = P.astype(np.float64) @ R_lev.T + t_lev
        if axis_xy is not None and len(P):
            P = deskew_swing(P, T, t_ref, gyro_t, gz, axis_xy)
        r = np.linalg.norm(P[:, :2], axis=1)
        nearp = P[(r > 2.0) & (r < 25.0)]
        zg = float(np.percentile(nearp[:, 2], 10)) if len(nearp) > 50 else -1.6
        h = P[:, 2] - zg
        above = voxel_first(P[(h >= H_GROUND) & (h < H_MAX) & (r > 1.5)], CELL_ABOVE)
        ground = voxel_first(P[(h < H_GROUND) & (h > -1.0) & (r > 1.5)], CELL_GROUND)
        rng = np.random.default_rng(0)
        if len(above) > MAX_ABOVE:
            above = above[rng.choice(len(above), MAX_ABOVE, replace=False)]
        if len(ground) > MAX_GROUND:
            ground = ground[rng.choice(len(ground), MAX_GROUND, replace=False)]
        Q = np.concatenate([above, ground])
        xy_all.append(np.clip(np.round(Q[:, :2] * 100), -32000, 32000).astype(np.int16))
        h_all.append(np.clip(np.round((Q[:, 2] - zg) / 0.05), 0, 255).astype(np.uint8))
        offs.append(offs[-1] + len(Q))
        n_above.append(len(above))
        zgs.append(zg)
    # ---- radar: the scan each GT frame was extracted with (same pairing as label_extract: iterate_bag stamps)
    rxy, rz, rv, roffs = [], [], [], [0]
    if not a.no_radar and m.get("config"):
        tic2 = time.time()
        cfg = yaml.safe_load(open(ROOT / m["config"]))
        tp = cfg["topics"]
        stamps = np.array([f["stamp"] for f in g.frames])
        scans = {}
        try:
            for _, rf, _ in iterate_bag(bag, cal, tp["radar"], tp["image"], tp.get("imu"), start=max(m["start"] - 0.5, 0.0),
                                        duration=m["end"] - m["start"] + 1.0):
                k = int(np.argmin(np.abs(stamps - rf.stamp)))
                d = abs(stamps[k] - rf.stamp)
                if d < 0.12 and (k not in scans or d < scans[k][0]):
                    scans[k] = (d, rf.xyz[:, :2].copy(), rf.v_r.copy(), rf.xyz[:, 2].copy())
        except Exception as ex:                                  # noqa: BLE001 - radar is optional in the panel
            print(f"{seg.name}: radar not cached: {ex}")
        for k in range(len(g.frames)):
            if k in scans:
                _, q, v, zz = scans[k]
                rxy.append(np.clip(np.round(q * 100), -32000, 32000).astype(np.int16))
                rv.append(v.astype(np.float16))
                rz.append(zz.astype(np.float16))
            roffs.append(roffs[-1] + (len(scans[k][1]) if k in scans else 0))
        print(f"{seg.name}: radar scans matched {len(scans)}/{len(g.frames)}, pts/frame median "
              f"{int(np.median([len(s[1]) for s in scans.values()])) if scans else 0} ({time.time() - tic2:.0f} s)", flush=True)
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, key=key, xy=np.concatenate(xy_all), h=np.concatenate(h_all), off=np.array(offs, np.int64),
                        n_above=np.array(n_above, np.int32), zg=np.array(zgs, np.float32), deskew=axis_xy is not None,
                        rxy=np.concatenate(rxy) if rxy else np.zeros((0, 2), np.int16), rv=np.concatenate(rv) if rv else np.zeros(0, np.float16),
                        rz=np.concatenate(rz) if rz else np.zeros(0, np.float16),
                        roff=np.array(roffs, np.int64))
    print(f"{seg.name}: {len(g.frames)} frames, above-ground pts/frame median {int(np.median(n_above))}, ground z median "
          f"{np.median(zgs):.2f} m, deskew={axis_xy is not None} -> {out} ({out.stat().st_size / 1e6:.1f} MB, {time.time() - tic:.0f} s)")


if __name__ == "__main__":
    main()
