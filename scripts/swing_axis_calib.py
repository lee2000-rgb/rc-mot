"""Swing-axis tilt in the level frame of a calib (docs/25 §3.1) -> calib JSON "swing_axis_level".

The level frame of a field calib is the LiDAR ground plane in front of the rig (field_calib.py reground). On a sloped
site that plane is tilted against the excavator's swing axis, and the work map, which turns the level frame about its
own z, then tilts with the heading: on 155530 the same ground read 0.25-0.30 m lower 73 deg to the right of the start
heading and 0.3 m higher to the left. The tilt k = (axis - z) (small angles, level frame) is found from LiDAR height
maps of still periods at different headings (Bosch gyro yaw, the profile's lever arm): the height difference of two
periods i, j over their common cells is a plane d = c + s . p_i with s = (R(theta_j - theta_i) - I) k; all pairs ->
least squares for k. The work monitor turns its points about the swing axis by the inverse (rcmot.work.monitor).

  python3 scripts/swing_axis_calib.py --profile hdx1026_work_155530 [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.sensors.bag_source import pc2_to_struct  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics  # noqa: E402
from rcmot.world import rot2  # noqa: E402

CELL, HALF = 0.2, 15.0


def still_periods(t, gz, *, rate_dps=1.0, min_s=3.0):
    """Gyro-integrated heading (bias = median rate while still, yaw held while still) and still periods
    [(t0, t1, heading)]."""
    k = max(int(round(1.0 / max(np.median(np.diff(t)), 1e-3))), 1)
    sm = np.convolve(gz, np.ones(k) / k, mode="same")
    still = np.abs(sm - np.median(sm)) < np.radians(rate_dps)
    bias = float(np.median(gz[still]))
    w = np.where(still, 0.0, gz - bias)
    yaw = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    out, i = [], 0
    while i < len(t):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and still[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            out.append((t[i] + 0.5, t[j] - 0.5, float(np.median(yaw[i:j + 1]))))
        i = j + 1
    return out, bias


def height_map(scans, R_lev, t_lev):
    """Per-cell median height (level frame, centred on the level origin) of a few scans."""
    n = int(2 * HALF / CELL)
    acc: dict[int, list] = {}
    for P, frame in scans:
        Rl, tl = lidar_to_sensor({}, frame)
        L = (P @ Rl.T + tl) @ R_lev.T + t_lev
        r = np.linalg.norm(L[:, :2], axis=1)
        L = L[(L[:, 2] > -1.5) & (L[:, 2] < 1.5) & (r > 3.0) & (r < 12.0)]
        ij = np.floor((L[:, :2] + HALF) / CELL).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        key, z = ij[ok, 0] * n + ij[ok, 1], L[ok, 2]
        o = np.lexsort((z, key))
        key, z = key[o], z[o]
        uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
        for kk, zz in zip(uk, z[st + (cnt - 1) // 2]):
            acc.setdefault(int(kk), []).append(float(zz))
    H = np.full((n, n), np.nan)
    for kk, v in acc.items():
        if len(v) >= 2:
            H.flat[kk] = np.median(v)
    return H


def pair_slope(Hi, Hj, th_i, th_j, c):
    """Plane fit of z_i - z_j at the cells common to both maps (frame of period i): (slope s (2,), MAD, n)."""
    n = Hi.shape[0]
    g = (np.arange(n) + 0.5) * CELL - HALF
    X, Y = np.meshgrid(g, g, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    # cell of map i -> the same ground in the level frame of period j (rotation about the swing axis c)
    Q = (P - c) @ rot2(th_i - th_j).T + c
    ij = np.floor((Q + HALF) / CELL).astype(int)
    ok = (ij >= 0).all(1) & (ij < n).all(1)
    zj = np.full(len(P), np.nan)
    zj[ok] = Hj[ij[ok, 0], ij[ok, 1]]
    d = Hi.ravel() - zj
    m = np.isfinite(d)
    if m.sum() < 300:
        return None
    M = np.c_[np.ones(m.sum()), P[m]]
    dm = d[m]
    w = np.abs(dm - np.median(dm)) < 0.3                           # dug / dumped cells, people, the boom
    for _ in range(3):                                              # trimmed least squares
        sol = np.linalg.lstsq(M[w], dm[w], rcond=None)[0]
        res = dm - M @ sol
        mad = float(np.median(np.abs(res[w])))
        w = np.abs(res) < max(4 * 1.48 * mad, 0.05)
    return sol[1:], mad, int(w.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="configs/datasets.yaml profile (bag, calib, world.lever_arm)")
    ap.add_argument("--imu", default="/bosch_imu/imu")
    ap.add_argument("--lidar", default="/ouster/points")
    ap.add_argument("--scans", type=int, default=5, help="LiDAR scans per still period")
    ap.add_argument("--min-deg", type=float, default=30.0, help="pairs at least this far apart in heading")
    ap.add_argument("--max-dt", type=float, default=120.0, help="pairs at most this far apart in time [s] (gyro drift)")
    ap.add_argument("--write", action="store_true", help="store swing_axis_level in the profile's calib JSON")
    a = ap.parse_args()

    prof = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))["datasets"][a.profile]
    bag, calp = Path(prof["bag"]), ROOT / prof["calib"]
    d = json.load(open(calp))
    R_lev, t_lev = RigExtrinsics.from_calib_json(d).level()
    cal = RigCalib.load(str(calp))
    la = (prof.get("overrides") or {}).get("world.lever_arm")
    if la is None:
        cfg = yaml.safe_load(open(ROOT / prof["config"]))
        la = (cfg.get("world") or {}).get("lever_arm")
    if la is None:
        raise SystemExit("no world.lever_arm for this profile: the swing axis position is needed")
    c = -(np.asarray(la, float)[:2] - cal.t_lr[:2])                   # swing axis in the level frame

    with AnyReader([bag]) as r:
        imu = [x for x in r.connections if x.topic == a.imu]
        T, GZ = [], []
        for con, tn, raw in r.messages(connections=imu):
            T.append(tn * 1e-9)
            GZ.append(r.deserialize(raw, con.msgtype).angular_velocity.z)
        T, GZ = np.array(T), np.array(GZ)
        per, bias = still_periods(T, GZ)
        print(f"{len(per)} still periods (>= 3 s), gyro bias {np.degrees(bias):+.3f} dps, "
              f"headings {np.degrees(min(p[2] for p in per)):+.0f}..{np.degrees(max(p[2] for p in per)):+.0f} deg")
        lid = [x for x in r.connections if x.topic == a.lidar]
        maps = []
        for t0, t1, th in per:
            want = np.linspace(t0, t1, a.scans)
            scans, k = [], 0
            for con, tn, raw in r.messages(connections=lid, start=int(t0 * 1e9), stop=int((t1 + 0.2) * 1e9)):
                if k < len(want) and tn * 1e-9 >= want[k]:
                    m = r.deserialize(raw, con.msgtype)
                    s = pc2_to_struct(m)
                    P = np.stack([s["x"], s["y"], s["z"]], 1).astype(float)
                    scans.append((P[np.isfinite(P).all(1) & (np.linalg.norm(P, axis=1) > 1.0)], m.header.frame_id))
                    k += 1
            if scans:
                maps.append((0.5 * (t0 + t1), th, height_map(scans, R_lev, t_lev)))

    rows_A, rows_b, info = [], [], []
    for i in range(len(maps)):
        for j in range(len(maps)):
            ti, thi, Hi = maps[i]
            tj, thj, Hj = maps[j]
            dth = (thj - thi + np.pi) % (2 * np.pi) - np.pi
            if i == j or abs(tj - ti) > a.max_dt or not (np.radians(a.min_deg) <= abs(dth) <= np.radians(180 - a.min_deg)):
                continue
            f = pair_slope(Hi, Hj, thi, thj, c)
            if f is None or f[1] > 0.06:
                continue
            rows_A.append(rot2(dth) - np.eye(2))
            rows_b.append(f[0])
            info.append((ti - T[0], np.degrees(dth), f[1], f[2]))
    if len(rows_A) < 3:
        raise SystemExit(f"only {len(rows_A)} usable pairs: need still periods at headings >= {a.min_deg} deg apart")
    Am, bm = np.concatenate(rows_A), np.concatenate(rows_b)
    keep = np.ones(len(rows_A), bool)
    for _ in range(3):                                             # drop pairs that disagree (scene changed, drove)
        km = np.repeat(keep, 2)
        kv = np.linalg.lstsq(Am[km], bm[km], rcond=None)[0]
        res = np.linalg.norm((Am @ kv - bm).reshape(-1, 2), axis=1)
        keep = res < max(3 * np.median(res[keep]), np.radians(0.15))
    print(f"{keep.sum()} / {len(rows_A)} pairs, residual slope median {np.degrees(np.median(res[keep])):.2f} deg")
    print(f"swing-axis tilt in the level frame: x {np.degrees(kv[0]):+.2f} deg, y {np.degrees(kv[1]):+.2f} deg "
          f"(|k| {np.degrees(np.linalg.norm(kv)):.2f} deg; height error it causes at 7 m: up to "
          f"{2 * 7 * np.linalg.norm(kv):.2f} m between opposite headings)")
    ax = np.array([kv[0], kv[1], 1.0])
    ax /= np.linalg.norm(ax)
    if a.write:
        d["swing_axis_level"] = [round(float(v), 6) for v in ax]
        d.setdefault("report", {})["swing_axis"] = {
            "tilt_deg": [round(float(np.degrees(v)), 3) for v in kv], "pairs": int(keep.sum()),
            "residual_slope_deg": round(float(np.degrees(np.median(res[keep]))), 3), "lever_arm": list(map(float, la)),
            "note": "scripts/swing_axis_calib.py: LiDAR still periods at different headings (docs/25 §3.1)"}
        json.dump(d, open(calp, "w"), indent=2)
        print("written to", calp)


if __name__ == "__main__":
    main()
