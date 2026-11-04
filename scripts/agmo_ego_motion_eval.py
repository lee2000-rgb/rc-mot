#!/usr/bin/env python3
"""Relative ego-motion accuracy for the tracker (tracker frame = sensor head, base_link + lever arm), agmo rig.

Methods per radar_mid scan pair (t_k, t_k+h), h = 1 scan (~0.066 s), 0.5 s, 1.5 s:
  (a) current rc-mot pipeline: ScanStats v_sensor EMA 0.4 as dp = v*dt, dyaw = mean(last 10 /ouster/imu gz)*dt,
      per-frame steps composed (exactly what tracker.ego_compensate accumulates), no lever arm, no time offset
  (a2) (a) but /agmo/imu yaw rate
  (b) EgoMotionProvider (CausalBackbone: ScanStats ego vx + /agmo/imu, lever 2.09, offset -0.04), no GNSS
  (d) (b) + 2D EKF with RTK FIXED position updates (10 Hz)
  (0) no ego compensation
Reference (c): /agmo/odom RTK position + yaw (= gyro-integrated /agmo/yaw_deg, constant offset to the GNSS
track removed), tracker frame at base + lever_ref, evaluated at radar stamp + lag_ref.
Error of a static object at 20 m (az 0, +-30 deg) after the predicted frame change is reported.

usage: PYTHONPATH=src python3 scripts/agmo_ego_motion_eval.py agmoA.npz [--lag-ref -0.1] [--lever-ref 2.09]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).parent))
from agmo_ego_motion_provider import EgoMotionProvider  # noqa: E402

R_E = 6378137.0


def R2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compose(steps):
    """steps: list of (dp(2,), dyaw) -> total (dp, dyaw) in the first frame."""
    p, w = np.zeros(2), 0.0
    for dp, dw in steps:
        p = p + R2(w) @ dp
        w += dw
    return p, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--lag-ref", type=float, default=-0.10, help="true time of a radar scan = stamp + lag_ref")
    ap.add_argument("--lever-ref", type=float, default=2.09)
    ap.add_argument("--max-pairs", type=int, default=4000)
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    od, im, fx, gv = d["/agmo/odom"], d["/agmo/imu"], d["/agmo/gnss/fix"], d["/agmo/gnss/vel"]
    th = od[:, 1]
    st = d["str:/agmo/gnss/status"]
    ts = np.array([float(s.split("|")[0]) for s in st])
    rtk = np.array([int(s.split("rtk=")[1][0]) for s in st])
    rk = rtk[np.clip(np.searchsorted(ts, od[:, 0]), 0, len(ts) - 1)]
    lat, lon = fx[:, 2], fx[:, 3]
    E = np.radians(lon - lon[0]) * R_E * np.cos(np.radians(lat[0]))
    N = np.radians(lat - lat[0]) * R_E
    E = np.interp(th, fx[:, 1], E)
    N = np.interp(th, fx[:, 1], N)
    yaw = np.unwrap(Rot.from_quat(od[:, 5:9]).as_euler("xyz")[:, 2])
    sp = np.hypot(gv[:, 2], gv[:, 3])
    sp = np.interp(th, gv[:, 1], sp)
    # constant yaw_deg offset vs GNSS track (agmoA: fusion not initialised -> ~135 deg)
    cog = np.arctan2(np.interp(th, gv[:, 1], gv[:, 3]), np.interp(th, gv[:, 1], gv[:, 2]))
    R = d["radar_mid"]
    fwd = np.interp(th, R[:, 1], uniform_filter1d(R[:, 2], 5))
    cog = np.where(fwd < 0, cog + np.pi, cog)
    wz1 = uniform_filter1d(im[:, 4], 50)
    m = (sp > 0.2) & (np.abs(np.degrees(np.interp(th, im[:, 1], wz1))) < 2) & (rk == 4)
    off = np.median(wrap(cog[m] - yaw[m]))
    yaw_ref = yaw + off
    print(f"yaw_deg -> track offset removed: {np.degrees(off):+.2f} deg (median over {m.sum()} straight RTK-FIXED samples)")

    tr = R[:, 1]
    vR = R[:, 2:4]
    okR = R[:, 5] > 0

    # ---------- (a) pipeline ----------
    ou = d["/ouster/imu"]
    ema = np.zeros(2)
    steps_a, steps_a2 = [], []
    k_ou = np.searchsorted(ou[:, 0], R[:, 0], side="right")          # bag time ordering (bag_source)
    k_ag = np.searchsorted(im[:, 0], R[:, 0], side="right")
    for i in range(len(tr)):
        if okR[i]:
            ema = 0.6 * ema + 0.4 * vR[i]
        dt = tr[i] - tr[i - 1] if i else 0.0
        wz_o = ou[max(k_ou[i] - 10, 0):k_ou[i], 4].mean() if k_ou[i] else 0.0
        wz_a = im[max(k_ag[i] - 3, 0):k_ag[i], 4].mean() if k_ag[i] else 0.0   # same 0.06 s window at 50 Hz
        steps_a.append((ema * dt, wz_o * dt))
        steps_a2.append((ema * dt, wz_a * dt))

    # ---------- (b)/(d) provider ----------
    def run_provider(use_gnss):
        prov = EgoMotionProvider(lever=2.09, radar_time_offset=-0.04, use_gnss=use_gnss)
        ev = [(t, 0, i) for i, t in enumerate(im[:, 1])] + [(t, 1, i) for i, t in enumerate(tr)]
        if use_gnss:
            sel = np.arange(0, len(th), 5)                              # 10 Hz
            ev += [(th[i], 2, i) for i in sel]
        ev.sort()
        for t, kind, i in ev:
            if kind == 0:
                prov.add_imu(t, im[i, 5:8], im[i, 2:5])
            elif kind == 1:
                prov.add_radar_ego(t, vR[i], okR[i])
            else:
                prov.add_gnss(t, E[i], N[i], int(rk[i]), (np.interp(t, gv[:, 1], gv[:, 2]), np.interp(t, gv[:, 1], gv[:, 3])))
        return prov
    prov_b = run_provider(False)
    prov_d = run_provider(True)

    # ---------- reference ----------
    def ref_pose(t):
        tq = t + a.lag_ref
        x, y, w = np.interp(tq, th, E), np.interp(tq, th, N), np.interp(tq, th, yaw_ref)
        return np.array([x, y]) + R2(w) @ np.array([a.lever_ref, 0.0]), w

    def ref_rel(t0, t1):
        p0, w0 = ref_pose(t0)
        p1, w1 = ref_pose(t1)
        return R2(-w0) @ (p1 - p0), wrap(w1 - w0)

    objs = [20 * np.array([np.cos(np.radians(az)), np.sin(np.radians(az))]) for az in (0, 30, -30)]

    def obj_err(dp_e, dw_e, dp_r, dw_r):
        return max(np.linalg.norm(R2(-dw_e) @ (o - dp_e) - R2(-dw_r) @ (o - dp_r)) for o in objs)

    t0 = th[0]
    spd_r = np.interp(tr + a.lag_ref, th, sp)
    rk_r = rk[np.clip(np.searchsorted(th, tr), 0, len(th) - 1)]
    awz = np.abs(np.degrees(np.interp(tr, im[:, 1], wz1)))
    start = tr[0] + 15.0                                                # after bias freeze / EKF init
    print(f"\nrelative motion of the tracker frame (base + {a.lever_ref} m); reference lag {a.lag_ref:+.2f} s; RTK FIXED & moving (>0.1 m/s) pairs")
    print("columns: translation error [m] p50/p95 | yaw error [deg] p50/p95 | 20 m object error [m] p50/p95")
    for H in (1, 8, 23):                                                 # 15.1 Hz: 0.066 / 0.53 / 1.52 s
        idx = np.flatnonzero((tr > start) & (spd_r > 0.1) & (rk_r == 4))
        idx = idx[idx + H < len(tr)]
        if len(idx) > a.max_pairs:
            idx = idx[np.linspace(0, len(idx) - 1, a.max_pairs).astype(int)]
        res = {k: [] for k in ("0 none", "a pipeline(ouster)", "a2 pipeline(agmo imu)", "b DR provider", "d DR+RTK EKF")}
        turn = []
        for i in idx:
            j = i + H
            dp_r, dw_r = ref_rel(tr[i], tr[j])
            cand = {"0 none": (np.zeros(2), 0.0),
                    "a pipeline(ouster)": compose(steps_a[i + 1:j + 1]),
                    "a2 pipeline(agmo imu)": compose(steps_a2[i + 1:j + 1])}
            sb = prov_b.step(tr[i], tr[j])
            sd = prov_d.step(tr[i], tr[j])
            cand["b DR provider"] = (sb.dp, sb.dyaw)
            cand["d DR+RTK EKF"] = (sd.dp, sd.dyaw)
            for k, (dp_e, dw_e) in cand.items():
                res[k].append((np.linalg.norm(dp_e - dp_r), abs(np.degrees(wrap(dw_e - dw_r))), obj_err(dp_e, dw_e, dp_r, dw_r)))
            turn.append(awz[i] >= 5)
        turn = np.array(turn)
        dt_h = np.median(tr[idx + H] - tr[idx])
        for lab, sel in (("all", np.ones(len(idx), bool)), ("straight", ~turn), ("turning>=5deg/s", turn)):
            if not sel.any():
                continue
            print(f"\n  h={dt_h:.3f} s  [{lab}, n={sel.sum()}]")
            for k, v in res.items():
                v = np.array(v)[sel]
                print(f"    {k:24s} trans {np.median(v[:, 0]):.3f}/{np.percentile(v[:, 0], 95):.3f} m | yaw {np.median(v[:, 1]):.2f}/{np.percentile(v[:, 1], 95):.2f} deg | obj@20m {np.median(v[:, 2]):.3f}/{np.percentile(v[:, 2], 95):.3f} m")
    # world-frame check (absolute pose of the DR+RTK EKF vs RTK, and pure DR drift)
    print("\nworld frame: EKF base position vs RTK (FIXED, moving):", end=" ")
    sel = (th > start) & (rk == 4)
    e = [np.linalg.norm(prov_d.sensor_pose(t)[:2] - (np.array([E[i], N[i]]) + R2(yaw_ref[i]) @ np.array([a.lever_ref, 0.0])))
         for i, t in zip(np.flatnonzero(sel)[::50], th[sel][::50])]
    print(f"p50 {np.median(e):.3f} m p95 {np.percentile(e, 95):.3f} m (note: same RTK used as input)")
    eh = [abs(np.degrees(wrap(prov_d.sensor_pose(t)[2] - yaw_ref[i]))) for i, t in zip(np.flatnonzero(sel)[::50], th[sel][::50])]
    print(f"EKF heading vs track-aligned yaw: p50 {np.median(eh):.2f} deg p95 {np.percentile(eh, 95):.2f} deg")


    # ---------- (e) DR in an odom frame + causal map->odom alignment from RTK FIXED (REP-105 style) ----------
    # relative motion = DR (b) by construction; only the world anchoring comes from RTK.
    L = 2.09
    tq = th[::10]                                                         # 5 Hz evaluation
    Pdr = np.array([prov_b.sensor_pose(t) for t in tq])
    Bdr = Pdr[:, :2] - L * np.stack([np.cos(Pdr[:, 2]), np.sin(Pdr[:, 2])], 1)   # DR base position
    Brt = np.stack([E[::10], N[::10]], 1)
    fixed = rk[::10] == 4
    T = None
    werr, herr = [], []
    last_fit = -1e9
    for i, t in enumerate(tq):
        if t - last_fit >= 1.0:                                           # refit once per second on the past 30 s
            w = np.flatnonzero((tq > t - 30) & (tq <= t) & fixed)
            if len(w) > 20 and np.ptp(Brt[w], axis=0).max() > 3.0:
                A, B = Bdr[w], Brt[w]
                ma, mb = A.mean(0), B.mean(0)
                Hm = (A - ma).T @ (B - mb)
                ang = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
                T = (ang, mb - R2(ang) @ ma)
                last_fit = t
        if T is not None and fixed[i] and t > start and np.interp(t, th, sp) > 0.1:
            ang, tt_ = T
            werr.append(np.linalg.norm(R2(ang) @ Bdr[i] + tt_ - Brt[i]))
            herr.append(abs(np.degrees(wrap(Pdr[i, 2] + ang - np.interp(t, th, yaw_ref)))))
    if werr:
        print(f"(e) DR + causal map->odom (30 s RTK FIXED window, 1 Hz): world base-position error p50 {np.median(werr):.3f} m p95 {np.percentile(werr, 95):.3f} m; "
              f"heading vs track-aligned yaw p50 {np.median(herr):.2f} deg p95 {np.percentile(herr, 95):.2f} deg")
    # pure DR drift (no anchoring) over the bag, for reference
    w = np.flatnonzero(fixed & (tq > start))
    A, B = Bdr[w], Brt[w]
    ma, mb = A.mean(0), B.mean(0)
    Hm = (A - ma).T @ (B - mb)
    ang = np.arctan2(Hm[0, 1] - Hm[1, 0], Hm[0, 0] + Hm[1, 1])
    e = np.linalg.norm((A - ma) @ R2(ang).T + mb - B, axis=1)
    print(f"pure DR (whole bag, one SE(2) alignment): base-position error mean {e.mean():.2f} m, max {e.max():.2f} m")

if __name__ == "__main__":
    main()
