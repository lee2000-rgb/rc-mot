#!/usr/bin/env python3
"""Radar ego-velocity under 'ego moving + moving objects' and candidate windows (agmoA).
Inputs: agmo_extract.py npz + agmo_yolo_scan.py csv. Reference ego = |GNSS vel| with radar sign along body x
+ lever*wz lateral, rotated by the Doppler mount yaw (see agmo_analyze.py).

usage: python3 scripts/agmo_movers.py agmoA.npz agmoA_yolo.csv [--psi -1.31 --lever 1.81 --lag -0.105]
"""
import argparse

import numpy as np
from scipy.ndimage import uniform_filter1d


def runs(mask):
    m = np.concatenate([[0], mask.astype(int), [0]])
    dd = np.diff(m)
    return list(zip(np.flatnonzero(dd == 1), np.flatnonzero(dd == -1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("csv")
    ap.add_argument("--psi", type=float, default=-1.31)
    ap.add_argument("--lever", type=float, default=1.81)
    ap.add_argument("--lag", type=float, default=-0.105)
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    y = np.loadtxt(a.csv, delimiter=",")
    od, gv, im = d["/agmo/odom"], d["/agmo/gnss/vel"], d["/agmo/imu"]
    th = od[:, 1]
    bag0 = od[0, 0] - (od[0, 0] - d["radar_mid"][0, 0]) if False else None
    sp = np.interp(th, gv[:, 1], np.hypot(gv[:, 2], gv[:, 3]))
    wz5 = uniform_filter1d(im[:, 4], 5)
    wz1 = uniform_filter1d(im[:, 4], 50)
    R = d["radar_mid"]
    t_bag0 = min(R[0, 0], od[0, 0], d["/agmo/imu"][0, 0])              # ~ bag start (csv is bag-relative)
    tr = R[:, 1]
    tq = tr + a.lag
    ps = np.radians(a.psi)
    vx = np.sign(R[:, 2]) * np.interp(tq, th, sp)
    vyb = a.lever * np.interp(tq, th, wz5)
    ref = np.stack([np.cos(ps) * vx + np.sin(ps) * vyb, -np.sin(ps) * vx + np.cos(ps) * vyb], 1)
    err = np.linalg.norm(R[:, 2:4] - ref, axis=1)
    ok = R[:, 5] > 0
    trel = R[:, 0] - t_bag0
    npers = np.interp(trel, y[:, 0], y[:, 1], left=0, right=0)
    nveh = np.interp(trel, y[:, 0], y[:, 2], left=0, right=0)
    hpx = np.interp(trel, y[:, 0], y[:, 4], left=0, right=0)
    spd = np.interp(tq, th, sp)
    awz = np.degrees(np.abs(np.interp(tq, th, wz1)))
    dynf = R[:, 9] / np.maximum(R[:, 8], 1)
    mv = (spd > 0.1) & ok
    print("## radar_mid ego velocity error |v_radar - v_ref| [m/s] vs scene content (moving scans)")
    groups = [("no person in image", mv & (npers < 0.5)), (">=1 person", mv & (npers >= 0.5)), (">=3 persons", mv & (npers >= 2.5)),
              ("person tall >120 px (near)", mv & (hpx > 120)), ("vehicle/equipment", mv & (nveh >= 0.5)),
              ("dyn points >=10/scan", mv & (R[:, 9] >= 10)), ("dyn points >=30/scan", mv & (R[:, 9] >= 30)),
              ("dyn frac >=15%", mv & (dynf >= 0.15)), (">=1 person & turning>=5", mv & (npers >= 0.5) & (awz >= 5)),
              (">=1 person & speed<0.2", mv & (npers >= 0.5) & (spd < 0.2))]
    for lab, sel in groups:
        if sel.sum() == 0:
            print(f"  {lab:28s} n=0")
            continue
        e = err[sel]
        print(f"  {lab:28s} n={sel.sum():5d}: p50 {np.median(e):.3f} p95 {np.percentile(e, 95):.3f} p99 {np.percentile(e, 99):.3f} max {e.max():.3f}; "
              f">0.1: {100 * np.mean(e > 0.1):.2f}%  >0.2: {100 * np.mean(e > 0.2):.2f}%; inlier p50 {np.median(R[sel, 6]):.2f}; dyn pts p50 {np.median(R[sel, 9]):.0f}")
    print(f"  corr(err, dyn points) moving: {np.corrcoef(err[mv], R[mv, 9])[0, 1]:.3f}; corr(err, persons) {np.corrcoef(err[mv], npers[mv])[0, 1]:.3f}")
    worst = np.flatnonzero(mv & (dynf >= 0.15))
    print("  scans with dyn frac >=15% (t_rel, err, dyn, n, inlier, persons, speed):")
    for i in worst[np.argsort(-err[worst])][:10]:
        print(f"    {trel[i]:7.2f}s err {err[i]:.3f} dyn {R[i, 9]:.0f}/{R[i, 8]:.0f} inl {R[i, 6]:.2f} pers {npers[i]:.0f} speed {spd[i]:.2f}")

    # candidate windows: ego moving & person(s) visible, >=4 s (1 s bins)
    print("\n## candidate windows (bag-relative s): ego moving (>0.15 m/s) and >=1 person detected (YOLO11s conf>=0.35, 2 Hz), >=4 s")
    T = np.arange(0, y[-1, 0] + 1, 1.0)
    pb = np.array([y[(y[:, 0] >= t) & (y[:, 0] < t + 1), 1].max(initial=0) for t in T])
    hb = np.array([y[(y[:, 0] >= t) & (y[:, 0] < t + 1), 4].max(initial=0) for t in T])
    vb = np.array([y[(y[:, 0] >= t) & (y[:, 0] < t + 1), 2].max(initial=0) for t in T])
    thr = th - t_bag0
    sb = np.interp(T + 0.5, thr, sp)
    wb = np.degrees(np.abs(np.interp(T + 0.5, thr, wz1)))
    good = (pb >= 1) & (sb > 0.15)
    good = uniform_filter1d(good.astype(float), 3) > 0.3                 # bridge 1 s gaps
    rows = []
    for s0, e0 in runs(good):
        if e0 - s0 >= 4:
            sel = (trel >= T[s0]) & (trel < T[e0 - 1] + 1) & mv
            rows.append((T[s0], T[e0 - 1] + 1, pb[s0:e0].max(), hb[s0:e0].max(), vb[s0:e0].max(), sb[s0:e0].mean(), wb[s0:e0].max(),
                         np.median(R[sel, 9]) if sel.any() else np.nan, np.percentile(err[sel], 95) if sel.any() else np.nan))
    for r in rows:
        print(f"  {r[0]:6.0f}-{r[1]:6.0f}s  persons max {r[2]:.0f}, tallest {r[3]:.0f} px, vehicles {r[4]:.0f}, speed {r[5]:.2f} m/s, |wz| max {r[6]:.1f} deg/s, "
              f"radar dyn pts/scan p50 {r[7]:.0f}, ego err p95 {r[8]:.3f} m/s")


if __name__ == "__main__":
    main()
