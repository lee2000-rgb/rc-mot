#!/usr/bin/env python3
"""Analyse the .npz written by agmo_extract.py: GNSS/RTK quality, frame conventions, heading sources,
motion segmentation, radar Doppler ego velocity vs GNSS/odom, IMU yaw-rate comparison, timing.

usage: python3 scripts/agmo_analyze.py agmoD.npz [--lever-x 2.09]
"""
import argparse
import re
from collections import Counter

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.spatial.transform import Rotation as Rot

R_E = 6378137.0
RADARS = ["radar_left", "radar_mid", "radar_right"]
# radar columns written by agmo_extract.py
T_BAG, T_HDR, VX, VY, VZ, VALID, INL, SIG, NPT, NDYN, AZ2, AZ98, RNG50, P70 = range(14)


def wrapd(a):
    return (a + 180) % 360 - 180


def qyaw(q):
    return Rot.from_quat(q).as_euler("xyz")[:, 2]


def enu(fix):
    lat, lon, alt = fix[:, 2], fix[:, 3], fix[:, 4]
    x = np.radians(lon - lon[0]) * R_E * np.cos(np.radians(lat[0]))
    y = np.radians(lat - lat[0]) * R_E
    return np.stack([x, y, alt - alt[0]], 1)


def runs(mask):
    """(start, end) index pairs of True runs."""
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def pct(a, q=(5, 50, 95)):
    return " / ".join(f"{v:.3f}" for v in np.percentile(a, q)) if len(a) else "n/a"


def section(s):
    print(f"\n## {s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--lever-x", type=float, default=None, help="radar x offset from base_link (default: fitted)")
    a = ap.parse_args()
    d = dict(np.load(a.npz))
    fx, od, gv, im = d["/agmo/gnss/fix"], d["/agmo/odom"], d["/agmo/gnss/vel"], d["/agmo/imu"]
    common = np.intersect1d(np.intersect1d(fx[:, 1], od[:, 1]), np.intersect1d(gv[:, 1], im[:, 1]))
    fx, od, gv, im = [x[np.isin(x[:, 1], common)] for x in (fx, od, gv, im)]   # same 50 Hz header stamps
    fx, od, gv, im = [x[np.unique(x[:, 1], return_index=True)[1]] for x in (fx, od, gv, im)]
    t0 = fx[0, 1]
    th = od[:, T_HDR]
    tb = th - t0
    dur = th[-1] - th[0]
    sp = np.hypot(gv[:, 2], gv[:, 3])
    wz = im[:, 4]
    wz1 = uniform_filter1d(wz, 50)                       # 1 s mean (50 Hz)
    yaw_odom = qyaw(od[:, 5:9])
    # body-x speed = |GNSS velocity| with the sign of the radar Doppler forward velocity. The odom twist is
    # R(yaw)^T * gnss/vel and therefore wrong whenever yaw_deg is not aligned (agmoA: fusion not initialised).
    rm = d["radar_mid"]
    sgn_r = np.sign(np.interp(th, rm[:, T_HDR], uniform_filter1d(rm[:, VX], 5)))
    sgn = np.where(np.abs(np.interp(th, rm[:, T_HDR], rm[:, VX])) > 0.05, sgn_r, np.sign(od[:, 9]))
    vbx = sp * np.where(sgn == 0, 1, sgn)
    print(f"[info] odom twist.x sign disagrees with radar forward sign on {100 * np.mean((np.sign(od[:, 9]) != sgn) & (sp > 0.1)):.1f}% of moving samples")

    # ---------------- GNSS status ----------------
    section("GNSS / RTK status")
    st = d["str:/agmo/gnss/status"]
    ts = np.array([float(s.split("|")[0]) for s in st])
    rtk = np.array([int(re.search(r"rtk=(\d)", s).group(1)) for s in st])
    sats = np.array([int(re.search(r"sats=(\d+)", s).group(1)) for s in st])
    names = {4: "FIXED", 5: "FLOAT", 2: "DGNSS", 1: "SINGLE", 0: "NONE", 3: "?3"}
    for c, n in sorted(Counter(rtk).items(), key=lambda x: -x[1]):
        r = runs(rtk == c)
        L = np.array([(ts[min(e, len(ts) - 1)] - ts[s]) for s, e in r])
        print(f"  rtk={c}({names.get(c, '?')}): {100 * n / len(rtk):5.1f}%  runs={len(r)}  run length s p50={np.median(L):.1f} max={L.max():.1f}  sats p50={np.median(sats[rtk == c]):.0f}")
    print("  raw_status:", dict(Counter(s.split("|", 1)[1].split(" sats")[0].split("pos_type=")[1] for s in d["str:/agmo/gnss/raw_status"])))
    hs = d["str:/agmo/heading_track_status"]
    print("  heading_track_status:", dict(Counter(re.sub(r"[-\d.]+", "#", s.split("|", 1)[1]).split(" (")[0] for s in hs)))
    print("  heading_qual values:", dict(Counter(re.search(r"heading_qual=(\S+)", s).group(1) for s in st)))
    print("  motion flag:", dict(Counter(re.search(r"motion=(\S+)", s).group(1) for s in st)))
    rk = rtk[np.clip(np.searchsorted(ts, od[:, 0]), 0, len(ts) - 1)].astype(float)   # nearest status (bag time)

    # position covariance reported
    print("  fix position_covariance[0] (m^2) by rtk:", {int(c): float(np.median(fx[rk == c, 6])) for c in np.unique(rk)})

    # ---------------- frames / conventions ----------------
    section("Frame conventions")
    print("  odom frames:", d["str:odom_frames"][0])
    P = enu(fx)
    print(f"  odom position - ENU(fix, origin first fix) [m]: dx {pct(od[:, 2] - P[:, 0])}  dy {pct(od[:, 3] - P[:, 1])}")
    ydm = d["/agmo/yaw_deg"]
    yd = ydm[np.clip(np.searchsorted(ydm[:, 0], od[:, 0]), 0, len(ydm) - 1), 1]   # nearest (bag time)
    print(f"  odom yaw(ENU, CCW from E) - (90 - yaw_deg): {pct(wrapd(np.degrees(yaw_odom) - (90 - yd)))} deg  -> yaw_deg = compass (CW from N)")
    yi = qyaw(im[:, 8:12])
    print(f"  /agmo/imu orientation yaw - odom yaw: {pct(wrapd(np.degrees(yi - yaw_odom)))} deg  (imu quaternion = NWU-like, yaw = -compass)")
    c, s = np.cos(yaw_odom), np.sin(yaw_odom)
    bx, by = c * gv[:, 2] + s * gv[:, 3], -s * gv[:, 2] + c * gv[:, 3]
    print(f"  odom twist - R(yaw)^T gnss/vel(ENU): |dx| max {np.abs(od[:, 9] - bx).max():.2e}, |dy| max {np.abs(od[:, 10] - by).max():.2e}  -> twist in base_link (body), from GNSS vel + yaw")
    print(f"  odom angular.z == /agmo/imu gyro z: max diff {np.abs(od[:, 14] - wz).max():.2e}")
    print(f"  header stamps identical fix/odom/vel/imu: {np.abs(fx[:, 1] - od[:, 1]).max() == 0 and np.abs(fx[:, 1] - im[:, 1]).max() == 0}")
    print(f"  bag-receive minus header stamp [s] (p5/p50/p95): gnss/odom/imu {pct(od[:, 0] - od[:, 1])}")
    for k in RADARS:
        print(f"    {k}: {pct(d[k][:, T_BAG] - d[k][:, T_HDR], (5, 50, 99))}")
    if "/oak/imu/data" in d:
        o = d["/oak/imu/data"]
        print(f"    oak imu: {pct(o[:, 0] - o[:, 1])}")
    if "/ouster/imu" in d:
        o = d["/ouster/imu"]
        print(f"    ouster imu header stamp is sensor clock (offset {np.median(o[:, 0] - o[:, 1]):.0f} s) -> use bag time")

    # ---------------- headings ----------------
    section("Heading sources (compass deg, moving straight: speed>0.2, |wz_1s|<2 deg/s)")
    mv = (sp > 0.2) & (np.abs(np.degrees(wz1)) < 2)
    cog = (90 - np.degrees(np.arctan2(gv[:, 3], gv[:, 2]))) % 360
    cog = np.where(vbx < 0, (cog + 180) % 360, cog)
    print(f"  COG(gnss/vel) - yaw_deg: p5/p50/p95 {pct(wrapd(cog[mv] - yd[mv]))}")
    ht = d["/agmo/heading_track_deg"]
    yh = ydm[np.clip(np.searchsorted(ydm[:, 0], ht[:, 0]), 0, len(ydm) - 1), 1]
    mh = np.interp(ht[:, 0], od[:, 0], mv.astype(float)) > 0.5
    print(f"  heading_track_deg - yaw_deg: {pct(wrapd(ht[mh, 1] - yh[mh]))}  (n={mh.sum()}, ~10 Hz)")
    rh = d["/agmo/gnss/raw_heading_deg"]
    yr = ydm[np.clip(np.searchsorted(ydm[:, 0], rh[:, 0]), 0, len(ydm) - 1), 1]
    mr = np.interp(rh[:, 0], od[:, 0], mv.astype(float)) > 0.5
    print(f"  raw_heading_deg - yaw_deg: {pct(wrapd(rh[mr, 1] - yr[mr]))}  (n={mr.sum()}, ~{len(rh) / dur:.2f} Hz, integer deg)")
    # yaw_deg vs integrated gyro: drift / jumps
    yaw_int = np.cumsum(np.concatenate([[0], wz[1:] * np.diff(th)]))
    dev = wrapd(np.degrees(np.unwrap(np.radians(90 - yd)) - yaw_int) - 0)
    dev = dev - dev[0]
    step = np.abs(np.diff(np.unwrap(np.radians(dev))))
    print(f"  (90-yaw_deg) - integral(gyro z): drift over bag {np.degrees(np.unwrap(np.radians(dev)))[-1]:+.2f} deg; max 20 ms step {np.degrees(step.max()):.3f} deg")
    yaw_rate_from_yawdeg = np.gradient(np.unwrap(np.radians(90 - yd)), th)
    stp = np.degrees(np.diff(np.unwrap(np.radians(dev))))
    for thr in (0.1, 0.3, 0.5):
        print(f"  yaw_deg corrections (step vs gyro > {thr} deg per 20 ms): {np.sum(np.abs(stp) > thr)} events")
    print(f"  d/dt(90-yaw_deg) vs gyro z: corr {np.corrcoef(uniform_filter1d(yaw_rate_from_yawdeg, 25), uniform_filter1d(wz, 25))[0, 1]:.4f}")

    # ---------------- static noise ----------------
    section("Static periods (speed<0.05 m/s for >=3 s): position noise")
    stat = uniform_filter1d((sp < 0.05).astype(float), 25) > 0.999
    for s0, e0 in runs(stat):
        if th[e0 - 1] - th[s0] < 3:
            continue
        seg = slice(s0, e0)
        sd = P[seg].std(0)
        drift = np.linalg.norm(P[e0 - 1, :2] - P[s0, :2])
        print(f"  t={tb[s0]:6.1f}-{tb[e0 - 1]:6.1f}s rtk={dict(Counter(rk[seg].astype(int)))}: std E/N/U = {sd[0]:.3f}/{sd[1]:.3f}/{sd[2]:.3f} m, start-end {drift:.3f} m; "
              f"|v| p95 {np.percentile(sp[seg], 95):.3f} m/s; gyro z mean {np.degrees(wz[seg].mean()):+.3f} deg/s std {np.degrees(wz[seg].std()):.3f}; yaw_deg std {yd[seg].std():.3f} deg")

    # GNSS position jumps vs velocity integration (all data)
    section("GNSS position consistency: |dp - v dt| per 20 ms step, by RTK state & at state changes")
    dp = np.diff(P[:, :2], axis=0)
    vdt = 0.5 * (gv[1:, 2:4] + gv[:-1, 2:4]) * np.diff(th)[:, None]
    inc = np.linalg.norm(dp - vdt, axis=1)
    for cc in np.unique(rk):
        m = rk[1:] == cc
        print(f"  rtk={int(cc)}: p50 {np.median(inc[m]) * 100:.1f} cm, p99 {np.percentile(inc[m], 99) * 100:.1f} cm, max {inc[m].max() * 100:.1f} cm")
    ch = np.flatnonzero(np.diff(rk) != 0)
    jumps = []
    for i in ch:                                        # 1 s window around the change
        w = slice(max(i - 25, 0), min(i + 25, len(inc)))
        jumps.append((tb[i], int(rk[i]), int(rk[i + 1]), np.linalg.norm((dp[w] - vdt[w]).sum(0))))
    J = np.array([j[3] for j in jumps] or [0.0])
    print(f"  {len(jumps)} state changes; net 1 s position jump p50 {np.median(J):.2f} m, p90 {np.percentile(J, 90):.2f} m, max {J.max():.2f} m")
    for j in sorted(jumps, key=lambda x: -x[3])[:6]:
        print(f"    t={j[0]:6.1f}s {names[j[1]]}->{names[j[2]]}: {j[3]:.2f} m")

    # ---------------- motion segmentation ----------------
    section("Motion segmentation (50 Hz samples; |wz| = 1 s mean gyro)")
    aw = np.abs(np.degrees(wz1))
    cls = np.where(sp < 0.1, 0, np.where(aw >= 5, 2, 1))
    cls = np.where((cls == 1) & (vbx < -0.05), 3, cls)
    lab = {0: "static (<0.1 m/s)", 1: "driving straight (|wz|<5 deg/s)", 2: "turning (|wz|>=5 deg/s)", 3: "reversing"}
    dts = np.diff(th, append=th[-1])
    for k in range(4):
        print(f"  {lab[k]:35s}: {dts[cls == k].sum():6.1f} s ({100 * dts[cls == k].sum() / dur:4.1f}%)")
    mov = sp >= 0.1
    print(f"  moving speed p5/p50/p95/max: {pct(sp[mov])} / {sp.max():.2f} m/s; distance {np.sum(sp * dts):.0f} m")
    print(f"  |yaw rate| (1 s mean) while moving p50/p90/p99/max: {np.percentile(aw[mov], 50):.1f} / {np.percentile(aw[mov], 90):.1f} / {np.percentile(aw[mov], 99):.1f} / {aw[mov].max():.1f} deg/s; "
          f"raw gyro |wz| p99 {np.percentile(np.abs(np.degrees(wz[mov])), 99):.1f} deg/s")
    print("  turning episodes (|wz_1s|>=5 deg/s for >=2 s):")
    for s0, e0 in runs(aw >= 5):
        if th[e0 - 1] - th[s0] >= 2:
            dyaw = np.degrees(np.sum(wz[s0:e0] * dts[s0:e0]))
            print(f"    t={tb[s0]:6.1f}-{tb[e0 - 1]:6.1f}s  dyaw={dyaw:+6.1f} deg  peak {aw[s0:e0].max():.1f} deg/s  speed {sp[s0:e0].mean():.2f} m/s")

    # ---------------- IMU yaw-rate comparison ----------------
    section("IMU yaw rate comparison vs /agmo/imu (== odom twist.angular.z)")
    others = []
    if "/ouster/imu" in d:
        o = d["/ouster/imu"]
        others.append(("ouster imu z (bag time)", o[:, 0] - np.median(od[:, 0] - od[:, 1]), o[:, 4]))
    if "/oak/imu/data" in d:
        o = d["/oak/imu/data"]
        others.append(("oak imu -y (optical frame, hdr)", o[:, 1], -o[:, 3]))
    if "/realsense/imu" in d:
        o = d["/realsense/imu"]
        others.append(("realsense imu -y (optical, hdr)", o[:, 1], -o[:, 3]))
    wz5 = uniform_filter1d(wz, 5)
    for name, t, w in others:
        # 10 Hz box-filtered comparison on the agmo time base
        k = max(int(round(0.1 / np.median(np.diff(t)))), 1)
        ws = uniform_filter1d(w, k)
        best = None
        for lag in np.arange(-0.3, 0.301, 0.005):
            wi = np.interp(th + lag, t, ws)
            r = np.sqrt(np.mean((wi[mov] - wz5[mov]) ** 2))
            if best is None or r < best[1]:
                best = (lag, r)
        wi = np.interp(th + best[0], t, ws)
        A = np.stack([wz5[mov], np.ones(mov.sum())], 1)
        g, b = np.linalg.lstsq(A, wi[mov], rcond=None)[0]
        sb = np.degrees(np.interp(th, t, w)[stat].mean()) if stat.any() else np.nan
        print(f"  {name:32s}: best lag {best[0]:+.3f} s, RMS diff {np.degrees(best[1]):.2f} deg/s, scale {g:.3f}, offset {np.degrees(b):+.3f} deg/s, "
              f"static bias {sb:+.3f} deg/s, corr {np.corrcoef(wi[mov], wz5[mov])[0, 1]:.3f}")
    print(f"  /agmo/imu static bias {np.degrees(wz[stat].mean()):+.3f} deg/s, static std (50 Hz) {np.degrees(wz[stat].std()):.3f} deg/s")
    wv = np.degrees(wz[mov & (aw < 2)])
    print(f"  /agmo/imu raw gyro z while driving straight: std {wv.std():.2f} deg/s (vibration), 1 s mean std {np.degrees(wz1[mov & (aw < 2)]).std():.2f} deg/s")

    # ---------------- radar ego velocity ----------------
    section("Radar Doppler ego velocity (ScanStatistician, planar) vs GNSS/odom")
    for k in RADARS:
        R = d[k]
        print(f"  {k}: {len(R)} scans ({len(R) / dur:.1f} Hz), pts p50 {np.median(R[:, NPT]):.0f}, ego_valid {100 * R[:, VALID].mean():.1f}%, "
              f"az 2-98% p50 {np.median(R[:, AZ2]):.0f}..{np.median(R[:, AZ98]):.0f} deg, range p50 {np.median(R[:, RNG50]):.1f} m")
    lever_fit = {}
    for k in RADARS:
        R = d[k]
        t = R[:, T_HDR]
        ok = R[:, VALID] > 0
        v = R[:, VX:VY + 1]
        s_ = np.interp(t, th, sp)
        w_ = np.interp(t, th, wz1)
        strt = ok & (s_ > 0.2) & (np.abs(np.degrees(w_)) < 2) & (np.interp(t, th, vbx) > 0)
        ang = np.degrees(np.arctan2(v[strt, 1], v[strt, 0]))
        psi = -np.median(ang)
        # lag + lever arm: vy_rot = L*wz(t+lag) ; |v| = speed(t+lag)
        cps, sps = np.cos(np.radians(psi)), np.sin(np.radians(psi))
        vxr, vyr = cps * v[:, 0] - sps * v[:, 1], sps * v[:, 0] + cps * v[:, 1]  # radar velocity rotated to body axes
        rk_r = rk[np.clip(np.searchsorted(th, t), 0, len(th) - 1)]
        strt &= rk_r == 4
        ang = np.degrees(np.arctan2(v[strt, 1], v[strt, 0]))
        psi = -np.median(ang)
        cps, sps = np.cos(np.radians(psi)), np.sin(np.radians(psi))
        vxr, vyr = cps * v[:, 0] - sps * v[:, 1], sps * v[:, 0] + cps * v[:, 1]
        m = ok & (s_ > 0.2) & (rk_r == 4)                     # fit on RTK FIXED only (DGNSS velocity is biased)
        res = []
        for lag in np.arange(-0.3, 0.301, 0.005):
            w2 = np.interp(t + lag, th, wz5)
            L = np.linalg.lstsq(w2[m, None], vyr[m], rcond=None)[0][0]
            s2 = np.interp(t + lag, th, vbx)
            res.append((lag, L, np.sqrt(np.mean((vyr[m] - L * w2[m]) ** 2)), np.sqrt(np.mean((vxr[m] - s2[m]) ** 2))))
        res = np.array(res)
        i1, i2 = res[:, 2].argmin(), res[:, 3].argmin()
        lever_fit[k] = (psi, res[i1, 1], res[i1, 0])
        print(f"  {k}: mount yaw from Doppler (straight fwd, n={strt.sum()}) = {psi:+.2f} deg (IQR {np.percentile(-ang, 25):+.1f}..{np.percentile(-ang, 75):+.1f}); "
              f"lever-arm x fit {res[i1, 1]:.2f} m; lag(vy~wz) {res[i1, 0]:+.3f} s; lag(vx~speed) {res[i2, 0]:+.3f} s; speed scale |v_radar|/|v_gnss| {np.median(np.hypot(*v[strt].T) / s_[strt]):.4f}")
    # error statistics for radar_mid with (a) tf-only reference and (b) calibrated reference
    for k in RADARS:
        R = d[k]
        t = R[:, T_HDR]
        ok = R[:, VALID] > 0
        v = R[:, VX:VY + 1]
        psi, Lfit, lag = lever_fit[k]
        configs = [("tf_static only (yaw 0, L=tf x, lag 0)", 0.0, a.lever_x or 2.09, 0.0),
                   ("calibrated (yaw, L, lag fitted)", psi, Lfit, lag)]
        if k != "radar_mid":
            configs = configs[1:]
        for name, ps, L, lg in configs:
            tt = t + lg
            vx_b = np.interp(tt, th, vbx)
            vy_b = np.interp(tt, th, od[:, 10]) * 0           # assume no side slip (odom vy comes from biased yaw)
            w_b = np.interp(tt, th, wz5)
            ref_b = np.stack([vx_b, vy_b + L * w_b], 1)
            cps, sps = np.cos(np.radians(ps)), np.sin(np.radians(ps))
            ref = np.stack([cps * ref_b[:, 0] + sps * ref_b[:, 1], -sps * ref_b[:, 0] + cps * ref_b[:, 1]], 1)  # body -> radar
            spd = np.interp(tt, th, sp)
            aw_ = np.abs(np.degrees(np.interp(tt, th, wz1)))
            rkk = rk[np.clip(np.searchsorted(th, tt), 0, len(th) - 1)]
            # pipeline EMA (ema_alpha 0.4, only valid scans update)
            ema = np.zeros_like(v)
            e_ = np.zeros(2)
            for i in range(len(v)):
                if ok[i]:
                    e_ = 0.6 * e_ + 0.4 * v[i]
                ema[i] = e_
            for lab_, vv in (("raw", v), ("EMA0.4", ema)):
                err = vv - ref
                mm = spd > 0.1
                en = np.linalg.norm(err, axis=1)
                print(f"\n  [{k} | {name} | {lab_}] moving scans n={mm.sum()}: bias ex {err[mm, 0].mean():+.3f} ey {err[mm, 1].mean():+.3f} m/s; "
                      f"RMS ex {np.sqrt(np.mean(err[mm, 0] ** 2)):.3f} ey {np.sqrt(np.mean(err[mm, 1] ** 2)):.3f}; |e| p50/p95/p99 {pct(en[mm], (50, 95, 99))}; "
                      f">0.1 m/s {100 * np.mean(en[mm] > 0.1):.1f}%, >0.3 m/s {100 * np.mean(en[mm] > 0.3):.2f}%")
                if lab_ == "raw" and name.startswith("calibrated"):
                    for nm, sel in [("speed 0.1-0.3", (spd >= 0.1) & (spd < 0.3)), ("speed 0.3-0.6", (spd >= 0.3) & (spd < 0.6)),
                                    ("speed >=0.6", spd >= 0.6), ("straight |wz|<2", mm & (aw_ < 2)),
                                    ("turn 5-15 deg/s", mm & (aw_ >= 5) & (aw_ < 15)), ("turn >=15 deg/s", mm & (aw_ >= 15)),
                                    ("static (<0.05)", spd < 0.05),
                                    ("RTK FIXED", mm & (rkk == 4)), ("RTK FLOAT", mm & (rkk == 5)),
                                    ("DGNSS", mm & (rkk == 2)), ("SINGLE", mm & (rkk == 1)),
                                    ("FIXED & straight", mm & (rkk == 4) & (aw_ < 2)),
                                    ("FIXED & turn>=5", mm & (rkk == 4) & (aw_ >= 5))]:
                        if sel.sum():
                            print(f"      {nm:18s} n={sel.sum():5d}: |e| p50 {np.median(en[sel]):.3f} p95 {np.percentile(en[sel], 95):.3f} m/s; ex bias {err[sel, 0].mean():+.3f}; >0.3: {100 * np.mean(en[sel] > 0.3):.2f}%")
                    dynf = R[:, NDYN] / np.maximum(R[:, NPT], 1)
                    for lo, hi in [(0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 1.01)]:
                        sel = mm & (dynf >= lo) & (dynf < hi)
                        if sel.sum():
                            print(f"      dyn frac {lo:.2f}-{hi:.2f}  n={sel.sum():5d}: |e| p50 {np.median(en[sel]):.3f} p95 {np.percentile(en[sel], 95):.3f}; >0.3: {100 * np.mean(en[sel] > 0.3):.2f}%")
                    for lo, hi in [(0, 0.5), (0.5, 0.7), (0.7, 1.01)]:
                        sel = mm & (R[:, INL] >= lo) & (R[:, INL] < hi)
                        if sel.sum():
                            print(f"      inlier ratio {lo:.1f}-{hi:.1f} n={sel.sum():5d}: |e| p50 {np.median(en[sel]):.3f} p95 {np.percentile(en[sel], 95):.3f}")
                    print(f"      sigma_v (scan Doppler noise) moving p50 {np.median(R[mm, SIG]):.3f} static p50 {np.median(R[spd < 0.05, SIG]) if (spd < 0.05).any() else float('nan'):.3f} m/s; "
                          f"dyn points/scan moving p50 {np.median(R[mm, NDYN]):.0f} p95 {np.percentile(R[mm, NDYN], 95):.0f} (static p50 {np.median(R[spd < 0.05, NDYN]) if (spd < 0.05).any() else float('nan'):.0f})")
                    bad = np.flatnonzero(mm & (en > 0.3))
                    if len(bad):
                        print(f"      worst scans (t, |e|, speed, |wz|, pts, inlier, ndyn, rtk):")
                        for i in bad[np.argsort(-en[bad])][:8]:
                            print(f"        {t[i] - t0:7.2f}s {en[i]:.2f} {spd[i]:.2f} {aw_[i]:5.1f} {R[i, NPT]:.0f} {R[i, INL]:.2f} {R[i, NDYN]:.0f} {int(rkk[i])}")
    section("Radar ego-velocity availability while moving (sparse scenes)")
    for k in RADARS:
        R = d[k]
        t = R[:, T_HDR]
        spd = np.interp(t, th, sp)
        mm = spd > 0.1
        vmag = np.hypot(R[:, VX], R[:, VY])
        bad = (R[:, VALID] == 0) | (R[:, NPT] < 20)
        absurd = (R[:, VALID] > 0) & (vmag > 3.0)          # vehicle max ~1 m/s
        rr = [(t[e - 1] - t[s0]) for s0, e in runs(bad & mm)]
        print(f"  {k}: moving scans with invalid fit or <20 pts: {100 * np.mean(bad[mm]):.2f}% ; longest run {max(rr) if rr else 0:.1f} s; "
              f"'valid' fits with |v|>3 m/s (absurd): {absurd.sum()} (n_pts {R[absurd, NPT].astype(int).tolist()}, sigma_v {np.round(R[absurd, SIG], 2).tolist()})")
        for s0, e in runs(bad):
            if t[e - 1] - t[s0] >= 1.0 and np.any(mm[s0:e]):
                print(f"     t={t[s0] - t0:7.1f}-{t[e - 1] - t0:7.1f}s: speed {spd[s0:e].mean():.2f} m/s (min {np.interp(t[s0:e], th, vbx).min():+.2f}, max {np.interp(t[s0:e], th, vbx).max():+.2f} body-x), |wz| {np.degrees(np.abs(np.interp(t[s0:e], th, wz1))).mean():.1f} deg/s")

    # dead reckoning (radar mid + gyro) vs GNSS
    section("Dead reckoning radar_mid (calibrated) + /agmo/imu gyro vs GNSS ENU (RTK FIXED windows)")
    R = d["radar_mid"]
    psi, Lfit, lag = lever_fit["radar_mid"]
    t = R[:, T_HDR] + lag
    ok = R[:, VALID] > 0
    v = R[:, VX:VY + 1]
    cps, sps = np.cos(np.radians(psi)), np.sin(np.radians(psi))
    vb = np.stack([cps * v[:, 0] - sps * v[:, 1], sps * v[:, 0] + cps * v[:, 1]], 1)
    vb[:, 1] -= Lfit * np.interp(t, th, wz5)                    # remove lever-arm term -> base velocity
    yaw0 = np.unwrap(yaw_odom)
    Pe = P[:, :2]
    print("  (raw = heading initialised from yaw_deg; aligned = after best 2D rotation, i.e. heading-bias removed)")
    for W in (5.0, 10.0, 20.0, 30.0):
        rows = []
        for ts_ in np.arange(t[0] + 5, t[-1] - W, W / 2):
            sel = (t >= ts_) & (t < ts_ + W) & ok
            i0, i1 = np.searchsorted(th, ts_), np.searchsorted(th, ts_ + W)
            if sel.sum() < 10 or np.interp(ts_, th, sp) < 0.1 or np.any(rk[i0:i1] != 4):
                continue
            yaw = yaw0[i0] + np.cumsum(wz[i0:i1] * np.diff(th[i0:i1 + 1]))
            tt = t[sel]
            y_r = np.interp(tt, th[i0:i1], yaw)
            dtt = np.diff(np.concatenate([[ts_], tt]))
            ve = np.stack([np.cos(y_r) * vb[sel, 0] - np.sin(y_r) * vb[sel, 1],
                           np.sin(y_r) * vb[sel, 0] + np.cos(y_r) * vb[sel, 1]], 1)
            traj = np.cumsum(ve * dtt[:, None], 0)
            ref = np.stack([np.interp(tt, th, Pe[:, 0]), np.interp(tt, th, Pe[:, 1])], 1) - Pe[i0]
            e_raw = np.linalg.norm(traj[-1] - ref[-1])
            H = traj.T @ ref                                    # best rotation (Kabsch, 2D)
            ang = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
            Rr = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            e_al = np.linalg.norm(Rr @ traj[-1] - ref[-1])
            rows.append((e_raw, e_al, np.degrees(ang), np.linalg.norm(ref[-1]), abs(wrapd(np.degrees(yaw[-1] - yaw0[i1 - 1])))))
        if rows:
            r_ = np.array(rows)
            print(f"  window {W:4.0f} s n={len(r_):3d}: endpoint error raw p50 {np.median(r_[:, 0]):.2f} m p90 {np.percentile(r_[:, 0], 90):.2f}; "
                  f"aligned p50 {np.median(r_[:, 1]):.2f} m p90 {np.percentile(r_[:, 1], 90):.2f} ({100 * np.median(r_[:, 1] / np.maximum(r_[:, 3], .1)):.1f}% of {np.median(r_[:, 3]):.1f} m); "
                  f"heading-bias rotation p50 {np.median(r_[:, 2]):+.1f} deg; gyro-int vs yaw_deg end diff p50 {np.median(r_[:, 4]):.2f} deg")

if __name__ == "__main__":
    main()
