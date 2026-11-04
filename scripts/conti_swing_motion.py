"""Continental swing analysis, questions 1-2: motion segmentation (IMU yaw rate vs radar Doppler sensor
velocity), radar-IMU time offset, slew lever arm.

Usage: python scripts/conti_swing_motion.py --data <dir with *_imu.npz, *_radar.npz> --out <dir> [--lidar <npz>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.utils.calib import RigCalib                                         # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics                           # noqa: E402

BAGS = ["20251126_4R_conti_drivemid-front_LC", "20251126_4R_conti_drivemid-side_LC"]
BIN = 0.5
C = {"stop": "#9e9e9e", "straight": "#4c72b0", "swing": "#dd8452", "mixed": "#c8c8c8"}


def moving_avg(t, x, win):
    n = max(1, int(round(win / np.median(np.diff(t)))))
    return np.convolve(x, np.ones(n) / n, mode="same")


def radar_time(rd, mode="hdr"):
    """Radar scan time on the bag clock: receive time, or header (sensor clock, regular) + robust offset."""
    if mode == "bag":
        return rd["t"]
    d = rd["t"] - rd["hdr"]
    return rd["hdr"] + np.percentile(d, 5)                           # min-latency alignment of the sensor clock


def load(data, bag, calib):
    imu = np.load(data / f"{bag}_imu.npz")
    rd = np.load(data / f"{bag}_radar.npz")
    ext = RigExtrinsics.from_calib_json(json.load(open(calib)))
    R_lev, _ = ext.level()
    w_lev = np.stack([imu["wx"], imu["wy"], imu["wz"]], 1) @ R_lev.T
    return imu, rd, w_lev


def segment_bins(t_imu, wz_deg, t_rad, speed, valid, t_end):
    edges = np.arange(0, t_end + BIN, BIN)
    rows = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        mi = (t_imu >= e0) & (t_imu < e1)
        mr = (t_rad >= e0) & (t_rad < e1) & valid
        yr = float(np.median(wz_deg[mi])) if mi.any() else np.nan
        sp = float(np.median(speed[mr])) if mr.any() else np.nan
        if np.isnan(yr):
            cls = "mixed"
        elif abs(yr) > 5:
            cls = "swing"
        elif abs(yr) < 2 and (np.isnan(sp) or sp <= 0.2):
            cls = "stop"
        elif abs(yr) < 2 and sp > 0.2:
            cls = "straight"
        else:
            cls = "mixed"
        rows.append((e0, e1, yr, sp, cls))
    return rows


def merge(rows, min_len=1.0):
    segs = []
    for e0, e1, yr, sp, cls in rows:
        if segs and segs[-1]["cls"] == cls:
            segs[-1]["t1"] = e1
        else:
            segs.append({"cls": cls, "t0": e0, "t1": e1})
    # absorb short segments into neighbours
    out = []
    for s in segs:
        if out and (s["t1"] - s["t0"] < min_len) and s["cls"] != "swing":
            out[-1]["t1"] = s["t1"]
        elif out and out[-1]["t1"] - out[-1]["t0"] < min_len and out[-1]["cls"] != "swing":
            out[-1] = {"cls": s["cls"], "t0": out[-1]["t0"], "t1": s["t1"]}
        else:
            out.append(dict(s))
    merged = []
    for s in out:
        if merged and merged[-1]["cls"] == s["cls"]:
            merged[-1]["t1"] = s["t1"]
        else:
            merged.append(s)
    return merged


def xcorr_lag(t_a, a, t_b, b, lags, fs=100.0, windows=None):
    """Lag (s) maximising the correlation of a(t) with b(t + lag); both resampled at fs (a: radar, b: IMU),
    restricted to the given time windows (else the whole overlap). lag < 0: the IMU shows the feature earlier."""
    if windows is None:
        windows = [(max(t_a[0], t_b[0]), min(t_a[-1], t_b[-1]))]
    t = np.concatenate([np.arange(w0, w1, 1 / fs) for w0, w1 in windows])
    ra = np.interp(t, t_a, a)
    cc = []
    for L in lags:
        rb = np.interp(t + L, t_b, b)
        ra_, rb_ = ra - ra.mean(), rb - rb.mean()
        cc.append(float((ra_ * rb_).sum() / np.sqrt((ra_ ** 2).sum() * (rb_ ** 2).sum() + 1e-12)))
    cc = np.array(cc)
    k = int(np.argmax(cc))
    if 0 < k < len(lags) - 1:                                        # parabolic refinement
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12)
        return float(lags[k] + d * (lags[1] - lags[0])), float(y1), cc
    return float(lags[k]), float(cc[k]), cc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="configs/calib/hdx_boryeong_pg_conti1126.json")
    ap.add_argument("--radar-time", default="hdr", choices=["hdr", "bag"])
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = RigCalib.load(a.calib)
    R_lr = calib.R_lr
    result = {}
    all_pairs = []                                                   # (bag, seg idx, t, v_level xy, wz)
    for bag in BAGS:
        if not (data / f"{bag}_radar.npz").exists():
            print("skip", bag); continue
        imu, rd, w_lev = load(data, bag, a.calib)
        t_imu = imu["t"]
        wz = w_lev[:, 2]
        wz_s = np.degrees(moving_avg(t_imu, wz, 0.2))
        wz_raw_s = np.degrees(moving_avg(t_imu, imu["wz"], 0.1))     # what the pipeline uses (os_imu z, 10 samples)
        t_rad = radar_time(rd, a.radar_time)
        v = np.stack([rd["vx"], rd["vy"], rd["vz"]], 1)
        valid = rd["valid"].astype(bool) & (rd["ratio"] > 0.35)
        speed = np.linalg.norm(v[:, :2], axis=1)
        ang = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        t_end = float(max(t_imu[-1], t_rad[-1]))
        rows = segment_bins(t_imu, wz_s, t_rad, speed, valid, t_end)
        segs = merge(rows)
        dur = {c: sum(s["t1"] - s["t0"] for s in segs if s["cls"] == c) for c in C}
        lat = rd["t"] - rd["hdr"]
        hdr_per = np.diff(rd["hdr"])
        info = {"duration_s": t_end, "n_scans": int(len(rd["t"])), "radar_rate_hz": float(1 / np.median(hdr_per)),
                "radar_hdr_period_ms": {"median": float(np.median(hdr_per) * 1e3), "p99": float(np.percentile(hdr_per, 99) * 1e3),
                                        "max": float(hdr_per.max() * 1e3)},
                "radar_receive_latency_ms": {"p50_minus_p5": float((np.percentile(lat, 50) - np.percentile(lat, 5)) * 1e3),
                                             "p95_minus_p5": float((np.percentile(lat, 95) - np.percentile(lat, 5)) * 1e3),
                                             "max_minus_p5": float((lat.max() - np.percentile(lat, 5)) * 1e3),
                                             "robust_std": float(1.4826 * np.median(np.abs(lat - np.median(lat))) * 1e3)},
                "radar_receive_dt_ms": {"robust_std": float(1.4826 * np.median(np.abs(np.diff(rd["t"]) - np.median(np.diff(rd["t"])))) * 1e3),
                                        "p5": float(np.percentile(np.diff(rd["t"]), 5) * 1e3), "p95": float(np.percentile(np.diff(rd["t"]), 95) * 1e3)},
                "imu_rate_hz": float(1 / np.median(np.diff(t_imu))),
                "imu_gyro_z_noise_deg_s_raw": float(np.degrees(np.std(imu["wz"][(t_imu > 5) & (t_imu < 15)]))),
                "imu_tilt_deg": float(np.degrees(np.arccos(RigExtrinsics.from_calib_json(json.load(open(a.calib))).ground_normal[2]))),
                "ego_valid_frac": float(valid.mean()),
                "segments": segs, "durations": dur,
                "bins": [{"t0": r[0], "t1": r[1], "yaw_rate_deg_s": r[2], "speed": r[3], "cls": r[4]} for r in rows]}

        # ---------- time offset: |v_sensor| vs |wz| in the swing segments (per swing and pooled) ----------
        lags = np.arange(-0.6, 0.6001, 0.01)
        swings = [s for s in segs if s["cls"] == "swing" and s["t1"] - s["t0"] >= 3.0]
        per, per_cc = [], []
        m_all = np.zeros(len(t_rad), bool)
        for s in swings:
            m = (t_rad >= s["t0"] - 1.5) & (t_rad <= s["t1"] + 1.5) & valid
            m_all |= m
            if m.sum() < 40:
                continue
            L, c, cc_s = xcorr_lag(t_rad[m], speed[m], t_imu, np.abs(wz_s), lags)
            per_cc.append(cc_s)
            per.append({"t0": s["t0"], "t1": s["t1"], "lag_s": L, "corr": c, "n": int(m.sum()),
                        "peak_yaw_deg_s": float(np.max(np.abs(wz_s[(t_imu >= s["t0"]) & (t_imu <= s["t1"])]))),
                        "peak_speed": float(np.percentile(speed[m], 98))})
        good = [p for p in per if p["corr"] > 0.99]                 # the pooled lag: median of the clean swings
        Lp = float(np.median([p["lag_s"] for p in good])) if good else (float(np.median([p["lag_s"] for p in per])) if per else np.nan)
        cp = float(np.median([p["corr"] for p in good])) if good else np.nan
        cc_pool = np.mean(per_cc, axis=0) if per_cc else None
        info["time_offset"] = {"per_swing": per, "pooled_lag_s": Lp, "pooled_corr": cp,
                               "lag_std_across_swings_s": float(np.std([p["lag_s"] for p in per])) if len(per) > 1 else None,
                               "note": "lag < 0: the IMU shows the feature BEFORE the radar does on the bag clock (radar time = header + min latency), i.e. the radar scan describes the state at t_radar + lag; corrected radar time = t_radar + lag"}

        # ---------- lever arm from (v_level, wz) pairs with the pooled lag applied ----------
        lag = Lp if np.isfinite(Lp) else 0.0
        v_lev = v @ R_lr.T
        wz_at = np.interp(t_rad + lag, t_imu, wz_s)                   # deg/s at the (corrected) radar time
        arms = []
        for s in swings:
            m = (t_rad >= s["t0"]) & (t_rad <= s["t1"]) & valid & (np.abs(wz_at) > 5)
            if m.sum() < 20:
                continue
            w = np.radians(wz_at[m])
            vx, vy = v_lev[m, 0], v_lev[m, 1]
            # v = w x r = w * (-r_y, r_x)
            r_x = float((w * vy).sum() / (w * w).sum())
            r_y = float(-(w * vx).sum() / (w * w).sum())
            res = np.hypot(vx + w * r_y, vy - w * r_x)
            # direction check: angle between v and r (should be 90 deg)
            dot = (vx * r_x + vy * r_y) / (np.hypot(vx, vy) * np.hypot(r_x, r_y) + 1e-9)
            arms.append({"t0": s["t0"], "t1": s["t1"], "n": int(m.sum()), "r_x": r_x, "r_y": r_y,
                         "r_norm": float(np.hypot(r_x, r_y)), "r_dir_deg": float(np.degrees(np.arctan2(r_y, r_x))),
                         "resid_rms": float(np.sqrt((res ** 2).mean())), "v_r_angle_deg": float(np.degrees(np.arccos(np.clip(np.median(dot), -1, 1)))),
                         "sense": float(np.sign(np.median(np.sign(w) * np.sign(vy)))), "mean_abs_w_deg_s": float(np.degrees(np.abs(w).mean()))})
            for ti, vxi, vyi, wi in zip(t_rad[m], vx, vy, w):
                all_pairs.append((bag, s["t0"], ti, vxi, vyi, wi))
        # pooled fit
        m = m_all & (np.abs(wz_at) > 5)
        w = np.radians(wz_at[m])
        vx, vy = v_lev[m, 0], v_lev[m, 1]
        if m.sum() > 20:
            r_x = float((w * vy).sum() / (w * w).sum()); r_y = float(-(w * vx).sum() / (w * w).sum())
            res = np.hypot(vx + w * r_y, vy - w * r_x)
            pooled = {"r_x": r_x, "r_y": r_y, "r_norm": float(np.hypot(r_x, r_y)), "r_dir_deg": float(np.degrees(np.arctan2(r_y, r_x))),
                      "resid_rms": float(np.sqrt((res ** 2).mean())), "n": int(m.sum()),
                      "resid_rms_no_lag": None}
            # same without lag (for the report)
            wz0 = np.radians(np.interp(t_rad, t_imu, wz_s))[m]
            r_x0 = float((wz0 * vy).sum() / (wz0 * wz0).sum()); r_y0 = float(-(wz0 * vx).sum() / (wz0 * wz0).sum())
            pooled["resid_rms_no_lag"] = float(np.sqrt((np.hypot(vx + wz0 * r_y0, vy - wz0 * r_x0) ** 2).mean()))
            pooled["r_no_lag"] = [r_x0, r_y0]
        else:
            pooled = None
        info["lever_arm"] = {"per_swing": arms, "pooled": pooled,
                             "std_across_swings": {k: float(np.std([x[k] for x in arms])) for k in ("r_x", "r_y", "r_norm", "r_dir_deg")} if len(arms) > 1 else None}
        result[bag] = info

        # ---------- LiDAR odometry (swing windows only) ----------
        lid_path = data / f"{bag.split('_')[3].split('-')[1]}_lidar.npz"
        lid = None
        if lid_path.exists():
            L = np.load(lid_path)
            tl, yr_l, sp_l, win_stats = [], [], [], []
            for wi in np.unique(L["win"]):
                m = L["win"] == wi
                T, tt = L["T"][m], L["t"][m]
                yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
                yr = np.degrees(np.diff(yaw) / np.diff(tt)); sp = np.linalg.norm(np.diff(T[:, :2, 3], axis=0), axis=1) / np.diff(tt)
                tm = 0.5 * (tt[1:] + tt[:-1])
                iy = np.interp(tm, t_imu, wz_s)
                tl.append(tm); yr_l.append(yr); sp_l.append(sp)
                d = yr - iy
                win_stats.append({"t0": float(tt[0]), "t1": float(tt[-1]), "yaw_total_lidar_deg": float(np.degrees(yaw[-1] - yaw[0])),
                                  "yaw_total_imu_deg": float(np.trapz(np.interp(tt, t_imu, wz_s), tt)),
                                  "yawrate_lidar_minus_imu_med": float(np.median(d)), "yawrate_lidar_minus_imu_rstd": float(1.4826 * np.median(np.abs(d - np.median(d)))),
                                  "fit_p5": float(np.percentile(L["fit"][m], 5)), "rmse_med": float(np.median(L["rmse"][m]))})
            lid = (np.concatenate(tl), np.concatenate(yr_l), np.concatenate(sp_l))
            info["lidar_odometry"] = win_stats
        # ---------- timeline figure ----------
        fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
        for s in segs:
            for axx in ax:
                axx.axvspan(s["t0"], s["t1"], color=C[s["cls"]], alpha=0.18, lw=0)
        ax[0].plot(t_imu, np.degrees(wz), color="#bbbbbb", lw=0.3, label="IMU wz raw (level)")
        ax[0].plot(t_imu, wz_s, color="k", lw=0.8, label="IMU wz 0.2 s avg")
        if lid is not None:
            ax[0].plot(lid[0], lid[1], ".", ms=2, color="#1f77b4", label="LiDAR odometry yaw rate")
            ax[1].plot(lid[0], lid[2], ".", ms=2, color="#17becf", label="LiDAR odometry speed")
        ax[0].set_ylabel("yaw rate [deg/s]"); ax[0].legend(loc="upper right", fontsize=8); ax[0].set_ylim(-40, 40)
        ax[1].plot(t_rad[valid], speed[valid], ".", ms=2, color="#1f77b4", label="|v_sensor| radar (valid)")
        ax[1].plot(t_rad[~valid], speed[~valid], "x", ms=3, color="#d62728", label="ego fit invalid")
        ax[1].set_ylabel("sensor speed [m/s]"); ax[1].legend(loc="upper right", fontsize=8); ax[1].set_ylim(0, 2.5)
        ax[2].plot(t_rad[valid & (speed > 0.15)], ang[valid & (speed > 0.15)], ".", ms=2, color="#2ca02c")
        ax[2].set_ylabel("v_sensor dir [deg]\n(radar frame, from +x)"); ax[2].set_ylim(-180, 180)
        ax[3].plot(t_rad[valid], rd["ratio"][valid], ".", ms=2, color="#7f7f7f", label="static inlier ratio")
        ax[3].plot(t_rad, rd["n"] / rd["n"].max(), "-", lw=0.5, color="#9467bd", label="n points / max")
        ax[3].legend(loc="upper right", fontsize=8); ax[3].set_xlabel("bag time [s]")
        for c, col in C.items():
            ax[0].plot([], [], color=col, lw=6, alpha=0.4, label=f"{c} {dur[c]:.1f}s")
        ax[0].legend(loc="upper right", fontsize=8, ncol=3)
        fig.suptitle(f"{bag}: motion timeline (radar time = {a.radar_time})")
        fig.tight_layout()
        fig.savefig(out / f"timeline_{bag.split('_')[3]}.png", dpi=110)
        plt.close(fig)

        # ---------- cross-correlation + lever arm figure ----------
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        if cc_pool is not None:
            for p, cc_s in zip(per, per_cc):
                ax[0].plot(lags, cc_s, lw=0.8, label=f"swing {p['t0']:.0f}s: {p['lag_s'] * 1e3:.0f} ms (corr {p['corr']:.3f})")
            ax[0].axvline(Lp, color="r", label=f"median lag {Lp * 1e3:.0f} ms")
            ax[0].set_xlabel("lag [s] (IMU later than radar)"); ax[0].set_ylabel("corr(|v_sensor|, |wz|)"); ax[0].legend()
        m = m_all & valid
        sc = ax[1].scatter(np.radians(wz_at[m]), v_lev[m, 1], c=t_rad[m], s=4, cmap="viridis")
        if pooled:
            ww = np.linspace(-0.6, 0.6, 10)
            ax[1].plot(ww, ww * pooled["r_x"], "r-", label=f"slope r_x = {pooled['r_x']:.2f} m")
        ax[1].set_xlabel("wz [rad/s] (lag applied)"); ax[1].set_ylabel("v_y level [m/s]"); ax[1].legend(); ax[1].grid(alpha=0.3)
        ax[2].scatter(np.radians(wz_at[m]), v_lev[m, 0], c=t_rad[m], s=4, cmap="viridis")
        if pooled:
            ax[2].plot(ww, -ww * pooled["r_y"], "r-", label=f"slope -r_y = {-pooled['r_y']:.2f} m")
        ax[2].set_xlabel("wz [rad/s] (lag applied)"); ax[2].set_ylabel("v_x level [m/s]"); ax[2].legend(); ax[2].grid(alpha=0.3)
        fig.suptitle(f"{bag}: radar-IMU lag and lever arm fit v = w x r")
        fig.tight_layout()
        fig.savefig(out / f"leverarm_{bag.split('_')[3]}.png", dpi=110)
        plt.close(fig)
        print(bag, json.dumps({k: info[k] for k in ("durations", "time_offset", "lever_arm", "radar_receive_latency_ms", "radar_hdr_period_ms")}, indent=1, default=float))

    json.dump(result, open(out / "motion_result.json", "w"), indent=1, default=float)
    np.save(out / "lever_pairs.npy", np.array([(p[2], p[3], p[4], p[5]) for p in all_pairs]))


if __name__ == "__main__":
    main()
