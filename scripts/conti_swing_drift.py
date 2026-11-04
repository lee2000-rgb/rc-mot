"""Continental swing analysis, question 3: static-scene drift of accumulated radar points under several
ego-motion compensations, with LiDAR odometry (conti_swing_lidar.py) as the reference.

Compensation variants (fixed frame = level frame of the first radar scan of the window):
  none      : no compensation
  pipe      : the pipeline as it runs: dp = R_lr ego_vel_EMA * dt, dyaw = mean(last 10 IMU wz before the frame is
              EMITTED, i.e. when the first image with header >= radar receive time arrives) * dt, dt from receive times
  pipe_t0   : same, but the IMU window ends at the radar receive time (no image-pairing delay)
  lag       : pipe_t0 + radar-IMU lag (IMU sampled around t_radar + lag), header-clock dt
  lag_noema : lag, with the per-scan v_sensor (no EMA) and a centred 50 ms IMU average
  lidar     : LiDAR odometry pose chain (reference: what perfect ego motion would give)
Metric per static object (LiDAR cluster) and 2 s window: robust 2D spread of the accumulated radar points and the
apparent velocity (linear drift of the per-scan centroid).

Usage: python scripts/conti_swing_drift.py --data <dir> --out <dir> --lag -0.067
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
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.utils.calib import RigCalib                                         # noqa: E402

BAGS = {"front": "20251126_4R_conti_drivemid-front_LC", "side": "20251126_4R_conti_drivemid-side_LC"}
VARIANTS = ["none", "pipe", "pipe_t0", "lag", "lag_noema", "lidar"]
COL = {"none": "#9e9e9e", "pipe": "#d62728", "pipe_t0": "#ff7f0e", "lag": "#2ca02c", "lag_noema": "#17becf", "lidar": "#1f77b4"}
MAD = 1.4826


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def pose_interp(tL, T, t):
    """Interpolate the LiDAR pose chain (yaw unwrapped, xy linear) at times t -> (yaw, xy)."""
    yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    return np.interp(t, tL, yaw), np.stack([np.interp(t, tL, T[:, 0, 3]), np.interp(t, tL, T[:, 1, 3])], 1)


def imu_mean(t_imu, w, t_end, n=10):
    k = np.searchsorted(t_imu, t_end)
    return float(np.mean(w[max(0, k - n):k])) if k > 0 else 0.0


def imu_centred(t_imu, w, t_c, half=0.025):
    m = (t_imu >= t_c - half) & (t_imu <= t_c + half)
    return float(np.mean(w[m])) if m.any() else float(np.interp(t_c, t_imu, w))


def chain(dyaw, dp):
    """Pose of each scan's level frame in the fixed (first-scan) frame, from per-step (dyaw, dp old-frame)."""
    yaw, pos = 0.0, np.zeros(2)
    out = []
    for dy, d in zip(dyaw, dp):
        out.append((yaw, pos.copy()))
        pos = pos + rotz(yaw) @ d[:2]
        yaw = yaw + dy
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def robust_spread(xy):
    if len(xy) < 5:
        return np.nan
    med = np.median(xy, 0)
    return float(np.hypot(MAD * np.median(np.abs(xy[:, 0] - med[0])), MAD * np.median(np.abs(xy[:, 1] - med[1]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", default="configs/calib/hdx_boryeong_pg_conti1126.json")
    ap.add_argument("--lag", type=float, default=-0.067, help="radar-IMU lag [s]: IMU time = radar hdr time + lag")
    ap.add_argument("--vcomp-max", type=float, default=0.3)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--attr-radius", type=float, default=1.0)
    ap.add_argument("--max-objects", type=int, default=5)
    ap.add_argument("--ext", action="append", default=[], help="name=<chain.json from conti_pose_dump.py> (per bag key: name:front=..,side=..)")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calib = RigCalib.load(a.calib)
    ext_chains = {}                                                 # name -> {bag key -> rows}
    for spec in a.ext:
        name, rest = spec.split("=", 1)
        for part in rest.split(","):
            key_, path_ = part.split(":", 1)
            ext_chains.setdefault(name, {})[key_] = json.load(open(path_))
    for name in ext_chains:
        if name not in VARIANTS:
            VARIANTS.append(name)
    results = {}
    rows_all = []                                                    # per (bag, win, object, 2 s window, variant)
    pose_rows_all, geo_curves = [], []

    for key, bag in BAGS.items():
        imu = np.load(data / f"{bag}_imu.npz"); rd = np.load(data / f"{bag}_radar.npz"); cam = np.load(data / f"{bag}_cam.npz")
        L = np.load(data / f"{key}_lidar.npz")
        t_imu, wz_raw = imu["t"], imu["wz"]
        t_bag = rd["t"]
        lat5 = np.percentile(rd["t"] - rd["hdr"], 5)
        t_hdr = rd["hdr"] + lat5                                     # regular sensor clock on the bag timeline
        v_s = np.stack([rd["vx"], rd["vy"], rd["vz"]], 1)
        valid = rd["valid"].astype(bool)
        cam_t0 = float(cam["t0"])
        cam_hdr_rel = cam["hdr"] - cam_t0
        cam_bag_t = cam["t"]
        # emission time of each radar frame in bag_source.iterate_bag: first image (in receive order) whose header >= radar receive time
        order = np.argsort(cam_bag_t)
        hdr_sorted_max = np.maximum.accumulate(cam_hdr_rel[order])    # running max of header along receive order
        emit = np.array([cam_bag_t[order][np.searchsorted(hdr_sorted_max, tb)] if np.searchsorted(hdr_sorted_max, tb) < len(order) else np.inf for tb in t_bag])
        pxyz = calib.radar_to_level(np.stack([rd["p_x"], rd["p_y"], rd["p_z"]], 1).astype(float))
        pscan = rd["p_scan"]; pv = rd["p_v_comp"]; pkeep = rd["p_keep"].astype(bool)
        bag_res = {"windows": [], "emission_delay_s": {"p50": float(np.median((emit - t_bag)[np.isfinite(emit)])),
                                                        "p95": float(np.percentile((emit - t_bag)[np.isfinite(emit)], 95))}}
        for wi in np.unique(L["win"]):
            mL = L["win"] == wi
            tL, TL = L["t"][mL], L["T"][mL]
            w0, w1 = L["windows"][wi]
            # ---- radar scans of the window (margin 2 s for the EMA warm-up) ----
            js = np.flatnonzero((t_hdr >= w0 - 2.0) & (t_hdr <= w1 + 0.5))
            dt_bag = np.diff(t_bag[js], prepend=t_bag[js[0]] - 0.05)
            dt_hdr = np.diff(t_hdr[js], prepend=t_hdr[js[0]] - 0.05)
            # pipeline EMA of the ego velocity (jump gate as in pipeline._radar)
            ego, age, ema_v = np.zeros(3), 10.0, []
            for j, dt in zip(js, dt_bag):
                age += dt
                ok = valid[j] and np.isfinite(v_s[j]).all() and np.linalg.norm(v_s[j]) <= 15 and \
                    (age > 1.0 or np.linalg.norm(v_s[j] - ego) <= 1.0 * (1 + age / 0.066))
                if ok:
                    ego = 0.6 * ego + 0.4 * v_s[j]; age = 0.0
                ema_v.append(ego.copy())
            ema_v = np.array(ema_v)
            v_lvl_ema = ema_v @ calib.R_lr.T
            v_lvl_raw = np.where(valid[js, None], v_s[js] @ calib.R_lr.T, v_lvl_ema)
            yaw_pipe = np.array([imu_mean(t_imu, wz_raw, e) for e in emit[js]])
            yaw_t0 = np.array([imu_mean(t_imu, wz_raw, tb) for tb in t_bag[js]])
            yaw_lag_win = np.array([imu_mean(t_imu, wz_raw, th + a.lag + 0.05) for th in t_hdr[js]])   # 100 ms window centred on t+lag
            yaw_lag_c = np.array([imu_centred(t_imu, wz_raw, th + a.lag) for th in t_hdr[js]])
            chains = {
                "none": chain(np.zeros(len(js)), np.zeros((len(js), 3))),
                "pipe": chain(yaw_pipe * dt_bag, v_lvl_ema * dt_bag[:, None]),
                "pipe_t0": chain(yaw_t0 * dt_bag, v_lvl_ema * dt_bag[:, None]),
                "lag": chain(yaw_lag_win * dt_hdr, v_lvl_ema * dt_hdr[:, None]),
                "lag_noema": chain(yaw_lag_c * dt_hdr, v_lvl_raw * dt_hdr[:, None]),
            }
            for name, per_key in ext_chains.items():               # the real pipeline's increments (conti_pose_dump.py)
                if key not in per_key:
                    continue
                et = np.array([r["t_bag"] for r in per_key[key]]) - float(rd["t0"])   # dump = epoch, table = bag-relative
                dy = np.zeros(len(js)); dpp = np.zeros((len(js), 3))
                for jj, j in enumerate(js):
                    k_ = int(np.argmin(np.abs(et - t_bag[j])))
                    if abs(et[k_] - t_bag[j]) < 0.03:
                        dy[jj] = per_key[key][k_]["dyaw"]; dpp[jj] = per_key[key][k_]["dp"]
                chains[name] = chain(dy, dpp)
                if name not in COL:
                    COL[name] = "#8c564b"
            # LiDAR reference pose at the corrected radar times, re-based to the first radar scan of the window
            yawL, xyL = pose_interp(tL, TL, t_hdr[js] + a.lag)
            R0 = rotz(yawL[0]).T
            chains["lidar"] = (yawL - yawL[0], (xyL - xyL[0]) @ R0.T)
            # ---- static objects from the LiDAR map (fixed frame of the first LiDAR scan -> re-based like above) ----
            mp = L["pts"][np.isin(L["scan"], np.flatnonzero(mL))]
            ms = L["scan"][np.isin(L["scan"], np.flatnonzero(mL))] - np.flatnonzero(mL)[0]
            mapxy = np.einsum("nij,nj->ni", TL[ms][:, :2, :2], mp[:, :2].astype(float)) + TL[ms][:, :2, 3]
            mapxy = (mapxy - xyL[0]) @ R0.T                          # into the first-radar-scan frame
            # object candidates: coarse grid downsample then DBSCAN
            g = np.round(mapxy / 0.3).astype(int)
            _, ui = np.unique(g, axis=0, return_index=True)
            cand = mapxy[ui]
            lab = DBSCAN(eps=0.8, min_samples=12).fit_predict(cand)
            objs = []
            for l in np.unique(lab[lab >= 0]):
                c = cand[lab == l]
                ext = c.max(0) - c.min(0)
                r = np.linalg.norm(c.mean(0))
                if not (6 <= r <= 40) or ext.max() > 14 or ext.max() < 1.0:
                    continue
                near = cKDTree(c).query(mapxy, distance_upper_bound=0.5)[0] < 0.5
                full = mapxy[near]
                # static check: centroid of early vs late LiDAR scans
                sc = ms[near]
                if len(np.unique(sc)) < 15:
                    continue
                q1, q3 = np.percentile(sc, [33, 66])
                drift = np.linalg.norm(full[sc <= q1].mean(0) - full[sc >= q3].mean(0))
                hz = mp[near, 2]
                objs.append({"xy": c.mean(0), "pts": c, "range": float(r), "extent": ext.tolist(), "lidar_drift_m": float(drift),
                             "height": float(np.percentile(hz, 95)), "n_lidar": int(len(full))})
            objs = [o for o in objs if o["lidar_drift_m"] < 0.4]
            # ---- attribute static radar points to objects via the LiDAR reference pose ----
            per_scan_pts = {}
            for jj, j in enumerate(js):
                m = (pscan == j) & (np.abs(pv) < a.vcomp_max) & pkeep
                per_scan_pts[jj] = pxyz[m][:, :2]
            for o in objs:
                tree = cKDTree(o["pts"])
                o["hits"] = []                                       # (jj, local xy of attributed points)
                for jj in range(len(js)):
                    p = per_scan_pts[jj]
                    if len(p) == 0:
                        continue
                    pw = (rotz(chains["lidar"][0][jj]) @ p.T).T + chains["lidar"][1][jj]
                    d, _ = tree.query(pw, distance_upper_bound=a.attr_radius)
                    if (d < a.attr_radius).sum() >= 1:
                        o["hits"].append((jj, p[d < a.attr_radius]))
                o["n_radar"] = sum(len(h[1]) for h in o["hits"])
                o["n_scans"] = len(o["hits"])
            objs = sorted([o for o in objs if o["n_scans"] >= 15], key=lambda o: -o["n_radar"])[:a.max_objects]
            # ---- pose-based metric: chain vs LiDAR pose, 2 s windows ----
            tj = t_hdr[js]
            om_j = np.degrees(np.array([imu_centred(t_imu, wz_raw, t + a.lag, 0.1) for t in tj]))
            pose_rows = []
            for vname, (yaws, poss) in chains.items():
                ye = np.degrees(yaws - chains["lidar"][0]); pe = poss - chains["lidar"][1]
                for ws in np.arange(w0, w1 - a.win * 0.5, a.win * 0.5):
                    m = (tj >= ws) & (tj < ws + a.win)
                    if m.sum() < 8:
                        continue
                    A = np.c_[tj[m] - tj[m].mean(), np.ones(m.sum())]
                    sy = np.linalg.lstsq(A, ye[m], rcond=None)[0][0]
                    sp_ = np.linalg.lstsq(A, pe[m], rcond=None)[0][0]
                    omega = float(np.mean(om_j[m]))
                    domega = float((om_j[m][-1] - om_j[m][0]) / (tj[m][-1] - tj[m][0]))
                    v20 = float(np.hypot(20 * np.radians(sy), np.linalg.norm(sp_)))
                    e20 = float(np.hypot(20 * np.radians(ye[m][-1] - ye[m][0]), np.linalg.norm(pe[m][-1] - pe[m][0])))
                    pose_rows.append((key, int(wi), vname, float(ws), omega, domega, float(sy), float(np.linalg.norm(sp_)), v20, e20))
            pose_rows_all.extend(pose_rows)
            # per-step yaw increment noise (chain vs lidar), steady swing only
            steady = np.abs(om_j) > 10
            step_noise = {}
            for vname, (yaws, poss) in chains.items():
                d = np.degrees(np.diff(yaws) - np.diff(chains["lidar"][0]))[steady[1:]]
                step_noise[vname] = {"yaw_step_err_deg_rstd": float(MAD * np.median(np.abs(d - np.median(d)))) if len(d) else None,
                                     "yaw_step_err_deg_median": float(np.median(d)) if len(d) else None}
            # ---- geometric lag scan: static radar points vs LiDAR map as a function of the assumed radar time shift ----
            map_tree = cKDTree(mapxy)
            taus = np.arange(-0.25, 0.2501, 0.01)
            med_d = []
            fast = [jj for jj in range(len(js)) if abs(om_j[jj]) > 10 and len(per_scan_pts[jj]) >= 3]
            for tau in taus:
                yq, xyq = pose_interp(tL, TL, t_hdr[js[fast]] + tau)
                dd = []
                for k_, jj in enumerate(fast):
                    p = per_scan_pts[jj]
                    rg = np.linalg.norm(p, axis=1)
                    p = p[(rg > 5) & (rg < 40)]
                    if len(p) == 0:
                        continue
                    pw = ((rotz(yq[k_]) @ p.T).T + xyq[k_] - xyL[0]) @ R0.T
                    d, _ = map_tree.query(pw, distance_upper_bound=3.0)
                    dd.append(np.minimum(d, 3.0))
                dd = np.concatenate(dd)
                med_d.append(float(np.mean(dd[dd < 3.0])) if (dd < 3.0).any() else np.nan)
            med_d = np.array(med_d)
            kb = int(np.nanargmin(med_d))
            tau_geo = float(taus[kb])
            if 0 < kb < len(taus) - 1:
                y0, y1, y2 = med_d[kb - 1], med_d[kb], med_d[kb + 1]
                tau_geo = float(taus[kb] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12) * 0.01)
            geo_curves.append((key, int(wi), taus, med_d, tau_geo))
            # ---- metrics per object, 2 s windows, variant ----
            wres = {"t0": float(w0), "t1": float(w1), "objects": [], "n_candidates": len(objs), "step_noise": step_noise,
                    "tau_geo_s": tau_geo, "tau_geo_min_dist_m": float(np.nanmin(med_d)), "tau_geo_dist_at_0": float(med_d[np.argmin(np.abs(taus))]),
                    "tau_geo_dist_at_lag": float(np.interp(a.lag, taus, med_d))}
            for oi, o in enumerate(objs):
                orow = {"range": o["range"], "xy": o["xy"].tolist(), "extent": o["extent"], "height": o["height"],
                        "n_radar": o["n_radar"], "n_scans": o["n_scans"], "lidar_drift_m": o["lidar_drift_m"], "windows": []}
                hit_t = np.array([t_hdr[js[h[0]]] for h in o["hits"]])
                for ws in np.arange(hit_t.min(), hit_t.max() - a.win * 0.5, a.win * 0.5):
                    sel = [h for h, t in zip(o["hits"], hit_t) if ws <= t < ws + a.win]
                    if len(sel) < 6:
                        continue
                    tt = np.array([t_hdr[js[h[0]]] for h in sel])
                    omega = float(np.degrees(np.mean([imu_centred(t_imu, wz_raw, t + a.lag, 0.1) for t in tt])))
                    rowv = {"t": float(tt.mean()), "yaw_rate_deg_s": omega, "n": int(sum(len(h[1]) for h in sel)), "variants": {}}
                    for vname, (yaws, poss) in chains.items():
                        acc, cents = [], []
                        for h in sel:
                            jj, p = h
                            pw = (rotz(yaws[jj]) @ p.T).T + poss[jj]
                            acc.append(pw); cents.append(pw.mean(0))
                        acc = np.concatenate(acc); cents = np.array(cents)
                        A = np.c_[tt - tt.mean(), np.ones(len(tt))]
                        vel = np.linalg.lstsq(A, cents, rcond=None)[0][0]
                        rowv["variants"][vname] = {"spread_m": robust_spread(acc), "v_app_m_s": float(np.linalg.norm(vel)),
                                                   "disp_m": float(np.linalg.norm(cents[-1] - cents[0]))}
                        rows_all.append((key, int(wi), oi, o["range"], omega, vname, robust_spread(acc), float(np.linalg.norm(vel)), float(max(o["extent"]))))
                    orow["windows"].append(rowv)
                wres["objects"].append(orow)
                o["tt"] = hit_t
            bag_res["windows"].append(wres)

            # ---- figure: map + accumulations for 4 variants ----
            fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
            for k, vname in enumerate(["none", "pipe", "lag", "lidar"]):
                yaws, poss = chains[vname]
                axx = ax[k]
                axx.scatter(mapxy[::4, 0], mapxy[::4, 1], s=0.3, c="#cfcfcf", label="LiDAR map (odometry)")
                for oi, o in enumerate(objs):
                    acc = np.concatenate([(rotz(yaws[jj]) @ p.T).T + poss[jj] for jj, p in o["hits"]])
                    axx.scatter(acc[:, 0], acc[:, 1], s=4, color=plt.cm.tab10(oi % 10), label=f"obj{oi} r={o['range']:.0f}m")
                    axx.annotate(f"{oi}", o["xy"], fontsize=9, weight="bold")
                sp = poss
                axx.plot(sp[:, 0], sp[:, 1], "k-", lw=1, label="sensor path")
                axx.set_aspect("equal"); axx.set_title(f"{vname}"); axx.set_xlim(-45, 45); axx.set_ylim(-45, 45)
                axx.grid(alpha=0.3)
            ax[0].legend(fontsize=7, loc="lower left")
            fig.suptitle(f"{bag} swing {w0:.0f}-{w1:.0f}s: static radar points (|v_comp|<{a.vcomp_max}) accumulated in the first-scan frame")
            fig.tight_layout()
            fig.savefig(out / f"drift_map_{key}_{int(w0)}.png", dpi=100)
            plt.close(fig)
            print(f"{key} win {wi} {w0:.1f}-{w1:.1f}: {len(objs)} objects; emission delay p50 {bag_res['emission_delay_s']['p50']:.2f}s", flush=True)
            for oi, o in enumerate(objs):
                print(f"   obj{oi} r={o['range']:.1f} ext={np.round(o['extent'], 1)} h={o['height']:.1f} nradar={o['n_radar']} nscan={o['n_scans']} ldrift={o['lidar_drift_m']:.2f}")
        results[key] = bag_res

    # ---------- summary over all windows ----------
    rows = np.array(rows_all, dtype=object)
    summary = {}
    for vname in VARIANTS:
        m = rows[:, 5] == vname
        sp = rows[m, 6].astype(float); va = rows[m, 7].astype(float); om = np.abs(rows[m, 4].astype(float)); rg = rows[m, 3].astype(float)
        ok = np.isfinite(sp)
        summary[vname] = {"n_windows": int(m.sum()), "spread_med": float(np.nanmedian(sp)), "spread_p90": float(np.nanpercentile(sp, 90)),
                          "v_app_med": float(np.median(va)), "v_app_p90": float(np.percentile(va, 90)),
                          "v_app_med_fast(>12deg/s)": float(np.median(va[om > 12])) if (om > 12).any() else None,
                          "v_app_med_ramp(<12deg/s)": float(np.median(va[om <= 12])) if (om <= 12).any() else None,
                          "v_app_at20m_per_deg_s": None}
        # apparent velocity normalised to 20 m vs |yaw rate|: slope through origin
        if ok.any():
            x = om[ok]; y = va[ok] * 20.0 / rg[ok]
            summary[vname]["v_app_at20m_per_deg_s"] = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else None
    results["summary"] = summary
    small = rows[:, 8].astype(float) <= 5.0
    for vname in VARIANTS:
        m = (rows[:, 5] == vname) & small
        sp = rows[m, 6].astype(float); va = rows[m, 7].astype(float)
        summary[vname]["small_objects"] = {"n": int(m.sum()), "spread_med": float(np.nanmedian(sp)), "spread_p90": float(np.nanpercentile(sp, 90)),
                                           "v_app_med": float(np.median(va)), "v_app_p90": float(np.percentile(va, 90))}
    P = np.array(pose_rows_all, dtype=object)
    pose_sum = {}
    for vname in VARIANTS:
        m = P[:, 2] == vname
        om = np.abs(P[m, 4].astype(float)); dom = np.abs(P[m, 5].astype(float)); v20 = P[m, 8].astype(float); e20 = P[m, 9].astype(float)
        sy = P[m, 6].astype(float); sp_ = P[m, 7].astype(float)
        steady = (om > 10) & (dom < 3); ramp = ~steady
        pose_sum[vname] = {"n": int(m.sum()),
                           "yaw_drift_deg_s_med": float(np.median(np.abs(sy))), "yaw_drift_deg_s_p90": float(np.percentile(np.abs(sy), 90)),
                           "pos_drift_m_s_med": float(np.median(sp_)), "pos_drift_m_s_p90": float(np.percentile(sp_, 90)),
                           "v_app20_med": float(np.median(v20)), "v_app20_p90": float(np.percentile(v20, 90)),
                           "err20_2s_med": float(np.median(e20)), "err20_2s_p90": float(np.percentile(e20, 90)),
                           "v_app20_steady_med": float(np.median(v20[steady])) if steady.any() else None,
                           "v_app20_ramp_med": float(np.median(v20[ramp])) if ramp.any() else None,
                           "err20_2s_steady_p90": float(np.percentile(e20[steady], 90)) if steady.any() else None,
                           "err20_2s_ramp_p90": float(np.percentile(e20[ramp], 90)) if ramp.any() else None,
                           "v20_per_deg_s_steady": float((om[steady] * v20[steady]).sum() / (om[steady] ** 2).sum()) if steady.any() else None}
        # yaw rate at which the 2 s error at 20 m exceeds 0.5 m: from the per-window errors binned by |omega|
        bins = [0, 5, 10, 15, 20, 30, 60]
        pb = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mb = (om >= b0) & (om < b1)
            if mb.sum() >= 2:
                pb.append({"omega_bin": [b0, b1], "n": int(mb.sum()), "err20_2s_med": float(np.median(e20[mb])), "err20_2s_p90": float(np.percentile(e20[mb], 90)),
                           "v_app20_med": float(np.median(v20[mb]))})
        pose_sum[vname]["by_omega"] = pb
    results["pose_summary"] = pose_sum
    results["tau_geo"] = [{"bag": g[0], "win": g[1], "tau_geo_s": g[4]} for g in geo_curves]
    json.dump(results, open(out / "drift_result.json", "w"), indent=1, default=float)

    # ---------- figure: pose-based drift vs yaw rate, and the geometric lag curves ----------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for vname in VARIANTS:
        if vname == "lidar":
            continue
        m = P[:, 2] == vname
        om = np.abs(P[m, 4].astype(float)); v20 = P[m, 8].astype(float); e20 = P[m, 9].astype(float)
        ax[0].scatter(om, v20, s=12, color=COL[vname], alpha=0.75, label=f"{vname} (med {np.median(v20):.2f} m/s)")
        ax[1].scatter(om, e20, s=12, color=COL[vname], alpha=0.75, label=f"{vname} (p90 {np.percentile(e20, 90):.2f} m)")
    ax[0].set_xlabel("|yaw rate| [deg/s] (2 s window mean)"); ax[0].set_ylabel("apparent velocity of a static point at 20 m [m/s]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_yscale("log")
    ax[1].set_xlabel("|yaw rate| [deg/s]"); ax[1].set_ylabel("position error at 20 m accumulated over 2 s [m]"); ax[1].axhline(0.5, color="k", ls="--", lw=0.8); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_yscale("log")
    for g in geo_curves:
        ax[2].plot(g[2] * 1e3, g[3], label=f"{g[0]} {g[1]}: tau* = {g[4] * 1e3:.0f} ms")
    ax[2].axvline(a.lag * 1e3, color="k", ls="--", label=f"Doppler xcorr lag {a.lag * 1e3:.0f} ms")
    ax[2].set_xlabel("assumed radar time shift [ms] (IMU/LiDAR time = radar header time + shift)"); ax[2].set_ylabel("mean radar-point-to-LiDAR-map distance [m]"); ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle("pose-chain error vs LiDAR odometry (left, centre) and geometric radar time-shift scan (right)")
    fig.tight_layout()
    fig.savefig(out / "drift_pose_vs_yawrate.png", dpi=110)
    plt.close(fig)
    print(json.dumps(pose_sum, indent=1))
    print("tau_geo", results["tau_geo"])

    # ---------- figure: spread / apparent velocity vs yaw rate ----------
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for vname in VARIANTS:
        m = rows[:, 5] == vname
        om = np.abs(rows[m, 4].astype(float)); sp = rows[m, 6].astype(float); va = rows[m, 7].astype(float); rg = rows[m, 3].astype(float)
        ax[0].scatter(om, sp, s=10, color=COL[vname], label=f"{vname} (med {np.nanmedian(sp):.2f} m)", alpha=0.7)
        ax[1].scatter(om, va * 20 / rg, s=10, color=COL[vname], label=f"{vname} (med {np.median(va * 20 / rg):.2f} m/s)", alpha=0.7)
    ax[0].set_xlabel("|yaw rate| [deg/s]"); ax[0].set_ylabel("robust 2D spread of static object points over 2 s [m]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].set_xlabel("|yaw rate| [deg/s]"); ax[1].set_ylabel("apparent velocity of a static object, scaled to 20 m [m/s]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].axhline(0.5, color="k", ls="--", lw=0.8)
    fig.suptitle("static-object drift under each compensation (both bags, all swings)")
    fig.tight_layout()
    fig.savefig(out / "drift_vs_yawrate.png", dpi=110)
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
