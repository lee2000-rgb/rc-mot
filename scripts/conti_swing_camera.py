"""Continental swing analysis, question 4: global image shift between consecutive camera frames (phase
correlation, from conti_swing_extract.py) -> yaw/pitch per frame interval, compared with the IMU yaw rate
integrated over the same interval; camera-IMU time offset by scanning the offset for the minimum residual.

Usage: python scripts/conti_swing_camera.py --data <dir> --out <dir> --motion <motion_result.json>
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


def integ(t_imu, w, t0, t1):
    """Integral of w (uniform samples) between t0 and t1 via the cumulative trapezoid."""
    c = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t_imu))])
    return np.interp(t1, t_imu, c) - np.interp(t0, t_imu, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--calib", default="configs/calib/hdx_boryeong_pg_conti1126.json")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    calib = RigCalib.load(a.calib)
    fx, fy = calib.K[0, 0], calib.K[1, 1]
    motion = json.load(open(a.motion))
    ext = RigExtrinsics.from_calib_json(json.load(open(a.calib)))
    R_lev, _ = ext.level()
    result = {}
    fig, axs = plt.subplots(2, 3, figsize=(18, 8))
    for bi, bag in enumerate(BAGS):
        cam = np.load(data / f"{bag}_cam.npz"); imu = np.load(data / f"{bag}_imu.npz")
        t0 = float(cam["t0"])
        hdr = cam["hdr"] - t0
        tb = cam["t"]
        dx, dy, resp = cam["dx"], cam["dy"], cam["resp"]
        w_lev = np.stack([imu["wx"], imu["wy"], imu["wz"]], 1) @ R_lev.T
        t_imu, wz, wy = imu["t"], w_lev[:, 2], w_lev[:, 1]
        # camera optical axis ~ level x; image x to the right ~ -level y: positive yaw (left) moves the scene right (+dx)
        # phase correlation measures the mean pixel shift over the image; for a rotation by theta a pixel at angle phi
        # moves f*theta*sec^2(phi), whose mean over the image width is 1 + W^2/(12 f^2) (= 1.126 here)
        W, H = calib.image_size
        k_yaw = 1 + (W / 2) ** 2 / (3 * fx ** 2)
        k_pitch = 1 + (H / 2) ** 2 / (3 * fy ** 2)
        yaw_cam = np.degrees(dx / fx) / k_yaw                          # deg per frame interval, sign checked below
        pitch_cam = np.degrees(dy / fy) / k_pitch
        segs = motion[bag]["segments"]
        swing = np.zeros(len(hdr), bool)
        for s in segs:
            if s["cls"] == "swing":
                swing |= (hdr >= s["t0"] - 1.0) & (hdr <= s["t1"] + 1.0)
        ok = np.isfinite(dx) & (resp > 0.2) & (np.diff(hdr, prepend=hdr[0] - 0.09) < 0.3)
        m = ok & swing
        taus = np.arange(-0.6, 0.6001, 0.005)
        res = {"hdr": [], "bag": []}
        for stamp_name, ts in (("hdr", hdr), ("bag", tb)):
            for tau in taus:
                yi = np.degrees(integ(t_imu, wz, ts[:-1] + tau, ts[1:] + tau))
                yi = np.concatenate([[np.nan], yi])
                r = yaw_cam[m] - yi[m]
                res[stamp_name].append(np.sqrt(np.nanmean(r ** 2)))
            res[stamp_name] = np.array(res[stamp_name])
        # sign of dx: check the correlation sign with the IMU at the best offset (hdr)
        kb = int(np.argmin(res["hdr"])); tau_best = float(taus[kb])
        if 0 < kb < len(taus) - 1:
            y0, y1, y2 = res["hdr"][kb - 1], res["hdr"][kb], res["hdr"][kb + 1]
            tau_best = float(taus[kb] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-12) * 0.005)
        yi_best = np.concatenate([[np.nan], np.degrees(integ(t_imu, wz, hdr[:-1] + tau_best, hdr[1:] + tau_best))])
        pi_best = np.concatenate([[np.nan], np.degrees(integ(t_imu, wy, hdr[:-1] + tau_best, hdr[1:] + tau_best))])
        sign = float(np.sign(np.nansum(yaw_cam[m] * yi_best[m])))
        r_yaw = (yaw_cam - yi_best)[m]
        r_yaw0 = (yaw_cam - np.concatenate([[np.nan], np.degrees(integ(t_imu, wz, hdr[:-1], hdr[1:]))]))[m]
        r_pitch = (-pitch_cam - pi_best)[m]
        slope = float(np.nansum(yaw_cam[m] * yi_best[m]) / np.nansum(yi_best[m] ** 2))
        # per-swing offsets
        per = []
        for s in segs:
            if s["cls"] != "swing" or s["t1"] - s["t0"] < 3:
                continue
            ms = ok & (hdr >= s["t0"] - 1.0) & (hdr <= s["t1"] + 1.0)
            if ms.sum() < 20:
                continue
            rr = []
            for tau in taus:
                yi = np.concatenate([[np.nan], np.degrees(integ(t_imu, wz, hdr[:-1] + tau, hdr[1:] + tau))])
                rr.append(np.sqrt(np.nanmean((yaw_cam[ms] - yi[ms]) ** 2)))
            per.append({"t0": s["t0"], "t1": s["t1"], "tau_s": float(taus[int(np.argmin(rr))]), "rms_deg": float(np.min(rr)), "n": int(ms.sum())})
        fr = np.median(np.diff(hdr))
        result[bag] = {"frame_period_s": float(fr), "n_frames_swing": int(m.sum()),
                       "tau_best_hdr_s": tau_best, "rms_deg_per_frame_best": float(np.sqrt(np.nanmean(r_yaw ** 2))),
                       "rms_deg_per_frame_tau0": float(np.sqrt(np.nanmean(r_yaw0 ** 2))),
                       "rms_deg_s_best": float(np.sqrt(np.nanmean(r_yaw ** 2)) / fr),
                       "robust_std_deg_per_frame_best": float(1.4826 * np.nanmedian(np.abs(r_yaw - np.nanmedian(r_yaw)))),
                       "median_resid_deg_per_frame": float(np.nanmedian(r_yaw)),
                       "scale_cam_over_imu": slope, "dx_sign_vs_imu": sign, "perspective_factor": float(k_yaw),
                       "tau_best_bag_s": float(taus[int(np.argmin(res["bag"]))]), "rms_bag_best_deg": float(np.min(res["bag"])),
                       "pitch_rms_deg_per_frame": float(np.sqrt(np.nanmean(r_pitch ** 2))),
                       "per_swing": per,
                       "image_header_minus_bag_s": {"p5": float(np.percentile(hdr - tb, 5)), "p50": float(np.percentile(hdr - tb, 50)), "p95": float(np.percentile(hdr - tb, 95))},
                       "max_abs_yaw_per_frame_deg": float(np.nanmax(np.abs(yaw_cam[m]))),
                       "max_dx_px": float(np.nanmax(np.abs(dx[m])))}
        ax = axs[bi]
        ax[0].plot(taus * 1e3, res["hdr"], label="image header stamp")
        ax[0].plot(taus * 1e3, res["bag"], label="bag receive stamp", alpha=0.6)
        ax[0].axvline(tau_best * 1e3, color="k", ls="--", label=f"best {tau_best * 1e3:.0f} ms (hdr)")
        ax[0].set_xlabel("camera time offset tau [ms] (IMU time = image header + tau)"); ax[0].set_ylabel("RMS(yaw_cam - yaw_imu) [deg/frame]"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        ax[0].set_title(bag.split("_")[3])
        ax[1].plot(hdr[m], yaw_cam[m] / fr, ".", ms=3, label="camera phase-corr yaw rate")
        ax[1].plot(hdr[m], yi_best[m] / fr, "-", lw=0.8, label=f"IMU yaw rate (tau={tau_best * 1e3:.0f} ms)")
        ax[1].set_xlabel("bag time [s]"); ax[1].set_ylabel("yaw rate [deg/s]"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
        ax[2].plot(yi_best[m], yaw_cam[m], ".", ms=3)
        lim = np.nanmax(np.abs(yi_best[m]))
        ax[2].plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax[2].set_xlabel("IMU yaw per frame [deg]"); ax[2].set_ylabel("camera yaw per frame [deg]"); ax[2].grid(alpha=0.3)
        ax[2].set_title(f"scale {slope:.3f}, resid rms {np.sqrt(np.nanmean(r_yaw ** 2)):.3f} deg/frame")
        print(bag, json.dumps(result[bag], indent=1))
    fig.tight_layout()
    fig.savefig(out / "camera_imu_offset.png", dpi=110)
    json.dump(result, open(out / "camera_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
