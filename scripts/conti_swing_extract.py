"""Pass 1 of the Continental swing analysis: extract IMU, radar ego-velocity (ScanStatistician) and camera
global image shift (phase correlation) from a bag into npz files (CPU only, no YOLO).

Usage: PYTHONPATH=src python scripts/conti_swing_extract.py --bag <bag dir> --out <dir> [--no-cam]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.radar.scan_stats import ScanStatistician, ScanStatsParams          # noqa: E402
from rcmot.sensors.bag_source import image_to_bgr, pc2_to_struct               # noqa: E402
from rcmot.utils.calib import RigCalib                                         # noqa: E402

RADAR = "/radar/radar_detection_image"
IMU = "/ouster/imu"
IMG = "/oak/rgb/image_raw"


def _hs(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", default="configs/calib/hdx_boryeong_pg_conti1126.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-cam", action="store_true")
    ap.add_argument("--cam-scale", type=float, default=0.5)
    a = ap.parse_args()
    calib = RigCalib.load(a.calib)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.bag).name
    stats = ScanStatistician(ScanStatsParams())

    imu = {k: [] for k in ("t", "hdr", "wx", "wy", "wz", "ax", "ay", "az")}
    scan = {k: [] for k in ("t", "hdr", "n", "vx", "vy", "vz", "sigma_v", "ratio", "valid", "n_inl")}
    pts = {k: [] for k in ("scan", "x", "y", "z", "v_r", "rcs", "v_comp", "inl", "keep", "vr_var", "az_var", "pdh0", "rr_amb")}
    cam = {k: [] for k in ("t", "hdr", "dx", "dy", "resp")}
    prev_img, k_scan = None, 0

    with AnyReader([Path(a.bag)]) as r:
        cs = {c.topic: c for c in r.connections}
        t0 = r.start_time * 1e-9
        want = [cs[RADAR], cs[IMU]] + ([] if a.no_cam else [cs[IMG]])
        for conn, t_ns, raw in r.messages(connections=want):
            t = t_ns * 1e-9 - t0
            m = r.deserialize(raw, conn.msgtype)
            if conn.topic == IMU:
                imu["t"].append(t); imu["hdr"].append(_hs(m))
                imu["wx"].append(m.angular_velocity.x); imu["wy"].append(m.angular_velocity.y); imu["wz"].append(m.angular_velocity.z)
                imu["ax"].append(m.linear_acceleration.x); imu["ay"].append(m.linear_acceleration.y); imu["az"].append(m.linear_acceleration.z)
            elif conn.topic == RADAR:
                arr = pc2_to_struct(m)
                xyz = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(float)
                ok = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.5)
                xyz = xyz[ok]
                v_r = arr["radial_velocity"][ok].astype(float)
                rcs = arr["radar_cross_section"][ok].astype(float)
                snr = arr["signal_noise_ratio"][ok].astype(float)
                st = stats.compute(t, xyz, v_r, snr, rcs, None, calib.radar_to_level(xyz))
                scan["t"].append(t); scan["hdr"].append(_hs(m)); scan["n"].append(len(v_r))
                scan["vx"].append(st.v_sensor[0]); scan["vy"].append(st.v_sensor[1]); scan["vz"].append(st.v_sensor[2])
                scan["sigma_v"].append(st.sigma_v); scan["ratio"].append(st.inlier_ratio); scan["valid"].append(st.ego_valid)
                scan["n_inl"].append(int(st.static_inlier.sum()))
                pts["scan"].append(np.full(len(v_r), k_scan, np.int32))
                pts["x"].append(xyz[:, 0].astype(np.float32)); pts["y"].append(xyz[:, 1].astype(np.float32)); pts["z"].append(xyz[:, 2].astype(np.float32))
                pts["v_r"].append(v_r.astype(np.float32)); pts["rcs"].append(rcs.astype(np.float32))
                pts["v_comp"].append(st.v_comp.astype(np.float32)); pts["inl"].append(st.static_inlier); pts["keep"].append(st.keep)
                pts["vr_var"].append(arr["range_rate_var"][ok].astype(np.float32)); pts["az_var"].append(arr["azimuth_var"][ok].astype(np.float32))
                pts["pdh0"].append(arr["pdh0"][ok].astype(np.uint8)); pts["rr_amb"].append(arr["range_rate_ambiguity"][ok].astype(np.float32))
                k_scan += 1
            else:
                img = image_to_bgr(m)
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                g = cv2.resize(g, None, fx=a.cam_scale, fy=a.cam_scale, interpolation=cv2.INTER_AREA).astype(np.float32)
                if prev_img is not None:
                    win = cv2.createHanningWindow(g.shape[::-1], cv2.CV_32F)
                    (dx, dy), resp = cv2.phaseCorrelate(prev_img, g, win)
                    cam["dx"].append(dx / a.cam_scale); cam["dy"].append(dy / a.cam_scale); cam["resp"].append(resp)
                else:
                    cam["dx"].append(np.nan); cam["dy"].append(np.nan); cam["resp"].append(np.nan)
                cam["t"].append(t); cam["hdr"].append(_hs(m))
                prev_img = g
            if k_scan % 500 == 0 and conn.topic == RADAR:
                print(f"{name}: t={t:.1f}s scans={k_scan} imu={len(imu['t'])} cam={len(cam['t'])}", flush=True)

    np.savez_compressed(out / f"{name}_imu.npz", t0=t0, **{k: np.asarray(v) for k, v in imu.items()})
    np.savez_compressed(out / f"{name}_radar.npz", t0=t0, **{k: np.asarray(v) for k, v in scan.items()},
                        **{"p_" + k: np.concatenate(v) for k, v in pts.items()})
    if not a.no_cam:
        np.savez_compressed(out / f"{name}_cam.npz", t0=t0, **{k: np.asarray(v) for k, v in cam.items()})
    print(f"done {name}: scans={k_scan} points={sum(len(v) for v in pts['x'])} imu={len(imu['t'])} cam={len(cam['t'])}")


if __name__ == "__main__":
    main()
