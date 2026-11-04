"""Online drive reconstruction (rcmot.work.recon, the GUI's "재구성" tab) replayed on a cached bag read and scored
against the LiDAR map exactly like the offline tool (docs/26 §6).

The cached read (scripts/drive_recon.py --reuse: data/drive_recon/<profile>/read_v3_0_None.pkl) holds the 6 Hz
images, every radar scan's Doppler velocity + static points, the gyro and the LiDAR odometry / map scans. The world
pose is dead-reckoned the way the pipeline does it (rcmot.world with the profile's lever arm, yaw rate = gyro mean
+-50 ms around the scan), the people masks come from the same detector the GUI uses (yolo11s-seg), and the frames
go through DriveRecon.update() in order (sync mode: each chunk runs inline).

  python3 scripts/recon_online_eval.py --profile hdx1026_excA_142258 [--set chunk=12 ...]
  -> data/drive_recon/<profile>/online_report.json (+ online_topview.png), numbers next to the offline report.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import drive_recon as dr  # noqa: E402  offline tool: profile, LiDAR reference, top view
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.sensors.bag_source import bag_camera_info  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics  # noqa: E402
from rcmot.work.recon import DriveRecon, ReconParams, apply, se3, voxel_first  # noqa: E402
from rcmot.world import WorldParams, WorldPose  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--cache", default=None, help="read_v3 pickle (default: data/drive_recon/<profile>/read_v3_0_None.pkl)")
    ap.add_argument("--set", action="append", default=[], help="ReconParams override key=value")
    ap.add_argument("--no-people", action="store_true", help="no detector masks")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix of online_report<tag>.json")
    a = ap.parse_args()

    prof, cfg, topics = dr.load_profile(a.profile)
    out = Path(a.out or ROOT / "data/drive_recon" / a.profile)
    out.mkdir(parents=True, exist_ok=True)
    calp = ROOT / prof["calib"]
    calib = RigCalib.load(str(calp))
    bag_camera_info(prof["bag"], topics["image"], calib)
    lever = (cfg.get("world") or {}).get("lever_arm")
    R_lev, t_lev = RigExtrinsics.from_calib_json(json.load(open(calp))).level()
    o = pickle.load(open(a.cache or out / "read_v3_0_None.pkl", "rb"))
    R_ls, t_ls = lidar_to_sensor({}, o["lidar_frame"])
    A = se3(R_lev @ R_ls, R_lev @ t_ls + t_lev)
    T_lvl_cam = np.linalg.inv(se3(calib.R_cl, calib.t_cl))

    rp = dict(cfg.get("recon") or {})
    rp.update(enable=True, sync=True, mask_people=not a.no_people)
    for kv in a.set:
        k, v = kv.split("=", 1)
        rp[k] = type(getattr(ReconParams(), k))(v) if not isinstance(getattr(ReconParams(), k), bool) else v.lower() in ("1", "true")
    p = ReconParams(**rp)
    recon = DriveRecon(p, calib, ROOT)
    det = None
    if p.mask_people:
        from rcmot.detection.yolo2d import Yolo2D

        det = Yolo2D(str(ROOT / "models/yolo11s-seg.pt"), {"person": "worker"}, imgsz=960, conf=0.25, device="cuda")

    # ---- replay: world pose per radar scan (as the pipeline), images in order
    wp = WorldPose(WorldParams(lever_arm=lever), calib.t_lr)
    rad_t, rad_v, rad_ok = np.asarray(o["rad_t"]), np.asarray(o["rad_v"]), np.asarray(o["rad_ok"])
    gyro_t, gyro_z = np.asarray(o["gyro_t"]), np.asarray(o["gyro_z"])

    def yaw_rate_at(t, half=0.05):                       # as rcmot.sensors.bag_source.yaw_rate_at: mean within +-50 ms
        m = np.abs(gyro_t - t) <= half
        return float(gyro_z[m].mean()) if m.any() else float(gyro_z[np.argsort(np.abs(gyro_t - t))[:10]].mean())
    img_t = np.asarray(o["img_t"])
    states, prev = [], None
    for t, v, ok in zip(rad_t, rad_v, rad_ok):
        dt = 0.0 if prev is None else min(t - prev, 0.5)
        states.append(wp.update(t, dt, yaw_rate_at(t), v[:2], bool(ok)))
        prev = t
    tic = time.time()
    n_det = 0
    for i, t in enumerate(img_t):
        k = int(np.argmin(np.abs(rad_t - t)))
        img = cv2.imdecode(np.frombuffer(o["img"][i], np.uint8), cv2.IMREAD_COLOR)
        dets = det(img) if det is not None else None
        n_det += len(dets) if dets else 0
        st = recon.update(t, img, states[k], o["rad_pts"][k], dets, img_dt=t - rad_t[k])
        if st.error:
            sys.exit(f"recon error: {st.error}")
    recon.flush(min_views=p.overlap + 1)
    st = recon.state(img_t[-1])
    print(f"replayed {len(img_t)} images in {time.time() - tic:.0f} s: {st.n_keyframes} keyframes, {st.n_chunks} chunks, "
          f"{st.points} points, people dets {n_det}", flush=True)
    s = recon.s
    rec, rec_c = s.P.astype(np.float64), s.C
    ks = sorted(s.kf_T)
    kf_t = np.array([s.kf[j].t for j in ks])
    T_cam = [s.kf_T[j] for j in ks]

    # ---- LiDAR reference in the same world frame (the online map is anchored at the first keyframe's world pose)
    T_W_lvl0 = s.kf[0].T_W_lvl
    T_lid = dr.refine_lidar(o, dr.lidar_poses(o, A, s.kf[0].t))
    true_cam = T_W_lvl0 @ T_lid(kf_t) @ np.linalg.inv(A) @ T_lvl_cam
    L = [apply(T_W_lvl0 @ T_lid(ts)[0], Q.astype(np.float64)).astype(np.float32) for ts, _, Q in o["map"]]
    lid = voxel_first(np.concatenate(L), p.voxel)[0]
    traj_rec, traj_true = np.array([T[:3, 3] for T in T_cam]), true_cam[:, :3, 3]
    e_rec = np.linalg.norm(traj_rec - traj_true, axis=1)
    path = float(np.linalg.norm(np.diff(traj_true, axis=0), axis=1).sum())
    up = lambda T: T[:3, :3].T @ np.array([0.0, 0.0, 1.0])               # noqa: E731
    tilt = [np.degrees(np.arccos(np.clip(up(T1) @ up(T2), -1, 1))) for T1, T2 in zip(T_cam, true_cam)]
    hd = lambda T: np.arctan2(T[1, 2], T[0, 2])                          # noqa: E731
    yaw = [np.degrees((hd(T1) - hd(T2) + np.pi) % (2 * np.pi) - np.pi) for T1, T2 in zip(T_cam, true_cam)]
    d_rec, _ = cKDTree(lid).query(rec, distance_upper_bound=5.0)
    has = np.isfinite(d_rec)
    seen = np.zeros(len(lid), bool)
    W_, H_ = calib.image_size
    for T in true_cam[::2]:
        pc = apply(np.linalg.inv(T), lid)
        m = (pc[:, 2] > 1.0) & (pc[:, 2] < p.max_range)
        uv = np.full((len(pc), 2), -1.0)
        uv[m] = cv2.projectPoints(pc[m], np.zeros(3), np.zeros(3), calib.K, calib.D)[0].reshape(-1, 2)
        seen |= m & (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H_)
    d_lid, _ = cKDTree(rec).query(lid[seen], distance_upper_bound=5.0)
    pct = lambda x: round(100.0 * float(np.mean(x)), 1)                 # noqa: E731
    report = dict(
        profile=a.profile, mode="online", params={k: v for k, v in rp.items() if k not in ("enable", "sync")},
        keyframes=len(ks), chunks=st.n_chunks, scales=[round(v, 3) for v in s.scales], chunk_log=s.log,
        trajectory=dict(path_m=round(path, 1), end_err_m=round(float(e_rec[-1]), 2), max_err_m=round(float(e_rec.max()), 2),
                        median_err_m=round(float(np.median(e_rec)), 2), end_yaw_err_deg=round(float(yaw[-1]), 1),
                        end_tilt_err_deg=round(float(tilt[-1]), 1), max_tilt_err_deg=round(float(np.max(tilt)), 1)),
        accuracy=dict(points=int(len(rec)), median_m=round(float(np.median(d_rec[has])), 3),
                      within_0p5_pct=pct(d_rec[has] < 0.5), within_1_pct=pct(d_rec[has] < 1.0)),
        completeness=dict(lidar_points_in_view=int(seen.sum()), within_0p5_pct=pct(d_lid < 0.5), within_1_pct=pct(d_lid < 1.0)))
    off = out / "report.json"
    if off.exists():
        r = json.load(open(off))
        report["offline"] = dict(within_0p5_pct=r["accuracy"]["within_0p5_pct"], within_1_pct=r["accuracy"]["within_1_pct"],
                                 end_err_m=r["trajectory"]["end_err_m"], completeness_0p5_pct=r["completeness"]["within_0p5_pct"])
    json.dump(report, open(out / f"online_report{a.tag}.json", "w"), indent=1, default=float)
    print(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    traj = {"LiDAR": (traj_true, "k"), "online camera": (traj_rec, "#2a78d6")}
    dr.topview(out / f"online_topview{a.tag}.png", rec, rec_c, lid, seen, traj, f"{a.profile}: online reconstruction vs LiDAR")


if __name__ == "__main__":
    main()
