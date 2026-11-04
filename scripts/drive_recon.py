"""Wide-area camera 3D reconstruction of a moving bag, checked against the LiDAR (docs/26).

Keyframes (every ~0.6 m / 6 deg of radar-Doppler + gyro dead reckoning) go through VGGT-Omega in chunks of 16 with 4
shared frames: per chunk the model gives depth, confidence and the poses of all its cameras (up to scale). The metric
scale of a chunk is the median ratio of the static radar points' camera depth to the VGGT depth at their pixels;
consecutive chunks are joined on the poses of their shared frames. The world frame is the calibrated level frame of
the first keyframe, so the reconstruction uses camera + radar (+ gyro for the keyframe spacing) only.
The LiDAR is the reference: KISS-ICP odometry + accumulated scans in the same world frame -> trajectory drift, point
accuracy (camera point -> nearest LiDAR point), completeness (LiDAR points in the camera's view with a camera point
within 0.5 m), top views (topview.png) and an interactive viewer (viewer.html: time slider, colour by error).

  python3 scripts/drive_recon.py --profile hdx1026_excA_142258 [--t0 0 --t1 90] [--reuse]
  -> data/drive_recon/<profile>/  report.json, topview.png, viewer.html, recon.ply, lidar.ply (+ caches)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.radar.scan_stats import ScanStatistician, ScanStatsParams  # noqa: E402
from rcmot.sensors.bag_source import bag_camera_info, image_to_bgr, open_bag, pc2_to_struct, radar_frame  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics  # noqa: E402
from rcmot.world import WorldParams, WorldPose, rot2  # noqa: E402


def load_profile(name: str):
    prof = yaml.safe_load(open(ROOT / "configs/datasets.yaml"))["datasets"][name]
    if not Path(prof["bag"]).exists() and os.environ.get("RCMOT_BAG_ROOT"):     # server: bags under another root
        prof["bag"] = str(Path(os.environ["RCMOT_BAG_ROOT"]) / Path(prof["bag"]).name)
    cfg = yaml.safe_load(open(ROOT / prof["config"]))
    for k, v in (prof.get("overrides") or {}).items():
        d = cfg
        for part in k.split(".")[:-1]:
            d = d.setdefault(part, {})
        d[k.split(".")[-1]] = v
    return prof, cfg, dict(cfg["topics"], **(prof.get("topics") or {}))


def se3(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def apply(T, P) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def voxel_first(P, size, *extra):
    """Keep the first point per voxel (inputs ordered by time -> the earliest observation)."""
    _, i = np.unique(np.floor(P / size).astype(np.int64), axis=0, return_index=True)
    i = np.sort(i)
    return (P[i],) + tuple(e[i] for e in extra)


# ---------------------------------------------------------------------------------------------------- bag (one pass)
def read_bag(bag, calib, cfg, topics, t0, t1, a) -> dict:
    """Images (img_hz, JPEG), radar ego velocity + static points (level frame), gyro z, LiDAR odometry (KISS-ICP, every
    scan) and LiDAR map scans (every map_dt, 0.1 m voxels, LiDAR frame)."""
    from kiss_icp.config import KISSConfig
    from kiss_icp.kiss_icp import KissICP

    kc = KISSConfig()
    kc.data.max_range, kc.data.min_range, kc.data.deskew = 80.0, 2.0, True     # deskew: 35 deg/s swings
    kc.mapping.voxel_size = 0.5
    kiss = KissICP(kc)
    stats = ScanStatistician(ScanStatsParams(**cfg.get("scan_stats", {})))
    o = {k: [] for k in ("img_t", "img", "rad_t", "rad_v", "rad_ok", "rad_pts", "gyro_t", "gyro_z", "acc", "lid_t", "lid_pose", "map")}
    frame = {}
    with open_bag(bag) as r:
        conns = {c.topic: c for c in r.connections}
        want = [conns[t] for t in (topics["image"], topics["radar"], topics.get("imu"), a.lidar) if t in conns]
        s0 = r.start_time * 1e-9
        stop = int((s0 + t1) * 1e9) if t1 else None
        last_img = last_map = -1e9
        tic = time.time()
        for conn, tn, raw in r.messages(connections=want, start=int((s0 + t0) * 1e9), stop=stop):
            tb = tn * 1e-9
            if conn.topic == topics.get("imu"):
                m = r.deserialize(raw, conn.msgtype)
                o["gyro_t"].append(tb)
                o["gyro_z"].append(m.angular_velocity.z)
                o["acc"].append((m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
                o["imu_frame"] = m.header.frame_id
                continue
            m = r.deserialize(raw, conn.msgtype)
            hs = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            ts = hs if abs(hs - tb) < 5.0 else tb                    # sensor-clock headers -> bag time
            if conn.topic == topics["image"]:
                if ts - last_img >= 1.0 / a.img_hz:
                    o["img_t"].append(ts - calib.camera_time_offset_s)
                    o["img"].append(cv2.imencode(".jpg", image_to_bgr(m), [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                    last_img = ts
            elif conn.topic == topics["radar"]:
                rf = radar_frame(pc2_to_struct(m), ts + calib.radar_time_offset_s, calib)
                st = stats.compute(rf.stamp, rf.xyz_radar, rf.v_r, rf.snr, rf.rcs, rf.peer, rf.xyz)
                ok = bool(st.ego_valid and np.isfinite(st.v_sensor).all())
                stat = st.static_inlier & st.keep
                if st.below_ground is not None:
                    stat &= ~st.below_ground
                o["rad_t"].append(rf.stamp)
                o["rad_v"].append(calib.R_lr @ st.v_sensor if ok else np.zeros(3))
                o["rad_ok"].append(ok)
                o["rad_pts"].append(rf.xyz[stat].astype(np.float32))
            else:                                                     # LiDAR
                arr = pc2_to_struct(m)
                P = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(np.float64)
                rr = np.linalg.norm(P, axis=1)
                ok = np.isfinite(rr) & (rr > 2.0) & (rr < 80.0)
                P = P[ok]
                tp = arr["t"][ok].astype(np.float64) if "t" in arr.dtype.names else np.zeros(len(P))   # Ouster ns in scan
                span = float(np.ptp(tp)) * 1e-9 if len(tp) else 0.0
                tp = (tp - tp.min()) / max(np.ptp(tp), 1.0) if len(tp) else tp
                frame.setdefault("id", m.header.frame_id)
                kiss.register_frame(P, tp)
                o["lid_t"].append(ts + 0.5 * span)                          # KISS-ICP deskews to the mid-scan time
                o["lid_pose"].append(kiss.last_pose.copy())
                if ts - last_map >= a.map_dt:
                    q = P[np.linalg.norm(P, axis=1) < a.lidar_range]
                    o["map"].append((ts, len(o["lid_t"]) - 1, voxel_first(q, 0.1)[0].astype(np.float32)))
                    last_map = ts
                if len(o["lid_t"]) % 200 == 0:
                    print(f"  bag {ts - s0:6.1f} s  ({time.time() - tic:.0f} s)  images {len(o['img_t'])}, scans "
                          f"{len(o['rad_t'])}, LiDAR {len(o['lid_t'])}", flush=True)
    o["lidar_frame"] = frame.get("id", "os_sensor")
    o["bag_start"] = s0
    return o


# ---------------------------------------------------------------------------------------------------- poses
def dead_reckon(o, calib, lever):
    """Radar-Doppler + gyro pose of the level frame (rcmot.world): (t, theta, origin xy) at the radar scans."""
    gt, gz = np.asarray(o["gyro_t"]), np.asarray(o["gyro_z"])
    k = 9
    gs = np.convolve(gz, np.ones(k) / k, mode="same") if len(gz) > k else gz
    wp = WorldPose(WorldParams(lever_arm=lever), calib.t_lr)
    T, TH, XY, prev = [], [], [], None
    for t, v, ok in zip(o["rad_t"], o["rad_v"], o["rad_ok"]):
        dt = 0.0 if prev is None else min(t - prev, 0.5)
        st = wp.update(t, dt, float(np.interp(t, gt, gs)) if len(gt) else 0.0, v[:2], ok)
        prev = t
        T.append(t), TH.append(st.theta), XY.append(st.origin)
    return np.array(T), np.unwrap(np.array(TH)), np.array(XY)


def dr_camera_centres(dr, kf_t, cc) -> np.ndarray:
    """Camera centres of the dead reckoning in the level frame of the first keyframe (2D motion, height of the calib)."""
    T, TH, XY = dr
    th0, xy0 = np.interp(kf_t[0], T, TH), np.array([np.interp(kf_t[0], T, XY[:, i]) for i in (0, 1)])
    out = []
    for t in kf_t:
        th, xy = np.interp(t, T, TH), np.array([np.interp(t, T, XY[:, i]) for i in (0, 1)])
        out.append(np.r_[rot2(-th0) @ (xy + rot2(th) @ cc[:2] - xy0), cc[2]])
    return np.array(out)


def travel_scale(ck, dr_c):
    """Radar-Doppler travel of the chunk's cameras / VGGT travel (camera centres in chunk units): (scale, travel m)."""
    C = np.array([-E[:, :3].T @ E[:, 3] for E in ck["E"]])
    tv, td = np.linalg.norm(np.diff(C, axis=0), axis=1).sum(), np.linalg.norm(np.diff(dr_c[:, :2], axis=0), axis=1).sum()
    return (td / tv if tv > 1e-6 else float("nan")), float(td)


def pick_keyframes(img_t, dr, a) -> list[int]:
    T, TH, XY = dr
    th = np.interp(img_t, T, TH)
    xy = np.stack([np.interp(img_t, T, XY[:, 0]), np.interp(img_t, T, XY[:, 1])], 1)
    keep = [0]
    for i in range(1, len(img_t)):
        j = keep[-1]
        dt = img_t[i] - img_t[j]
        if dt >= a.kf_min_dt and (np.linalg.norm(xy[i] - xy[j]) >= a.kf_dist or abs(th[i] - th[j]) >= np.radians(a.kf_deg)
                                  or dt >= a.kf_max_dt):
            keep.append(i)
    return keep


def lidar_poses(o, A, t_ref):
    """T_W_lidar(t) = A P(t_ref)^-1 P(t): KISS-ICP odometry in the level frame of t_ref (interpolated)."""
    lt, k = np.unique(np.asarray(o["lid_t"]), return_index=True)
    LP = np.asarray(o["lid_pose"])[k]
    sl = Slerp(lt, Rotation.from_matrix(LP[:, :3, :3]))

    def at(t):
        t = np.clip(np.atleast_1d(t), lt[0], lt[-1])
        out = np.tile(np.eye(4), (len(t), 1, 1))
        out[:, :3, :3] = sl(t).as_matrix()
        for c in range(3):
            out[:, c, 3] = np.interp(t, lt, LP[:, c, 3])
        return out
    ref_inv = np.linalg.inv(at(t_ref)[0])
    return lambda t: A @ ref_inv @ at(t)


class Gravity:
    """Up direction in the level frame from the accelerometer (Ouster os_imu: os_sensor axes): mean specific force over
    +-half s minus the radar-Doppler linear acceleration; None while turning (centripetal terms of the swing)."""

    def __init__(self, o, R_lev, half=0.5, max_rate_dps=5.0):
        self.gt, self.acc, self.R_lev, self.half = np.asarray(o["gyro_t"]), np.asarray(o["acc"], float), R_lev, half
        gz = np.asarray(o["gyro_z"])
        gz = np.convolve(gz, np.ones(31) / 31, mode="same")                       # ~0.3 s: the vibration averages out
        self.turn = np.abs(gz - np.median(gz)) > np.radians(max_rate_dps)
        self.rt, self.rv = np.asarray(o["rad_t"]), np.asarray(o["rad_v"], float)

    def __call__(self, t):
        m = np.abs(self.gt - t) <= self.half
        if m.sum() < 20 or self.turn[m].mean() > 0.1:
            return None
        v = lambda x: np.array([np.interp(x, self.rt, self.rv[:, i]) for i in range(3)])   # noqa: E731
        a_lin = (v(t + self.half) - v(t - self.half)) / (2 * self.half)                     # level frame
        up = self.R_lev @ self.acc[m].mean(0) - a_lin
        return up / np.linalg.norm(up)


def rot_between(a, b) -> np.ndarray:
    """Minimal rotation taking unit vector a onto unit vector b."""
    v, c = np.cross(a, b), float(a @ b)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def overlap_points(ck_a, sa, sc_a, T_W_cam, ck_b, sb, sc_b, E_b, rays, a, max_d=25.0, step=4):
    """The same pixels of a shared frame in two chunks: world points of chunk a (P), chunk-b-world points of chunk b
    (Q) and weights 1/depth^2 (VGGT depth noise grows with range); pixels confident in both and agreeing within 20 %."""
    da, db = ck_a["depth"][sa].astype(np.float32) * sc_a, ck_b["depth"][sb].astype(np.float32) * sc_b
    ca, cb = ck_a["conf"][sa].astype(np.float32), ck_b["conf"][sb].astype(np.float32)
    m = (ca >= max(np.quantile(ca, a.conf_q), a.conf_min)) & (cb >= max(np.quantile(cb, a.conf_q), a.conf_min))
    m &= (da > 1.0) & (da < max_d) & (np.abs(db - da) < 0.2 * da)
    m[int(round((1.0 - a.crop_bottom) * m.shape[0])):] = False
    sub = np.zeros_like(m)
    sub[::step, ::step] = True
    m &= sub
    P = apply(T_W_cam, rays[m] * da[m, None])
    Q = apply(np.linalg.inv(E_b), rays[m] * db[m, None])
    return P, Q, 1.0 / da[m] ** 2


def align_rigid(P, Q, w, T0, delta=0.3, iters=8) -> np.ndarray:
    """T (P-frame <- Q-frame) minimising the Huber-weighted sum of w |P - T Q|^2 (weighted Kabsch, IRLS), from T0."""
    T = T0
    for _ in range(iters):
        r = np.linalg.norm(P - apply(T, Q), axis=1)
        wi = w * np.minimum(1.0, delta / np.maximum(r, 1e-9))
        wi = wi / wi.sum()
        pc, qc = wi @ P, wi @ Q
        H = (Q - qc).T @ ((P - pc) * wi[:, None])
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        T = se3(R, pc - R @ qc)
    return T


def refine_lidar(o, T_odo, near=60.0):
    """Scan-to-map ICP of the map scans (open3d point-to-plane, KISS-ICP odometry as the initial guess): the odometry
    alone drifts 1.4 m in height over 92 s on the open field (return pass 1.4 m below the outbound pass, 142258); against
    the accumulated map the return pass lands on the outbound ground. Returns T(t) = refined LiDAR pose in the world
    (the odometry between the nearest map scans)."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    ts_all = np.array([ts for ts, _, _ in o["map"]])
    poses, corr, M, tgt = [], np.eye(4), np.zeros((0, 3)), None
    for k, (ts, _, Q) in enumerate(o["map"]):
        T = corr @ T_odo(ts)[0]
        if tgt is not None:
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q.astype(np.float64))).voxel_down_sample(0.2)
            for dmax in (1.0, 0.4):
                r = reg.registration_icp(src, tgt, dmax, T, reg.TransformationEstimationPointToPlane(),
                                         reg.ICPConvergenceCriteria(max_iteration=30))
                if r.fitness > 0.3:
                    T = r.transformation
        corr = T @ np.linalg.inv(T_odo(ts)[0])
        poses.append(T)
        M = np.concatenate([M, apply(T, Q.astype(np.float64))])
        if k % 4 == 0:                                               # rebuild the target (cropped, 0.2 m, normals)
            M = voxel_first(M, 0.1)[0]
            c = M[np.linalg.norm(M[:, :2] - T[:2, 3], axis=1) < near]
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c)).voxel_down_sample(0.2)
            tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=20))
    poses = np.array(poses)

    def at(t):
        t = np.atleast_1d(t)
        k = np.clip(np.searchsorted(ts_all, t), 1, len(ts_all) - 1)
        k = np.where(np.abs(ts_all[k - 1] - t) < np.abs(ts_all[k] - t), k - 1, k)
        return poses[k] @ np.linalg.inv(T_odo(ts_all[k])) @ T_odo(t)
    return at


def pass_consistency(P, heading, h0, cell=0.5):
    """Ground of the return pass minus the outbound pass in their common cells (no reference needed): passes = points
    seen from headings within 45 deg of the start heading h0 / of h0 + 180 deg; ground = 20th height percentile per cell."""
    dh = (heading - h0 + np.pi) % (2 * np.pi) - np.pi
    big, off = 1 << 20, 1 << 19

    def ground(m):
        ij = np.floor(P[m, :2] / cell).astype(np.int64) + off
        key, z = ij[:, 0] * big + ij[:, 1], P[m, 2]
        o_ = np.lexsort((z, key))
        key, z = key[o_], z[o_]
        uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
        ok = cnt >= 5
        return dict(zip(uk[ok], z[st[ok] + cnt[ok] // 5]))
    a_, b_ = ground(np.abs(dh) < np.radians(45)), ground(np.abs(dh) > np.radians(135))
    com = np.array(sorted(set(a_) & set(b_)), np.int64)
    if len(com) < 50:
        return None
    dz = np.array([b_[k] - a_[k] for k in com])
    xy = (np.c_[com // big, com % big] - off + 0.5) * cell
    m = np.abs(dz - np.median(dz)) < 0.5
    sol = np.linalg.lstsq(np.c_[np.ones(m.sum()), xy[m]], dz[m], rcond=None)[0]
    return dict(cells=len(com), median_dz_m=round(float(np.median(dz)), 3), mad_m=round(float(np.median(np.abs(dz - np.median(dz)))), 3),
                slope_x_deg=round(float(np.degrees(np.arctan(sol[1]))), 2), slope_y_deg=round(float(np.degrees(np.arctan(sol[2]))), 2))


def person_masks(images, root, dilate=9):
    """YOLO11s-seg person masks per keyframe (full image resolution, dilated): dynamic objects smear across keyframes
    and VGGT depth on thin people is 3x worse than on the ground (err_anal: 0.22 vs 0.08 m)."""
    from rcmot.detection.yolo2d import Yolo2D

    import torch

    det = Yolo2D(str(root / "models/yolo11s-seg.pt"), {"person": "worker"}, imgsz=960, conf=0.25, device="cuda")
    out, n = [], 0
    k = np.ones((dilate, dilate), np.uint8)
    for im in images:
        m = np.zeros(im.shape[:2], np.uint8)
        try:
            dets = det(im)
        except torch.OutOfMemoryError:                                  # the GUI holds the GPU: CPU is fine for 200 frames
            torch.cuda.empty_cache()
            det = Yolo2D(str(root / "models/yolo11s-seg.pt"), {"person": "worker"}, imgsz=960, conf=0.25, device="cpu", half=False)
            dets = det(im)
        for d in dets:
            x1, y1 = max(int(d.box[0]), 0), max(int(d.box[1]), 0)
            if d.mask is not None:
                hh, ww = d.mask.shape
                m[y1:y1 + hh, x1:x1 + ww] |= d.mask.astype(np.uint8)
            else:
                m[y1:int(d.box[3]), x1:int(d.box[2])] = 1
            n += 1
        out.append(cv2.dilate(m, k) > 0 if n else m > 0)
    return out, n


def keyframe_cloud(ck, s, scale, rays, rgb, above, pmask, a, step=4, max_d=25.0):
    """Camera-frame points of a keyframe for registration (open3d cloud with normals), without sky / people."""
    import open3d as o3d

    d = ck["depth"][s].astype(np.float32) * scale
    cf = ck["conf"][s].astype(np.float32)
    m = (cf >= max(np.quantile(cf, a.conf_q), a.conf_min)) & (d > 1.0) & (d < max_d) & ~sky_mask(rgb, above)
    if pmask is not None:
        m &= ~pmask
    m[int(round((1.0 - a.crop_bottom) * m.shape[0])):] = False
    sub = np.zeros_like(m)
    sub[::step, ::step] = True
    m &= sub
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rays[m] * d[m, None])).voxel_down_sample(0.2)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=20))
    return pc


def pose_graph(T_cam, cks, chunks, sv, clouds, kf_t, a, log):
    """Pose-graph optimisation of all keyframe cameras (open3d): odometry edges = VGGT relative poses inside each chunk
    (scaled), loop edges = point-to-plane ICP between keyframes that are close in space but in different chunks
    (seams and the return pass). Information matrices from the ICP overlap, so flat-ground pairs constrain height /
    tilt but not the position along the road."""
    import open3d as o3d

    reg = o3d.pipelines.registration
    n = len(T_cam)
    pg = reg.PoseGraph()
    for T in T_cam:
        pg.nodes.append(reg.PoseGraphNode(T.copy()))
    inf_odo = np.diag([1 / np.radians(0.5) ** 2] * 3 + [1 / 0.05 ** 2] * 3)   # open3d information is (rot, trans)
    for c, (ck, (i0, i1)) in enumerate(zip(cks, chunks)):
        Ts = [np.linalg.inv(se3(E[:, :3], E[:, 3] * sv[c])) for E in ck["E"]]   # world_c <- cam
        for s in range(i1 - i0):
            for q in (1, 2):
                if s + q < i1 - i0:
                    rel = np.linalg.inv(Ts[s + q]) @ Ts[s]                     # open3d: T = inv(P_target) P_source
                    pg.edges.append(reg.PoseGraphEdge(i0 + s, i0 + s + q, rel, inf_odo, uncertain=False))
    # loop candidates: different chunk, closer than loop_m, not consecutive-chunk shared frames
    chunk_of = np.full(n, -1)
    for c, (i0, i1) in enumerate(chunks):
        chunk_of[i0:i1] = c
    C = np.array([T[:3, 3] for T in T_cam])
    cand = []
    for j in range(n):
        for k in range(j + 1, n):
            if abs(chunk_of[j] - chunk_of[k]) < 1 or np.linalg.norm(C[j] - C[k]) > a.loop_m:
                continue
            if abs(chunk_of[j] - chunk_of[k]) == 1 and (k - j) < 8:
                continue                                             # neighbours across a seam: odometry has them
            cand.append((j, k))
    rng = np.random.default_rng(0)
    if len(cand) > a.loop_max:
        cand = [cand[i] for i in sorted(rng.choice(len(cand), a.loop_max, replace=False))]
    n_ok = 0
    for j, k in cand:
        init = np.linalg.inv(T_cam[j]) @ T_cam[k]                      # cam_j <- cam_k
        src, tgt = clouds[k], clouds[j]
        T = init
        for dmax in (1.5, 0.6):
            r = reg.registration_icp(src, tgt, dmax, T, reg.TransformationEstimationPointToPlane(),
                                     reg.ICPConvergenceCriteria(max_iteration=30))
            T = r.transformation
        if r.fitness < a.loop_fitness or r.inlier_rmse > 0.3:
            continue
        info = reg.get_information_matrix_from_point_clouds(src, tgt, 0.6, T)
        pg.edges.append(reg.PoseGraphEdge(k, j, T, info, uncertain=True))     # T: cam_j <- cam_k (source k)
        n_ok += 1
    log(f"  pose graph: {n} nodes, {len(pg.edges) - n_ok} odometry edges, {n_ok} / {len(cand)} loop edges")
    opt = reg.GlobalOptimizationOption(max_correspondence_distance=0.6, edge_prune_threshold=0.25,
                                       preference_loop_closure=a.loop_pref, reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                                reg.GlobalOptimizationConvergenceCriteria(), opt)
    return [np.asarray(nd.pose) for nd in pg.nodes], n_ok, len(cand)


def ground_refine(T_cam, cks, chunks, owner, sv, rays, rgbs, above, pm, a, log, iters=4, rate=None):
    """Sequential scan-to-map refinement of each keyframe's roll, pitch and height (3 DoF, point-to-plane on the
    near points against the map of the earlier keyframes). 6-DoF ICP slides on the flat ground; tilt / height are
    what the ground constrains well, and a 1 deg tilt is 0.35 m at 20 m. The correction of keyframe j is carried
    to all later keyframes (chain), like refine_lidar."""
    from scipy.spatial import cKDTree

    n = len(T_cam)
    T_cam = [T.copy() for T in T_cam]
    local = []
    for j in range(n):
        c = owner[j]
        d = cks[c]["depth"][j - chunks[c][0]].astype(np.float32) * sv[c]
        cf = cks[c]["conf"][j - chunks[c][0]].astype(np.float32)
        m = (cf >= max(np.quantile(cf, a.conf_q), a.conf_min)) & (d > 1.5) & (d < a.gr_range) & ~sky_mask(rgbs[j], above)
        if pm[j] is not None:
            m &= ~pm[j]
        m[int(round((1.0 - a.crop_bottom) * m.shape[0])):] = False
        sub = np.zeros_like(m)
        sub[::4, ::4] = True
        m &= sub
        local.append(rays[m] * d[m, None])
    M = np.zeros((0, 3))
    tree = normals = None
    corr = np.eye(4)
    dz_all, tilt_all = [], []
    for j in range(n):
        T = corr @ T_cam[j]
        P0 = apply(T, local[j])
        turning = rate is not None and abs(rate[j]) > a.gr_max_rate        # in-place swing: the map seen from a
        if tree is not None and len(P0) > 200 and not turning:              # rotating viewpoint pulled 140718 6 deg off
            c0 = T[:3, 3]
            for _ in range(iters):
                P = apply(T, local[j])
                dist, idx = tree.query(P, distance_upper_bound=a.gr_dist)
                ok = np.isfinite(dist)
                if ok.sum() < 100:
                    break
                q, nn = M[idx[ok]], normals[idx[ok]]
                Pk = P[ok]
                r = np.einsum("ij,ij->i", nn, Pk - q)                     # point-to-plane residual
                # d r / d(theta_x, theta_y, dz): rotation about the horizontal axes through the camera centre
                lever = Pk - c0
                cols = [np.einsum("ij,ij->i", nn, np.cross(np.repeat([[1.0, 0, 0]], len(Pk), 0), lever)),
                        np.einsum("ij,ij->i", nn, np.cross(np.repeat([[0, 1.0, 0]], len(Pk), 0), lever)),
                        nn[:, 2]]
                if a.gr_dof == 6:                                           # + yaw, x, y: damped towards the init
                    cols += [np.einsum("ij,ij->i", nn, np.cross(np.repeat([[0, 0, 1.0]], len(Pk), 0), lever)), nn[:, 0], nn[:, 1]]
                J = np.stack(cols, 1)
                w = np.minimum(1.0, 0.15 / np.maximum(np.abs(r), 1e-6))     # Huber
                A_ = J * w[:, None]
                H = A_.T @ J + 1e-6 * np.eye(J.shape[1])
                if a.gr_dof == 6:                                           # prior sigma: 0.3 deg yaw, 0.1 m xy per keyframe
                    H[3, 3] += w.sum() * (0.15 / np.radians(0.3)) ** 2 * a.gr_prior
                    H[4, 4] += w.sum() * (0.15 / 0.1) ** 2 * a.gr_prior
                    H[5, 5] += w.sum() * (0.15 / 0.1) ** 2 * a.gr_prior
                x = np.linalg.solve(H, -(A_.T @ r))
                x[:2] = np.clip(x[:2], -np.radians(1.5), np.radians(1.5))   # per iteration
                rv, dt = (np.r_[x[:2], 0.0], np.r_[0, 0, x[2]]) if a.gr_dof == 3 else (np.r_[x[:2], x[3]], np.r_[x[4], x[5], x[2]])
                R = Rotation.from_rotvec(rv).as_matrix()
                T = se3(R, c0 - R @ c0 + dt) @ T
            T0 = corr @ T_cam[j]
            dR = Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T)
            if dR.magnitude() > np.radians(a.gr_max_deg):               # cap: a berm / people in the near field
                dR = Rotation.from_rotvec(dR.as_rotvec() * np.radians(a.gr_max_deg) / dR.magnitude())
                R = dR.as_matrix()
                T = se3(R, c0 - R @ c0 + [0, 0, T[2, 3] - T0[2, 3]]) @ T0
            if abs(T[2, 3] - T0[2, 3]) > a.gr_max_dz:
                T[2, 3] = T0[2, 3] + np.sign(T[2, 3] - T0[2, 3]) * a.gr_max_dz
            dz_all.append(T[2, 3] - T0[2, 3])
            tilt_all.append(np.degrees(Rotation.from_matrix(T[:3, :3] @ T0[:3, :3].T).magnitude()))
        corr = T @ np.linalg.inv(T_cam[j])
        T_cam[j] = T
        M = np.concatenate([M, apply(T, local[j])])
        if j % 2 == 0 or j == n - 1:
            M = voxel_first(M[np.linalg.norm(M[:, :2] - T[:2, 3], axis=1) < 60.0], 0.2)[0]
            tree = cKDTree(M)
            _, nb = tree.query(M, k=12)
            Q = M[nb] - M[nb].mean(1, keepdims=True)
            normals = np.linalg.svd(Q, full_matrices=False)[2][:, 2, :]
            normals *= np.sign(normals[:, 2:3] + 1e-9)
    if dz_all:
        log(f"  ground refine: per-keyframe |tilt| p50/p90 {np.percentile(tilt_all, 50):.2f}/{np.percentile(tilt_all, 90):.2f} deg, "
            f"|dz| p50/p90 {np.percentile(np.abs(dz_all), 50):.3f}/{np.percentile(np.abs(dz_all), 90):.3f} m")
    ground_refine.last = dict(tilt_deg=np.round(tilt_all, 3).tolist(), dz_m=np.round(dz_all, 4).tolist())
    return T_cam


def multiview_filter(P, kf, voxel, min_views):
    """Keep the points whose voxel was seen from at least min_views keyframes."""
    key = np.floor(P / voxel).astype(np.int64)
    key = (key[:, 0] + (1 << 20)) * (1 << 42) + (key[:, 1] + (1 << 20)) * (1 << 21) + (key[:, 2] + (1 << 20))
    pair = np.unique(np.c_[key, kf.astype(np.int64)], axis=0)
    uk, cnt = np.unique(pair[:, 0], return_counts=True)
    good = uk[cnt >= min_views]
    return np.isin(key, good)


def mean_pose(Ts) -> np.ndarray:
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    return se3(R, np.mean([T[:3, 3] for T in Ts], 0))


# ---------------------------------------------------------------------------------------------------- VGGT-Omega
def run_chunks(images, chunks, ckpt, res=512):
    import torch
    from rcmot.work.depth_engine import VGGTOmegaEngine

    eng = VGGTOmegaEngine(ckpt, ROOT, sync=True, res=res)
    eng._load()                                                        # also puts third_party/vggt-omega on sys.path
    from vggt_omega.utils.pose_enc import encoding_to_camera
    out = []
    for c, (i0, i1) in enumerate(chunks):
        X = torch.stack([eng._prep(images[i]) for i in range(i0, i1)]).to("cuda")
        tic = time.perf_counter()
        with torch.inference_mode():
            p = eng.model(X)
        h, w = X.shape[-2:]
        E, Kp = encoding_to_camera(p["pose_enc"], (h, w))
        out.append(dict(depth=p["depth"][0, ..., 0].float().cpu().numpy().astype(np.float16),
                        conf=p["depth_conf"][0].float().cpu().numpy().astype(np.float16),
                        E=E[0].float().cpu().numpy(), fx=float(Kp[0, :, 0, 0].mean())))
        print(f"  chunk {c + 1}/{len(chunks)}: {i1 - i0} views, {time.perf_counter() - tic:.1f} s, "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        del X, p
        torch.cuda.empty_cache()
    del eng
    torch.cuda.empty_cache()                                           # the GPU is shared with the GUI (5.9 GB)
    return out


def pixel_rays(calib, h, w):
    """Undistorted camera rays (z = 1) of the model-resolution pixel centres."""
    W, H = calib.image_size
    u, v = np.meshgrid((np.arange(w) + 0.5) * W / w - 0.5, (np.arange(h) + 0.5) * H / h - 0.5)
    n = cv2.undistortPoints(np.stack([u, v], -1).reshape(-1, 1, 2).astype(np.float64), calib.K, calib.D).reshape(h, w, 2)
    return np.dstack([n, np.ones((h, w))])


def radar_scale(ck, kf_idx, o, calib, kf_t):
    """Median (radar camera depth / VGGT depth) at the static radar points of the scans nearest the chunk's frames."""
    rt = np.asarray(o["rad_t"])
    W, H = calib.image_size
    h, w = ck["depth"].shape[1:]
    ratios = []
    for s, k in enumerate(kf_idx):
        near = np.argsort(np.abs(rt - kf_t[k]))[:2]
        P = np.concatenate([o["rad_pts"][j] for j in near if abs(rt[j] - kf_t[k]) < 0.1] or [np.zeros((0, 3))])
        if not len(P):
            continue
        pc = P @ calib.R_cl.T + calib.t_cl
        pc = pc[(pc[:, 2] > 2.0) & (pc[:, 2] < 40.0)]
        if not len(pc):
            continue
        uv = cv2.projectPoints(pc, np.zeros(3), np.zeros(3), calib.K, calib.D)[0].reshape(-1, 2)
        x = np.round((uv[:, 0] + 0.5) * w / W - 0.5).astype(int)
        y = np.round((uv[:, 1] + 0.5) * h / H - 0.5).astype(int)
        m = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        d = ck["depth"][s].astype(np.float32)
        cf = ck["conf"][s].astype(np.float32)
        good = m.copy()
        good[m] = cf[y[m], x[m]] >= np.quantile(cf, 0.3)
        ratios.append(pc[good, 2] / np.maximum(d[y[good], x[good]], 1e-3))
    r = np.concatenate(ratios) if ratios else np.zeros(0)
    return (float(np.median(r)) if len(r) >= 15 else float("nan")), len(r)


def sky_mask(rgb, above) -> np.ndarray:
    """Sky: blue or bright grey (cloud) pixels above the horizon, connected to the image top (VGGT puts part of the sky
    at 10-50 m with confidence > 1.2: 10-14 % of the points 4-15 m above the ground without this)."""
    r, g, b = (rgb[..., i].astype(int) for i in range(3))
    skyish = (((b > r + 15) & (b > 110)) | ((np.minimum(np.minimum(r, g), b) > 190) & (np.ptp(rgb, axis=2) < 30))) & above
    _, lab = cv2.connectedComponents(skyish.astype(np.uint8), connectivity=4)
    top = np.unique(lab[0][skyish[0]])
    return np.isin(lab, top[top > 0])


def chunk_points(ck, s, scale, T_W_cam, rays, img_rgb, a, above=None, pmask=None):
    """Back-project frame s of a chunk: metric depth along the calibrated rays -> world points + colours + range.
    above: pixels whose ray points above the horizon (level frame) -> sky removal."""
    d = ck["depth"][s].astype(np.float32) * scale
    cf = ck["conf"][s].astype(np.float32)
    gy, gx = np.gradient(np.log(np.maximum(d, 1e-3)))
    m = (cf >= max(np.quantile(cf, a.conf_q), a.conf_min)) & (d > 1.0) & (d < a.max_range) & (np.hypot(gx, gy) < 0.08)
    m[int(round((1.0 - a.crop_bottom) * m.shape[0])):] = False          # the machine's own bar / bucket at the bottom
    if above is not None:
        m &= ~sky_mask(img_rgb, above)
    if pmask is not None:
        m &= ~pmask
    sub = np.zeros_like(m)
    sub[::a.stride, ::a.stride] = True
    m &= sub
    Pc = rays[m] * d[m, None]
    return apply(T_W_cam, Pc).astype(np.float32), img_rgb[m], d[m].astype(np.float16)


# ---------------------------------------------------------------------------------------------------- outputs
def write_ply(path, P, C=None):
    with open(path, "wb") as f:
        hdr = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
        if C is not None:
            hdr += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f.write((hdr + "end_header\n").encode())
        dt = [("p", "<f4", 3)] + ([("c", "u1", 3)] if C is not None else [])
        a = np.empty(len(P), dt)
        a["p"] = P
        if C is not None:
            a["c"] = C
        f.write(a.tobytes())


def topview(path, rec, rec_c, lid, seen, traj, title, cell=0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    allp = np.concatenate([rec[:, :2], lid[seen][:, :2]])
    lo, hi = np.percentile(allp, 0.5, 0) - 3, np.percentile(allp, 99.5, 0) + 3
    nx, ny = np.ceil((hi - lo) / cell).astype(int)

    def grid(P, val, how):
        ij = np.floor((P[:, :2] - lo) / cell).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < nx) & (ij[:, 1] < ny)
        key, v = ij[ok, 0] * ny + ij[ok, 1], val[ok]
        G = np.full(nx * ny, np.nan)
        o = np.lexsort((v, key))
        key, v = key[o], v[o]
        uk, st, cnt = np.unique(key, return_index=True, return_counts=True)
        G[uk] = v[st + (cnt - 1) // 2] if how == "median" else v[st + cnt - 1]
        return G.reshape(nx, ny).T
    Hr, Hl = grid(rec, rec[:, 2], "median"), grid(lid[seen], lid[seen][:, 2], "median")
    ij = np.floor((rec[:, :2] - lo) / cell).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < nx) & (ij[:, 1] < ny)
    RGB = np.full((ny, nx, 3), 255, np.uint8)
    o = np.argsort(rec[ok, 2])                                           # highest point's colour on top
    RGB[ij[ok][o, 1], ij[ok][o, 0]] = rec_c[ok][o]
    ext = (lo[0], lo[0] + nx * cell, lo[1], lo[1] + ny * cell)
    z0 = np.nanpercentile(Hl, 5)
    fig, ax = plt.subplots(2, 2, figsize=(16, float(np.clip(16 * ny / nx + 1.5, 6, 30))), constrained_layout=True)
    ax[0, 0].imshow(RGB, origin="lower", extent=ext)
    ax[0, 0].set_title("camera reconstruction (colour of the highest point)")
    for a_, H_, t_ in ((ax[0, 1], Hr, "camera height [m]"), (ax[1, 0], Hl, "LiDAR height [m] (cells in the camera's view)")):
        im = a_.imshow(H_ - z0, origin="lower", extent=ext, cmap="viridis", vmin=-0.5, vmax=4.0)
        a_.set_title(t_)
        fig.colorbar(im, ax=a_, shrink=0.8)
    im = ax[1, 1].imshow(Hr - Hl, origin="lower", extent=ext, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax[1, 1].set_title("camera - LiDAR height [m] (median per 0.25 m cell)")
    fig.colorbar(im, ax=ax[1, 1], shrink=0.8)
    for a_ in ax.flat:
        for name, (P, col) in traj.items():
            a_.plot(P[:, 0], P[:, 1], color=col, lw=1.5, label=name)
        a_.set_aspect("equal")
        a_.set_xlabel("x [m] (first keyframe's level frame)")
    ax[0, 0].legend(loc="upper right", fontsize=9)
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def viewer_html(path, title, rec, rec_c, rec_kf, rec_err, lid, lid_t, kf_t, traj, report):
    """Self-contained three.js viewer: int16 cm positions, time slider (keyframe index), colour by RGB / height / error."""
    c0 = np.round(np.mean(rec, 0), 1)
    span = max(np.abs(rec - c0).max(), np.abs(lid - c0).max())
    q = 0.01 if span < 320 else 0.02
    b64 = lambda x: base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()   # noqa: E731
    pack = lambda P: b64(np.clip(np.round((P - c0) / q), -32767, 32767).astype(np.int16))   # noqa: E731
    lkf = np.searchsorted(kf_t, lid_t).astype(np.uint16)                   # LiDAR scan -> first keyframe after it
    err = np.where(np.isfinite(rec_err), np.clip(rec_err * 100, 0, 254), 255).astype(np.uint8)
    data = dict(q=q, c0=c0.tolist(), n_kf=len(kf_t), kf_t=(np.asarray(kf_t) - kf_t[0]).round(2).tolist(),
                rec=pack(rec), rec_c=b64(rec_c.astype(np.uint8)), rec_kf=b64(rec_kf.astype(np.uint16)), rec_err=b64(err),
                lid=pack(lid), lid_kf=b64(lkf),
                traj={k: dict(p=(P - c0).round(3).tolist(), col=col) for k, (P, col) in traj.items()},
                report=report)
    html = VIEWER.replace("__TITLE__", title).replace("__DATA__", json.dumps(data))
    Path(path).write_text(html)


VIEWER = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#121316;--panel:#1d1f23;--ink:#e6e6e6;--muted:#9aa0a6;--line:#33363c}
body{margin:0;background:var(--bg);color:var(--ink);font:13px/1.4 system-ui,sans-serif;overflow:hidden}
#ui{position:absolute;top:10px;left:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px 12px;width:320px;max-width:calc(100vw - 44px)}
#ui h1{font-size:14px;margin:0 0 6px}#ui label{display:block;margin:3px 0}#ui .row{display:flex;gap:8px;align-items:center;white-space:nowrap}
#ui input[type=range]{width:100%}#stats{color:var(--muted);font-size:12px;white-space:pre-line;margin-top:6px}
#legend{position:absolute;bottom:10px;left:10px;background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:12px;color:var(--muted)}
.sw{display:inline-block;width:18px;height:3px;vertical-align:middle;margin:0 4px 0 10px}
</style>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
</head><body>
<div id="ui"><h1>__TITLE__</h1>
<label><input type="checkbox" id="showRec" checked> 카메라 재구성 (VGGT-Ω + 레이더 축척)</label>
<label><input type="checkbox" id="showLid"> LiDAR 지도 (기준)</label>
<label><input type="checkbox" id="showTraj" checked> 궤적</label>
<div class="row">색 <select id="mode"><option value="rgb">카메라 색</option><option value="h">높이</option><option value="err">LiDAR까지 거리 (0-1 m)</option></select></div>
<div>시간 <span id="tlab"></span><input type="range" id="t" min="0" value="0"></div>
<div class="row">점 크기 <input type="range" id="ps" min="1" max="8" step="0.5" value="2"></div>
<div id="stats"></div></div>
<div id="legend"></div>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const pos = s => { const a = dec(s, Int16Array), f = new Float32Array(a.length); for (let i = 0; i < a.length; i++) f[i] = a[i] * D.q; return f; };
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x121316);
const cam = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.1, 3000); cam.up.set(0, 0, 1);
const ren = new THREE.WebGLRenderer({antialias: true}); ren.setPixelRatio(devicePixelRatio); ren.setSize(innerWidth, innerHeight); document.body.appendChild(ren.domElement);
const ctl = new OrbitControls(cam, ren.domElement);
function cloud(P, kf) {
  const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(P, 3));
  const c = new Float32Array(P.length); g.setAttribute('color', new THREE.BufferAttribute(c, 3));
  const m = new THREE.PointsMaterial({size: 2, sizeAttenuation: false, vertexColors: true});
  const o = new THREE.Points(g, m); o.userData = {kf, n: P.length / 3}; scene.add(o); return o;
}
const recP = pos(D.rec), recC = dec(D.rec_c, Uint8Array), recKf = dec(D.rec_kf, Uint16Array), recE = dec(D.rec_err, Uint8Array);
const lidP = pos(D.lid), lidKf = dec(D.lid_kf, Uint16Array);
// draw order = keyframe order, so the time slider is a draw range
const ord = a => Array.from(a.keys()).sort((i, j) => a[i] - a[j]);
function reorder(P, kf, extra) { const o = ord(kf), P2 = new Float32Array(P.length), k2 = new Uint16Array(kf.length), e2 = extra.map(([x, w]) => new x.constructor(x.length));
  o.forEach((s, d) => { for (let c = 0; c < 3; c++) P2[3 * d + c] = P[3 * s + c]; k2[d] = kf[s]; extra.forEach(([x, w], j) => { for (let c = 0; c < w; c++) e2[j][w * d + c] = x[w * s + c]; }); });
  return [P2, k2, ...e2]; }
const [rP, rK, rC, rE] = reorder(recP, recKf, [[recC, 3], [recE, 1]]);
const [lP, lK] = reorder(lidP, lidKf, []);
const rec = cloud(rP, rK), lid = cloud(lP, lK);
function hcol(z, lo, hi) { const x = Math.min(Math.max((z - lo) / (hi - lo), 0), 1); return [0.15 + 0.85 * x, 0.25 + 0.6 * x * (1 - x) * 2.5, 0.85 - 0.75 * x]; }
const zs = Array.from(lK.length ? lP.filter((_, i) => i % 3 == 2) : rP.filter((_, i) => i % 3 == 2)).sort((a, b) => a - b);
const zlo = zs[Math.floor(zs.length * 0.05)], zhi = zlo + 4;
function paint() {
  const mode = document.getElementById('mode').value, c = rec.geometry.attributes.color.array;
  for (let i = 0; i < rK.length; i++) {
    let v;
    if (mode == 'rgb') v = [rC[3 * i] / 255, rC[3 * i + 1] / 255, rC[3 * i + 2] / 255];
    else if (mode == 'h') v = hcol(rP[3 * i + 2], zlo, zhi);
    else { const e = rE[i]; v = e == 255 ? [0.35, 0.35, 0.35] : (e < 25 ? [0.16, 0.47, 0.84] : e < 50 ? [0.55, 0.75, 0.3] : e < 100 ? [0.95, 0.7, 0.2] : [0.9, 0.25, 0.2]); }
    c.set(v, 3 * i);
  }
  rec.geometry.attributes.color.needsUpdate = true;
  const lc = lid.geometry.attributes.color.array;
  for (let i = 0; i < lK.length; i++) { const v = mode == 'h' ? hcol(lP[3 * i + 2], zlo, zhi) : [0.62, 0.62, 0.66]; lc.set(v, 3 * i); }
  lid.geometry.attributes.color.needsUpdate = true;
  document.getElementById('legend').innerHTML = mode == 'err'
    ? '<span class="sw" style="background:#2a78d6"></span>&lt;0.25 m<span class="sw" style="background:#8cbf4d"></span>0.25-0.5<span class="sw" style="background:#f2b233"></span>0.5-1<span class="sw" style="background:#e64033"></span>&gt;1 m<span class="sw" style="background:#595959"></span>LiDAR 없음'
    : Object.entries(D.traj).map(([k, v]) => `<span class="sw" style="background:${v.col}"></span>${k}`).join('');
}
const lines = Object.entries(D.traj).map(([k, v]) => { const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(v.p.flat()), 3));
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({color: v.col})); scene.add(l); return l; });
const ts = document.getElementById('t'); ts.max = D.n_kf - 1; ts.value = D.n_kf - 1;
const firstAfter = (arr, k) => { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= k) lo = m + 1; else hi = m; } return lo; };
function setT() {
  const k = +ts.value; document.getElementById('tlab').textContent = `${D.kf_t[k].toFixed(1)} s (키프레임 ${k + 1}/${D.n_kf})`;
  rec.geometry.setDrawRange(0, firstAfter(rK, k)); lid.geometry.setDrawRange(0, firstAfter(lK, k));
  lines.forEach(l => l.geometry.setDrawRange(0, k + 1));
}
const $ = id => document.getElementById(id);
const vis = () => { rec.visible = $('showRec').checked; lid.visible = $('showLid').checked; lines.forEach(l => l.visible = $('showTraj').checked); };
['showRec', 'showLid', 'showTraj'].forEach(id => document.getElementById(id).onchange = vis);
document.getElementById('mode').onchange = paint; ts.oninput = setT;
document.getElementById('ps').oninput = e => { rec.material.size = lid.material.size = +e.target.value; };
const R = D.report; document.getElementById('stats').textContent =
  `키프레임 ${R.keyframes}, 이동 ${R.trajectory.path_m} m\n궤적 끝 오차 ${R.trajectory.end_err_m} m (${R.trajectory.end_err_pct} %), 최대 ${R.trajectory.max_err_m} m\n` +
  `점 정확도: 중앙값 ${R.accuracy.median_m} m, 0.5 m 이내 ${R.accuracy.within_0p5_pct} %\n완전성(LiDAR 점 중 0.5 m 안에 카메라 점): ${R.completeness.within_0p5_pct} %`;
const tr = D.traj[Object.keys(D.traj)[0]].p, mid = tr[Math.floor(tr.length / 2)];
ctl.target.set(mid[0], mid[1], mid[2]); cam.position.set(mid[0] - 25, mid[1] - 35, mid[2] + 35); ctl.update();
paint(); setT(); vis();
addEventListener('resize', () => { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); ren.setSize(innerWidth, innerHeight); });
(function loop() { requestAnimationFrame(loop); ctl.update(); ren.render(scene, cam); })();
</script></body></html>
"""


# ---------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="configs/datasets.yaml profile (bag, calib, topics, world.lever_arm)")
    ap.add_argument("--t0", type=float, default=0.0, help="segment start [s from the bag start]")
    ap.add_argument("--t1", type=float, default=None, help="segment end [s from the bag start] (default: bag end)")
    ap.add_argument("--lidar", default="/ouster/points")
    ap.add_argument("--img-hz", type=float, default=6.0, help="images kept from the bag (keyframe candidates)")
    ap.add_argument("--kf-dist", type=float, default=1.0, help="new keyframe after this much travel [m] ...")
    ap.add_argument("--kf-deg", type=float, default=8.0, help="... or this much turn [deg] ...")
    ap.add_argument("--kf-max-dt", type=float, default=8.0, help="... or this long [s]")
    ap.add_argument("--kf-min-dt", type=float, default=0.15)
    ap.add_argument("--chunk", type=int, default=16, help="VGGT-Omega views per inference")
    ap.add_argument("--res", type=int, default=512, help="VGGT-Omega input resolution (long side)")
    ap.add_argument("--overlap", type=int, default=4, help="frames shared by consecutive chunks (pose chaining)")
    ap.add_argument("--conf-q", type=float, default=0.5, help="drop the least confident share of each depth map ...")
    ap.add_argument("--conf-min", type=float, default=1.2, help="... and pixels below this VGGT confidence (sky ~1.0)")
    ap.add_argument("--crop-bottom", type=float, default=0.07, help="image share at the bottom not used (own machine)")
    ap.add_argument("--scale", choices=("auto", "travel", "depth"), default="auto",
                    help="chunk scale: radar-Doppler travel / radar depth ratio / travel when >= --min-travel")
    ap.add_argument("--min-travel", type=float, default=3.0, help="[m]")
    ap.add_argument("--no-lidar-refine", dest="lidar_refine", action="store_false",
                    help="LiDAR reference from the KISS-ICP odometry only (no scan-to-map ICP)")
    ap.add_argument("--no-ground-refine", dest="ground_refine", action="store_false",
                    help="no per-keyframe roll / pitch / height refinement against the map of the earlier keyframes")
    ap.add_argument("--gr-range", type=float, default=20.0, help="ground refine: points up to this depth [m]")
    ap.add_argument("--gr-dist", type=float, default=1.0, help="ground refine: correspondence distance [m]")
    ap.add_argument("--gr-dof", type=int, choices=(3, 6), default=3, help="ground refine: roll/pitch/z, or all 6 (damped)")
    ap.add_argument("--gr-max-deg", type=float, default=0.5, help="ground refine: largest tilt change per keyframe [deg]")
    ap.add_argument("--gr-max-dz", type=float, default=0.1, help="ground refine: largest height change per keyframe [m]")
    ap.add_argument("--gr-max-rate", type=float, default=1e9, help="ground refine: skipped while turning faster [deg/s] (off: skipping lost the swing-section height fix)")
    ap.add_argument("--gr-prior", type=float, default=1.0, help="6-DoF damping weight (1 = sigma 0.3 deg / 0.1 m)")
    ap.add_argument("--pgo", action="store_true", help="pose-graph optimisation with ICP loop edges (did not help)")
    ap.add_argument("--pgo-iters", type=int, default=2, help="pose graph passes (ICP loop edges re-found each pass)")
    ap.add_argument("--loop-m", type=float, default=8.0, help="loop-edge candidates: cameras closer than this [m]")
    ap.add_argument("--loop-max", type=int, default=600)
    ap.add_argument("--loop-fitness", type=float, default=0.5)
    ap.add_argument("--loop-pref", type=float, default=0.3, help="open3d preference_loop_closure")
    ap.add_argument("--no-people", dest="people", action="store_false", help="keep people (no YOLO mask)")
    ap.add_argument("--min-views", type=int, default=2, help="keep points seen from >= this many keyframes (0.3 m voxels)")
    ap.add_argument("--join", choices=("hybrid", "points", "cameras"), default="hybrid",
                    help="join chunks on the shared cameras (heading, xy) + the shared frames' depth maps (tilt, height) / "
                         "on the depth maps only (rigid, Huber) / on the camera poses only")
    ap.add_argument("--gravity", action="store_true",
                    help="tilt each chunk to the accelerometer's up (excavator A's Ouster IMU scatters 4-7 deg vs the "
                         "LiDAR while driving: off by default)")
    ap.add_argument("--max-range", type=float, default=35.0, help="camera points up to this depth [m]")
    ap.add_argument("--stride", type=int, default=2, help="pixel stride at the model resolution")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--map-dt", type=float, default=0.5, help="LiDAR map scan spacing [s]")
    ap.add_argument("--lidar-range", type=float, default=50.0)
    ap.add_argument("--reuse", action="store_true", help="reuse the cached bag read / VGGT chunks")
    ap.add_argument("--out", default=str(ROOT / "data/drive_recon"))
    a = ap.parse_args()

    prof, cfg, topics = load_profile(a.profile)
    out = Path(a.out) / a.profile
    out.mkdir(parents=True, exist_ok=True)
    calp = ROOT / prof["calib"]
    calib = RigCalib.load(str(calp))
    bag_camera_info(prof["bag"], topics["image"], calib)
    lever = (cfg.get("world") or {}).get("lever_arm")
    R_lev, t_lev = RigExtrinsics.from_calib_json(json.load(open(calp))).level()

    # ---- bag
    cache = out / f"read_v3_{a.t0:g}_{a.t1}.pkl"                   # v3: + accelerometer, deskewed LiDAR odometry
    shared = ROOT / "data/drive_recon" / a.profile / cache.name        # sweeps: --out elsewhere, one bag read
    if a.reuse and not cache.exists() and shared.exists():
        cache = shared
    if a.reuse and cache.exists():
        o = pickle.load(open(cache, "rb"))
    else:
        print(f"reading {prof['bag']} ...", flush=True)
        o = read_bag(prof["bag"], calib, cfg, topics, a.t0, a.t1, a)
        pickle.dump(o, open(cache, "wb"), protocol=5)
    R_ls, t_ls = lidar_to_sensor({}, o["lidar_frame"])
    A = se3(R_lev @ R_ls, R_lev @ t_ls + t_lev)                       # LiDAR -> level
    T_lvl_cam = np.linalg.inv(se3(calib.R_cl, calib.t_cl))             # camera -> level
    img_t = np.asarray(o["img_t"])
    print(f"{len(img_t)} images, {len(o['rad_t'])} radar scans, {len(o['lid_t'])} LiDAR scans", flush=True)

    # ---- keyframes, VGGT chunks
    dr = dead_reckon(o, calib, lever)
    kf = pick_keyframes(img_t, dr, a)
    kf_t = img_t[kf]
    step = a.chunk - a.overlap
    starts = list(range(0, max(len(kf) - a.overlap, 1), step))
    chunks = [(s, min(s + a.chunk, len(kf))) for s in starts]
    print(f"{len(kf)} keyframes over {kf_t[-1] - kf_t[0]:.0f} s -> {len(chunks)} chunks", flush=True)
    images = [cv2.imdecode(np.frombuffer(o["img"][i], np.uint8), cv2.IMREAD_COLOR) for i in kf]
    ccache = out / f"chunks_{a.t0:g}_{a.t1}_{len(kf)}_{a.chunk}_{a.overlap}{'' if a.res == 512 else f'_r{a.res}'}.pkl"
    if a.reuse and ccache.exists():
        cks = pickle.load(open(ccache, "rb"))
    else:
        ckpt = (cfg.get("work") or {}).get("vggt_ckpt", "models/vggt/vggt_omega_1b_512.pt")
        cks = run_chunks(images, chunks, ckpt, a.res)
        pickle.dump(cks, open(ccache, "wb"), protocol=5)

    # ---- scale + chaining (camera + radar only)
    h, w = cks[0]["depth"].shape[1:]
    rays = pixel_rays(calib, h, w)
    above = (rays @ calib.R_cl)[..., 2] > 0                              # ray direction in the level frame points up
    cc = calib.camera_center()
    traj_dr = dr_camera_centres(dr, kf_t, cc)
    scales, sv = [], []
    for ck, (i0, i1) in zip(cks, chunks):
        s_d, n = radar_scale(ck, list(range(i0, i1)), o, calib, kf_t)
        s_t, travel = travel_scale(ck, traj_dr[i0:i1])
        # travel (Doppler speed, ~1 %) while the chunk drives; the radar depth ratio for swings / standing still
        use_t = a.scale == "travel" or (a.scale == "auto" and travel >= a.min_travel and np.isfinite(s_t))
        scales.append(dict(depth=round(s_d, 3), travel=round(s_t, 3), travel_m=round(travel, 1), points=n,
                           used="travel" if use_t else "depth"))
        sv.append(s_t if use_t else s_d)
    sv = np.array(sv)
    for c in range(1, len(cks)):                                        # short travel: baselines of the shared cameras
        if scales[c]["travel_m"] < a.min_travel:                        # in the previous (metric) chunk / in this one
            i0, p1 = chunks[c][0], chunks[c - 1][1]
            Cp = np.array([-E[:, :3].T @ E[:, 3] for E in cks[c - 1]["E"]])[i0 - chunks[c - 1][0]:] * sv[c - 1]
            Cc = np.array([-E[:, :3].T @ E[:, 3] for E in cks[c]["E"]])[:p1 - i0]
            dp, dc = np.linalg.norm(Cp[1:] - Cp[:-1], axis=1), np.linalg.norm(Cc[1:] - Cc[:-1], axis=1)
            if dc.sum() > 1e-3:
                sv[c] = dp.sum() / dc.sum()
                scales[c].update(shared=round(float(sv[c]), 3), used="shared")
    fill = np.nanmedian(sv)
    sv = np.where(np.isfinite(sv), sv, fill)                           # chunks without radar points: the median
    # gravity: the first chunk's cameras define "up" in the world frame; every later chunk is turned about a horizontal
    # axis through its first camera so that its cameras agree with the accelerometer (chained tilt drift otherwise)
    grav = Gravity(o, R_lev) if a.gravity and o.get("imu_frame") in ("os_imu", "os_sensor") else None
    ups = [grav(t) if grav is not None else None for t in kf_t]
    u_W, tilt_fix = None, []
    pose_ck = []                                                       # per chunk: {keyframe: T_W_cam}
    for c, (ck, (i0, i1)) in enumerate(zip(cks, chunks)):
        Es = [se3(E[:, :3], E[:, 3] * sv[c]) for E in ck["E"]]          # cam <- chunk world (metric)
        if c == 0:
            T_W_C = T_lvl_cam @ Es[0]
        else:
            shared = range(i0, chunks[c - 1][1])
            T_W_C = mean_pose([pose_ck[c - 1][j] @ Es[j - i0] for j in shared])   # on the shared cameras, then
            if a.join != "cameras":                                      # on the shared frames' depth maps
                p0 = chunks[c - 1][0]
                PQW = [overlap_points(cks[c - 1], j - p0, sv[c - 1], pose_ck[c - 1][j], ck, j - i0, sv[c], Es[j - i0], rays, a)
                       for j in shared]
                P_, Q_, w_ = (np.concatenate(x) for x in zip(*PQW))
                if len(P_) >= 200:
                    T_pts = align_rigid(P_, Q_, w_, T_W_C)
                    if a.join == "points":
                        T_W_C = T_pts
                    else:
                        # hybrid: the shared points (mostly ground) fix tilt and height, the shared cameras heading and
                        # xy (142258: cameras alone 1.5 deg tilt / points alone 2.9 deg heading at the end)
                        e3 = np.array([0.0, 0.0, 1.0])
                        Rt = rot_between(T_W_C[:3, :3] @ (T_pts[:3, :3].T @ e3), e3)
                        q0 = (T_W_C @ np.linalg.inv(Es[0]))[:3, 3]           # about the first shared camera
                        T_W_C = se3(Rt, q0 - Rt @ q0) @ T_W_C
                        T_W_C[2, 3] += (T_pts @ np.linalg.inv(Es[0]))[2, 3] - q0[2]
        poses = {i0 + s: T_W_C @ np.linalg.inv(E) for s, E in enumerate(Es)}
        wu = [poses[j][:3, :3] @ (calib.R_cl @ ups[j]) for j in range(i0, i1) if ups[j] is not None]
        if len(wu) >= 3:
            m = np.mean(wu, 0)
            m /= np.linalg.norm(m)
            if u_W is None:
                u_W = m
            else:
                Rc, piv = rot_between(m, u_W), poses[i0][:3, 3]
                Tc = se3(Rc, piv - Rc @ piv)
                poses = {j: Tc @ T for j, T in poses.items()}
                tilt_fix.append(round(float(np.degrees(np.arccos(np.clip(m @ u_W, -1, 1)))), 2))
        pose_ck.append(poses)
    half = a.overlap // 2
    owner = {}
    for c, (i0, i1) in enumerate(chunks):
        lo = i0 if c == 0 else i0 + half
        hi = i1 if c == len(chunks) - 1 else i1 - (a.overlap - half)
        for j in range(lo, hi):
            owner[j] = c
    T_cam = [pose_ck[owner[j]][j] for j in range(len(kf))]
    pm = [None] * len(kf)
    if a.people:
        pm_full, n_people = person_masks(images, ROOT)
        pm = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in pm_full]
        print(f"  people: {n_people} detections masked", flush=True)
    rgbs = [cv2.cvtColor(cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB) for im in images]
    n_loop = (0, 0)
    if a.ground_refine:
        th_kf = np.unwrap(np.interp(kf_t, dr[0], dr[1]))                 # gyro heading rate at the keyframes
        rate = np.degrees(np.gradient(th_kf, kf_t))
        T_cam = ground_refine(T_cam, cks, chunks, owner, sv, rays, rgbs, above, pm, a, lambda m: print(m, flush=True), rate=rate)
    if a.pgo:
        clouds = [keyframe_cloud(cks[owner[j]], j - chunks[owner[j]][0], sv[owner[j]], rays, rgbs[j], above, pm[j], a)
                  for j in range(len(kf))]
        for it in range(a.pgo_iters):
            T_cam, *n_loop = pose_graph(T_cam, cks, chunks, sv, clouds, kf_t, a, lambda m: print(m, flush=True))

    # ---- camera points
    P, C, K_, Rg = [], [], [], []
    for j in range(len(kf)):
        c = owner[j]
        p, col, rng = chunk_points(cks[c], j - chunks[c][0], sv[c], T_cam[j], rays, rgbs[j], a, above, pm[j])
        P.append(p), C.append(col), K_.append(np.full(len(p), j, np.uint16)), Rg.append(rng)
    P, C, K_, Rg = (np.concatenate(x) for x in (P, C, K_, Rg))
    if a.min_views > 1:
        keep = multiview_filter(P, K_, 0.3, a.min_views)
        print(f"  multi-view filter: kept {100 * keep.mean():.1f} % of the points", flush=True)
        P, C, K_, Rg = P[keep], C[keep], K_[keep], Rg[keep]
    rec, rec_c, rec_kf, rec_rng = voxel_first(P, a.voxel, C, K_, Rg)

    # ---- LiDAR reference
    T_W_lidar = lidar_poses(o, A, kf_t[0])
    if a.lidar_refine:
        T_W_lidar = refine_lidar(o, T_W_lidar)
    L, LT = [], []
    for ts, _, Q in o["map"]:
        L.append(apply(T_W_lidar(ts)[0], Q.astype(np.float64)).astype(np.float32))
        LT.append(np.full(len(Q), ts))
    lid, lid_t = voxel_first(np.concatenate(L), a.voxel, np.concatenate(LT))
    true_cam = T_W_lidar(kf_t) @ np.linalg.inv(A) @ T_lvl_cam           # camera poses from the LiDAR odometry

    # outbound vs return ground of each map (headings of the camera / of the LiDAR at the time of the points)
    hd_cam = np.array([np.arctan2(T[1, 2], T[0, 2]) for T in T_cam])
    hd_lid = np.array([np.arctan2(T[1, 2], T[0, 2]) for T in true_cam])
    consist = dict(camera=pass_consistency(rec, hd_cam[rec_kf], hd_cam[0]),
                   lidar=pass_consistency(lid, np.interp(lid_t, kf_t, np.unwrap(hd_lid)), hd_lid[0]))

    # trajectories (camera centres)
    traj_rec, traj_true = np.array([T[:3, 3] for T in T_cam]), true_cam[:, :3, 3]
    path = float(np.linalg.norm(np.diff(traj_true, axis=0), axis=1).sum())
    e_rec = np.linalg.norm(traj_rec - traj_true, axis=1)
    e_dr = np.linalg.norm(traj_dr[:, :2] - traj_true[:, :2], axis=1)
    ang = [np.degrees(Rotation.from_matrix(T1[:3, :3].T @ T2[:3, :3]).magnitude()) for T1, T2 in zip(T_cam, true_cam)]
    up = lambda T: T[:3, :3].T @ np.array([0.0, 0.0, 1.0])               # noqa: E731  world up in the camera frame
    tilt = [np.degrees(np.arccos(np.clip(up(T1) @ up(T2), -1, 1))) for T1, T2 in zip(T_cam, true_cam)]
    hd = lambda T: np.arctan2(T[1, 2], T[0, 2])                          # noqa: E731  optical axis heading
    yaw = [np.degrees((hd(T1) - hd(T2) + np.pi) % (2 * np.pi) - np.pi) for T1, T2 in zip(T_cam, true_cam)]

    # accuracy / completeness
    tree = cKDTree(lid)
    d_rec, _ = tree.query(rec, distance_upper_bound=5.0)
    has = np.isfinite(d_rec)
    seen = np.zeros(len(lid), bool)
    W_, H_ = calib.image_size
    for T in true_cam[::2]:
        pc = apply(np.linalg.inv(T), lid)
        m = (pc[:, 2] > 1.0) & (pc[:, 2] < a.max_range)
        uv = np.full((len(pc), 2), -1.0)
        uv[m] = cv2.projectPoints(pc[m], np.zeros(3), np.zeros(3), calib.K, calib.D)[0].reshape(-1, 2)
        seen |= m & (uv[:, 0] >= 0) & (uv[:, 0] < W_) & (uv[:, 1] >= 0) & (uv[:, 1] < H_)
    d_lid, _ = cKDTree(rec).query(lid[seen], distance_upper_bound=5.0)
    pct = lambda x: round(100.0 * float(np.mean(x)), 1)                 # noqa: E731
    by_range = {}
    for lo_, hi_ in ((0, 10), (10, 20), (20, a.max_range)):
        m = has & (rec_rng >= lo_) & (rec_rng < hi_)
        if m.sum() > 100:
            by_range[f"{lo_}-{hi_:g} m"] = dict(n=int(m.sum()), median_m=round(float(np.median(d_rec[m])), 3),
                                                 within_0p5_pct=pct(d_rec[m] < 0.5))
    report = dict(
        profile=a.profile, segment_s=[a.t0, a.t1], keyframes=len(kf), chunks=len(chunks),
        gravity_tilt_fix_deg=tilt_fix if grav is not None else "off",
        pose_graph=dict(on=a.pgo, loop_edges=n_loop[0], loop_candidates=n_loop[1], iters=a.pgo_iters) if a.pgo else "off",
        ground_refine=dict(on=a.ground_refine, range_m=a.gr_range, dist_m=a.gr_dist, dof=a.gr_dof, max_deg=a.gr_max_deg,
                           max_dz=a.gr_max_dz, **(getattr(ground_refine, "last", {}) if a.ground_refine else {})),
        filters=dict(people=a.people, min_views=a.min_views),
        scale=dict(per_chunk=scales, vggt_fx_px=[round(ck["fx"], 1) for ck in cks],
                   calib_fx_px=round(float(calib.K[0, 0]) * w / calib.image_size[0], 1)),
        trajectory=dict(path_m=round(path, 1), end_err_m=round(float(e_rec[-1]), 2), end_err_pct=round(100 * float(e_rec[-1]) / max(path, 1e-6), 1),
                        max_err_m=round(float(e_rec.max()), 2), median_err_m=round(float(np.median(e_rec)), 2),
                        end_rot_err_deg=round(float(ang[-1]), 1), max_rot_err_deg=round(float(np.max(ang)), 1),
                        end_yaw_err_deg=round(float(yaw[-1]), 1), end_tilt_err_deg=round(float(tilt[-1]), 1),
                        max_tilt_err_deg=round(float(np.max(tilt)), 1),
                        path_ratio=round(float(np.linalg.norm(np.diff(traj_rec, axis=0), axis=1).sum()) / max(path, 1e-6), 3),
                        radar_gyro_dr_end_err_m=round(float(e_dr[-1]), 2), radar_gyro_dr_max_err_m=round(float(e_dr.max()), 2)),
        accuracy=dict(points=int(len(rec)), with_lidar_within_5m_pct=pct(has), median_m=round(float(np.median(d_rec[has])), 3),
                      p80_m=round(float(np.percentile(d_rec[has], 80)), 3), within_0p25_pct=pct(d_rec[has] < 0.25),
                      within_0p5_pct=pct(d_rec[has] < 0.5), within_1_pct=pct(d_rec[has] < 1.0), by_camera_range=by_range),
        pass_consistency=dict(consist, note="return-pass ground minus outbound-pass ground in common 0.5 m cells "
                                            "(no reference needed; 0 = the map closes on itself)"),
        completeness=dict(lidar_points_in_view=int(seen.sum()), within_0p5_pct=pct(d_lid < 0.5), within_1_pct=pct(d_lid < 1.0),
                          note="LiDAR map points inside some keyframe's image (0-max_range m, occlusion ignored)"),
        per_keyframe=dict(t=(kf_t - kf_t[0]).round(2).tolist(), cam_err_m=e_rec.round(3).tolist(), dr_err_m=e_dr.round(3).tolist(),
                          rot_err_deg=np.round(ang, 2).tolist(), yaw_err_deg=np.round(yaw, 2).tolist(),
                          tilt_err_deg=np.round(tilt, 2).tolist()))
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "per_keyframe"}, indent=1, ensure_ascii=False))

    traj = {"LiDAR 기준": (traj_true, "#e6e6e6"), "카메라 (VGGT-Ω)": (traj_rec, "#2a78d6"), "레이더+자이로": (traj_dr, "#eb6834")}
    topview(out / "topview.png", rec, rec_c, lid, seen, {k: (v[0], {"#e6e6e6": "k"}.get(v[1], v[1])) for k, v in traj.items()},
            f"{a.profile}: camera reconstruction vs LiDAR")
    write_ply(out / "recon.ply", rec, rec_c)
    write_ply(out / "lidar.ply", lid)
    vr, vc, vk, ve = rec, rec_c, rec_kf, np.where(has, d_rec, np.nan)
    if len(vr) > 1_500_000:                                            # viewer budget
        vr, vc, vk, ve = voxel_first(vr, 0.15, vc, vk, ve)
    vl, vt = (lid, lid_t) if len(lid) < 1_000_000 else voxel_first(lid, 0.2, lid_t)
    viewer_html(out / "viewer.html", f"주행 3D 재구성 {a.profile}", vr, vc, vk, ve, vl, vt, kf_t, traj,
                {k: v for k, v in report.items() if k != "per_keyframe"})
    print("wrote", out)


if __name__ == "__main__":
    main()
