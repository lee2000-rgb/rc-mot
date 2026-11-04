"""Field calibration solve: one stationary capture -> camera / radar extrinsics + report + overlays.

Same methods as the HDX bag calibration (docs/06_hdx_dataset_calibration.html), plus corner reflectors:
  ground   LiDAR RANSAC plane
  camera   person-mask purity x contrast (camera_lidar), coarse +-camera_search_deg then fine search
  radar    joint point-to-plane ICP + horizontal reflector anchors (radar_lidar, reflectors), two rounds
           (re-match the reflectors after the first solve)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation

from rcmot.calib import camera_lidar as cl
from rcmot.calib import radar_lidar as rl
from rcmot.calib import reflectors as rf
from rcmot.calib.common import DEFAULT_TOPICS, Topics, ground_plane, ground_plane_near, lidar_to_sensor, stationary_windows, tf_prior
from rcmot.sensors.bag_source import pc2_to_struct
from rcmot.utils.rig_extrinsics import RigExtrinsics, rot_rpy

CAM_FRAMES = ("oak-d-base-frame", "oak_rgb_camera_frame", "camera_link", "camera")
RADAR_FRAMES = ("bosch_4d_frame", "radar", "radar_link")


def _rpy(R) -> list[float]:
    y, p, r = Rotation.from_matrix(R).as_euler("ZYX", degrees=True)
    return [float(r), float(p), float(y)]


def initial_extrinsics(cfg: dict) -> tuple[RigExtrinsics, list[str]]:
    ini, notes = cfg["initial"], []
    ex = RigExtrinsics(cam_rpy_deg=list(map(float, ini["cam_rpy_deg"])), cam_t=list(map(float, ini["cam_t"])))
    if ini.get("radar_points_frame", "os_sensor") == "radar":
        ex.radar_rpy_deg, ex.radar_t = list(map(float, ini["radar_rpy_deg"])), list(map(float, ini["radar_t"]))
    if ini.get("init_from_tf_bag"):
        tf = tf_prior(ini["init_from_tf_bag"])
        for name in CAM_FRAMES:
            if ("os_sensor", name) in tf:
                R = Rotation.from_euler("xyz", tf[("os_sensor", name)]["rpy_deg"], degrees=True).as_matrix()
                t = np.array(tf[("os_sensor", name)]["t"])
                ex.cam_rpy_deg, ex.cam_t = _rpy(R.T), (R.T @ t).tolist()       # p_body = R^T (p_s - t)
                notes.append(f"camera init from tf os_sensor->{name}")
                break
        if ini.get("radar_points_frame", "os_sensor") == "radar":
            for name in RADAR_FRAMES:
                if ("os_sensor", name) in tf:
                    R = Rotation.from_euler("xyz", tf[("os_sensor", name)]["rpy_deg"], degrees=True).as_matrix()
                    ex.radar_rpy_deg, ex.radar_t = _rpy(R), list(tf[("os_sensor", name)]["t"])
                    notes.append(f"radar init from tf os_sensor->{name}")
                    break
    return ex, notes


def solve(bag: str, cfg: dict, out_json: str, seg_weights: str, *, overlay_dir: str | None = None,
          windows: list[tuple[float, float]] | None = None, log=print) -> dict:
    """`windows`: explicit (t0, t1) seconds-from-start; use when one reflector is carried to several spots
    (one window per spot, reflectors are detected per window) instead of the automatic stationary split."""
    topics = Topics(**cfg["topics"])
    sv = cfg["solve"]
    ex, notes = initial_extrinsics(cfg)
    warn = []
    ex0 = RigExtrinsics(**{k: list(v) if isinstance(v, list) else v for k, v in vars(ex).items()})

    # 1. stationary part of the capture
    with AnyReader([Path(bag)]) as r:
        dur = (r.end_time - r.start_time) / 1e9
    ws = [(float(a), float(b)) for a, b in windows] if windows else stationary_windows(bag, topics=topics, max_len=max(dur, 60))
    if windows:
        notes.append(f"windows given by hand: {[(round(a, 1), round(b, 1)) for a, b in ws]}")
    elif not ws:
        warn.append("radar Doppler says the machine was MOVING during the whole capture; result uses all of it")
        ws = [(0.5, dur - 0.5)]
    elif sum(b - a for a, b in ws) < 0.7 * dur:
        warn.append(f"only {sum(b - a for a, b in ws):.0f} of {dur:.0f} s stationary; using the stationary part")
    log(f"stationary windows: {[(round(a, 1), round(b, 1)) for a, b in ws]}")
    windows = [rl.load_window(bag, a, b, topics=topics, keep_raw=True) for a, b in ws]

    # 2. ground
    n, h, n_inl = ground_plane(np.concatenate([w.lidar_raw for w in windows]))
    ex.ground_normal, ex.sensor_height = n.tolist(), h
    log(f"ground: sensor height {h:.3f} m, tilt {np.degrees(np.arccos(n[2])):.2f} deg")

    # 3. radar: reflectors + ICP, two rounds
    R, t = rot_rpy(ex.radar_rpy_deg), np.array(ex.radar_t, float)
    fit0 = [rl._residuals((R, t), w, 0.4)[3].mean() for w in windows]
    refs = []
    for rnd in range(3):
        refs = [x for w in windows for x in rf.detect(w, ex, n, h)]
        if rnd:                                   # drop matches that disagree with the current solution
            for x in refs:
                if x.q_lidar is not None and np.linalg.norm((R @ x.p_radar + t - x.q_lidar)[:2]) > sv.get("max_reflector_residual_m", 0.3):
                    x.q_lidar, x.note = None, "rejected: radar-LiDAR mismatch > max_reflector_residual_m"
        anc = rf.anchors(refs)
        use = anc if len(anc) >= sv["min_reflectors"] else None
        R, t, st = rl.joint_icp(windows, R, t, anchors=use, anchor_weight=sv["reflector_weight"])
        ex.radar_rpy_deg, ex.radar_t = rl.rpy_deg(R).tolist(), t.tolist()
        log(f"radar round {rnd + 1}: {len(refs)} reflector candidates, {len(anc)} matched -> rpy "
            f"{np.round(ex.radar_rpy_deg, 2)} t {np.round(t, 3)}")
    R_icp, t_icp, _ = rl.joint_icp(windows, R, t)                       # ICP alone, for the agreement check
    ref_rep = []
    for x in refs:
        d = {"radar_xy": (R @ x.p_radar + t)[:2].round(3).tolist(), "rcs": round(x.rcs, 1),
             "persistence": round(x.persistence, 2), "note": x.note}
        if x.q_lidar is not None:
            d["lidar_xy"] = x.q_lidar[:2].round(3).tolist()
            d["residual_xy_cm"] = round(float(np.linalg.norm((R @ x.p_radar + t - x.q_lidar)[:2])) * 100, 1)
        ref_rep.append(d)
    n_anc = len(rf.anchors(refs))
    if n_anc < sv["min_reflectors"]:
        warn.append(f"only {n_anc} reflector(s) matched (< {sv['min_reflectors']}): radar from ICP only")
    radar_rep = {"windows": st, "fitness_initial": float(np.mean(fit0)), "reflectors": ref_rep, "n_anchors": n_anc,
                 "icp_only_minus_final_rpy_deg": (rl.rpy_deg(R_icp) - rl.rpy_deg(R)).round(3).tolist(),
                 "icp_only_minus_final_t_m": (t_icp - t).round(3).tolist()}
    res = [d["residual_xy_cm"] for d in ref_rep if "residual_xy_cm" in d]
    log(f"radar: ICP fitness {np.mean(fit0):.2f} -> {np.mean([s['fitness'] for s in st]):.2f}"
        + (f", reflector residual xy {np.mean(res):.1f} cm (max {np.max(res):.1f})" if res else ""))

    # 4. camera
    times = [x for a, b in ws for x in np.arange(a, b, 1.0 / sv["camera_fps"])]
    if len(times) > sv["max_camera_frames"]:
        times = list(np.array(times)[np.linspace(0, len(times) - 1, sv["max_camera_frames"]).astype(int)])
    frames, K, D, size, _ = cl.load_frames(bag, times, topics=topics)
    frames = cl.prepare(frames, n, h, seg_weights)
    n_masks = sum(len(f.masks) for f in frames)
    cam_rep = {"n_frames": len(frames), "n_masks": n_masks}
    if n_masks == 0:
        warn.append("no person found by the detector: camera rotation NOT estimated (kept the initial value)")
    else:
        c1 = cl.refine_rotation(ex, frames, K, D, span=sv["camera_search_deg"], step=0.5)
        c2 = cl.refine_rotation(ex, frames, K, D, span=1.0, step=0.125)
        cam_rep.update({"score_initial": c1["score_before"], "score_final": c2["score_after"],
                        "sharpness_1deg": c2["sharpness_1deg"]})
        log(f"camera: rpy {np.round(ex.cam_rpy_deg, 2)}  score {c1['score_before']:.3f} -> {c2['score_after']:.3f}  "
            f"({len(frames)} frames, {n_masks} person masks)")
        if n_masks < sv["min_person_masks"]:
            warn.append(f"only {n_masks} person masks (< {sv['min_person_masks']}): camera result unreliable")
        if max(abs(a - b) for a, b in zip(ex.cam_rpy_deg[1:], ex0.cam_rpy_deg[1:])) > sv["camera_search_deg"] - 0.6:
            warn.append("camera optimum at the edge of the search range: initial rotation too far off, "
                        "increase solve.camera_search_deg")
        if c2["sharpness_1deg"]["yaw"] < 0.05:
            warn.append("camera score has no clear yaw peak (few / far people?)")

    # 5. save
    report = {"bag": str(bag), "notes": notes, "warnings": warn, "ground": {"height": h, "normal": n.tolist()},
              "initial": vars(ex0), "camera": cam_rep, "radar": radar_rep}
    d = ex.to_calib_json(K, D, size, name=Path(bag).name, distortion_model="rational_polynomial")
    d["report"] = report
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(d, indent=1, default=float))
    log(f"saved {out_json}")
    for w_ in warn:
        log("WARNING: " + w_)
    if overlay_dir:
        _overlays(Path(overlay_dir), frames, windows, refs, ex, ex0, K, D, R, t)
        log(f"overlays in {overlay_dir}")
    return d


def reground(bag: str, base_json: str, out_json: str, *, step: float = 2.0, topics: Topics = DEFAULT_TOPICS,
             log=print) -> dict:
    """The calibrated rig moved to another mount (tripod -> excavator): keep the camera / radar -> os_sensor
    extrinsics of `base_json` and re-estimate only the ground (sensor height + normal) from the LiDAR of `bag`,
    median over one scan every `step` s. The spread over the bag (swing over uneven ground) goes to the report."""
    base = json.loads(Path(base_json).read_text())
    ex = RigExtrinsics.from_calib_json(base)
    prior, rows = np.asarray(ex.ground_normal, float), []
    with AnyReader([Path(bag)]) as r:
        cs = {c.topic: c for c in r.connections}
        tf = {}
        for c, _, raw in r.messages(connections=[cs[x] for x in ("/tf_static",) if x in cs]):
            for t in r.deserialize(raw, c.msgtype).transforms:
                q, tr = t.transform.rotation, t.transform.translation
                tf[(t.header.frame_id, t.child_frame_id)] = (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                                                             np.array([tr.x, tr.y, tr.z]))
        nxt = 0.0
        for c, t_ns, raw in r.messages(connections=[cs[topics.lidar]]):
            s = (t_ns - r.start_time) * 1e-9
            if s < nxt:
                continue
            nxt = s + step
            m = r.deserialize(raw, c.msgtype)
            a = pc2_to_struct(m)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
            xyz = xyz[np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1.5)]
            Rm, tm = lidar_to_sensor(tf, m.header.frame_id)
            P = xyz @ Rm.T + tm
            if not rows:                                   # first scan: twice, the new mount may be far from the base
                prior = ground_plane_near(P, ground_plane_near(P, prior)[0])[0]
            n, h, _ = ground_plane_near(P, prior)
            rows.append((s, *n, h))
    A = np.array(rows)
    n = np.median(A[:, 1:4], axis=0)
    n /= np.linalg.norm(n)
    h = float(np.median(A[:, 4]))
    roll, pitch = np.degrees(np.arctan2(A[:, 2], A[:, 3])), np.degrees(np.arctan2(-A[:, 1], A[:, 3]))
    p10_90 = lambda v, nd: [round(float(x), nd) for x in np.percentile(v, [10, 90])]  # noqa: E731
    ground = {"height": h, "normal": n.tolist(), "tilt_deg": float(np.degrees(np.arccos(n[2]))),
              "roll_deg": float(np.degrees(np.arctan2(n[1], n[2]))), "pitch_deg": float(np.degrees(np.arctan2(-n[0], n[2]))),
              "height_p10_p90": p10_90(A[:, 4], 3), "roll_p10_p90": p10_90(roll, 2), "pitch_p10_p90": p10_90(pitch, 2),
              "n_scans": len(A), "step_s": step}
    warn = []
    if np.ptp(ground["pitch_p10_p90"]) > 1.0 or np.ptp(ground["roll_p10_p90"]) > 1.0 or np.ptp(ground["height_p10_p90"]) > 0.05:
        warn.append("rig pose to the ground changes during the bag (p10-p90 above): one ground per bag is approximate")
    log(f"ground: sensor height {h:.3f} m, roll {ground['roll_deg']:+.2f} deg, pitch {ground['pitch_deg']:+.2f} deg "
        f"({len(A)} scans; p10-p90 height {ground['height_p10_p90']}, pitch {ground['pitch_p10_p90']})")
    ex.ground_normal, ex.sensor_height = n.tolist(), h
    size = (base["image_size"]["width"], base["image_size"]["height"])
    d = ex.to_calib_json(base["K"], base["D"], size, name=Path(bag).name, distortion_model=base.get("distortion_model", ""))
    for k in ("camera_time_offset_s", "radar_time_offset_s"):
        if k in base:
            d[k] = base[k]
    d["report"] = {"bag": str(bag), "notes": [f"reground of {Path(base_json).name}: extrinsics kept, ground from this bag"],
                   "warnings": warn, "ground": ground, "base_report": base.get("report")}
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(d, indent=1, default=float))
    log(f"saved {out_json}")
    for w_ in warn:
        log("WARNING: " + w_)
    return d


def _overlays(out: Path, frames, windows, refs, ex, ex0, K, D, R, t):
    """camera_initial_vs_final.jpg (up to 4 frames: initial | final) and radar_bev.jpg."""
    out.mkdir(parents=True, exist_ok=True)
    radar_s = windows[0].p[-600:] @ R.T + t
    radar_s0 = ex0.radar_to_sensor(windows[0].p[-600:])
    if frames:
        pick = np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)
        rows = [np.hstack([cv2.resize(draw_camera(frames[i], ex0, K, D, radar_s0), (640, 360)),
                           cv2.resize(draw_camera(frames[i], ex, K, D, radar_s), (640, 360))]) for i in pick]
        cv2.imwrite(str(out / "camera_initial_vs_final.jpg"), np.vstack(rows))
    cv2.imwrite(str(out / "radar_bev.jpg"), draw_bev(windows[0], refs, ex, R, t))


def draw_camera(frame, ex, K, D, radar_pts=None, lmax=30.0):
    img = cl.enhance_dark(frame.image)
    for _, core, _ in frame.masks:
        cnt, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)

    def proj(pc):
        k = pc[:, 2] > 0.5
        uv = np.full((len(pc), 2), -1.0)
        if k.any():
            uv[k] = cv2.projectPoints(pc[k], np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        return uv, k

    H, W = img.shape[:2]
    for pts, rad, fill in ((frame.xyz, 2, True), (radar_pts, 5, False)):
        if pts is None or not len(pts):
            continue
        uv, k = proj(ex.sensor_to_cam(pts))
        col = cv2.applyColorMap(np.uint8(np.clip(np.linalg.norm(pts, axis=1) / lmax * 255, 0, 255))[:, None],
                                cv2.COLORMAP_TURBO)[:, 0]
        for (u, v), c, kk in zip(uv, col, k):
            if kk and 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), rad, tuple(int(x) for x in c), -1 if fill else 2)
    return img


def draw_bev(window, refs, ex, R, t, rng_m=40.0, px=800):
    img = np.full((px, px, 3), 25, np.uint8)
    s = px / rng_m
    to = lambda p: np.c_[px / 2 - p[:, 1] * s, px - p[:, 0] * s].astype(int)
    for r_ in range(5, int(rng_m) + 1, 5):
        cv2.circle(img, (px // 2, px), int(r_ * s), (60, 60, 60), 1)
    n = np.asarray(ex.ground_normal)
    L = window.lidar_raw[(window.lidar_raw @ n + ex.sensor_height) > 0.25]
    q = to(L)
    ok = (q[:, 0] >= 0) & (q[:, 0] < px) & (q[:, 1] >= 0) & (q[:, 1] < px)
    img[q[ok, 1], q[ok, 0]] = (150, 150, 150)
    q = to(window.p @ R.T + t)
    for u, v in q:
        if 0 <= u < px and 0 <= v < px:
            img[v, u] = (60, 60, 255)
    for x in refs:
        c = to((R @ x.p_radar + t)[None])[0]
        cv2.circle(img, tuple(c), 10, (0, 255, 255) if x.q_lidar is not None else (0, 140, 255), 2)
        if x.q_lidar is not None:
            cv2.drawMarker(img, tuple(to(x.q_lidar[None])[0]), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
    cv2.putText(img, "grey LiDAR (above ground), red radar (static), yellow ring = reflector (radar),"
                " green + = LiDAR match, orange = unmatched", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return img
