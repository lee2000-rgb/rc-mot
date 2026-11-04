"""Target-less camera <- LiDAR extrinsic refinement with instance masks ("mask purity").

For each image, instance masks (YOLO-seg, COCO person / car / truck / bus) give object regions. With the
right extrinsic the non-ground LiDAR points projected INSIDE a mask belong to one range cluster (the
object: purity) and the points in a thin BAND around the mask are farther away (background: contrast).
The per-mask score is purity x contrast, so a mask that only covers background (pure but no contrast) or
straddles the object boundary (impure) scores low; masks are weighted by sqrt(#points) and summed over
frames. (A first version scored "fg inside / bg outside" over an enlarged ROI; it rewarded projecting the
object out of the ROI and had its minimum at the true extrinsic.) Unlike image-edge alignment it ignores texture (vegetation, gravel), which made the
edge score flat on the 32-beam rig, and it works at night as long as the detector finds the objects.

Only the rotation is searched by default (camera-LiDAR lever arm is a few cm and is kept at its nominal
value): coarse yaw/pitch grid, then Nelder-Mead on roll/pitch/yaw.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from rcmot.calib.common import DEFAULT_TOPICS, Topics, lidar_to_sensor
from rcmot.sensors.bag_source import image_to_bgr, pc2_to_struct
from rcmot.utils.rig_extrinsics import RigExtrinsics

FG_TOL = {"person": 0.8}           # range extent of the foreground cluster [m]; default for vehicles below
FG_TOL_DEFAULT = 3.0


@dataclass
class CalFrame:
    stamp: float
    image: np.ndarray
    xyz: np.ndarray                 # (N,3) os_sensor, non-ground, range-limited
    rng: np.ndarray                 # (N,)
    masks: list = field(default_factory=list)   # [(cls, core mask bool HxW, band mask bool HxW)]
    window: int = -1                             # index of the stationary radar window it belongs to


def _tf_static(r, cs) -> dict:
    tf = {}
    for c, _, raw in r.messages(connections=[cs["/tf_static"]]):
        for t in r.deserialize(raw, c.msgtype).transforms:
            tr, q = t.transform.translation, t.transform.rotation
            tf[(t.header.frame_id, t.child_frame_id)] = (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                                                         np.array([tr.x, tr.y, tr.z]))
    return tf


def _stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def lidar_clock_offset(r, conn, n: int = 200, sweep: float = 0.1) -> float:
    """Ouster header stamps are sensor time (since boot), not ROS time. offset = min(bag time - header)
    minus one sweep (the scan is published right after it ends), so ROS time = header + offset."""
    d = []
    for _, tn, raw in r.messages(connections=[conn]):
        d.append(tn * 1e-9 - _stamp(r.deserialize(raw, conn.msgtype)))
        if len(d) >= n:
            break
    return float(np.min(d)) - sweep


def header_clock_shift(r, radar_conn, n: int = 50) -> float:
    """bag time - header time of the radar (its header is the receive time): ~0 for a live recording, large
    for a re-recorded replay (ros2 bag play keeps the old headers). Subtract it from bag-time-derived times."""
    d = []
    for _, tn, raw in r.messages(connections=[radar_conn]):
        d.append(tn * 1e-9 - _stamp(r.deserialize(raw, radar_conn.msgtype)))
        if len(d) >= n:
            break
    return float(np.median(d)) if d else 0.0


def load_frames(bag: str, times: list[float], *, max_dt: float = 0.03, scan_mid: float = 0.05,
                topics: Topics = DEFAULT_TOPICS):
    """Image + time-nearest LiDAR scan (mid-sweep, converted to ROS time) at each requested time; returns
    (frames, K, D, image_size, os_sensor points of the first scans for the ground plane)."""
    frames, ground_pts = [], []
    with AnyReader([Path(bag)]) as r:
        cs = {c.topic: c for c in r.connections}
        tf = _tf_static(r, cs)
        T_IMG, T_LIDAR, T_INFO = topics.image, topics.lidar, topics.camera_info
        info = r.deserialize(next(r.messages(connections=[cs[T_INFO]]))[2], cs[T_INFO].msgtype)
        off = lidar_clock_offset(r, cs[T_LIDAR]) - (header_clock_shift(r, cs[topics.radar]) if topics.radar in cs else 0.0)
        for s in times:
            imgs, scans = [], []
            for c, tn, raw in r.messages(connections=[cs[T_IMG], cs[T_LIDAR]], start=r.start_time + int(s * 1e9)):
                m = r.deserialize(raw, c.msgtype)
                (imgs if c.topic == T_IMG else scans).append(m)
                if len(scans) >= 4 and len(imgs) >= 6:
                    break
            best = None
            for im in imgs:
                for sc in scans:
                    dt = abs(_stamp(im) - (_stamp(sc) + off + scan_mid))
                    if best is None or dt < best[0]:
                        best = (dt, im, sc)
            if best is None or best[0] > max_dt:
                continue
            _, im, sc = best
            a = pc2_to_struct(sc)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
            xyz = xyz[np.linalg.norm(xyz, axis=1) > 1.0]
            if sc.header.frame_id != "os_sensor":
                R, t = lidar_to_sensor(tf, sc.header.frame_id)
                xyz = xyz @ R.T + t
            if len(ground_pts) < 5:
                ground_pts.append(xyz)
            frames.append(CalFrame(_stamp(im), image_to_bgr(im), xyz, np.linalg.norm(xyz, axis=1)))
    return frames, np.array(info.k, float).reshape(3, 3), np.array(info.d, float), (info.width, info.height), \
        np.concatenate(ground_pts)


def enhance_dark(img: np.ndarray, target: float = 90.0) -> np.ndarray:
    """Gamma + CLAHE on the luminance of dark (night) images, for the detector only."""
    m = img.mean()
    if m >= 60:
        return img.copy()
    g = np.log(target / 255) / np.log(max(m, 1) / 255)
    out = np.uint8(255 * (img / 255.0) ** g)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def prepare(frames: list[CalFrame], normal: np.ndarray, height: float, seg_weights: str, *,
            classes=("person",), conf: float = 0.4, min_area: int = 1500,
            max_range: float = 40.0, ground_clear: float = 0.25, device=None) -> list[CalFrame]:
    """Drop ground / far points and attach instance masks; frames without masks are removed."""
    from ultralytics import YOLO

    seg = YOLO(seg_weights)
    keep_cls = [i for i, n in seg.names.items() if n in classes]
    out = []
    for f in frames:
        hag = f.xyz @ normal + height
        k = (hag > ground_clear) & (f.rng < max_range)
        f.xyz, f.rng = f.xyz[k], f.rng[k]
        r = seg.predict(enhance_dark(f.image), conf=conf, classes=keep_cls, retina_masks=True, verbose=False,
                        device=device)[0]
        if r.masks is None:
            continue
        H, W = f.image.shape[:2]
        for m, c, b in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                           r.boxes.xyxy.cpu().numpy()):
            if m.sum() < min_area:
                continue
            w, h = b[2] - b[0], b[3] - b[1]
            k = int(np.clip(0.08 * max(w, h), 6, 25))                  # band width ~8 % of the object size
            band = cv2.dilate(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool) & ~m
            core = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            f.masks.append((seg.names[c], core, band))
        if f.masks:
            out.append(f)
    return out


def _fg_cluster(r: np.ndarray, tol: float, gap: float = 1.0, min_pts: int = 3) -> np.ndarray:
    """Mask of the nearest range cluster with at least `min_pts` points (1D gap clustering)."""
    order = np.argsort(r)
    rs = r[order]
    splits = np.flatnonzero(np.diff(rs) > gap) + 1
    for g in np.split(np.arange(len(rs)), splits):
        if len(g) >= min_pts:
            lo = rs[g[0]]
            return (r >= lo) & (r <= lo + tol)
    return np.zeros(len(r), bool)


def purity_score(ex: RigExtrinsics, frames: list[CalFrame], K, D, *, per_frame: bool = False, min_pts: int = 4):
    """sum_w (purity_in * contrast_band) / sum_w over masks, w = sqrt(min(n_in, n_band))."""
    num = den = 0.0
    scores = []
    for f in frames:
        pc = ex.sensor_to_cam(f.xyz)
        front = pc[:, 2] > 0.5
        uv = np.full((len(pc), 2), -1, int)
        if front.any():
            uv[front] = cv2.projectPoints(pc[front], np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2).round().astype(int)
        H, W = f.image.shape[:2]
        ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        idx = np.flatnonzero(ok)
        fn = fd = 0.0
        for cls, core, band in f.masks:
            i_in = idx[core[uv[idx, 1], uv[idx, 0]]]
            i_bd = idx[band[uv[idx, 1], uv[idx, 0]]]
            w = np.sqrt(max(min(len(i_in), len(i_bd)), 0))
            if len(i_in) < min_pts or len(i_bd) < min_pts:
                fd += 1.0                                            # a mask without support counts as a miss
                continue
            tol = FG_TOL.get(cls, FG_TOL_DEFAULT)
            fg = _fg_cluster(f.rng[i_in], tol)
            purity = fg.mean()
            r_obj = f.rng[i_in][fg].max() if fg.any() else np.inf
            contrast = (f.rng[i_bd] > r_obj + 0.5).mean()
            fn += w * purity * contrast
            fd += w
        num += fn
        den += fd
        scores.append(fn / fd if fd else np.nan)
    s = num / max(den, 1e-9)
    return (s, scores) if per_frame else s


def refine_rotation(ex: RigExtrinsics, frames, K, D, *, span: float = 4.0, step: float = 0.25) -> dict:
    """Coarse yaw/pitch grid around the current value, then Nelder-Mead on (roll, pitch, yaw)."""
    r0, p0, y0 = ex.cam_rpy_deg

    def score(rpy):
        e = RigExtrinsics(cam_rpy_deg=list(rpy), cam_t=list(ex.cam_t))
        return purity_score(e, frames, K, D)

    before = score([r0, p0, y0])
    grid = np.arange(-span, span + 1e-9, step)
    G = np.array([[score([r0, p0 + dp, y0 + dy]) for dy in grid] for dp in grid])
    ip, iy = np.unravel_index(np.argmax(G), G.shape)
    x0 = np.array([r0, p0 + grid[ip], y0 + grid[iy]])
    res = minimize(lambda x: -score(x), x0, method="Nelder-Mead",
                   options={"initial_simplex": np.vstack([x0, x0 + np.diag([0.3, 0.3, 0.3])]),
                            "xatol": 0.01, "fatol": 1e-5, "maxiter": 300})
    best = res.x if -res.fun >= G.max() else x0
    ex.cam_rpy_deg = [float(v) for v in best]
    # peak sharpness: score drop 1 deg away from the optimum (yaw / pitch)
    s_best = score(best)
    sharp = {ax: s_best - np.mean([score(best + d * np.eye(3)[i]) for d in (-1, 1)])
             for i, ax in enumerate(("roll", "pitch", "yaw"))}
    return {"score_before": before, "score_after": s_best, "grid": G, "grid_axis": grid, "sharpness_1deg": sharp,
            "n_frames": len(frames), "n_masks": int(sum(len(f.masks) for f in frames))}
