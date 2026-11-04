#!/usr/bin/env python3
"""Extrinsic calibration of the HDX rig from stationary data (camera <- os_sensor, radar -> os_sensor).

  python3 scripts/calib_tune.py --bags <bag> [<bag> ...] --calib configs/calib/<name>.json --auto [--gui]

Pipeline (docs/06_hdx_dataset_calibration.html):
  1. stationary windows per bag from radar Doppler ego speed (halt bags: whole bag in 60 s pieces)
  2. ground plane from the LiDAR (os_sensor height / tilt)
  3. camera rotation: person-mask purity x contrast of the LiDAR points (rcmot.calib.camera_lidar), dark
     frames brightened for the detector only; translation fixed to the measured tf prior (weakly observable)
  4. radar: joint robust point-to-plane ICP of Doppler-static radar points to the LiDAR over all windows
     (rcmot.calib.radar_lidar), leave-one-window-out jackknife
  5. report (scores, 1-deg sharpness, jackknife), json, overlay image; --gui for manual checks/adjustment

The camera-LiDAR rotation agreed day vs night within 0.1 deg; the radar did not (about 1 deg), so calibrate
the radar per recording session (pass only that session's bags).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.calib import camera_lidar as cl  # noqa: E402
from rcmot.calib import radar_lidar as rl  # noqa: E402
from rcmot.calib.common import ground_plane, stationary_windows  # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics, rot_rpy  # noqa: E402

# measured translations from the HDX detection bags' /tf_static (rotations there are all 0)
TF_CAM_T = [0.08, 0.0, -0.05]      # os_sensor -> oak-d-base-frame
TF_RADAR_T = [0.14, 0.0, -0.17]    # os_sensor -> bosch_4d_frame; NOT applied by the old pipeline (radar points
                                   # are published as os_sensor) and ICP finds z ~ 0, so kept as reference only
SEG = str(ROOT / "models/yolo11s-seg.pt")


def collect(bags, fps, max_frames_per_bag, topics=None):
    from rcmot.calib.common import DEFAULT_TOPICS
    topics = topics or DEFAULT_TOPICS
    frames, windows, gpts, K = [], [], [], None
    for bag in bags:
        ws = stationary_windows(bag, topics=topics)
        print(f"{Path(bag).name}: stationary windows {[(round(a, 1), round(b, 1)) for a, b in ws]}")
        if not ws:
            continue
        times = [t for a, b in ws for t in np.arange(a, b, 1.0 / fps)]
        if len(times) > max_frames_per_bag:
            times = list(np.array(times)[np.linspace(0, len(times) - 1, max_frames_per_bag).astype(int)])
        fr, K, D, size, gp = cl.load_frames(bag, times)
        with AnyReader([Path(bag)]) as r:
            t_start = r.start_time * 1e-9
        for f in fr:                                              # radar window shown with this frame in the GUI
            rel = f.stamp - t_start
            f.window = len(windows) + int(np.argmin([0 if a <= rel <= b else min(abs(rel - a), abs(rel - b)) for a, b in ws]))
        frames += fr
        gpts.append(gp)
        windows += [rl.load_window(bag, a, b, topics=topics) for a, b in ws]
    return frames, windows, np.concatenate(gpts), K, D, size


def overlay(frame: cl.CalFrame, ex: RigExtrinsics, K, D, radar_xyz=None, show_lidar=True, show_masks=True):
    img = cl.enhance_dark(frame.image)
    if show_masks:
        for _, core, _ in frame.masks:
            cnt, _ = cv2.findContours(core.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnt, -1, (255, 255, 0), 1)

    def draw(pc, rng, radius, fixed=None):
        k = pc[:, 2] > 0.5
        if not k.any():
            return
        uv = cv2.projectPoints(pc[k], np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        col = cv2.applyColorMap(np.uint8(np.clip(rng[k] / 30 * 255, 0, 255))[:, None], cv2.COLORMAP_TURBO)[:, 0]
        for (u, v), c in zip(uv, col):
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), radius, fixed or tuple(int(x) for x in c), -1 if fixed is None else 2)

    if show_lidar:
        draw(ex.sensor_to_cam(frame.xyz), frame.rng, 2)
    if radar_xyz is not None and len(radar_xyz):
        p = ex.radar_to_sensor(radar_xyz)
        draw(ex.sensor_to_cam(p), np.linalg.norm(p, axis=1), 5, (255, 255, 255))
    return img


def run_gui(frames, windows, ex, K, D, save):
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
    from PyQt5 import QtCore, QtGui, QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QWidget()
    win.setWindowTitle("RC-MOT extrinsic tuner")
    lay = QtWidgets.QHBoxLayout(win)
    view = QtWidgets.QLabel()
    view.setMinimumSize(1100, 620)
    lay.addWidget(view, 1)
    panel = QtWidgets.QFormLayout()
    side = QtWidgets.QWidget()
    side.setLayout(panel)
    side.setFixedWidth(330)
    lay.addWidget(side)
    spins = {}
    for grp, keys in (("camera <- os_sensor", ("cam_rpy_deg", "cam_t")), ("radar -> os_sensor", ("radar_rpy_deg", "radar_t"))):
        panel.addRow(QtWidgets.QLabel(f"<b>{grp}</b>"))
        for key in keys:
            for i, ax in enumerate(("roll", "pitch", "yaw") if "rpy" in key else ("x", "y", "z")):
                sb = QtWidgets.QDoubleSpinBox()
                deg = "rpy" in key
                sb.setRange(-30 if deg else -3, 30 if deg else 3)
                sb.setDecimals(2 if deg else 3)
                sb.setSingleStep(0.05 if deg else 0.005)
                sb.setSuffix(" deg" if deg else " m")
                sb.setValue(getattr(ex, key)[i])
                spins[(key, i)] = sb
                panel.addRow(f"{key.split('_')[0]} {ax}", sb)
    frame = QtWidgets.QSpinBox()
    frame.setRange(0, max(len(frames) - 1, 0))
    panel.addRow("frame", frame)
    chk = {k: QtWidgets.QCheckBox(k) for k in ("lidar", "radar", "masks")}
    for c in chk.values():
        c.setChecked(True)
        panel.addRow(c)
    score = QtWidgets.QLabel()
    score.setWordWrap(True)
    panel.addRow(score)
    btn = QtWidgets.QPushButton("Save")
    panel.addRow(btn)
    panel.addRow(QtWidgets.QLabel("dots = LiDAR (colour = range, 0-30 m)<br>white rings = radar (static, window of "
                                  "the frame)<br>cyan = person masks used by the score"))

    def pull():
        for (key, i), sb in spins.items():
            getattr(ex, key)[i] = sb.value()

    def redraw(*_):
        pull()
        k = frame.value()
        rad = windows[frames[k].window].p[-2000:] if (windows and chk["radar"].isChecked()) else None
        img = overlay(frames[k], ex, K, D, rad, chk["lidar"].isChecked(), chk["masks"].isChecked())
        rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        q = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
        view.setPixmap(QtGui.QPixmap.fromImage(q.copy()).scaled(view.width(), view.height(), QtCore.Qt.KeepAspectRatio))
        score.setText(f"person-mask score (all frames): {cl.purity_score(ex, frames, K, D):.3f}")

    for sb in spins.values():
        sb.valueChanged.connect(redraw)
    frame.valueChanged.connect(redraw)
    for c in chk.values():
        c.stateChanged.connect(redraw)
    btn.clicked.connect(lambda: (pull(), score.setText("saved " + save(ex))))
    win.show()
    redraw()
    app.exec_()


def _topics(a):
    from dataclasses import replace
    from rcmot.calib.common import DEFAULT_TOPICS
    return replace(DEFAULT_TOPICS, radar=a.radar_topic) if a.radar_topic else DEFAULT_TOPICS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True, help="bags of ONE recording session")
    ap.add_argument("--calib", required=True, help="json to read (if it exists) and write")
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--fps", type=float, default=2.0, help="camera frames per second sampled in stationary windows")
    ap.add_argument("--max-frames-per-bag", type=int, default=60)
    ap.add_argument("--cam-6dof", action="store_true", help="also refine the camera translation (weakly observable)")
    ap.add_argument("--overlay", help="write an overlay image of a few frames")
    ap.add_argument("--radar-topic", help="radar PointCloud2 topic (default: Bosch off-highway driver)")
    a = ap.parse_args()

    frames, windows, gpts, K, D, size = collect(a.bags, a.fps, a.max_frames_per_bag, topics=_topics(a))
    n, h, _ = ground_plane(gpts)
    frames = cl.prepare(frames, n, h, SEG)
    print(f"ground: height {h:.3f} m, tilt {np.degrees(np.arccos(n[2])):.2f} deg | {len(frames)} frames with person masks "
          f"({sum(len(f.masks) for f in frames)} masks) | {len(windows)} radar windows")

    p = Path(a.calib)
    if p.exists() and "extrinsics" in json.loads(p.read_text()):
        ex = RigExtrinsics.from_calib_json(json.loads(p.read_text()))
        print("loaded", p)
    else:
        ex = RigExtrinsics(cam_rpy_deg=[0.0, 0.0, 0.0], cam_t=list(TF_CAM_T))
        print("initial: measured tf (rotation 0)")
    ex.ground_normal, ex.sensor_height = n.tolist(), h
    report = {"bags": [Path(b).name for b in a.bags], "ground": {"height": h, "normal": n.tolist()}}

    def save(e: RigExtrinsics) -> str:
        d = e.to_calib_json(K, D, size, name="+".join(Path(b).name for b in a.bags), distortion_model="rational_polynomial")
        d["report"] = report
        p.write_text(json.dumps(d, indent=1, default=float))
        return str(p)

    if a.auto:
        if frames:
            res = cl.refine_rotation(ex, frames, K, D)
            report["camera"] = {k: res[k] for k in ("score_before", "score_after", "sharpness_1deg", "n_frames", "n_masks")}
            print("camera rotation:", np.round(ex.cam_rpy_deg, 3), report["camera"])
            if a.cam_6dof:
                from scipy.optimize import minimize
                x0 = np.r_[ex.cam_rpy_deg, ex.cam_t]
                f6 = lambda x: -cl.purity_score(RigExtrinsics(cam_rpy_deg=list(x[:3]), cam_t=list(x[3:])), frames, K, D)
                r6 = minimize(f6, x0, method="Nelder-Mead", options={"initial_simplex": np.vstack(
                    [x0, x0 + np.diag([.3, .3, .3, .05, .05, .05])]), "xatol": 0.002, "fatol": 1e-5, "maxiter": 800})
                ex.cam_rpy_deg, ex.cam_t = [float(v) for v in r6.x[:3]], [float(v) for v in r6.x[3:]]
                report["camera"]["score_6dof"] = -r6.fun
                print("camera 6dof:", np.round(r6.x, 3), f"score {-r6.fun:.3f}")
        if windows:
            R, t, st = rl.joint_icp(windows, rot_rpy(ex.radar_rpy_deg), np.array(ex.radar_t, float))
            sr, stt = rl.jackknife(windows, R, t)
            ex.radar_rpy_deg, ex.radar_t = rl.rpy_deg(R).tolist(), t.tolist()
            report["radar"] = {"rpy_jackknife_std_deg": sr.tolist(), "t_jackknife_std_m": stt.tolist(), "windows": st}
            print("radar:", np.round(ex.radar_rpy_deg, 3), "+-", np.round(sr, 2), "t", np.round(t, 3), "+-", np.round(stt, 3),
                  f"fitness {np.mean([s['fitness'] for s in st]):.2f} rmse {np.nanmean([s['rmse'] for s in st]):.3f}")
        print("saved", save(ex))
    if a.overlay and frames:
        pick = np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int)
        tiles = [cv2.resize(overlay(frames[i], ex, K, D, windows[frames[i].window].p[-2000:] if windows else None),
                            (640, 360)) for i in pick]
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(a.overlay, np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))
        print("overlay", a.overlay)
    if a.gui:
        run_gui(frames, windows, ex, K, D, save)


if __name__ == "__main__":
    main()
