#!/usr/bin/env python3
"""Frame-by-frame calibration viewer: camera image + synced LiDAR scan + synced radar frame(s), and a BEV.

  python3 scripts/calib_viewer.py [--bag data/bags/hdx_boryeong_pg/<bag>] [--calib configs/calib/<json>]

Keys: Right/D next frame, Left/A previous, Up/Down +-10 frames, Space play/pause, S save.
Sync per frame (header times): camera header = capture time (ROS); LiDAR header = Ouster boot time, converted
with the bag-time offset (scan mid = header + offset + 0.05 s); radar header = receive time (ROS).
Radar colour: range on the LiDAR colour scale (default) or ego-compensated radial velocity (white = static,
red = moving away, blue = approaching).
The calib json of the bag's session is picked automatically (20251125 -> day1125, 20251126 -> night1126).
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
from rcmot.calib.camera_lidar import enhance_dark, header_clock_shift, lidar_clock_offset  # noqa: E402
from rcmot.calib.common import lidar_to_sensor  # noqa: E402
from rcmot.radar.ego_velocity import compensate_doppler, estimate_ego_velocity  # noqa: E402
from rcmot.sensors.bag_source import image_to_bgr, pc2_to_struct  # noqa: E402
from rcmot.utils.rig_extrinsics import RigExtrinsics  # noqa: E402

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)        # opencv-python ships broken Qt plugins
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

T_IMG, T_LIDAR, T_INFO = "/oak/rgb/image_raw", "/ouster/points", "/oak/rgb/camera_info"
T_RADAR = "/off_highway_premium_radar_sample_driver/locations"
BAG_DIR = ROOT / "data/bags/hdx_boryeong_pg"
SESSION_CALIB = {"20251125": "hdx_boryeong_pg_day1125.json", "20251126": "hdx_boryeong_pg_night1126.json"}


def _stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


class BagFrames:
    """Random-ish access to camera frames of one bag plus the LiDAR / radar data synced to each."""

    def __init__(self, path: str):
        self.path = path
        self.r = AnyReader([Path(path)])
        self.r.open()
        self.cs = {c.topic: c for c in self.r.connections}
        self.t0, self.t1 = self.r.start_time, self.r.end_time
        for t in (T_IMG, T_RADAR, T_INFO):
            if t not in self.cs:
                raise KeyError(f"topic {t} not in the bag; available: {sorted(self.cs)} (use --config with the rig's topics)")
        self.has_lidar = T_LIDAR in self.cs                       # no LiDAR (radar + camera rig): image + radar only
        self.lidar_off = (lidar_clock_offset(self.r, self.cs[T_LIDAR]) if self.has_lidar else 0.0) \
            - header_clock_shift(self.r, self.cs[T_RADAR])
        info = next(self.r.messages(connections=[self.cs[T_INFO]]))
        ci = self.r.deserialize(info[2], self.cs[T_INFO].msgtype)
        self.K, self.D, self.size = np.array(ci.k, float).reshape(3, 3), np.array(ci.d, float), (ci.width, ci.height)
        tf = {}
        for c, _, raw in self.r.messages(connections=[self.cs[k] for k in ("/tf_static",) if k in self.cs]):
            for t in self.r.deserialize(raw, c.msgtype).transforms:
                q, tr = t.transform.rotation, t.transform.translation
                from scipy.spatial.transform import Rotation
                tf[(t.header.frame_id, t.child_frame_id)] = (Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(),
                                                             np.array([tr.x, tr.y, tr.z]))
        self.tf = tf

    def close(self):
        self.r.close()

    def image_after(self, t_bag_ns: int, inclusive: bool = False):
        """First camera message with bag time > t (>= t when inclusive) -> (bag_ns, header_s, bgr) or None."""
        start = t_bag_ns if inclusive else t_bag_ns + 1
        for c, tn, raw in self.r.messages(connections=[self.cs[T_IMG]], start=start):
            m = self.r.deserialize(raw, c.msgtype)
            return tn, _stamp(m), image_to_bgr(m)
        return None

    def synced(self, t_img: float, t_bag_ns: int, n_radar: int):
        """LiDAR scan whose mid-sweep is closest to the image time, and the n radar frames closest to it."""
        lo, hi = t_bag_ns - int(1.0e9), t_bag_ns + int(0.6e9)
        scans, radars = [], []
        conns = [self.cs[T_RADAR]] + ([self.cs[T_LIDAR]] if self.has_lidar else [])
        for c, tn, raw in self.r.messages(connections=conns, start=lo, stop=hi):
            m = self.r.deserialize(raw, c.msgtype)
            if c.topic == T_LIDAR:
                scans.append((_stamp(m) + self.lidar_off + 0.05, m))
            else:
                radars.append((_stamp(m), m))
        out = {"lidar": None, "dt_lidar": np.nan, "radar": [], "dt_radar": np.nan}
        if scans:
            ts, m = min(scans, key=lambda s: abs(s[0] - t_img))
            a = pc2_to_struct(m)
            xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
            xyz = xyz[np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 1.0)]
            R, t = lidar_to_sensor(self.tf, m.header.frame_id)
            out["lidar"], out["dt_lidar"] = xyz @ R.T + t, ts - t_img
        if radars:
            radars.sort(key=lambda s: abs(s[0] - t_img))
            out["dt_radar"] = radars[0][0] - t_img
            for ts, m in radars[:n_radar]:
                a = pc2_to_struct(m)
                xyz = np.stack([a["x"], a["y"], a["z"]], 1).astype(float)
                vr = a["radial_velocity"].astype(float)
                k = np.isfinite(xyz).all(1) & (np.linalg.norm(xyz, axis=1) > 0.5)
                xyz, vr = xyz[k], vr[k]
                ego = estimate_ego_velocity(xyz, vr, inlier_thr=0.2, planar=True)
                v_s = ego.v_sensor if ego.valid else np.zeros(3)
                if np.percentile(np.abs(vr), 70) < 0.05:
                    v_s = np.zeros(3)
                out["radar"].append((xyz, compensate_doppler(xyz, vr, v_s), np.linalg.norm(v_s[:2])))
        return out


# ------------------------------------------------------------------------------------------------ drawing
def _proj(pc, K, D):
    k = pc[:, 2] > 0.5
    uv = np.full((len(pc), 2), -1.0)
    if k.any():
        uv[k] = cv2.projectPoints(pc[k], np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    return uv, k


def _vel_colour(v):
    """BGR: white static, red receding (+), blue approaching (-), saturating at 1.5 m/s."""
    a = np.clip(np.abs(v) / 1.5, 0, 1)[:, None]
    base = np.where(v[:, None] > 0, [[40, 40, 255]], [[255, 80, 40]])
    return (255 * (1 - a) + base * a).astype(np.uint8)


def _range_colour(xyz, max_range):
    """BGR turbo by range (same scale as the LiDAR overlay, so aligned radar rings match the dots below)."""
    r = np.linalg.norm(xyz, axis=1)
    return cv2.applyColorMap(np.uint8(np.clip(r / max_range * 255, 0, 255))[:, None], cv2.COLORMAP_TURBO)[:, 0]


def _radar_colours(p_sensor, vc, mode, max_range):
    return _range_colour(p_sensor, max_range) if mode == "range" else _vel_colour(vc)


def draw_image(img, ex, K, D, lidar, radar, *, show_lidar, show_radar, lidar_max=40.0, dot=2, ring=5,
               moving_only=False, moving_thr=0.3, radar_colour="range"):
    out = img.copy()
    H, W = out.shape[:2]
    if show_lidar and lidar is not None:
        pc = ex.sensor_to_cam(lidar)
        uv, k = _proj(pc, K, D)
        rng = np.linalg.norm(lidar, axis=1)
        ok = k & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        col = cv2.applyColorMap(np.uint8(np.clip(rng[ok] / lidar_max * 255, 0, 255))[:, None], cv2.COLORMAP_TURBO)[:, 0]
        u, v = uv[ok, 0].astype(int), uv[ok, 1].astype(int)
        for du in range(-dot + 1, dot):
            for dv in range(-dot + 1, dot):
                out[np.clip(v + dv, 0, H - 1), np.clip(u + du, 0, W - 1)] = col
    if show_radar:
        for xyz, vc, _ in radar:
            ps = ex.radar_to_sensor(xyz)
            uv, k = _proj(ex.sensor_to_cam(ps), K, D)
            for (u, v), c, kk, vv in zip(uv, _radar_colours(ps, vc, radar_colour, lidar_max), k, vc):
                if kk and 0 <= u < W and 0 <= v < H and (not moving_only or abs(vv) > moving_thr):
                    cv2.circle(out, (int(u), int(v)), ring, (0, 0, 0), 3)
                    cv2.circle(out, (int(u), int(v)), ring, tuple(int(x) for x in c), 2)
    return out


def draw_bev(ex, K, D, size, lidar, radar, *, show_lidar, show_radar, rng_m=40.0, px=720, moving_only=False,
             moving_thr=0.3, radar_colour="range", colour_range=30.0):
    img = np.full((px, px, 3), 25, np.uint8)
    s = px / rng_m
    to = lambda p: np.c_[px / 2 - p[:, 1] * s, px - p[:, 0] * s].astype(int)
    for r in range(5, int(rng_m) + 1, 5):                       # range rings
        cv2.circle(img, (px // 2, px), int(r * s), (60, 60, 60), 1)
        cv2.putText(img, f"{r}m", (px // 2 + 4, px - int(r * s) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)
    # camera field of view (edges of the image projected to the ground direction)
    W, H = size
    for u in (0, W - 1):
        ray = cv2.undistortPoints(np.array([[[u, H / 2]]], float), K, D).reshape(2)
        d_cam = np.array([ray[0], ray[1], 1.0])
        from rcmot.utils.rig_extrinsics import R_OPT, rot_rpy
        d_s = rot_rpy(ex.cam_rpy_deg).T @ (R_OPT.T @ d_cam)
        p = np.array([[0, 0, 0], d_s / np.linalg.norm(d_s[:2]) * rng_m])
        q = to(p)
        cv2.line(img, tuple(q[0]), tuple(q[1]), (90, 90, 0), 1)
    if show_lidar and lidar is not None:
        n = np.asarray(ex.ground_normal, float)
        hag = lidar @ n + ex.sensor_height
        keep = hag > 0.25
        q = to(lidar[keep])
        col = cv2.applyColorMap(np.uint8(np.clip(hag[keep] / 4.0 * 255, 0, 255))[:, None], cv2.COLORMAP_VIRIDIS)[:, 0]
        ok = (q[:, 0] >= 1) & (q[:, 0] < px - 1) & (q[:, 1] >= 1) & (q[:, 1] < px - 1)
        for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
            img[q[ok, 1] + dv, q[ok, 0] + du] = col[ok]
    if show_radar:
        for xyz, vc, _ in radar:
            ps = ex.radar_to_sensor(xyz)
            q = to(ps)
            for (u, v), c, vv in zip(q, _radar_colours(ps, vc, radar_colour, colour_range), vc):
                if 0 <= u < px and 0 <= v < px and (not moving_only or abs(vv) > moving_thr):
                    cv2.circle(img, (int(u), int(v)), 4, (0, 0, 0), -1)
                    cv2.circle(img, (int(u), int(v)), 3, tuple(int(x) for x in c), -1)
    cv2.putText(img, "BEV (x fwd): LiDAR above ground (colour = height 0-4 m), dots = radar", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return img


def to_pixmap(bgr, w, h):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    q = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(q.copy()).scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)


# --------------------------------------------------------------------------------------------------- GUI
class Viewer(QtWidgets.QWidget):
    def __init__(self, bag: str, calib: str | None):
        super().__init__()
        self.setWindowTitle("RC-MOT calibration viewer")
        self.resize(1880, 1000)
        self.bag, self.calib_override = None, calib
        self.hist: list[int] = []          # bag times (ns) of visited frames
        self.cur = None                    # (bag_ns, header, bgr, synced)

        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 1)
        views = QtWidgets.QHBoxLayout()
        left.addLayout(views, 1)
        self.view_img = QtWidgets.QLabel()
        self.view_img.setMinimumSize(960, 540)
        self.view_img.setAlignment(QtCore.Qt.AlignCenter)
        self.view_img.setStyleSheet("background:#111")
        views.addWidget(self.view_img, 3)
        self.view_bev = QtWidgets.QLabel()
        self.view_bev.setMinimumSize(420, 420)
        self.view_bev.setAlignment(QtCore.Qt.AlignCenter)
        self.view_bev.setStyleSheet("background:#111")
        views.addWidget(self.view_bev, 2)

        nav = QtWidgets.QHBoxLayout()
        left.addLayout(nav)
        self.btn = {}
        for k, txt in (("p10", "<< 10"), ("prev", "< prev (A)"), ("play", "play (Space)"), ("next", "next (D) >"), ("n10", "10 >>")):
            b = QtWidgets.QPushButton(txt)
            nav.addWidget(b)
            self.btn[k] = b
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        nav.addWidget(self.slider, 1)
        self.t_label = QtWidgets.QLabel()
        self.t_label.setMinimumWidth(90)
        nav.addWidget(self.t_label)
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("font-family:monospace;font-size:12px")
        left.addWidget(self.info)

        panel = QtWidgets.QFormLayout()
        side = QtWidgets.QWidget()
        side.setLayout(panel)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(side)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(360)
        root.addWidget(scroll)
        self.bag_box = QtWidgets.QComboBox()
        for p in sorted(BAG_DIR.iterdir()):
            if p.is_dir():
                self.bag_box.addItem(p.name, str(p))
        panel.addRow("bag", self.bag_box)
        self.calib_label = QtWidgets.QLabel()
        self.calib_label.setStyleSheet("font-size:11px")
        self.calib_label.setWordWrap(True)
        panel.addRow("calib", self.calib_label)
        self.spins = {}
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
                    sb.setKeyboardTracking(False)
                    self.spins[(key, i)] = sb
                    panel.addRow(f"{key.split('_')[0]} {ax}", sb)
        panel.addRow(QtWidgets.QLabel("<b>display</b>"))
        self.n_radar = QtWidgets.QSpinBox()
        self.n_radar.setRange(1, 15)
        self.n_radar.setValue(1)
        panel.addRow("radar frames", self.n_radar)
        self.dot = QtWidgets.QSpinBox()
        self.dot.setRange(1, 4)
        self.dot.setValue(2)
        panel.addRow("LiDAR dot", self.dot)
        self.rcol = QtWidgets.QComboBox()
        self.rcol.addItem("range", "range")
        self.rcol.addItem("Doppler", "doppler")
        panel.addRow("radar colour", self.rcol)
        self.ring = QtWidgets.QSpinBox()
        self.ring.setRange(2, 12)
        self.ring.setValue(5)
        panel.addRow("radar ring", self.ring)
        self.lmax = QtWidgets.QSpinBox()
        self.lmax.setRange(5, 80)
        self.lmax.setValue(30)
        self.lmax.setSuffix(" m")
        panel.addRow("LiDAR colour range", self.lmax)
        self.chk = {k: QtWidgets.QCheckBox(k) for k in ("LiDAR", "radar", "radar: moving only", "brighten night")}
        for k, c in self.chk.items():
            c.setChecked(k != "radar: moving only")
            panel.addRow(c)
        self.btn_reset = QtWidgets.QPushButton("Reset to file")
        self.btn_save = QtWidgets.QPushButton("Save (S)")
        panel.addRow(self.btn_reset)
        panel.addRow(self.btn_save)
        self.status = QtWidgets.QLabel()
        self.status.setWordWrap(True)
        panel.addRow(self.status)
        legend = QtWidgets.QLabel(
            "<small>LiDAR colour = range (near blue → far red).<br>"
            "Radar ring colour: range (same scale as LiDAR: an aligned ring has the colour of the LiDAR dots under it) "
            "or Doppler (white static, red moving away, blue approaching).<br>"
            "Check: LiDAR range jumps on object outlines; radar rings on people / loader / cones; "
            "in the BEV radar dots on LiDAR objects.<br>"
            "Keys: D/→ next, A/← prev, ↑/↓ ±10, Space play, S save.</small>")
        legend.setWordWrap(True)
        panel.addRow(legend)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.next_frame)
        self.btn["next"].clicked.connect(self.next_frame)
        self.btn["prev"].clicked.connect(self.prev_frame)
        self.btn["n10"].clicked.connect(lambda: self.skip(10))
        self.btn["p10"].clicked.connect(lambda: self.skip(-10))
        self.btn["play"].clicked.connect(self.toggle_play)
        self.slider.sliderReleased.connect(self.seek_slider)
        for sb in self.spins.values():
            sb.valueChanged.connect(self.redraw)
        self.rcol.currentIndexChanged.connect(self.redraw)
        for w in (self.dot, self.lmax, self.ring):
            w.valueChanged.connect(self.redraw)
        for c in self.chk.values():
            c.stateChanged.connect(self.redraw)
        self.n_radar.valueChanged.connect(self.reload_sync)
        self.bag_box.currentIndexChanged.connect(lambda _: self.open_bag(self.bag_box.currentData()))
        self.btn_reset.clicked.connect(self.load_calib)
        self.btn_save.clicked.connect(self.save)
        for key, fn in ((QtCore.Qt.Key_D, self.next_frame), (QtCore.Qt.Key_Right, self.next_frame),
                        (QtCore.Qt.Key_A, self.prev_frame), (QtCore.Qt.Key_Left, self.prev_frame),
                        (QtCore.Qt.Key_Up, lambda: self.skip(10)), (QtCore.Qt.Key_Down, lambda: self.skip(-10)),
                        (QtCore.Qt.Key_Space, self.toggle_play), (QtCore.Qt.Key_S, self.save)):
            QtWidgets.QShortcut(QtGui.QKeySequence(key), self, activated=fn)

        if bag and self.bag_box.findData(str(Path(bag).resolve())) < 0:      # e.g. a field capture
            self.bag_box.insertItem(0, Path(bag).name, str(Path(bag).resolve()))
        i = self.bag_box.findData(str(Path(bag).resolve())) if bag else 0
        self.bag_box.setCurrentIndex(max(i, 0))
        self.open_bag(self.bag_box.currentData())

    # ---------------------------------------------------------------------------------------- data
    def calib_path(self) -> Path:
        if self.calib_override:
            return Path(self.calib_override)
        name = Path(self.bag.path).name
        return ROOT / "configs/calib" / SESSION_CALIB.get(name[:8], "hdx_boryeong_pg_night1126.json")

    def load_calib(self):
        p = self.calib_path()
        self.calib_json = json.loads(p.read_text())
        self.ex = RigExtrinsics.from_calib_json(self.calib_json)
        self.calib_label.setText(p.name)
        for (key, i), sb in self.spins.items():
            sb.blockSignals(True)
            sb.setValue(getattr(self.ex, key)[i])
            sb.blockSignals(False)
        self.redraw()

    def open_bag(self, path):
        if self.bag is not None:
            self.bag.close()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.bag = BagFrames(path)
        QtWidgets.QApplication.restoreOverrideCursor()
        self.slider.setRange(0, int((self.bag.t1 - self.bag.t0) / 1e9))
        self.hist, self.cur = [], None
        self.load_calib()
        self.goto(self.bag.t0, inclusive=True)

    def goto(self, t_bag_ns, inclusive=False, record=True):
        f = self.bag.image_after(t_bag_ns, inclusive)
        if f is None:
            self.timer.stop()
            self.status.setText("end of bag")
            return False
        tn, th, img = f
        self.cur = [tn, th, img, self.bag.synced(th, tn, self.n_radar.value())]
        if record:
            self.hist.append(tn)
        self.slider.blockSignals(True)
        self.slider.setValue(int((tn - self.bag.t0) / 1e9))
        self.slider.blockSignals(False)
        self.redraw()
        return True

    def reload_sync(self):
        if self.cur:
            self.cur[3] = self.bag.synced(self.cur[1], self.cur[0], self.n_radar.value())
            self.redraw()

    def next_frame(self):
        if self.cur:
            self.goto(self.cur[0])

    def prev_frame(self):
        if len(self.hist) >= 2:
            self.hist.pop()
            self.goto(self.hist[-1], inclusive=True, record=False)

    def skip(self, n):
        if n > 0:
            for _ in range(n - 1):                               # walk image headers only (cheap), then load
                f = self.bag.image_after(self.cur[0])
                if f is None:
                    break
                self.cur[0] = f[0]
                self.hist.append(f[0])
            self.next_frame()
        else:
            for _ in range(min(-n, len(self.hist) - 1)):
                self.hist.pop()
            self.goto(self.hist[-1], inclusive=True, record=False)

    def seek_slider(self):
        self.hist = []
        self.goto(self.bag.t0 + int(self.slider.value() * 1e9), inclusive=True)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.btn["play"].setText("play (Space)")
        else:
            self.timer.start(30)
            self.btn["play"].setText("pause (Space)")

    # ---------------------------------------------------------------------------------------- view
    def redraw(self, *_):
        if not self.cur or not hasattr(self, "ex"):
            return
        for (key, i), sb in self.spins.items():
            getattr(self.ex, key)[i] = sb.value()
        tn, th, img, s = self.cur
        base = enhance_dark(img) if self.chk["brighten night"].isChecked() else img
        b = self.bag
        im = draw_image(base, self.ex, b.K, b.D, s["lidar"], s["radar"], show_lidar=self.chk["LiDAR"].isChecked(),
                        show_radar=self.chk["radar"].isChecked(), lidar_max=self.lmax.value(), dot=self.dot.value(),
                        ring=self.ring.value(), moving_only=self.chk["radar: moving only"].isChecked(),
                        radar_colour=self.rcol.currentData())
        bev = draw_bev(self.ex, b.K, b.D, b.size, s["lidar"], s["radar"], show_lidar=self.chk["LiDAR"].isChecked(),
                       show_radar=self.chk["radar"].isChecked(), moving_only=self.chk["radar: moving only"].isChecked(),
                       radar_colour=self.rcol.currentData(), colour_range=self.lmax.value())
        self.view_img.setPixmap(to_pixmap(im, self.view_img.width(), self.view_img.height()))
        self.view_bev.setPixmap(to_pixmap(bev, self.view_bev.width(), self.view_bev.height()))
        rel = (tn - b.t0) / 1e9
        self.t_label.setText(f"{rel:7.2f} s")
        ego = s["radar"][0][2] if s["radar"] else float("nan")
        n_r = sum(len(r[0]) for r in s["radar"])
        self.info.setText(f"{Path(b.path).name}   frame #{len(self.hist)}   t={rel:.3f}s   "
                          f"sync: LiDAR {s['dt_lidar'] * 1000:+.0f} ms, radar {s['dt_radar'] * 1000:+.0f} ms   "
                          f"radar pts {n_r}   ego speed {ego:.2f} m/s   img mean {img.mean():.0f}")

    def save(self):
        p = self.calib_path()
        d = self.ex.to_calib_json(self.bag.K, self.bag.D, self.bag.size, name=self.calib_json.get("bag_key", ""),
                                  distortion_model=self.calib_json.get("distortion_model", "rational_polynomial"))
        if "report" in self.calib_json:
            d["report"] = self.calib_json["report"]
        d.setdefault("report", {})["manual_edit"] = {"by": "calib_viewer", "frame_bag": Path(self.bag.path).name}
        p.write_text(json.dumps(d, indent=1, default=float))
        self.calib_json = d
        self.status.setText(f"saved {p.name}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.redraw()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", help="bag directory (default: first in data/bags/hdx_boryeong_pg)")
    ap.add_argument("--calib", help="calib json (default: the bag's session file)")
    ap.add_argument("--config", help="yaml with a 'topics:' section (e.g. configs/field_calib.yaml)")
    a = ap.parse_args()
    if a.config:
        import yaml

        global T_IMG, T_LIDAR, T_RADAR, T_INFO
        tp = yaml.safe_load(open(a.config))["topics"]
        T_IMG, T_LIDAR, T_RADAR, T_INFO = tp["image"], tp["lidar"], tp["radar"], tp["camera_info"]
    app = QtWidgets.QApplication(sys.argv)
    w = Viewer(a.bag, a.calib)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
