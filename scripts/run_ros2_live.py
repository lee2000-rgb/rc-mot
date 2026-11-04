#!/usr/bin/env python3
"""Live ROS 2 runner (Orin), headless with --no-gui; without it opens scripts/rcmot_app.py --live.
Subscribes radar / compressed image / IMU, runs the pipeline per radar frame
and shows the GUI. Source ROS first:  source /opt/ros/<distro>/setup.bash
  python3 scripts/run_ros2_live.py [--config configs/default.yaml] [--no-gui]
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rclpy  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CompressedImage, Image, Imu, PointCloud2  # noqa: E402

from rcmot.pipeline import Pipeline, build_pipeline  # noqa: E402
from rcmot.sensors.bag_source import image_to_bgr, pc2_to_struct, radar_frame  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


class RcMotNode(Node):
    def __init__(self, cfg: dict, calib: RigCalib, pipe: Pipeline):
        super().__init__("rc_mot")
        self.calib, self.pipe = calib, pipe
        self.img: tuple[float, np.ndarray] | None = None
        self.gyro_z = 0.0
        self.latest = None                     # (image, FrameResult) for the GUI thread
        t = cfg["topics"]
        img_type = CompressedImage if t["image"].endswith("/compressed") else Image
        self.create_subscription(img_type, t["image"], self.on_image, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, t["radar"], self.on_radar, qos_profile_sensor_data)
        if t.get("imu"):
            self.create_subscription(Imu, t["imu"], self.on_imu, qos_profile_sensor_data)

    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def on_image(self, msg):
        self.img = (self._stamp(msg) - self.calib.camera_time_offset_s, image_to_bgr(msg))

    def on_imu(self, msg):
        self.gyro_z = 0.9 * self.gyro_z + 0.1 * msg.angular_velocity.z

    def on_radar(self, msg):
        if self.img is None:
            return
        rf = radar_frame(pc2_to_struct(msg), self._stamp(msg), self.calib)
        res = self.pipe.step(self.img[1], rf, self.gyro_z)
        self.latest = (self.img[1], res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--no-gui", action="store_true")
    a = ap.parse_args()
    if not a.no_gui:                                       # GUI (DVR timeline) lives in rcmot_app.py
        from rcmot_app import main as app_main

        sys.argv = [sys.argv[0], "--live", "--config", a.config]
        return app_main()
    cfg = yaml.safe_load(open(a.config))
    calib = RigCalib.load(str((Path(a.config).parent / cfg["calib"]).resolve()))
    pipe = build_pipeline(cfg, calib, ROOT)

    rclpy.init()
    node = RcMotNode(cfg, calib, pipe)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    if a.no_gui:
        while rclpy.ok():
            time.sleep(1.0)
            if node.latest:
                r = node.latest[1]
                node.get_logger().info(f"ego={r.ego_vel[:2].round(2)} tracks={[t.tid for t in r.tracks]}")
        return

    from PyQt5 import QtCore, QtWidgets
    from rcmot.gui.viewer import Viewer

    app = QtWidgets.QApplication(sys.argv)
    g = cfg.get("gui", {})
    win = Viewer(calib, max_range=g.get("bev_range", 30.0), danger_m=g.get("danger_m", 5.0),
                 zones=tuple(g.get("zones", (5.0, 10.0, 15.0))), title="live")
    win.show()
    shown = {"res": None}

    def refresh():
        if node.latest and node.latest[1] is not shown["res"]:
            img, res = node.latest
            win.show_frame(img, res)
            shown["res"] = res

    timer = QtCore.QTimer()
    timer.timeout.connect(refresh)
    timer.start(30)
    app.exec_()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
