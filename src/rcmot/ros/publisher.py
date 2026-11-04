"""ROS 2 publisher of the pipeline results (docs/20): tracks, alarm state, a vision_msgs mirror, a 1 Hz status
and, optionally, RViz markers. Called on the processing thread for every processed frame (worker sink).

Topics (namespace `/rcmot`, frame `base_link_level`):
  tracks      rcmot_msgs/TrackArray         every frame   full information (velocity, state, zone, source ...)
  alarm       rcmot_msgs/AlarmState         every frame   level after hysteresis, nearest track
  detections  vision_msgs/Detection3DArray  every frame   same tracks for RViz / Foxglove
  status      rcmot_msgs/Status             1 Hz          profile, model, calib, FPS, lag, drops
  markers     visualization_msgs/MarkerArray  (off by default)
  /tf_static  <radar frame> -> base_link_level, once (the HDX rig's tree ends at the radar frame)

`enabled` False = only the 1 Hz status goes out, with publishing=false (the HDX GUI's health lamp of /rcmot/tracks
then reads DEAD = OFF). Track zones use the zones the alarm decided the frame with (res.alarm.zones), so
/rcmot/tracks[].zone and /rcmot/alarm never disagree.
QoS: RELIABLE, depth 10, so `ros2 bag record` misses nothing.
"""
from __future__ import annotations

import math
import threading
import time
from collections import deque

import numpy as np

from rcmot.alarm import zone_of
from rcmot.types import TrackState

OBJ_SIZE = {"worker": (0.6, 0.6, 1.7), "person": (0.6, 0.6, 1.7), "machine": (4.0, 2.0, 2.5), "vehicle": (4.0, 2.0, 1.6)}
_STATE_CODE = {TrackState.TENTATIVE: 0, TrackState.CONFIRMED: 1, TrackState.LOST: 2, TrackState.RADAR_ONLY: 3}


def quat_from_matrix(R: np.ndarray) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a rotation matrix (Shepperd's method)."""
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def _stamp(t: float):
    from builtin_interfaces.msg import Time

    sec = int(math.floor(t))
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)) % 1_000_000_000)


class ResultPublisher:
    def __init__(self, calib, *, meta: dict | None = None, namespace: str = "/rcmot",
                 frame_id: str = "base_link_level", tf_parent: str | None = "radar", markers: bool = False,
                 status_hz: float = 1.0, enabled: bool = True, node=None, stats_fn=None):
        """calib: RigCalib (R_lr / t_lr give the radar -> level TF). meta: profile / detector / calib names for
        Status. stats_fn() -> dict(cam_lag_s, frames_dropped) from the live worker. node: existing rclpy node,
        else an own node `rc_mot_pub` is created (rclpy.init when needed)."""
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rcmot_msgs.msg import AlarmState, Status, TrackArray
        from tf2_msgs.msg import TFMessage
        from vision_msgs.msg import Detection3DArray
        from visualization_msgs.msg import MarkerArray

        self.calib, self.meta, self.frame_id, self.tf_parent = calib, dict(meta or {}), frame_id, tf_parent
        self.markers, self.status_period = markers, (1.0 / status_hz if status_hz > 0 else 0.0)
        self.enabled = bool(enabled)
        self.stats_fn = stats_fn
        self._lock = threading.Lock()
        if not rclpy.ok():
            from rcmot.gui.playback import _no_signal_handlers
            rclpy.init(signal_handler_options=_no_signal_handlers())
        self.own_node = node is None
        self.node = node or Node("rc_mot_pub")
        ns = namespace.rstrip("/")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub_tracks = self.node.create_publisher(TrackArray, f"{ns}/tracks", qos)
        self.pub_alarm = self.node.create_publisher(AlarmState, f"{ns}/alarm", qos)
        self.pub_det = self.node.create_publisher(Detection3DArray, f"{ns}/detections", qos)
        self.pub_status = self.node.create_publisher(Status, f"{ns}/status", qos)
        self.pub_markers = self.node.create_publisher(MarkerArray, f"{ns}/markers", qos) if markers else None
        self.pub_tf = self.node.create_publisher(
            TFMessage, "/tf_static", QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._tf_sent = False
        self._recent: deque = deque()                     # (wall time, total ms, det ms) of the last second
        self._last_status = 0.0
        self.n_published = 0

    # ------------------------------------------------------------------ control
    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)

    @staticmethod
    def _zones(res) -> tuple:
        return tuple(res.alarm.zones) if getattr(res, "alarm", None) is not None else ()

    def close(self) -> None:
        for p in (self.pub_tracks, self.pub_alarm, self.pub_det, self.pub_status, self.pub_markers, self.pub_tf):
            if p is not None:
                try:
                    self.node.destroy_publisher(p)
                except Exception:  # noqa: BLE001
                    pass
        if self.own_node:
            self.node.destroy_node()

    # ------------------------------------------------------------------ per frame
    def publish(self, res, tf_parent: str | None = None) -> bool:
        """res: FrameResult (with .alarm when the worker decided it). Returns True when something was published."""
        with self._lock:
            now = time.time()
            if not self.enabled:                             # off: 1 Hz status only (alive, publishing=false)
                if self.status_period and now - self._last_status >= self.status_period:
                    self._last_status = now
                    self._recent.clear()
                    self.pub_status.publish(self._status_msg(res, _stamp(float(res.stamp))))
                return False
            if not self._tf_sent:
                self._send_tf(tf_parent or self.tf_parent)
            stamp = _stamp(float(res.stamp))
            self.pub_tracks.publish(self._tracks_msg(res, stamp))
            self.pub_alarm.publish(self._alarm_msg(res, stamp))
            self.pub_det.publish(self._detections_msg(res, stamp))
            if self.pub_markers is not None:
                self.pub_markers.publish(self._markers_msg(res, stamp))
            self.n_published += 1
            tm = res.timing_ms or {}
            self._recent.append((now, float(tm.get("total", sum(tm.values()))), float(tm.get("detector", 0.0))))
            while self._recent and now - self._recent[0][0] > 1.0:
                self._recent.popleft()
            if self.status_period and now - self._last_status >= self.status_period:
                self._last_status = now
                self.pub_status.publish(self._status_msg(res, stamp))
        return True

    # ------------------------------------------------------------------ messages
    def _header(self, stamp):
        from std_msgs.msg import Header

        return Header(stamp=stamp, frame_id=self.frame_id)

    def _tracks_msg(self, res, stamp):
        from geometry_msgs.msg import Point, Vector3
        from rcmot_msgs.msg import Track, TrackArray

        msg = TrackArray(header=self._header(stamp))
        zones = self._zones(res)
        for t in res.tracks:
            m = Track()
            m.id = int(t.tid)
            m.cls = str(t.cls)
            m.state = _STATE_CODE.get(t.state, 0)
            m.score = float(t.score)
            m.exist = float(getattr(t, "exist", 1.0))
            p, v = np.asarray(t.pos, float), np.asarray(t.vel, float) if t.vel is not None else np.zeros(3)
            m.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]) if len(p) > 2 else 0.0)
            m.velocity = Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]) if len(v) > 2 else 0.0)
            cov = getattr(t, "pos_cov", None)
            m.position_covariance = [float(x) for x in np.asarray(cov, float).reshape(-1)[:9]] if cov is not None else [0.0] * 9
            m.is_static = bool(getattr(t, "static", False))
            r = float(np.hypot(p[0], p[1]))
            m.range = r
            m.zone = zone_of(r, zones) if zones else 0
            m.source = str(getattr(t, "source", ""))
            m.n_radar = int(getattr(t, "n_radar", 0))
            m.age = float(getattr(t, "age", 0.0))
            m.time_since_update = float(t.time_since_update)
            b = getattr(t, "box2d", None)
            m.box2d = [int(round(float(x))) for x in b] if b is not None else [-1, -1, -1, -1]
            msg.tracks.append(m)
        ev = np.asarray(res.ego_vel, float) if res.ego_vel is not None else np.zeros(3)
        msg.ego_velocity = Vector3(x=float(ev[0]), y=float(ev[1]), z=float(ev[2]) if len(ev) > 2 else 0.0)
        msg.ego_valid = bool(res.ego_valid)
        return msg

    def _alarm_msg(self, res, stamp):
        from rcmot_msgs.msg import AlarmState

        st = res.alarm
        m = AlarmState(header=self._header(stamp))
        if st is None:
            m.level, m.nearest_id, m.nearest_range = 0, -1, float("nan")
            m.zones = [0.0, 0.0, 0.0]
            return m
        m.level = int(st.level)
        m.nearest_id = -1 if st.nearest_tid is None else int(st.nearest_tid)
        m.nearest_range = float("nan") if st.nearest_range is None else float(st.nearest_range)
        m.zones = [float(z) for z in st.zones]
        m.sound_enabled = bool(st.sound_enabled)
        m.confirmed_only = bool(st.confirmed_only)
        return m

    def _detections_msg(self, res, stamp):
        from geometry_msgs.msg import Point, Pose, Vector3
        from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

        msg = Detection3DArray(header=self._header(stamp))
        for t in res.tracks:
            d = Detection3D(header=msg.header)
            d.id = str(int(t.tid))
            h = ObjectHypothesisWithPose()
            h.hypothesis.class_id = str(t.cls)
            h.hypothesis.score = float(t.score)
            p = np.asarray(t.pos, float)
            w, l, hgt = OBJ_SIZE.get(str(t.cls), (1.0, 1.0, 1.8))
            pose = Pose(position=Point(x=float(p[0]), y=float(p[1]), z=hgt / 2.0))
            pose.orientation.w = 1.0
            h.pose.pose = pose
            cov = getattr(t, "pos_cov", None)
            if cov is not None:
                c = np.asarray(cov, float).reshape(3, 3)
                c6 = np.zeros((6, 6)); c6[:3, :3] = c
                h.pose.covariance = [float(x) for x in c6.reshape(-1)]
            d.results.append(h)
            d.bbox.center = pose
            d.bbox.size = Vector3(x=float(l), y=float(w), z=float(hgt))
            msg.detections.append(d)
        return msg

    def _markers_msg(self, res, stamp):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray

        cols = {0: (0.7, 0.7, 0.7), 1: (1.0, 0.83, 0.3), 2: (1.0, 0.6, 0.23), 3: (1.0, 0.31, 0.31)}
        msg = MarkerArray()
        wipe = Marker(header=self._header(stamp)); wipe.action = Marker.DELETEALL
        msg.markers.append(wipe)
        zones = self._zones(res)
        for t in res.tracks:
            p = np.asarray(t.pos, float)
            w, l, hgt = OBJ_SIZE.get(str(t.cls), (1.0, 1.0, 1.8))
            z = zone_of(float(np.hypot(p[0], p[1])), zones) if zones else 0
            box = Marker(header=self._header(stamp), ns="rcmot", id=int(t.tid), type=Marker.CUBE, action=Marker.ADD)
            box.pose.position = Point(x=float(p[0]), y=float(p[1]), z=hgt / 2.0)
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = float(l), float(w), float(hgt)
            box.color.r, box.color.g, box.color.b, box.color.a = (*cols[z], 0.5)
            txt = Marker(header=self._header(stamp), ns="rcmot_id", id=int(t.tid), type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            txt.pose.position = Point(x=float(p[0]), y=float(p[1]), z=hgt + 0.3)
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.4
            txt.text = f"{t.tid} {t.cls}"
            txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
            msg.markers.extend((box, txt))
        return msg

    def _status_msg(self, res, stamp):
        from rcmot_msgs.msg import Status

        m = Status(header=self._header(stamp))
        m.profile = str(self.meta.get("profile", ""))
        m.detector = str(self.meta.get("detector", ""))
        m.calib = str(self.meta.get("calib", ""))
        rec = list(self._recent)
        m.fps = float(len(rec))                          # frames published in the last second
        m.total_ms = float(np.median([r[1] for r in rec])) if rec else 0.0
        m.det_ms = float(np.median([r[2] for r in rec])) if rec else 0.0
        st = self.stats_fn() if self.stats_fn else {}
        lag = st.get("cam_lag_s", float("nan"))
        m.cam_lag_s = float(lag) if lag is not None else float("nan")
        m.frames_dropped = int(st.get("frames_dropped", 0))
        m.publishing = bool(self.enabled)
        return m

    def _send_tf(self, parent: str | None) -> None:
        """radar frame -> base_link_level, once: T_parent_child = inv(T_level_radar) from the calib."""
        if not parent:
            return
        from geometry_msgs.msg import TransformStamped
        from tf2_msgs.msg import TFMessage

        R_lr, t_lr = np.asarray(self.calib.R_lr, float), np.asarray(self.calib.t_lr, float).reshape(3)
        R_rl, t_rl = R_lr.T, -R_lr.T @ t_lr
        ts = TransformStamped()
        ts.header.stamp = _stamp(time.time())
        ts.header.frame_id = parent
        ts.child_frame_id = self.frame_id
        ts.transform.translation.x, ts.transform.translation.y, ts.transform.translation.z = (float(v) for v in t_rl)
        qx, qy, qz, qw = quat_from_matrix(R_rl)
        ts.transform.rotation.x, ts.transform.rotation.y, ts.transform.rotation.z, ts.transform.rotation.w = qx, qy, qz, qw
        self.pub_tf.publish(TFMessage(transforms=[ts]))
        self._tf_sent = True
