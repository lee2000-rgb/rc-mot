"""Playback back-ends for the GUI (docs/11): a processed-frame cache plus a worker that fills it.

  FrameCache   time-ordered processed frames (JPEG image + FrameResult); the GUI plays from it at the
               chosen speed and can scrub / step backwards inside it without recomputing.
  BagWorker    runs the pipeline over a rosbag2 (mcap / sqlite3) ahead of the play cursor; seeking outside
               the cache resets the pipeline and restarts the reader at (target - warm-up).
  LiveWorker   ROS 2 subscriber: buffers radar scans until a camera frame with a later header stamp has
               arrived (the OAK-D stamps reach us ~1 s after the radar's), processes the newest such scan
               (older ones are dropped when slower than the sensor) and appends to the same cache
               (DVR-style rewind of the last minutes while live processing continues).

The tracker is causal, so a result at time t depends on the frames before it: after a seek the first
`warmup` seconds are processed but not shown, so tracks already exist when playback starts.
"""
from __future__ import annotations

import bisect
import threading
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Entry:
    t: float              # timeline time: bag time (replay) or radar header stamp (live)
    jpg: bytes
    res: object           # FrameResult

    def image(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.jpg, np.uint8), cv2.IMREAD_COLOR)


class FrameCache:
    """Thread-safe, bounded list of processed frames of ONE contiguous segment."""

    def __init__(self, max_frames: int = 3000, jpeg_quality: int = 88):
        self.max_frames, self.q = max_frames, jpeg_quality
        self.lock = threading.Lock()
        self.entries: list[Entry] = []
        self.gen = 0                                   # bumped on every clear (worker restart)

    def clear(self) -> int:
        with self.lock:
            self.entries.clear()
            self.gen += 1
            return self.gen

    def append(self, gen: int, t: float, image: np.ndarray, res, keep_after: float = -np.inf) -> bool:
        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.q])
        with self.lock:
            if gen != self.gen:                          # stale frame of a segment that was cleared meanwhile
                return False
            self.entries.append(Entry(t, jpg.tobytes(), res))
            while len(self.entries) > self.max_frames and self.entries[0].t < keep_after:
                self.entries.pop(0)
            return True

    def __len__(self):
        return len(self.entries)

    def span(self) -> tuple[float, float] | None:
        with self.lock:
            return (self.entries[0].t, self.entries[-1].t) if self.entries else None

    def index_at(self, t: float) -> int | None:
        """Index of the last frame at or before t (None if the cache does not reach t)."""
        with self.lock:
            if not self.entries or t < self.entries[0].t - 0.2 or t > self.entries[-1].t + 0.2:
                return None
            k = bisect.bisect_right([e.t for e in self.entries], t) - 1
            return max(k, 0)

    def get(self, i: int) -> Entry | None:
        with self.lock:
            return self.entries[i] if 0 <= i < len(self.entries) else None

    def find(self, t: float) -> int:
        """Index of the frame with timeline time t (after the list was trimmed at the front)."""
        with self.lock:
            k = bisect.bisect_left([e.t for e in self.entries], t)
            return min(k, len(self.entries) - 1)


class _Sinks:
    """Mixin: per-frame alarm decision + result sinks (ROS publisher ...) run on the worker thread after step()."""

    alarm = None                                     # rcmot.alarm.Alarm; decide() per frame -> res.alarm
    sinks: list = []                                 # callables f(res) (publisher.publish ...); replaced per instance

    def _after_step(self, res) -> None:
        if self.alarm is not None:
            res.alarm = self.alarm.decide(res.tracks)
        for f in self.sinks:
            try:
                f(res)
            except Exception as e:                   # noqa: BLE001  a sink must never stop processing
                if getattr(self, "sink_error", None) is None:
                    import traceback
                    traceback.print_exc()
                self.sink_error = repr(e)


class BagWorker(_Sinks, threading.Thread):
    def __init__(self, pipe, frames_from, cache: FrameCache, *, ahead_s: float = 20.0, warmup_s: float = 1.5,
                 t_begin: float = 0.0):
        """frames_from(t_abs) -> iterator of (image, RadarFrame with .bag_t, yaw_rate) starting at bag time t_abs."""
        super().__init__(daemon=True)
        self.pipe, self.frames_from, self.cache = pipe, frames_from, cache
        self.ahead_s, self.warmup_s, self.t_begin = ahead_s, warmup_s, t_begin
        self.cursor_t = t_begin                          # play position, set by the GUI
        self.request: float | None = t_begin
        self.done = False                                # reached the end of the bag
        self.busy = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self.stopped = False
        self.sinks, self.sink_error = [], None

    def stop(self):
        self.stopped = True
        self.join(timeout=3.0)

    def seek(self, t: float):
        with self._lock:
            self.request = t

    def run(self):
        it, gen = None, 0
        while not self.stopped:
            with self._lock:
                req, self.request = self.request, None
            if req is not None:
                gen = self.cache.clear()
                self.pipe.reset()
                if self.alarm is not None:
                    self.alarm.reset()
                it, self.done = self.frames_from(max(req - self.warmup_s, self.t_begin)), False
            span = self.cache.span()
            if it is None or self.done or (span and span[1] - self.cursor_t > self.ahead_s):
                self.busy = False
                time.sleep(0.02)
                continue
            self.busy = True
            try:
                img, rf, yaw = next(it)
            except StopIteration:
                self.done = True
                continue
            except Exception as e:                       # noqa: BLE001  keep the GUI alive, show the error
                self.error, self.done = repr(e), True
                continue
            res = self.pipe.step(img, rf, yaw)
            self._after_step(res)
            self.cache.append(gen, rf.bag_t if rf.bag_t is not None else rf.stamp, img, res,
                              keep_after=self.cursor_t - 30.0)


def _no_signal_handlers():
    """rclpy.init() would install its own SIGINT / SIGTERM handlers that shut the context down before the app's
    handler runs (in-flight publishes then fail with 'context is invalid'); the GUI quits cleanly on its own."""
    from rclpy.signals import SignalHandlerOptions
    return SignalHandlerOptions.NO


class LiveWorker(_Sinks, threading.Thread):
    """Subscribes radar / image / IMU topics (rclpy) and processes the newest scan whenever idle."""

    def __init__(self, pipe, calib, topics: dict, cache: FrameCache, *, max_dt: float = 0.06):
        super().__init__(daemon=True)
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu, PointCloud2

        self.pipe, self.calib, self.cache, self.max_dt = pipe, calib, cache, max_dt
        self.stopped = False
        if not rclpy.ok():
            rclpy.init(signal_handler_options=_no_signal_handlers())   # the app owns SIGTERM / SIGINT (clean Qt quit)
        self.node = Node("rc_mot")
        self.lock = threading.Lock()
        self.imgs: deque = deque(maxlen=15)              # (stamp, msg), decoded only when used
        self.radars: deque = deque(maxlen=60)            # (stamp, msg, receive time) waiting for a camera frame
        self.cam_lag = float("nan")                      # newest radar stamp - newest image stamp
        self.gyro: deque = deque(maxlen=600)              # (receive time, gyro z) for the yaw rate at the scan time
        self.n_in = self.n_done = self.n_drop = 0
        self.error: str | None = None
        self.rx: dict[str, float] = {}                   # topic -> last receive wall time
        self.sinks, self.sink_error = [], None
        self.radar_frame_id: str | None = None           # header.frame_id of the radar cloud (TF parent, docs/20)
        img_type = CompressedImage if topics["image"].endswith("/compressed") else Image
        self.topics = topics
        self.node.create_subscription(img_type, topics["image"], self._on_image, qos_profile_sensor_data)
        self.node.create_subscription(PointCloud2, topics["radar"], self._on_radar, qos_profile_sensor_data)
        if topics.get("imu"):
            self.node.create_subscription(Imu, topics["imu"], self._on_imu, qos_profile_sensor_data)
        self.depths: deque = deque(maxlen=10)            # (stamp, msg) stereo depth for the work monitor (docs/24)
        self.depth_K = None
        if topics.get("depth") and pipe.work is not None:
            from rcmot.sensors.bag_source import camera_info_topic
            dtype = CompressedImage if "compressed" in topics["depth"] else Image
            self.node.create_subscription(dtype, topics["depth"], self._on_depth, qos_profile_sensor_data)
            self.node.create_subscription(CameraInfo, camera_info_topic(topics["depth"]), self._on_depth_info,
                                          qos_profile_sensor_data)
        from rcmot.sensors.bag_source import camera_info_topic
        ci = topics.get("camera_info") or camera_info_topic(topics["image"])
        self.node.create_subscription(CameraInfo, ci, self._on_camera_info, qos_profile_sensor_data)
        self.intrinsics_live = False                     # K/D/size taken from /camera_info (docs/18: OAK day mode 2016x1512)
        threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True).start()

    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _on_image(self, msg):
        self.rx["image"] = time.time()
        with self.lock:
            self.imgs.append((self._stamp(msg) - self.calib.camera_time_offset_s, msg))

    def _on_camera_info(self, msg):
        """The rig's camera may run at another resolution than the calib file (OAK day 2016x1512 vs 1280x720):
        take K, D and the image size from the driver once; the extrinsic (level -> camera) stays."""
        if self.intrinsics_live or msg.width == 0:
            return
        from rcmot.sensors.bag_source import apply_camera_info
        apply_camera_info(self.calib, msg.width, msg.height, msg.k, msg.d)
        self.intrinsics_live = True

    def _on_depth(self, msg):
        with self.lock:
            self.depths.append((self._stamp(msg) - self.calib.camera_time_offset_s, msg))

    def _on_depth_info(self, msg):
        if self.depth_K is None and msg.width:
            self.depth_K = np.array(msg.k, float).reshape(3, 3)

    def _on_imu(self, msg):
        self.rx["imu"] = time.time()
        with self.lock:
            self.gyro.append((time.time(), msg.angular_velocity.z))

    def _on_radar(self, msg):
        self.rx["radar"] = time.time()
        if self.radar_frame_id is None and msg.header.frame_id:
            self.radar_frame_id = msg.header.frame_id
        with self.lock:
            self.n_in += 1
            if len(self.radars) == self.radars.maxlen:
                self.n_drop += 1                         # no camera frame for ~4 s
            # scan time = header + calib.radar_time_offset_s, exactly as bag replay (bag_source.iterate_bag) and the
            # pipeline's box-time model assume (docs/16 §10; was missing live until the 11-04 audit)
            self.radars.append((self._stamp(msg) + self.calib.radar_time_offset_s, msg, time.time()))

    def _pair(self):
        """Newest radar scan already covered by a camera frame + the nearest image; older scans are dropped."""
        with self.lock:
            if not self.radars or not self.imgs:
                return None
            newest = self.imgs[-1][0]
            self.cam_lag = self.radars[-1][0] - newest
            ready = [k for k, (rs, _, _) in enumerate(self.radars) if rs <= newest]
            if not ready:
                return None
            k = ready[-1]
            rs, rmsg, t_rx = self.radars[k]
            self.n_drop += k                             # processing slower than the sensor
            for _ in range(k + 1):
                self.radars.popleft()
            s, m = min(self.imgs, key=lambda x: abs(x[0] - rs))
        return (rs, rmsg, m, s, t_rx) if abs(s - rs) <= self.max_dt else None

    def run(self):
        from rcmot.sensors.bag_source import image_to_bgr, pc2_to_struct, radar_frame, yaw_rate_at

        while not self.stopped:
            p = self._pair()
            if p is None:
                time.sleep(0.003)
                continue
            rs, rmsg, imsg, s_img, t_rx = p
            try:
                img = image_to_bgr(imsg)
                with self.lock:
                    g = list(self.gyro)
                # yaw rate at THIS scan's arrival (gyro times are arrival times too), not at the newest scan's
                yaw = yaw_rate_at([x[0] for x in g], [x[1] for x in g], t_rx + self.calib.radar_time_offset_s)
                rf = radar_frame(pc2_to_struct(rmsg), rs, self.calib)
                rf.img_dt = s_img - rs                       # image time - scan time (docs/21 E2-c)
                if self.depth_K is not None:
                    with self.lock:
                        dd = min(self.depths, key=lambda x: abs(x[0] - s_img)) if self.depths else None
                    if dd is not None and abs(dd[0] - s_img) <= self.max_dt:
                        from rcmot.sensors.bag_source import depth_to_m
                        rf.depth = (depth_to_m(dd[1]), self.depth_K)
                res = self.pipe.step(img, rf, yaw)
                self._after_step(res)
            except Exception as e:                       # noqa: BLE001  keep running, show the last error
                if self.error is None:
                    import traceback
                    traceback.print_exc()
                self.error = repr(e)
                continue
            self.n_done += 1
            self.cache.append(self.cache.gen, rs, img, res, keep_after=np.inf)

    def join_worker(self):
        """Stop processing and wait for the in-flight step (incl. its sink calls); the node stays up."""
        self.stopped = True
        if self.is_alive():
            self.join(timeout=3.0)

    def stop(self):
        import rclpy

        self.join_worker()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def health(self) -> str:
        now = time.time()
        ages = "  ".join(f"{k} {now - self.rx[k]:.1f}s" if k in self.rx else f"{k} --"
                         for k in ("radar", "image", "imu") if k != "imu" or self.topics.get("imu"))
        err = f" | ERROR {self.error}" if self.error else ""
        return (f"rx age: {ages} | camera stamp lag {self.cam_lag:.2f}s | scans {self.n_in}, processed {self.n_done},"
                f" dropped {self.n_drop}{err}")
