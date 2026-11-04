"""ROS 2 bag (mcap / sqlite3) reader without a ROS install (uses `rosbags`).

Yields (image_bgr, RadarFrame, yaw_rate) per radar frame, paired with the nearest camera frame.
Bosch off-highway premium radar: PointCloud2 fields x,y,z (sensor frame), radial_velocity
(+ = away), radar_cross_section, signal_noise_ratio, ...
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

_MSG_DIR = Path(__file__).resolve().parents[3] / "ros" / "rcmot_msgs" / "msg"
_TYPESTORE = None


def rcmot_typestore():
    """rosbags typestore = ROS 2 Humble types + rcmot_msgs (ros/rcmot_msgs/msg, docs/20), so bags recorded by the
    HDX GUI with /rcmot/* topics decode on a PC without ROS. Built once."""
    global _TYPESTORE
    if _TYPESTORE is None:
        ts = get_typestore(Stores.ROS2_HUMBLE)
        types = {}
        for f in sorted(_MSG_DIR.glob("*.msg")) if _MSG_DIR.is_dir() else []:
            types.update(get_types_from_msg(f.read_text(), f"rcmot_msgs/msg/{f.stem}"))
        if types:
            ts.register(types)
        _TYPESTORE = ts
    return _TYPESTORE


def open_bag(bag) -> AnyReader:
    return AnyReader([Path(bag)], default_typestore=rcmot_typestore())

from rcmot.types import RadarFrame
from rcmot.utils.calib import RigCalib

_PF = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 8: "f8"}


# other radars' field names -> the Bosch names the pipeline uses (Continental ARS548 / PDK: docs/16 §10)
_ALIASES = {"range_rate": "radial_velocity", "rcs": "radar_cross_section", "doppler": "radial_velocity",
            "velocity": "radial_velocity", "snr": "signal_noise_ratio"}


def pc2_to_struct(msg) -> np.ndarray:
    """PointCloud2 -> structured array with the Bosch field names (aliases of other radars renamed; a radar
    without an SNR channel gets signal_noise_ratio = RCS so the per-scan statistics still work)."""
    names = [f.name for f in msg.fields]
    dt = np.dtype({"names": names,
                   "formats": [("<" if not msg.is_bigendian else ">") + _PF[f.datatype] for f in msg.fields],
                   "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dt, count=msg.width * msg.height)
    ren = {n: _ALIASES[n] for n in names if n in _ALIASES and _ALIASES[n] not in names}
    if ren or ("radial_velocity" in names and "signal_noise_ratio" not in names and "radar_cross_section" in names) \
            or ("rcs" in names and "signal_noise_ratio" not in names):
        a = _rename(a, ren)
    return a


def _rename(a: np.ndarray, ren: dict) -> np.ndarray:
    names = [ren.get(n, n) for n in a.dtype.names]
    out = np.empty(len(a), dtype=[(n, a.dtype.fields[o][0]) for n, o in zip(names, a.dtype.names)]
                   + ([("signal_noise_ratio", "f4")] if "signal_noise_ratio" not in names else []))
    for n, o in zip(names, a.dtype.names):
        out[n] = a[o]
    if "signal_noise_ratio" not in names:
        out["signal_noise_ratio"] = out["radar_cross_section"]
    return out


def _stamp(msg) -> float:
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


def image_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage or sensor_msgs/Image (bgr8 / rgb8 / mono8 / bayer) -> BGR."""
    if hasattr(msg, "format"):                                   # CompressedImage
        return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
    enc = msg.encoding.lower()
    ch = 1 if enc in ("mono8",) or enc.startswith("bayer") else 3
    img = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch) if ch == 3 else img.reshape(msg.height, msg.width)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "mono8":
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc.startswith("bayer"):
        code = {"bayer_rggb8": cv2.COLOR_BayerRG2BGR, "bayer_bggr8": cv2.COLOR_BayerBG2BGR,
                "bayer_gbrg8": cv2.COLOR_BayerGB2BGR, "bayer_grbg8": cv2.COLOR_BayerGR2BGR}[enc]
        return cv2.cvtColor(img, code)
    return img.copy()                                            # bgr8


def radar_frame(arr: np.ndarray, stamp: float, calib: RigCalib) -> RadarFrame:
    xyz_r = np.stack([arr["x"], arr["y"], arr["z"]], 1).astype(float)
    ok = np.isfinite(xyz_r).all(1) & (np.linalg.norm(xyz_r, axis=1) > 0.5)
    xyz_r = xyz_r[ok]
    g = lambda n: arr[n][ok].astype(float) if n in arr.dtype.names else np.zeros(ok.sum())
    rf = RadarFrame(stamp, calib.radar_to_level(xyz_r), g("radial_velocity"), g("radar_cross_section"),
                    g("signal_noise_ratio"), xyz_r)
    if "idx_azimuth_ambiguity_peer" in arr.dtype.names:              # peer indices refer to the unfiltered scan
        remap = np.full(len(arr), -1)
        remap[np.flatnonzero(ok)] = np.arange(ok.sum())
        p = arr["idx_azimuth_ambiguity_peer"][ok].astype(int)
        rf.peer = np.where((p >= 0) & (p < len(arr)), remap[np.clip(p, 0, len(arr) - 1)], -1)
    if "azimuth_angle_probability" in arr.dtype.names:
        rf.az_prob = g("azimuth_angle_probability")
    return rf


def yaw_rate_at(gyro_t, gyro_z, t: float, half: float = 0.05) -> float:
    """Mean gyro z of the samples within +-half s of t (the scan time). The frame is emitted only after a later
    image arrives (0.3-1.3 s later), so the samples around t are available; the last-10-samples rule sampled the
    yaw rate up to 1.4 s late and made standing objects swing at 1-2 m/s on swing ramps (docs/16 §10)."""
    if not gyro_t:
        return 0.0
    gt = np.asarray(gyro_t[-400:])
    gz = np.asarray(gyro_z[-400:])
    m = np.abs(gt - t) <= half
    if m.any():
        return float(gz[m].mean())
    k = np.argsort(np.abs(gt - t))[:10]                            # no sample in the window: nearest ones
    return float(gz[k].mean())


def camera_info_topic(image_topic: str) -> str:
    return image_topic.replace("/image_raw/compressedDepth", "/camera_info").replace("/image_raw/compressed", "/camera_info") \
        .replace("/image_raw", "/camera_info")


def depth_to_m(msg) -> np.ndarray:
    """sensor_msgs/CompressedImage '16UC1; compressedDepth' (12-byte header + PNG, mm) or Image 16UC1 / 32FC1 -> metres."""
    if hasattr(msg, "format"):
        raw = bytes(msg.data)
        png = raw[raw.find(b"\x89PNG"):]
        d = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / 1000.0
    if msg.encoding.lower() == "32fc1":
        return np.frombuffer(bytes(msg.data), np.float32).reshape(msg.height, msg.width).copy()
    return np.frombuffer(bytes(msg.data), np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0


def first_camera_k(r, conns: dict, topic: str):
    """K (3x3) of the first CameraInfo on `topic` in an open reader, or None."""
    c = conns.get(topic)
    if c is None:
        return None
    for conn, _, raw in r.messages(connections=[c]):
        return np.array(r.deserialize(raw, conn.msgtype).k, float).reshape(3, 3)
    return None


def apply_camera_info(calib: RigCalib, width: int, height: int, K, D) -> bool:
    """Take K / D / image size from the driver's CameraInfo when the camera runs at another resolution or
    intrinsics than the calib file (OAK day mode 2016x1512 vs the 1280x720 calibration; docs/18 §2). The
    extrinsic (level -> camera) is kept. Returns True when the calib was changed."""
    if width == 0:
        return False
    K = np.array(K, float).reshape(3, 3)
    if (int(width), int(height)) != tuple(calib.image_size) or np.abs(K - calib.K).max() > 1.0:
        calib.K, calib.D = K, np.array(D, float)
        calib.image_size = (int(width), int(height))
        return True
    return False


def bag_camera_info(bag: str | Path, image_topic: str, calib: RigCalib) -> bool:
    """Apply the bag's first CameraInfo (if recorded) to calib; see apply_camera_info."""
    with open_bag(bag) as r:
        conns = {c.topic: c for c in r.connections}
        ci = conns.get(camera_info_topic(image_topic))
        if ci is None:
            return False
        for conn, _, raw in r.messages(connections=[ci]):
            m = r.deserialize(raw, conn.msgtype)
            return apply_camera_info(calib, m.width, m.height, m.k, m.d)
    return False


SENSOR_CLOCK_GAP_S = 1e6                                          # |header - receive| above this: the header is a sensor clock


def _cdr_header_stamp(raw: bytes) -> float:
    """header.stamp of a CDR-serialised ROS 2 message whose first field is std_msgs/Header (4-byte encapsulation
    header, then int32 sec + uint32 nanosec); no deserialisation of the point cloud."""
    end = ">" if raw[1] == 0 else "<"                               # CDR representation id 0x0000 = big endian
    sec, nsec = struct.unpack_from(end + "iI", raw, 4)
    return sec + nsec * 1e-9


def _cache_dir() -> Path:
    return Path(os.environ.get("RCMOT_CACHE", Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rcmot")) / "radar_clock"


def _bag_signature(bag: Path) -> str:
    """Name / size / mtime of the bag files, so a re-recorded bag at the same path invalidates the cache."""
    files = sorted(bag.iterdir()) if bag.is_dir() else [bag]
    items = [(f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files if f.is_file()]
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:16]


def radar_clock_offset(bag: str | Path, radar_topic: str, *, max_msgs: int = 50000, cache: bool = True) -> dict:
    """Header-clock -> bag-clock offset of a radar topic, estimated ONCE from the whole bag and cached.

    Continental ARS548 headers carry the sensor's own clock (~1.76e9 s off the bag clock), so a scan's bag time is
    header + offset where offset = 5th percentile of (bag receive - header) over ALL radar messages of the bag (the
    receive delay floor; p5 absorbs the receive jitter, p95-p5 up to 0.1 s, docs/16d). Measured on the conti side
    bag: windowed p5 drifts 279-305 ms over the bag, the first 30 messages are biased +60 ms by the recorder's
    start-up backlog, and the old "p5 of the first 30 scans after the read start" moved by 10-40 ms with --start
    (changing the image pairing between runs). Reading every radar header costs ~1 min on an external disk, so the
    result is cached in $RCMOT_CACHE/radar_clock/ (default ~/.cache/rcmot/radar_clock/) per bag path + topic + bag
    file signature. Returns {sensor_clock, offset_s, n, p5, p50, min, ...}; sensor_clock False (epoch headers,
    Bosch) still gives the receive-delay floor but iterate_bag does not apply it then."""
    bag = Path(bag).resolve()
    sig = _bag_signature(bag)
    key = hashlib.sha1(f"{bag}|{radar_topic}".encode()).hexdigest()[:16]
    cpath = _cache_dir() / f"{key}.json"
    if cache and cpath.exists():
        try:
            rec = json.load(open(cpath))
            if rec.get("signature") == sig and rec.get("topic") == radar_topic and rec.get("n", 0) > 0:
                return rec
        except (OSError, ValueError):
            pass
    delays = []
    with open_bag(bag) as r:
        conns = {c.topic: c for c in r.connections}
        if radar_topic not in conns:
            raise KeyError(f"topic not in bag: {radar_topic}; available: {sorted(conns)}")
        checked = False
        for conn, t_ns, raw in r.messages(connections=[conns[radar_topic]]):
            hs = _cdr_header_stamp(raw)
            if not checked:                                        # the raw parse must agree with the typestore
                checked = True
                if abs(_stamp(r.deserialize(raw, conn.msgtype)) - hs) > 1e-6:
                    raise ValueError(f"{radar_topic}: message does not start with std_msgs/Header")
            delays.append(t_ns * 1e-9 - hs)
            if len(delays) >= max_msgs:
                break
    if not delays:
        raise ValueError(f"no messages on {radar_topic} in {bag}")
    d = np.asarray(delays)
    rec = {"bag": str(bag), "topic": radar_topic, "signature": sig, "n": int(len(d)),
           "sensor_clock": bool(abs(float(np.median(d))) > SENSOR_CLOCK_GAP_S),
           "offset_s": float(np.percentile(d, 5)), "p5": float(np.percentile(d, 5)), "p50": float(np.median(d)),
           "min": float(d.min()), "p95": float(np.percentile(d, 95)), "rule": "p5 of bag receive - header, whole bag"}
    if cache:
        try:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            tmp = cpath.with_suffix(".tmp")
            json.dump(rec, open(tmp, "w"), indent=1)
            tmp.replace(cpath)
        except OSError:
            pass
    return rec


def iterate_bag(bag: str | Path, calib: RigCalib, radar_topic: str, image_topic: str, imu_topic: str | None = None,
                *, max_dt: float = 0.06, start: float = 0.0, duration: float | None = None,
                radar_stamp: str = "auto", radar_time_offset_s: float = 0.0, use_camera_info: bool = True,
                depth_topic: str | None = None, radar_clock_offset_s: float | None = None) -> Iterator[tuple[np.ndarray, RadarFrame, float]]:
    """radar_stamp: "auto" (default; sensor-clock headers detected), "header" (epoch header stamps, Bosch) or "bag" (receive time; radars whose header is the
    sensor's own clock, e.g. Continental ARS548), shifted by radar_time_offset_s (sensor latency, negative).
    Sensor-clock headers are mapped to the bag clock with one offset per bag (radar_clock_offset, cached; the same
    value whatever `start` is), or with radar_clock_offset_s when given (e.g. the value a GT segment was extracted
    with, meta.radar_clock_offset_s, to reproduce its image pairing exactly); rf.clock_offset_s carries the value used.
    use_camera_info: intrinsics from the bag's camera_info when it differs from the calib (AGX recordings).
    depth_topic: stereo depth aligned to the RGB optical frame (OAK compressedDepth); the depth image nearest to the
    paired camera image is attached as rf.depth = (metres (h, w), K 3x3) for the work monitor (docs/24)."""
    if use_camera_info:
        bag_camera_info(bag, image_topic, calib)
    with open_bag(bag) as r:
        conns = {c.topic: c for c in r.connections}
        missing = [t for t in (radar_topic, image_topic) if t not in conns]
        if missing:
            raise KeyError(f"topics not in bag: {missing}; available: {sorted(conns)}")
        want = [conns[radar_topic], conns[image_topic]] + ([conns[imu_topic]] if imu_topic in conns else [])
        Kd = first_camera_k(r, conns, camera_info_topic(depth_topic)) if depth_topic in conns else None
        if Kd is not None:
            want.append(conns[depth_topic])
        depths: list[tuple[float, object]] = []
        t0 = r.start_time * 1e-9 + start
        t_end = t0 + duration if duration else np.inf
        lead = 0.5                                                 # images just before t0 to pair the first scans
        imgs: list[tuple[float, object]] = []                      # (stamp, image msg), decoded lazily
        gyro_t, gyro_z = [], []                                    # bag receive times (the IMU header may be a sensor clock)
        pending: list[tuple[float, np.ndarray]] = []
        clock_off = radar_clock_offset_s                           # header -> bag clock (sensor-clock radars), fixed per bag
        tau = calib.radar_time_offset_s + radar_time_offset_s
        for conn, t_ns, raw in r.messages(connections=want, start=int((t0 - lead) * 1e9)):
            t = t_ns * 1e-9
            if t < t0 and conn.topic != image_topic:
                continue
            if t > t_end:
                break
            msg = r.deserialize(raw, conn.msgtype)
            if conn.topic == image_topic:
                imgs.append((_stamp(msg) - calib.camera_time_offset_s, msg))
                imgs = imgs[-30:]
            elif conn.topic == depth_topic:
                depths.append((_stamp(msg) - calib.camera_time_offset_s, msg))
                depths = depths[-10:]
            elif conn.topic == imu_topic:
                gyro_t.append(t)
                gyro_z.append(msg.angular_velocity.z)
                if len(gyro_t) > 400:
                    del gyro_t[:200], gyro_z[:200]
            else:
                hs = _stamp(msg)
                if radar_stamp == "auto":                      # sensor-clock header (Continental: ~3600 s) vs epoch
                    radar_stamp = "bag" if abs(hs - t) > SENSOR_CLOCK_GAP_S else "header"
                if radar_stamp == "bag":                       # receive time jitters (p95 0.15 s): keep the header
                    if clock_off is None:                      # clock's spacing, shift it by the bag's fixed offset
                        clock_off = radar_clock_offset(bag, radar_topic)["offset_s"]
                    rs = hs + clock_off + tau
                else:
                    rs = hs + tau
                pending.append((rs, pc2_to_struct(msg), t))
            # emit radar frames once a later image exists (so the nearest image is known)
            while pending and imgs and imgs[-1][0] >= pending[0][0]:
                rs, arr, bag_t = pending.pop(0)
                ts = [s for s, _ in imgs]
                k = bisect.bisect_left(ts, rs)
                cand = [i for i in (k - 1, k) if 0 <= i < len(ts)]
                i = min(cand, key=lambda i: abs(ts[i] - rs))
                if abs(ts[i] - rs) > max_dt:
                    continue
                img = image_to_bgr(imgs[i][1])
                yaw = yaw_rate_at(gyro_t, gyro_z, rs)              # at the scan time, not at emission (docs/16 §10)
                rf = radar_frame(arr, rs, calib)
                rf.bag_t = bag_t                                   # bag receive time (timeline / seeking)
                rf.img_dt = ts[i] - rs                             # image time - scan time (docs/21 E2-c)
                rf.clock_offset_s = clock_off                      # None: epoch headers (Bosch), no mapping applied
                if depths:
                    sd, dm = min(depths, key=lambda x: abs(x[0] - ts[i]))
                    if abs(sd - ts[i]) <= max_dt:
                        rf.depth = (depth_to_m(dm), Kd)
                yield img, rf, yaw


def bag_span(bag: str | Path) -> tuple[float, float]:
    """(start, end) bag time in seconds."""
    with open_bag(bag) as r:
        return r.start_time * 1e-9, r.end_time * 1e-9
