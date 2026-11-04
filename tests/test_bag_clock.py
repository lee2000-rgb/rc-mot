"""Sensor-clock radar anchoring in rcmot.sensors.bag_source (docs/13 "프레임 대응"): the header -> bag clock offset
is one value per bag (p5 of the receive delay over the whole bag, cached), so the scan stamps - and with them the
image pairing - do not depend on where the read starts. Bosch-style epoch headers get no mapping."""
import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from rcmot.sensors import bag_source as bs
from rcmot.utils.calib import RigCalib

R_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)       # level -> camera optical
CAL = RigCalib(np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float), np.zeros(5), (8, 8),
               np.eye(3), np.zeros(3), R_OPT, -R_OPT @ np.array([0, 0, 1.5]), 0.0, -0.05)
T0 = 1.7e9                                                           # bag clock (epoch)
SENSOR0 = 4000.0                                                     # Continental-style header clock
N_RADAR, N_IMG = 200, 100                                            # 10 s at 20 Hz / 10 Hz


def _delays():
    """Receive delay per radar message: start-up backlog on the first 30, then 0.30 s + deterministic jitter."""
    rng = np.random.RandomState(0)
    d = 0.30 + np.abs(rng.standard_normal(N_RADAR)) * 0.02
    d[:30] += 0.06
    return d


def _write_bag(path, sensor_clock=True):
    ts = get_typestore(Stores.ROS2_HUMBLE)
    Header, Time = ts.types["std_msgs/msg/Header"], ts.types["builtin_interfaces/msg/Time"]
    PC2, PF, Img = ts.types["sensor_msgs/msg/PointCloud2"], ts.types["sensor_msgs/msg/PointField"], ts.types["sensor_msgs/msg/Image"]

    def tm(t):
        return Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))

    delays = _delays()
    with Writer(path, version=8) as w:
        cr = w.add_connection("/radar", PC2.__msgtype__, typestore=ts)
        ci = w.add_connection("/img", Img.__msgtype__, typestore=ts)
        fields = [PF(name=n, offset=4 * i, datatype=7, count=1)
                  for i, n in enumerate(("x", "y", "z", "radial_velocity", "radar_cross_section", "signal_noise_ratio"))]
        pt = np.array([10.0, 0.0, 0.0, 0.0, 5.0, 20.0], np.float32).tobytes()
        for k in range(N_RADAR):
            t_bag = T0 + 0.05 * k + delays[k]
            hs = (SENSOR0 if sensor_clock else T0) + 0.05 * k
            msg = PC2(header=Header(stamp=tm(hs), frame_id="radar"), height=1, width=1, fields=fields,
                      is_bigendian=False, point_step=24, row_step=24, data=np.frombuffer(pt, np.uint8), is_dense=True)
            w.write(cr, int(round(t_bag * 1e9)), ts.serialize_cdr(msg, PC2.__msgtype__))
        for k in range(N_IMG):
            hs = T0 + 0.1 * k + 0.013                                # epoch header, received 0.5 s later
            msg = Img(header=Header(stamp=tm(hs), frame_id="cam"), height=8, width=8, encoding="bgr8",
                      is_bigendian=False, step=24, data=np.zeros(8 * 24, np.uint8))
            w.write(ci, int(round((hs + 0.5) * 1e9)), ts.serialize_cdr(msg, Img.__msgtype__))
    return delays


@pytest.fixture
def bag(tmp_path, monkeypatch):
    monkeypatch.setenv("RCMOT_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "bag_conti"
    delays = _write_bag(path)
    return path, delays


def test_offset_is_whole_bag_p5_and_cached(bag, monkeypatch):
    path, delays = bag
    rec = bs.radar_clock_offset(path, "/radar")
    assert rec["sensor_clock"] and rec["n"] == N_RADAR
    assert abs(rec["offset_s"] - (T0 - SENSOR0 + np.percentile(delays, 5))) < 1e-6
    assert rec["offset_s"] < T0 - SENSOR0 + np.percentile(delays[:30], 5) - 0.03   # not the start-up window
    monkeypatch.setattr(bs, "open_bag", lambda *a, **k: (_ for _ in ()).throw(AssertionError("bag re-read")))
    assert bs.radar_clock_offset(path, "/radar") == rec                          # second call: cache only


def test_stamps_independent_of_read_start(bag):
    path, delays = bag
    runs = {}
    for start in (0.0, 2.3, 5.0):
        runs[start] = {round(rf.bag_t, 6): (rf.stamp, rf.clock_offset_s, rf.img_dt)
                       for _, rf, _ in bs.iterate_bag(path, CAL, "/radar", "/img", use_camera_info=False, start=start)}
    assert len(runs[0.0]) > 150
    rec = bs.radar_clock_offset(path, "/radar")
    for start, frames in runs.items():
        for bt, (st, off, _) in frames.items():
            assert off == rec["offset_s"]
            assert abs(st - runs[0.0][bt][0]) < 1e-9, f"start {start}: stamp differs for bag_t {bt}"
            assert abs(runs[0.0][bt][2] - frames[bt][2]) < 1e-9                   # same image paired
    # the stamp is header + offset + radar_time_offset_s
    _, (st0, _, _) = min(runs[0.0].items())
    rem = (st0 - CAL.radar_time_offset_s - rec["offset_s"] - SENSOR0) % 0.05
    assert min(rem, 0.05 - rem) < 1e-6


def test_explicit_offset_overrides_estimate(bag):
    path, _ = bag
    rec = bs.radar_clock_offset(path, "/radar")
    a = {round(rf.bag_t, 6): rf.stamp for _, rf, _ in bs.iterate_bag(path, CAL, "/radar", "/img", use_camera_info=False)}
    b = {round(rf.bag_t, 6): rf.stamp for _, rf, _ in
         bs.iterate_bag(path, CAL, "/radar", "/img", use_camera_info=False, radar_clock_offset_s=rec["offset_s"] + 0.02)}
    common = set(a) & set(b)
    assert common and all(abs(b[k] - a[k] - 0.02) < 1e-6 for k in common)   # 1.7e9 s: double eps ~2e-7


def test_epoch_headers_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("RCMOT_CACHE", str(tmp_path / "cache"))
    path = tmp_path / "bag_bosch"
    _write_bag(path, sensor_clock=False)
    assert not bs.radar_clock_offset(path, "/radar")["sensor_clock"]
    frames = list(bs.iterate_bag(path, CAL, "/radar", "/img", use_camera_info=False, start=1.0))
    assert frames
    for _, rf, _ in frames:
        assert rf.clock_offset_s is None
        rem = (rf.stamp - CAL.radar_time_offset_s - T0) % 0.05
        assert min(rem, 0.05 - rem) < 1e-6                                       # header stamp + tau, no offset
