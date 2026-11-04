"""ResultPublisher (docs/20) -- runs only where rclpy + rcmot_msgs are installed (AGX); skipped elsewhere."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("rclpy")
pytest.importorskip("rcmot_msgs")

from rcmot.ros.publisher import ResultPublisher, quat_from_matrix  # noqa: E402
from rcmot.types import AlarmState, TrackState  # noqa: E402


def _calib():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return SimpleNamespace(R_lr=R, t_lr=np.array([0.1, 0.0, -0.15]))


def _res(stamp=1700000000.25):
    tr = SimpleNamespace(tid=4, cls="worker", state=TrackState.CONFIRMED, pos=np.array([3.0, 4.0, 0.0]),
                         vel=np.array([0.5, 0.0, 0.0]), box2d=np.array([10.2, 20.0, 30.0, 40.0]), score=0.9, hits=5,
                         time_since_update=0.0, pos_cov=np.eye(3) * 0.04, age=1.5, source="radar", n_radar=7,
                         exist=0.8, static=True)
    return SimpleNamespace(stamp=stamp, tracks=[tr], ego_vel=np.zeros(3), ego_valid=True,
                           timing_ms={"detector": 19.0, "total": 60.0},
                           alarm=AlarmState(2, 4, 5.0, (3.0, 5.0, 10.0), True, True))


def test_quat_roundtrip():
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    x, y, z, w = quat_from_matrix(R)
    assert math.isclose(x * x + y * y + z * z + w * w, 1.0, abs_tol=1e-9)
    assert math.isclose(abs(w), math.cos(math.pi / 4), abs_tol=1e-9) and math.isclose(abs(z), math.sin(math.pi / 4), abs_tol=1e-9)


def test_messages_and_enable_toggle():
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    pub = ResultPublisher(_calib(), meta={"profile": "p", "detector": "d.engine", "calib": "c.json"},
                          namespace="/rcmot_test", markers=True, status_hz=1000.0,
                          stats_fn=lambda: {"cam_lag_s": 0.9, "frames_dropped": 3})
    try:
        from rcmot.ros.publisher import _stamp
        res = _res()
        tr = pub._tracks_msg(res, _stamp(res.stamp))
        assert (tr.header.stamp.sec, tr.header.frame_id) == (1700000000, "base_link_level")
        t = tr.tracks[0]
        assert (t.id, t.cls, t.state, t.is_static, t.source, t.n_radar) == (4, "worker", 1, True, "radar", 7)
        assert math.isclose(t.range, 5.0) and t.zone == 2 and list(t.box2d) == [10, 20, 30, 40]
        assert math.isclose(t.position_covariance[0], 0.04) and tr.ego_valid
        al = pub._alarm_msg(res, tr.header.stamp)
        assert (al.level, al.nearest_id, al.sound_enabled, al.confirmed_only) == (2, 4, True, True)
        assert math.isclose(al.nearest_range, 5.0) and list(al.zones) == [3.0, 5.0, 10.0]
        det = pub._detections_msg(res, tr.header.stamp)
        assert det.detections[0].id == "4" and det.detections[0].results[0].hypothesis.class_id == "worker"
        assert math.isclose(det.detections[0].bbox.size.z, 1.7)
        mk = pub._markers_msg(res, tr.header.stamp)
        assert len(mk.markers) == 3                      # DELETEALL + cube + text
        st = pub._status_msg(res, tr.header.stamp)
        assert (st.profile, st.detector, st.calib, st.frames_dropped) == ("p", "d.engine", "c.json", 3)
        assert math.isclose(st.cam_lag_s, 0.9, abs_tol=1e-6)

        assert pub.publish(res, "radar") and pub.n_published == 1 and pub._tf_sent
        pub.set_enabled(False)
        assert not pub.publish(res) and pub.n_published == 1
        assert pub._status_msg(res, tr.header.stamp).publishing is False and pub._status_msg(res, tr.header.stamp).fps == 0.0
        res2 = _res(); res2.alarm = None
        assert pub._tracks_msg(res2, tr.header.stamp).tracks[0].zone == 0      # no decision -> no zone claimed
        assert list(pub._alarm_msg(res2, tr.header.stamp).zones) == [0.0, 0.0, 0.0]
        pub.set_enabled(True)
        assert pub.publish(res) and pub.n_published == 2
        s = pub._status_msg(res, tr.header.stamp)
        assert s.fps == 1.0 and math.isclose(s.det_ms, 19.0) and math.isclose(s.total_ms, 60.0)   # window cleared while off
    finally:
        pub.close()


def test_stamp_split():
    from rcmot.ros.publisher import _stamp
    s = _stamp(1700000000.25)
    assert (s.sec, s.nanosec) == (1700000000, 250000000)
