"""LiveWorker._pair without ROS (docs/21 review): the scan processed is the newest one covered by a camera frame,
and its own arrival time and the paired image stamp come back with it (yaw rate at that scan, img_dt)."""
import threading
from collections import deque

from rcmot.gui.playback import LiveWorker


def _worker():
    w = object.__new__(LiveWorker)                    # no rclpy / node: only the pairing state
    w.lock = threading.Lock()
    w.radars, w.imgs = deque(maxlen=60), deque(maxlen=30)
    w.max_dt, w.n_drop, w.cam_lag = 0.06, 0, float("nan")
    return w


def test_pair_returns_scan_arrival_and_image_stamp():
    w = _worker()
    for k, rs in enumerate((10.00, 10.066, 10.133, 10.200)):        # 15 Hz radar, arrival 20 ms after the stamp
        w.radars.append((rs, f"scan{k}", rs + 0.02))
    w.imgs.extend([(10.05, "img_a"), (10.14, "img_b")])             # the newest image covers scans 0..2
    rs, rmsg, imsg, s_img, t_rx = w._pair()
    assert rmsg == "scan2" and abs(rs - 10.133) < 1e-9
    assert imsg == "img_b" and abs(s_img - 10.14) < 1e-9
    assert abs(t_rx - 10.153) < 1e-9                                # this scan's arrival, not the newest scan's
    assert w.n_drop == 2 and [r[1] for r in w.radars] == ["scan3"]


def test_pair_none_when_no_image_close_enough():
    w = _worker()
    w.radars.append((10.0, "scan0", 10.02))
    w.imgs.append((10.2, "img"))                                    # covers the scan but 0.2 s away
    assert w._pair() is None
    w2 = _worker()
    assert w2._pair() is None                                       # empty buffers
