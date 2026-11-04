"""Worker hook of docs/20: alarm decided per processed frame, sinks called, a failing sink never stops processing."""
import threading
import time
from types import SimpleNamespace

import numpy as np

from rcmot.alarm import Alarm
from rcmot.gui.playback import BagWorker, FrameCache
from rcmot.types import TrackState


class _Pipe:
    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def step(self, img, rf, yaw):
        self.n += 1
        r = 2.0 if self.n < 3 else 20.0
        tr = SimpleNamespace(tid=1, pos=np.array([r, 0.0, 0.0]), state=TrackState.CONFIRMED, time_since_update=0.0)
        return SimpleNamespace(stamp=rf.stamp, tracks=[tr], alarm=None)


def _frames(t):
    for k in range(5):
        yield np.zeros((8, 8, 3), np.uint8), SimpleNamespace(stamp=t + k * 0.1, bag_t=t + k * 0.1), 0.0


def test_bag_worker_decides_alarm_and_calls_sinks():
    pipe, cache = _Pipe(), FrameCache(100)
    w = BagWorker(pipe, _frames, cache, ahead_s=100.0, warmup_s=0.0, t_begin=0.0)
    w.alarm = Alarm(zones=(3, 5, 10), hysteresis=0.0, sounds={"danger": "/nonexistent.wav"})
    seen, lock = [], threading.Lock()

    def bad(res):
        raise RuntimeError("boom")

    def good(res):
        with lock:
            seen.append(res.alarm.level)

    w.sinks += [bad, good]
    w.start()
    t0 = time.time()
    while not w.done and time.time() - t0 < 5:
        time.sleep(0.02)
    w.stop()
    assert w.done and len(cache) == 5
    assert seen == [3, 3, 0, 0, 0]                      # decided on the worker thread, in processing order
    assert [cache.get(i).res.alarm.level for i in range(5)] == seen
    assert "boom" in (w.sink_error or "")
    w.alarm.stop()
