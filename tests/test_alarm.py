"""Alarm zone levels, hysteresis and the camera-confirmed-only filter (silent: sound files do not exist)."""
from types import SimpleNamespace

import numpy as np

from rcmot.alarm import Alarm
from rcmot.types import TrackState

SOUNDS = {"danger": "/nonexistent/d.wav", "warning": "/nonexistent/w.wav", "caution": "/nonexistent/c.wav"}


def mk(tid, r, state=TrackState.CONFIRMED, tsu=0.0, tcam=None):
    t = SimpleNamespace(tid=tid, pos=np.array([r, 0.0, 0.0]), state=state, time_since_update=tsu)
    if tcam is not None:
        t.time_since_camera = tcam
    return t


def alarm(**kw):
    a = Alarm(zones=(3, 5, 10), sounds=SOUNDS, **kw)
    a.set_enabled(True)
    return a


def test_zones():
    for r, lvl in [(3.0, 3), (3.01, 2), (5.0, 2), (5.01, 1), (10.0, 1), (10.01, 0)]:
        a = alarm(hysteresis=0.0)
        assert a.update([mk(1, r)]) == lvl, r
        a.stop()


def test_hysteresis():
    a = alarm(hysteresis=1.0)
    assert a.update([mk(1, 2.9)]) == 3
    assert a.update([mk(1, 3.5)]) == 3
    assert a.update([mk(1, 4.1)]) == 2
    a.stop()


def test_confirmed_only():
    a = alarm(hysteresis=0.0, confirmed_only=True)
    assert a.update([mk(1, 2.0, TrackState.RADAR_ONLY), mk(2, 4.0)]) == 2
    assert a.nearest_tid == 2
    a.set_enabled(False); a.update([]); a.set_enabled(True)
    assert a.update([mk(3, 2.0, TrackState.LOST, 0.2)]) == 3
    a.set_enabled(False); a.update([]); a.set_enabled(True)
    assert a.update([mk(3, 2.0, TrackState.LOST, 1.0)]) == 0
    assert a.nearest is None
    a.stop()


def test_radar_only_grace():
    """docs/38 §10.4: a radar-only track counts for radar_only_grace_s after the camera last saw it."""
    a = alarm(hysteresis=0.0, confirmed_only=True, radar_only_grace_s=1.0)
    assert a.update([mk(1, 2.0, TrackState.RADAR_ONLY, tcam=0.6), mk(2, 4.0)]) == 3
    a.set_enabled(False); a.update([]); a.set_enabled(True)
    assert a.update([mk(1, 2.0, TrackState.RADAR_ONLY, tcam=1.4), mk(2, 4.0)]) == 2
    a.stop()


def test_default_counting_rule():
    """Default (since 11-04) = the counting rule the alarm results were scored with (docs/38 §10.4): a radar-only
    track counts only within 1 s of the camera; confirmed_only=False restores 'every track counts'."""
    a = alarm(hysteresis=0.0)
    assert a.update([mk(1, 2.0, TrackState.RADAR_ONLY, tcam=0.5)]) == 3
    a.stop()
    a = alarm(hysteresis=0.0)
    assert a.update([mk(1, 2.0, TrackState.RADAR_ONLY, tcam=2.0)]) == 0
    a.stop()
    a = alarm(hysteresis=0.0, confirmed_only=False)
    assert a.update([mk(1, 2.0, TrackState.RADAR_ONLY)]) == 3
    a.stop()


def test_decide_is_stateful_and_play_is_separate():
    """docs/20: decide() per processed frame (hysteresis state), play() per displayed frame (sound only)."""
    a = alarm(hysteresis=1.0)
    st = a.decide([mk(7, 2.9)])
    assert (st.level, st.nearest_tid, st.nearest_range, st.zones) == (3, 7, 2.9, (3.0, 5.0, 10.0))
    assert st.sound_enabled and st.confirmed_only               # counting rule on by default (docs/38 §10.4)
    assert a.decide([mk(7, 3.5)]).level == 3            # hysteresis keeps danger
    assert a._want == 0                                 # nothing played yet
    a.play(3)
    assert a._want == 3
    a.set_enabled(False)
    a.play(3)
    assert a._want == 0                                 # disabled: silent, but decisions continue
    assert a.decide([mk(7, 3.5)]).level == 3
    a.reset()
    assert a.decide([mk(7, 3.5)]).level == 2            # hysteresis state forgotten (seek)
    assert a.decide([]).nearest_tid is None
    a.stop()


def test_zone_of():
    from rcmot.alarm import zone_of
    assert [zone_of(r, (3.0, 5.0, 10.0)) for r in (None, 1.0, 3.0, 4.0, 9.9, 10.1)] == [0, 3, 3, 2, 1, 0]
