import numpy as np

from rcmot.tracking.tracker import RadarCameraTracker, TrackerParams
from rcmot.types import Det3D, TrackState


def _det(p, v=None):
    return Det3D("worker", 0.9, np.array([0, 0, 50, 100.0]), np.asarray(p, float), np.eye(3) * 0.09,
                 vel=None if v is None else np.asarray(v, float), vel_cov=None if v is None else np.eye(3) * 0.04)


def test_absolute_velocity_with_moving_ego():
    """Ego drives +x at 2 m/s; a worker walks +y at 1 m/s. Track velocity must be absolute (0, 1)."""
    trk = RadarCameraTracker(TrackerParams(min_hits=2))
    dt, v_ego, v_obj = 0.066, np.array([2.0, 0, 0]), np.array([0, 1.0, 0])
    p_world = np.array([20.0, -3.0, 0.0])
    ego = np.zeros(3)
    for k in range(60):
        rel = p_world + v_obj * k * dt - ego                      # position in the current base frame
        out = trk.update([_det(rel)], dt, ego_dp=v_ego * dt if k else None)
        ego = ego + v_ego * dt
    assert len(out) == 1 and out[0].state == TrackState.CONFIRMED
    assert np.allclose(out[0].vel[:2], v_obj[:2], atol=0.15)


def test_track_survives_missed_detections():
    trk = RadarCameraTracker(TrackerParams(min_hits=2, max_lost_s=1.0))
    for k in range(10):
        out = trk.update([_det([10 + 0.1 * k, 0, 0], [1.5, 0, 0])], 0.066)
    tid = out[0].tid
    for _ in range(10):                                           # 0.66 s occlusion
        out = trk.update([], 0.066)
    assert out and out[0].state == TrackState.LOST
    out = trk.update([_det([10 + 0.1 * 20, 0, 0])], 0.066)
    assert out[0].tid == tid and out[0].state == TrackState.CONFIRMED


def test_radar_only_drift_is_reacquired_not_reborn():
    """S1/S2 (opt-in): a track kept alive by radar-only returns 2 m beside the person must hand its ID back to the
    camera detection of that person instead of a new ID being born (docs/12)."""
    trk = RadarCameraTracker(TrackerParams(min_hits=2, reacq=True, ro_floor0=0.3,
                                           ro_floor_rate={"worker": 0.6, "default": 0.6},
                                           ro_extra_sigma={"worker": 0.4, "default": 0.6}))
    for k in range(10):
        out = trk.update([_det([10.0, 0.0, 0.0])], 0.066)
    tid = out[0].tid
    t = trk.tracks[0]
    for _ in range(15):                                           # 1 s of radar-only on someone else's returns
        d = _det([10.0, 2.0, 0.0])
        d.source = "radar_only"
        d.pos_cov = np.eye(3) * 0.01
        trk.tracks[0].kf.predict(0.066)
        t.tsu += 0.066
        t.t_cam += 0.066
        t.update(d, trk.p, trk.t)
    out = trk.update([_det([10.0, 0.0, 0.0])], 0.066)
    for _ in range(2):
        out = trk.update([_det([10.0, 0.0, 0.0])], 0.066)
    assert [o.tid for o in out] == [tid] and out[0].state == TrackState.CONFIRMED
    assert abs(out[0].pos[1]) < 0.5


def test_display_ids_are_consecutive_despite_unconfirmed_births():
    trk = RadarCameraTracker(TrackerParams(min_hits=3))
    for k in range(4):                                            # one-frame false positives far away
        trk.update([_det([30.0 + 5 * k, 10.0, 0.0])], 0.066)
    for _ in range(4):
        out = trk.update([_det([10.0, 0.0, 0.0])], 0.066)
    first = out[0].tid
    for _ in range(4):
        out = trk.update([_det([10.0, 0.0, 0.0]), _det([5.0, -5.0, 0.0])], 0.066)
    assert sorted(o.tid for o in out) == [first, first + 1]


def test_standstill_zeroes_jittery_velocity_and_releases_on_motion():
    """docs/16: a standing worker whose measured position jitters by ~0.3 m shows ~0 velocity (static mode);
    once it really walks (radial Doppler + displacement) the mode is left and the velocity follows."""
    rng = np.random.default_rng(0)
    trk = RadarCameraTracker(TrackerParams(static_mode=True))
    out = None
    for k in range(40):                                       # 2.6 s standing at (10, 0), 0.3 m jitter, vr ~ 0
        d = _det(np.r_[10.0, 0.0, 0.9] + np.r_[rng.normal(0, 0.3, 2), 0.0])
        d.vr, d.vr_u, d.vr_var = float(rng.normal(0, 0.05)), np.array([1.0, 0.0, 0.0]), 0.05 ** 2
        out = trk.update([d], 0.066)
    assert out[0].static and np.linalg.norm(out[0].vel[:2]) < 0.1
    for k in range(30):                                       # walks away at 1.2 m/s radially
        x = 10.0 + 1.2 * 0.066 * (k + 1)
        d = _det(np.r_[x, 0.0, 0.9] + np.r_[rng.normal(0, 0.1, 2), 0.0])
        d.vr, d.vr_u, d.vr_var = 1.2 + float(rng.normal(0, 0.05)), np.array([1.0, 0.0, 0.0]), 0.05 ** 2
        out = trk.update([d], 0.066)
    assert not out[0].static and abs(np.linalg.norm(out[0].vel[:2]) - 1.2) < 0.4
