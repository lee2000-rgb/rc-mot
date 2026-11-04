"""Radar-Gated Recovery on a synthetic rig: a worker walks laterally 15 m ahead and the main detector misses
it for a while (occlusion). The crop detector either sees it (recovery by re-detection) or not (radar-only)."""
import numpy as np

from rcmot.fusion.radar_camera import fuse
from rcmot.radar.ego_velocity import compensate_doppler
from rcmot.tracking.recovery import RadarGatedRecovery, RGRParams
from rcmot.tracking.tracker import RadarCameraTracker, TrackerParams
from rcmot.types import Det2D, RadarFrame, TrackState
from rcmot.utils.calib import RigCalib

R_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)      # level (x fwd, z up) -> optical
CAL = RigCalib(np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float), np.zeros(5), (1280, 720),
               np.eye(3), np.zeros(3), R_OPT, -R_OPT @ np.array([0, 0, 1.5]))
DT, V_OBJ = 0.066, np.array([0.0, 1.0, 0.0])


def _obj(k):
    return np.array([15.0, -3.0, 0.0]) + V_OBJ * k * DT


def _box(p):
    c = np.array([[p[0] + dx, p[1] + dy, z] for dx in (-.3, .3) for dy in (-.3, .3) for z in (0, 1.8)])
    uv, _ = CAL.project(c)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def _radar(k, rng, with_obj=True):
    bg = rng.uniform([5, -20, 0], [50, 20, 2], (60, 3))
    xyz = bg if not with_obj else np.vstack([bg, _obj(k) + rng.normal(0, [0.15, 0.15, 0.3], (4, 3)) + [0, 0, 0.9]])
    d = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    vr = np.zeros(len(xyz))
    if with_obj:
        vr[-4:] = d[-4:] @ V_OBJ
    rf = RadarFrame(k * DT, xyz, vr, np.zeros(len(xyz)), np.zeros(len(xyz)), xyz)
    rf.v_r_comp = compensate_doppler(xyz, vr, np.zeros(3))
    rf.static_mask = vr == 0
    return rf


class FakeDetector:
    """Stand-in for Yolo2D; `detect_batch` is replaced in `_run` to return the true worker box per crop."""


def _run(params, crop_sees, occl=(20, 45), n=60):
    rng = np.random.default_rng(0)
    img = np.zeros((720, 1280, 3), np.uint8)
    trk = RadarCameraTracker(TrackerParams(min_hits=2))
    det = FakeDetector()
    rgr = RadarGatedRecovery(det, CAL, params)
    orig_crop = rgr._crop
    state = {}

    def crop(box):                                                   # remember crop offsets for the fake detector
        c, off = orig_crop(box)
        state.setdefault("offs", []).append(off)
        return c, off
    rgr._crop = crop

    def detect_batch(crops, imgsz=640, conf=0.1):
        offs, state["offs"] = state["offs"], []
        b = _box(_obj(state["k"]))
        return [[Det2D("worker", 0.35, b - off)] if crop_sees else [] for off in offs]
    det.detect_batch = detect_batch

    tids, states = set(), []
    for k in range(n):
        state["k"] = k
        rf = _radar(k, rng)
        dets2d = [] if occl[0] <= k < occl[1] else [Det2D("worker", 0.9, _box(_obj(k)))]
        dets3d = fuse(dets2d, rf, CAL, image=img)
        rgr.set_frame(img, rf, [d.box2d for d in dets3d])
        out = trk.update(dets3d, DT, recover=rgr)
        tids |= {t.tid for t in out}
        states.append(out[0].state if out else None)
    return tids, states, out, rgr.stats


def test_recovery_by_crop_redetection_keeps_id():
    tids, states, out, stats = _run(RGRParams(), crop_sees=True)
    assert len(tids) == 1 and stats["image"] > 0
    assert all(s == TrackState.CONFIRMED for s in states[20:45])
    assert np.allclose(out[0].vel[:2], V_OBJ[:2], atol=0.3)


def test_radar_only_coasting_survives_long_occlusion():
    # 25 frames = 1.65 s > max_lost_s (1.5 s): without radar the track would be deleted
    tids, states, out, stats = _run(RGRParams(), crop_sees=False)
    assert len(tids) == 1 and stats["radar_only"] > 0
    assert TrackState.RADAR_ONLY in states[20:45]
    assert out[0].state == TrackState.CONFIRMED                    # camera re-acquires with the same ID


def test_without_radar_long_occlusion_breaks_track():
    p = RGRParams(use_radar_roi=False, use_radar_gate=False, radar_only=False)
    tids, *_ = _run(p, crop_sees=False)
    assert len(tids) == 2                                            # ID switch: the baseline failure RGR fixes
