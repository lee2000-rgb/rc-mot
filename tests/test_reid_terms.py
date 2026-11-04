import numpy as np

from rcmot.tracking.tracker import RadarCameraTracker, TrackerParams, Track, appearance_terms
from rcmot.types import Det3D, TrackState


def _emb(seed, base=None, noise=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(size=64) if base is None else base + rng.normal(0, noise, 64)
    return e / np.linalg.norm(e)


def _det(pos, emb, score=0.9):
    d = Det3D("worker", score, np.array([0, 0, 40, 100.0]), np.asarray(pos, float), np.eye(3) * 0.04)
    d.emb = emb
    return d


def test_appearance_weight_vanishes_when_everyone_looks_alike():
    p = TrackerParams()
    vest = _emb(0)
    tr = Track(_det([10, 0, 0], _emb(1, vest, 0.02)), p, 0.0)
    alike = [_det([10, 0, 0], _emb(i, vest, 0.02)) for i in range(2, 6)]
    _, W = appearance_terms([tr], alike, p)
    assert W[0] < 0.3
    distinct = [_det([10, 0, 0], _emb(1, vest, 0.02))] + [_det([10, 0, 0], _emb(10 + i)) for i in range(4)]
    _, W = appearance_terms([tr], distinct, p)
    assert W[0] > 0.7


def test_relink_reuses_the_id_of_a_recently_deleted_track():
    trk = RadarCameraTracker(TrackerParams(min_hits=2, max_lost_s=0.5))
    me, other = _emb(0), _emb(99)
    for k in range(5):
        out = trk.update([_det([10, 0.1 * k, 0], _emb(k, me, 0.03)), _det([5, -4, 0], _emb(50 + k, other, 0.03))], 0.1)
    tid = next(t.tid for t in out if abs(t.pos[0] - 10) < 1)
    for _ in range(8):                                                    # 0.8 s gone -> deleted
        out = trk.update([_det([5, -4, 0], _emb(70, other, 0.03))], 0.1)
    assert all(t.tid != tid for t in out)
    out = trk.update([_det([10.5, 1.0, 0], _emb(7, me, 0.03)), _det([5, -4, 0], _emb(71, other, 0.03))], 0.1)
    assert any(t.tid == tid and t.state == TrackState.CONFIRMED for t in out) and trk.relinked == 1
