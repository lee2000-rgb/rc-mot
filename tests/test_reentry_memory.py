"""Swing re-entry memory (tracker.reentry_memory, docs/33 Track C step 2): OUT_OF_VIEW memories with a world anchor,
ID decided at confirmation -- and the switch-off path staying bit-identical."""
from types import SimpleNamespace

import numpy as np

from rcmot.tracking.tracker import RadarCameraTracker, Track, TrackerParams
from rcmot.types import Det3D, TrackState

DT = 0.066


def _emb(seed: int) -> np.ndarray:
    e = np.random.default_rng(seed).normal(size=64)
    return e / np.linalg.norm(e)


def _det(p, emb=None, k=0):
    x = 100.0 * k
    return Det3D("worker", 0.9, np.array([x, 100.0, x + 40.0, 200.0]), np.r_[np.asarray(p, float)[:2], 0.0],
                 np.eye(3) * 0.09, emb=None if emb is None else emb.copy())


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _run(params, script, world=True, seed=0):
    """script(frame) -> list of (world xy, emb) visible this frame; the ego turns at the scripted yaw rate.
    Returns the per-frame [(tid, state, pos)] and the tracker."""
    Track._next_id, Track._next_tmp = 1, -1                      # display IDs are a class counter
    trk = RadarCameraTracker(params)
    rng = np.random.default_rng(seed)
    th, out = 0.0, []
    for f in range(len(script)):
        yaw_rate, people = script[f]
        dyaw = yaw_rate * DT if f else 0.0
        th += dyaw
        R = _rot(th)
        dets = [_det(R.T @ (np.asarray(p, float) + rng.normal(0, 0.05, 2)), e, k) for k, (p, e) in enumerate(people)]
        w = SimpleNamespace(theta=th, origin=np.zeros(2)) if world else None
        res = trk.update(dets, DT, ego_dp=np.zeros(3) if f else None, ego_dyaw=dyaw, world=w)
        out.append([(o.tid, o.state, o.pos.copy()) for o in res])
    return out, trk


def _swing_script(gap_s=6.0, back=None, extra=None):
    """two standing workers at world (12, 3) and (12, -6); the ego swings away (60 deg/s) for gap_s and back
    (gap > max_lost_static_s 5 s: the tracks are deleted, only the memory remains)."""
    a, b = _emb(1), _emb(2)
    P = [((12.0, 3.0), a), ((12.0, -6.0), b)]
    n_gap = int(round(gap_s / DT))
    sc = [(0.0, P)] * 45                                          # 3 s seen, both enter standstill
    sc += [(np.radians(60.0), [])] * (n_gap // 2) + [(-np.radians(60.0), [])] * (n_gap - n_gap // 2)
    ret = P if back is None else back
    sc += [(0.0, ret + (extra or []))] * 30
    return sc


def test_switch_off_bit_identical():
    """reentry_memory False: the re-entry keys, the world argument and the new code paths change nothing."""
    sc = _swing_script(6.0)
    off = TrackerParams(min_hits=2)
    off2 = TrackerParams(min_hits=2, reentry_world=False, reentry_gate=1.0, reentry_coast_s=0.1, reentry_decide_s=5.0,
                         reentry_app_relink=True, reentry_life_stand_s=1.0)
    o1, t1 = _run(off, sc, world=True)
    o2, t2 = _run(off2, sc, world=False)
    assert len(o1) == len(o2)
    for a, b in zip(o1, o2):
        assert [(x[0], x[1]) for x in a] == [(x[0], x[1]) for x in b]
        assert all(np.array_equal(x[2], y[2]) for x, y in zip(a, b))
    assert t1.relinked == t2.relinked and t1.reentered == t2.reentered == 0


def _ids_at(out, f):
    return sorted(x[0] for x in out[f])


def test_reentry_inherits_after_swing_world_and_ego_frames():
    """Both workers come back after a 6 s swing (static memories, gap > max_lost_s): the IDs are inherited, with the
    world pose and with the tracker's own dead-reckoned frame."""
    sc = _swing_script(6.0)
    for world in (True, False):
        out, trk = _run(TrackerParams(min_hits=2, reentry_memory=True), sc, world=world)
        before = _ids_at(out, 44)
        after = _ids_at(out, len(sc) - 1)
        assert before == after and len(after) == 2, (world, before, after)
        assert trk.reentered == 2


def test_new_person_far_from_memories_gets_new_id():
    """A third worker appearing 15 m from both memories is new; the returning two keep their IDs."""
    c = _emb(3)
    sc = _swing_script(6.0, extra=[((25.0, 14.0), c)])
    out, trk = _run(TrackerParams(min_hits=2, reentry_memory=True), sc)
    before, after = _ids_at(out, 44), _ids_at(out, len(sc) - 1)
    assert set(before) < set(after) and len(after) == 3


def test_contradicting_appearance_is_not_inherited():
    """Someone who looks different (cosine distance >> bound) standing where worker a stood is not a."""
    a, b, c = _emb(1), _emb(2), _emb(3)
    sc = _swing_script(6.0, back=[((12.0, 3.0), c), ((12.0, -6.0), b)])
    out, trk = _run(TrackerParams(min_hits=2, reentry_memory=True), sc)
    before, after = _ids_at(out, 44), _ids_at(out, len(sc) - 1)
    assert len(set(before) & set(after)) == 1 and len(after) == 2
    assert trk.reentered == 1


def test_alive_lost_static_track_is_taken_over_not_duplicated():
    """A standing worker's track survives 5 s lost (max_lost_static_s); a re-entry 2 s later that the association
    misses (position shifted 1.5 m by a turning range error) continues that track instead of a second ID."""
    a = _emb(1)
    sc = [(0.0, [((12.0, 3.0), a)])] * 45 + [(np.radians(60.0), [])] * 15 + [(-np.radians(60.0), [])] * 15
    sc += [(0.0, [((13.5, 3.0), a)])] * 20
    kw = dict(min_hits=2, gate_pos=1.0, birth_suppress_chi2=1.0, app_reacq=False)
    out0, _ = _run(TrackerParams(**kw), sc)
    assert _ids_at(out0, 44) != _ids_at(out0, len(sc) - 1)          # without the memory: a second ID
    out, trk = _run(TrackerParams(reentry_memory=True, **kw), sc)
    assert _ids_at(out, 44) == _ids_at(out, len(sc) - 1)
    assert len(trk.tracks) == 1 and trk.reentered == 1


def test_no_embeddings_fragment_aware_does_not_crash():
    """Without ReID (no det.emb: reid off, eval_fusion_depth's GT-box pipeline) the re-entry memory with
    relink_fragment_aware must not call the appearance distance on None embeddings (11-03 crash on the Continental
    profile, tracker._maybe_same -> fused_distance(None, None)); position alone decides."""
    sc = [(yr, [(p, None) for p, _ in ppl]) for yr, ppl in _swing_script(6.0)]
    for mode in ("cos", "fused"):
        out, trk = _run(TrackerParams(min_hits=2, reentry_memory=True, relink_fragment_aware=True, app_mode=mode), sc)
        assert len(_ids_at(out, len(sc) - 1)) == 2


def test_maybe_same_without_embeddings_is_false():
    """Two fragments never alive together, one or both without an embedding: not judged the same (no crash)."""
    for mode in ("cos", "fused"):
        trk = RadarCameraTracker(TrackerParams(relink_fragment_aware=True, app_mode=mode))
        a = SimpleNamespace(spans=[], t_born=0.0, t_dead=1.0, emb=None, hist=None)
        b = SimpleNamespace(spans=[], t_born=2.0, t_dead=3.0, emb=None, hist=None)
        c = SimpleNamespace(spans=[], t_born=4.0, t_dead=5.0, emb=_emb(3), hist=None)
        assert not trk._maybe_same(a, b) and not trk._maybe_same(c, a) and not trk._maybe_same(a, c)
