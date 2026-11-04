"""Crowd-relative appearance (docs/32): part-colour histograms, crowd-mean step, tracker distance modes."""
from types import SimpleNamespace

import numpy as np

from rcmot.reid import appearance as A
from rcmot.tracking.tracker import TrackerParams, app_distance, app_norm, app_relink_max


def _det(emb, crop=None, mask=None, score=0.9, cls="worker"):
    return SimpleNamespace(cls=cls, score=score, emb=emb / np.linalg.norm(emb), crop=crop, crop_mask=mask, hist=None)


def _person(top, pants, h=120, w=48):
    im = np.zeros((h, w, 3), np.uint8)
    k = int(0.62 * h)                        # sleeve band ends at 0.60 of the height
    im[:k] = top
    im[k:] = pants
    return im


def test_part_hists_unit_rows_and_band_colours():
    h = A.part_hists(_person((0, 200, 255), (120, 40, 10)), np.ones((120, 48), bool))
    assert h.shape == (4, A.NBINS)
    assert np.allclose(np.linalg.norm(h, axis=1), 1.0, atol=1e-5)
    h2 = A.part_hists(_person((0, 200, 255), (20, 20, 20)), None)
    bc = np.einsum("pk,pk->p", h, h2)
    assert bc[0] < 0.5                      # pants band differs
    assert bc[1] > 0.99                     # sleeves (upper outer columns) identical


def test_crowd_step_removes_shared_component_and_holds():
    rng = np.random.default_rng(0)
    shared = rng.normal(size=64) * 3
    a, b = shared + rng.normal(size=64), shared + rng.normal(size=64)
    st = A.AppearanceStage(dict(crowd_mean=True))
    d = [_det(a), _det(b), _det(shared + rng.normal(size=64))]
    raw_cos = float(d[0].emb @ d[1].emb)
    st.crowd_step(d, 0.0)
    assert float(d[0].emb @ d[1].emb) < raw_cos - 0.3     # the shared uniform no longer makes them alike
    alone = _det(a)
    st.crowd_step([alone], 1.0)                            # alone: last crowd mean (hold 3 s)
    assert np.allclose(alone.emb, d[0].emb, atol=1e-5)
    late = _det(a)
    st.crowd_step([late], 10.0)                            # hold expired: raw
    assert np.allclose(late.emb, a / np.linalg.norm(a), atol=1e-6)


def test_crowd_excludes_low_scores_and_other_classes():
    rng = np.random.default_rng(1)
    st = A.AppearanceStage(dict(crowd_mean=True, crowd_min_score=0.5))
    d = [_det(rng.normal(size=16)), _det(rng.normal(size=16), score=0.2), _det(rng.normal(size=16), cls="heavy_equipment")]
    e0 = d[0].emb.copy()
    st.crowd_step(d, 0.0)                                  # only one qualifying worker -> no crowd, unchanged
    assert np.allclose(d[0].emb, e0)


def test_tracker_distance_modes():
    rng = np.random.default_rng(2)
    a, b = _det(rng.normal(size=32)), _det(rng.normal(size=32))
    p = TrackerParams()
    assert app_distance(a, b, p) == 1.0 - float(a.emb @ b.emb)          # cos path unchanged
    assert app_relink_max(p) == p.relink_max_dist and app_norm(p) == 0.5
    p.app_mode = "fused"
    a.hist = A.part_hists(_person((0, 200, 255), (120, 40, 10)), None)
    b.hist = A.part_hists(_person((0, 200, 255), (20, 20, 20)), None)
    same = SimpleNamespace(emb=a.emb, hist=a.hist)
    assert app_distance(a, same, p) < 1e-3
    assert app_distance(a, b, p) > 0.5
    assert app_relink_max(p) == p.relink_max_fused


def test_steep_camera_switches_part_colour_off():
    def calib(pitch_deg):
        th = -np.radians(pitch_deg)
        # level (x fwd, y left, z up) -> optical (x right, y down, z fwd), camera pitched down by th
        R0 = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
        Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        return SimpleNamespace(R_cl=R0 @ Ry)
    assert abs(A.camera_pitch_down_deg(calib(10.0)) - 10.0) < 1e-6
    flat = A.AppearanceStage(dict(crowd_mean=True, part_colour=True), calib(10.0))
    steep = A.AppearanceStage(dict(crowd_mean=True, part_colour=True), calib(26.0))
    assert flat.mode == "fused" and steep.mode == "crowd"


def test_fragment_aware_rivals():
    from rcmot.tracking.tracker import RadarCameraTracker
    rng = np.random.default_rng(3)
    e = rng.normal(size=32)
    trk = RadarCameraTracker(TrackerParams(relink_fragment_aware=True))
    trk.t = 20.0
    frag1 = SimpleNamespace(emb=e / np.linalg.norm(e), hist=None, spans=[], t_born=0.0, t_dead=5.0)
    frag2 = SimpleNamespace(emb=(e + 0.1 * rng.normal(size=32)) / np.linalg.norm(e), hist=None, spans=[],
                            t_born=6.0, t_dead=9.0)
    other = SimpleNamespace(emb=frag2.emb, hist=None, spans=[], t_born=4.0, t_dead=8.0)     # alive with frag1
    frag2.emb = frag2.emb / np.linalg.norm(frag2.emb)
    assert trk._maybe_same(frag1, frag2)            # never together, look alike -> may be the same person
    assert not trk._maybe_same(frag1, other)        # seen together -> a real rival
    trk.p.relink_fragment_aware = False
    assert not trk._maybe_same(frag1, frag2)
