"""docs/38: range as a distribution -- the online likelihood (rcmot.fusion.range_lik), the tracker's per-pair selection
(rcmot.tracking.range_select), the tracker's birth / hypothesis rules, and the near-zone evaluation view."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcmot.fusion.range_lik import (GRID, BoxCues, RangeLik, RangeLikModel, posterior, summarize,  # noqa: E402
                                    window_moments)
from rcmot.labeling.metrics import Frame, evaluate, near_zone  # noqa: E402
from rcmot.tracking.range_select import select  # noqa: E402
from rcmot.tracking.tracker import RadarCameraTracker, TrackerParams  # noqa: E402
from rcmot.types import Det3D  # noqa: E402

FULL = json.load(open(ROOT / "configs/range_lik/full.json"))


def _cues(d: dict) -> BoxCues:
    nan = lambda x: np.nan if x is None else x  # noqa: E731
    return BoxCues(**{k: nan(v) if k not in ("cut", "g_ok", "w_ok", "dm_ok", "clusters") else v for k, v in d.items()})


def test_port_equals_research_model():
    """The online model reproduces research/range_unc/model.py (fixture: 27 logged boxes of the docs/37 dumps, the
    research terms / summary computed with configs/range_lik/full.json)."""
    rows = json.load(open(ROOT / "tests/data/range_lik_fixture.json"))
    m = RangeLikModel(FULL)
    for row in rows:
        c = _cues(row["cues"])
        t = m.camera_terms(c)
        rt = m.radar_term(m.radar_parts(c.clusters))
        if rt is not None:
            t["radar"] = rt
        assert set(t) == set(row["terms"])
        for k, v in row["terms"].items():
            np.testing.assert_allclose(t[k][[0, 200, 500, 900, 1400, 1779]], v, rtol=1e-9, atol=1e-9)
        s = summarize(posterior(list(t.values())))
        for k, v in row["summ"].items():
            assert s[k] == pytest.approx(v, abs=1e-9)


def test_camera_only_box_centres_on_its_range():
    m = RangeLikModel(FULL)
    r_true, h_cam = 20.0, 1.6
    c = BoxCues(r_mono=r_true, h_px=80, w_px=30, aspect=0.375, r_g=r_true, g_ok=True, h_cam=h_cam)
    s = summarize(posterior(list(m.camera_terms(c).values())))
    assert abs(s["r"] - r_true) < 1.0 and s["sig"] < 0.1 * r_true and s["p_alt"] < 0.1


def test_cut_box_is_an_upper_bound_only():
    """A box cut at the image edge is shorter than the person and looks farther: its height prior keeps only 'not
    farther than r_mono' (one-sided)."""
    m = RangeLikModel(FULL)
    c = BoxCues(r_mono=15.0, h_px=60, w_px=30, aspect=0.5, cut=True)
    t = m.camera_terms(c)["mono"]
    assert t[np.searchsorted(GRID, 20.0)] < t[np.searchsorted(GRID, 15.0)] - 1.0   # farther: unlikely
    assert abs(t[np.searchsorted(GRID, 8.0)] - t[np.searchsorted(GRID, 11.0)]) < 0.05   # nearer: flat


def test_lone_weak_cluster_against_camera_is_a_minor_mode():
    """docs/37 §6.6 (a): a radar cluster behind the person, camera cues in front -> the camera range wins, the cluster
    is a second mode with small mass (the old fusion reported the cluster with sigma 0.34 m)."""
    m = RangeLikModel(FULL)
    c = BoxCues(r_mono=17.7, h_px=90, w_px=35, aspect=0.39, r_g=17.6, g_ok=True, r_w=17.0, w_ok=True,
                clusters=[{"r": 21.7, "f": None}])
    t = m.camera_terms(c)
    t["radar"] = m.radar_term(m.radar_parts(c.clusters))
    s = summarize(posterior(list(t.values())))
    assert abs(s["r"] - 17.7) < 1.0
    assert any(abs(md[0] - 21.7) < 0.5 for md in s["modes"]) or s["p_alt"] < 0.05


def _det(log_cam, log_rad=None, r_hat=None, radar_backed=False, clusters=()):
    """Det3D on the x axis (camera at the origin, 1.6 m high) with the given log-likelihoods."""
    post = posterior([t for t in (log_cam, log_rad) if t is not None])
    s = summarize(post)
    r = s["r"] if r_hat is None else r_hat
    C = np.array([0.0, 0.0, 1.6])
    ray = np.array([1.0, 0.0, -0.05])
    rl = RangeLik(C=C, ray=ray, log_cam=log_cam, log_rad=log_rad, summ=s, radar_backed=radar_backed,
                  b_sig=np.radians(0.3), sig_up=0.2, clusters=list(clusters))
    d = Det3D("worker", 0.9, np.array([600.0, 300.0, 640.0, 400.0]), rl.point(r), np.diag([s["sig"] ** 2, 0.01, 0.04]),
              source="lik+radar" if radar_backed else "lik")
    d.rlik = rl
    return d


def _gauss(mu, sd):
    return -0.5 * ((GRID - mu) / sd) ** 2


def test_select_gaussian_equals_mahalanobis():
    d = _det(_gauss(20.0, 0.5))
    P2 = np.diag([0.3 ** 2, 0.2 ** 2])
    for r_p in (19.0, 20.0, 21.5):
        s = select(d, np.array([r_p, 0.0]), P2, ext=0.0, eps=0.0)
        assert s.r == pytest.approx(20.0, abs=0.05)
        expect = (r_p - 20.0) ** 2 / (0.5 ** 2 + 0.3 ** 2)
        assert s.d2_along == pytest.approx(expect, rel=0.05, abs=0.05)


def test_select_follows_the_tracks_mode():
    """Bimodal box (camera 18 m strong, radar 22 m weaker): a track at 18 m measures 18, a track at 22 m measures 22
    (with a mass penalty in its distance), a track at 30 m measures neither cheaply."""
    d = _det(np.logaddexp(_gauss(18.0, 0.6), _gauss(22.0, 0.3) + np.log(0.25)))
    P2 = np.diag([0.4 ** 2, 0.2 ** 2])
    a = select(d, np.array([18.0, 0.0]), P2, ext=0.25)
    b = select(d, np.array([22.0, 0.0]), P2, ext=0.25)
    c = select(d, np.array([30.0, 0.0]), P2, ext=0.25)
    assert abs(a.r - 18.0) < 0.3 and abs(b.r - 22.0) < 0.3
    assert a.d2 < b.d2 < c.d2 and c.d2 > 11.34


def test_tracker_does_not_birth_on_a_vague_range_and_resolves_hypotheses():
    p = TrackerParams(range_lik=True, min_hits=3, min_hits_radar=2, app_reacq=False, w_app=0.0, max_lost_s=1.0)
    trk = RadarCameraTracker(p)
    vague = _det(_gauss(20.0, 4.0))                          # sigma / range = 0.2 -> no birth
    trk.update([vague], 0.1)
    assert not trk.tracks and trk.lik_counts["birth_wide"] == 1
    # two equally strong modes at 15 and 25 m: a hypothesis track; later boxes only support 25 m
    both = _det(np.logaddexp(_gauss(15.0, 0.4), _gauss(25.0, 0.4)))
    assert both.rlik.summ["p_alt"] > 0.1
    trk.update([both], 0.1)
    assert len(trk.tracks) == 1 and trk.tracks[0].hyps is not None
    for _ in range(6):
        trk.update([_det(_gauss(25.0, 0.4))], 0.1)
    t = trk.tracks[0]
    assert t.hyps is None
    assert abs(np.hypot(*t.kf.pos[:2]) - 25.0) < 0.6
    assert t.state.name == "CONFIRMED"


def test_legacy_tracker_ignores_rlik():
    """range_lik False: a box with an rlik attribute is tracked exactly like before (point position)."""
    p = TrackerParams()
    d = _det(_gauss(20.0, 4.0))
    trk = RadarCameraTracker(p)
    trk.update([d], 0.1)
    assert len(trk.tracks) == 1


def test_near_zone():
    gt_boxes = np.array([[0, 0, 10, 20], [100, 0, 110, 20]], float)
    f = Frame(np.array([1, 2]), gt_boxes, np.array([7, 8, 9]),
              np.array([[0, 0, 10, 20], [100, 0, 110, 20], [300, 0, 310, 20]], float),
              np.zeros(2, bool), gt_range=np.array([10.0, 40.0]), gt_xy=np.array([[10.0, 0], [40.0, 0]]),
              gt_quality=np.array(["dense", "dense"], object), pr_range=np.array([25.0, 40.0, 45.0]),
              pr_xy=np.array([[25.0, 0], [40.0, 0], [45.0, 0]]))
    g = near_zone(f, 20.0)
    assert list(g.gt_ignore) == [False, True]                # the 40 m person is outside the zone
    assert list(g.pr_ids) == [7]                             # 8 (40 m, only on the far person) and 9 (45 m, on nobody)
                                                             # are outside the zone; 7 stays (on the in-zone person)
    e = evaluate([g])
    assert e["TP"] == 1 and e["FP"] == 0 and e["FN"] == 0


def test_alarm_metrics():
    """docs/38 §10: per-band found / right-place / zone / range error, false objects, frame alarm level, delay."""
    from rcmot.labeling.metrics import alarm_metrics, alarm_summary

    def fr(gt_r, pr_r, pr_box_shift=0.0, extra_pred=None):
        gb = np.array([[100, 100, 140, 200]], float)
        pb = [[100 + pr_box_shift, 100, 140 + pr_box_shift, 200]]
        prr, pxy = [pr_r], [[pr_r, 0.0]]
        if extra_pred is not None:
            pb.append([500, 100, 540, 200])
            prr.append(extra_pred)
            pxy.append([extra_pred, 5.0])
        return Frame(np.array([1]), gb, np.arange(len(pb)) + 1, np.array(pb, float), np.zeros(1, bool),
                     gt_range=np.array([gt_r]), gt_xy=np.array([[gt_r, 0.0]]), gt_quality=np.array(["dense"], object),
                     pr_range=np.array(prr, float), pr_xy=np.array(pxy, float))
    frames = [fr(8.0, 300.0, pr_box_shift=300.0),        # person at 8 m, prediction elsewhere: not found
              fr(8.0, 12.5),                              # found, 4.5 m too far: wrong zone (under), not right place
              fr(8.0, 8.4, extra_pred=15.0)]              # found right + a false object at 15 m
    m = alarm_metrics(frames, stamps=[0.0, 0.1, 0.2])
    b = m["b10"]
    assert b["n"] == 3 and b["found"] == 2 and b["bev"] == 1 and b["zone_ok"] == 1 and b["under"] == 1
    assert m["b20"]["false"] == 1
    assert m["lvl_conf"][3][3] == 1 and m["lvl_conf"][3][2] == 1 and m["lvl_miss_danger"] == 2
    assert m["delay_s"] == [pytest.approx(0.2)] and m["never"] == 0
    s = alarm_summary([m])
    assert s["b10"]["found"] == pytest.approx(2 / 3) and s["lvl"]["miss_danger"] == pytest.approx(2 / 3)


def test_track_cannot_take_a_small_bump_inside_a_mode():
    """docs/38 §7.4 (conti_side_drive_b GT 40): one mode at 5.26 m plus a 0.5 % 'standing' bump at 8.7 m (below the 1 %
    peak threshold of the summary, so it is not a mode of its own). A track predicted at 8.4 m must not measure the bump:
    the basin mask drops it, the old per-mode mask ("mode") did not."""
    L = np.logaddexp(_gauss(5.26, 0.35), _gauss(8.7, 0.35) + np.log(0.006))
    d = _det(L)
    assert d.rlik.summ["n_modes"] == 1
    P2 = np.diag([0.75 ** 2, 0.2 ** 2])
    old = select(d, np.array([8.4, 0.0]), P2, ext=0.25, mask="mode")
    new = select(d, np.array([8.4, 0.0]), P2, ext=0.25, mask="basin")
    assert abs(old.r - 8.7) < 0.2
    assert abs(new.r - 5.26) < 0.2 and new.d2 > old.d2


def test_wedge_clusters_and_term():
    """docs/41: returns in the box's BEV azimuth wedge but outside the box (below it in the image, or under the ground)
    form range clusters with the wedge features; the person-like cluster (narrow, several returns, mirrors) gets the
    larger likelihood ratio and the radar term peaks at its range."""
    from types import SimpleNamespace
    from rcmot.fusion.radar_camera import _wedge_clusters
    rng = np.random.default_rng(0)
    C = np.array([0.0, 0.0, 1.6])
    # person at 15 m straight ahead: 6 returns at heights -2..2 m (3 of them below ground = mirrors), lateral +-0.2 m
    pr = 15.0 + rng.normal(0, 0.1, 6)
    pl = rng.normal(0, 0.15, 6)
    pz = np.array([1.0, 0.5, 1.5, -1.2, -1.8, -2.0])
    # background at 25 m: 3 returns spread laterally +-0.9 m
    br = 25.0 + rng.normal(0, 0.2, 3)
    bl = np.array([-0.9, 0.0, 0.9])
    bz = np.array([0.5, 1.0, 2.0])
    xyz = np.c_[np.r_[pr, br], np.r_[pl, bl], np.r_[pz, bz]]
    n = len(xyz)
    below = xyz[:, 2] < -1.0
    st = SimpleNamespace(keep=~below, below_ground=below, dynamic=np.r_[np.ones(6, bool), np.zeros(3, bool)],
                         z_rcs_scan=np.zeros(n))
    radar = SimpleNamespace(stats=st, xyz=xyz)
    uv = np.c_[np.full(n, 640.0), np.full(n, 700.0)]          # all project below a box ending at v = 500
    rng_c = np.hypot(xyz[:, 0], xyz[:, 1])
    W = FULL.get("wedge")
    if W is None:
        pytest.skip("no wedge block in configs/range_lik/full.json")
    cl = _wedge_clusters(W, radar, uv, rng_c, C, (600.0, 300.0, 680.0, 500.0), np.radians(2.0), np.radians(-2.0))
    assert len(cl) == 2
    person = min(cl, key=lambda c: abs(c["r"] - 15.0))
    assert person["f"]["n_mirror"] == 3 and person["f"]["n_out_low"] == 3 and person["f"]["n_in"] == 0
    m = RangeLikModel(FULL)
    lt, parts = m.wedge_term(cl)
    assert abs(GRID[int(np.argmax(lt))] - 15.0) < 0.5
