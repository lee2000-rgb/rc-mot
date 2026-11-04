"""docs/33 wave 3 (fusion side): soft instance-mask weights with weighted cluster features (Track A), the online
depth-map stage, its cluster-score feature (B-b) and fallback range (B-a) -- and the switch-off path staying
bit-identical."""
from types import SimpleNamespace

import numpy as np

from rcmot.fusion.depth_map import DepthFit, DepthMapParams, DepthMapResult, fit_disparity
from rcmot.fusion.depth_stage import DepthStage, DepthStageParams, make_depth_stage
from rcmot.fusion.radar_camera import (FusionParams as _FusionParams, cluster_candidates, fuse, load_cluster_model, mask_factors,
                                       weighted_median)
from rcmot.pipeline import Pipeline
from rcmot.types import Det2D, RadarFrame
from rcmot.utils.calib import RigCalib


def FusionParams(**kw):  # noqa: N802  (these tests cover the legacy point-range fusion; lik is the default since docs/38)
    return _FusionParams(**{"range_model": "legacy", **kw})

R_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
CAL = RigCalib(np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], float), np.zeros(5), (1280, 720),
               np.eye(3), np.zeros(3), R_OPT, -R_OPT @ np.array([0, 0, 1.6]))
WORKER, CLUTTER = np.array([20.0, 1.0, 0.0]), np.array([28.0, 1.4, 0.0])


class _NoDet:
    def __call__(self, img):
        return []


def _box(p, w=0.6, h=1.75):
    c = np.array([[p[0] + dx, p[1] + dy, z] for dx in (-w / 2, w / 2) for dy in (-w / 2, w / 2) for z in (0, h)])
    uv, _ = CAL.project(c)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def _frame(*clusters, n=3, seed=1):
    rng = np.random.default_rng(seed)
    pts = [c + rng.normal(0, [0.1, 0.05, 0.2], (n, 3)) + [0, 0, 0.9] for c in clusters]
    xyz = np.vstack(pts) if pts else np.zeros((0, 3))
    vr = np.zeros(len(xyz))
    rf = RadarFrame(0.0, xyz, vr, np.zeros(len(xyz)), np.zeros(len(xyz)), xyz)
    rf.v_r_comp = vr.copy()
    return rf


def _mask_left_of(box, u_cut):
    """Box-cropped mask covering the box's pixels left of image column u_cut."""
    x0, y0 = int(box[0]), int(box[1])
    m = np.zeros((int(np.ceil(box[3])) - y0, int(np.ceil(box[2])) - x0), bool)
    m[:, : max(int(u_cut) - x0, 1)] = True
    return m


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.source == y.source and np.array_equal(x.pos, y.pos) and np.array_equal(x.pos_cov, y.pos_cov)
        assert x.n_radar == y.n_radar


# ------------------------------------------------------------------ helpers
def test_weighted_median_equals_median_for_equal_weights():
    x = np.array([3.0, 1.0, 2.0, 10.0])
    assert weighted_median(x, np.ones(4)) == np.median(x)
    assert weighted_median(x, np.array([1.0, 1.0, 1.0, 10.0])) == 10.0
    assert weighted_median(x[:3], np.array([1.0, 2.0, 1.0])) == 1.5   # half the weight exactly at 1: midpoint
    assert weighted_median(x[:3], np.array([1.0, 3.0, 1.0])) == 1.0


def test_mask_factors_modes():
    m = np.zeros((20, 20), np.uint8)
    m[:, :10] = 1                                     # columns 0..9 on the mask
    u = np.array([5.0, 11.0, 14.0, 30.0])             # on, 2 px out (d 1.5), 5 px out (d 4.5), beyond the crop
    v = np.full(4, 10.0)
    assert np.allclose(mask_factors(m, u, v, 0, 0, "const", 5.0, out_weight=0.2), [1, 0.2, 0.2, 0.2])
    g = mask_factors(m, u, v, 0, 0, "gauss_px", 5.0)
    assert g[0] == 1.0 and abs(g[1] - np.exp(-0.5 * (1.5 / 5.0) ** 2)) < 1e-6
    assert abs(g[2] - np.exp(-0.5 * (4.5 / 5.0) ** 2)) < 1e-6 and g[3] < g[2]
    gm = mask_factors(m, u, v, 0, 0, "gauss_m", 5.0, sigma_m=0.2, rng=np.full(4, 20.0), fx=1000.0)
    assert abs(gm[1] - np.exp(-0.5 * (1.5 * 20 / 1000 / 0.2) ** 2)) < 1e-6


def test_weighted_cluster_features_reduce_to_unweighted():
    rf = _frame(WORKER, CLUTTER)
    box = _box(WORKER)
    uv, _ = CAL.project(rf.xyz)
    idx = np.arange(len(rf.xyz))
    model = load_cluster_model(FusionParams().cluster_score_model)
    a = cluster_candidates(idx, rf.xyz, uv, rf.v_r, box, 1000.0, model, 1.0)
    b = cluster_candidates(idx, rf.xyz, uv, rf.v_r, box, 1000.0, model, 1.0, w=np.ones(len(idx)))
    for x, y in zip(a, b):
        assert np.array_equal(x[0], y[0]) and abs(x[1] - y[1]) < 1e-9 and np.allclose(x[2], y[2]) and abs(x[3] - y[3]) < 1e-9
    w = np.ones(len(idx))
    w[0] = 0.1                                        # a down-weighted member moves the centroid / lowers the count
    c = cluster_candidates(idx, rf.xyz, uv, rf.v_r, box, 1000.0, model, 1.0, w=w)
    assert not np.allclose(c[0][2], a[0][2]) and c[0][3] != a[0][3]


# ------------------------------------------------------------------ switch-off identity
def test_switches_off_bit_identical():
    """mask_weight hard, cluster_dmap / dmap_fallback off: passing depth-map records (and a mask) changes nothing
    but the diagnostics; the new keys at their defaults leave fuse() as before."""
    rf = _frame(WORKER, CLUTTER)
    box = _box(WORKER)
    uv, _ = CAL.project(rf.xyz[:3])
    d1 = Det2D("worker", 0.9, box, mask=_mask_left_of(box, uv[:, 0].min() - 3))
    d2 = Det2D("worker", 0.9, _box(np.array([14.0, -3.0, 0.0])))
    dm = [{"r": 28.0, "r_org": 28.0, "sig": 0.5, "ok": True, "bias": 1.1, "sig_rel": 0.06, "age": 0},
          {"r": 18.0, "r_org": 18.0, "sig": 0.5, "ok": True, "bias": None, "sig_rel": 0.06, "age": 0}]
    p_old = FusionParams()
    p_keys = FusionParams(mask_weight="hard", mask_hard_below_m=15.0, cluster_dmap=False, dmap_fallback=False)
    a = fuse([d1, d2], rf, CAL, params=p_old)
    b = fuse([d1, d2], rf, CAL, params=p_keys, dmap=dm)
    _same(a, b)
    assert a[0].dbg is None or "dm" not in a[0].dbg
    pipe = Pipeline(_NoDet(), CAL, {"recovery": {"enable": False}, "movers": {"enable": False}, "world": {"enable": False},
                                    "fusion": {"range_model": "legacy"}})
    assert pipe.dstage is None                                          # no depth stage unless asked for


# ------------------------------------------------------------------ Track A
def test_soft_mask_keeps_far_points_hard_gate_near():
    """The worker's returns land 3 px beside its silhouette. Hard gate: the cluster is deleted (camera fallback).
    gauss_px beyond mask_hard_below_m: kept (radar range). A 10 m worker: the hard path applies in both modes."""
    for pos, far in ((WORKER, True), (np.array([10.0, 0.5, 0.0]), False)):
        rf = _frame(pos)
        box = _box(pos)
        uv, _ = CAL.project(rf.xyz)
        det = Det2D("worker", 0.9, box, mask=_mask_left_of(box, uv[:, 0].min() - 3))
        hard = fuse([det], rf, CAL, params=FusionParams())
        soft = fuse([det], rf, CAL, params=FusionParams(mask_weight="gauss_px"))
        assert hard[0].n_radar == 0
        if far:
            assert soft[0].n_radar == 3 and soft[0].source.startswith("radar") and soft[0].dbg.get("mask_soft") == 1
            assert abs(np.linalg.norm(soft[0].pos[:2]) - 20.0) < 0.5
        else:
            _same(hard, soft)


def test_soft_mask_far_points_dropped_below_min_weight():
    rf = _frame(WORKER)
    box = _box(WORKER)
    uv, _ = CAL.project(rf.xyz)
    det = Det2D("worker", 0.9, box, mask=_mask_left_of(box, uv[:, 0].min() - 40))    # 40 px off: factor < 0.05
    soft = fuse([det], rf, CAL, params=FusionParams(mask_weight="gauss_px"))
    assert soft[0].n_radar == 0


def test_soft_mask_veto_lone_off_mask_cluster():
    """diag_A (11-03): a lone return just outside the mask at a wrong range (28 m clutter in a 20 m worker's box)
    with a cluster probability below mask_soft_min_p leaves the box exactly as under the hard mask (camera range);
    with the veto off (0) it is used; a cluster with a point ON the mask is never vetoed."""
    rf = _frame(CLUTTER, n=1)
    box = _box(WORKER)
    uv, _ = CAL.project(rf.xyz)
    det = Det2D("worker", 0.9, box, mask=_mask_left_of(box, uv[:, 0].min() - 2))
    hard = fuse([det], rf, CAL, params=FusionParams())
    veto = fuse([det], rf, CAL, params=FusionParams(mask_weight="gauss_px", mask_soft_min_p=0.3))
    off = fuse([det], rf, CAL, params=FusionParams(mask_weight="gauss_px", mask_soft_min_p=0.0))
    assert hard[0].n_radar == 0
    _same(hard, veto)
    assert veto[0].dbg.get("soft_weak") == 1
    assert off[0].n_radar == 1 and abs(np.linalg.norm(off[0].pos[:2]) - 28.0) < 0.5
    on = Det2D("worker", 0.9, box, mask=_mask_left_of(box, uv[:, 0].max() + 2))
    onm = fuse([on], rf, CAL, params=FusionParams(mask_weight="gauss_px", mask_soft_min_p=0.3))
    assert onm[0].n_radar == 1 and "soft_weak" not in onm[0].dbg


def test_soft_mask_veto_with_track_gate_and_safety_net():
    """The veto can empty the box; with a track's 2-D gate and the moving-ego safety net active this must fall back
    to the camera / track range, not fail (11-03 server run: argmax of an empty sequence in _kde_mode)."""
    rf = _frame(CLUTTER, n=1)
    box = _box(WORKER)
    uv, _ = CAL.project(rf.xyz)
    det = Det2D("worker", 0.9, box, mask=_mask_left_of(box, uv[:, 0].min() - 2))
    kw = dict(range_priors=[(20.0, 0.5)], xy_priors=[(WORKER[:2].copy(), np.eye(2) * 0.25, 0.066)], ego_static=False)
    out = fuse([det], rf, CAL, params=FusionParams(mask_weight="gauss_px", mask_soft_min_p=0.3), **kw)
    hard = fuse([det], rf, CAL, params=FusionParams(), **kw)
    assert out[0].n_radar == 0 and out[0].dbg.get("soft_weak") == 1
    assert abs(np.linalg.norm(out[0].pos[:2]) - np.linalg.norm(hard[0].pos[:2])) < 1e-6


# ------------------------------------------------------------------ B-b / B-a in fuse()
def test_bb_depth_feature_used_only_with_an_offered_map():
    """Worker at 20 m (the height prior agrees) and a second cluster at 28 m: the plain scorer picks 20 m; with the
    depth map saying 28 m the dmap model picks 28 m; a map that is not offered (ok False) changes nothing."""
    rf = _frame(WORKER, CLUTTER)
    det = Det2D("worker", 0.9, _box(WORKER))
    rec = {"r": 28.0, "r_org": 28.0, "sig": 0.5, "ok": True, "bias": None, "sig_rel": 0.06, "age": 0}
    base = fuse([det], rf, CAL, params=FusionParams())
    on = fuse([det], rf, CAL, params=FusionParams(cluster_dmap=True), dmap=[rec])
    bad = fuse([det], rf, CAL, params=FusionParams(cluster_dmap=True), dmap=[dict(rec, ok=False)])
    assert abs(np.linalg.norm(base[0].pos[:2]) - 20.0) < 1.0
    assert abs(np.linalg.norm(on[0].pos[:2]) - 28.0) < 1.5 and on[0].dbg.get("dm_bb") == 1
    _same(base, bad)


def test_ba_fallback_range_with_bias_and_guards():
    rf = _frame()                                                      # no radar points at all
    det = Det2D("worker", 0.9, _box(WORKER))
    C = CAL.camera_center()
    rec = {"r": 20.0, "r_org": 20.0, "sig": 0.3, "ok": True, "bias": 1.05, "sig_rel": 0.06, "age": 1}
    p = FusionParams(dmap_fallback=True)
    a = fuse([det], rf, CAL, params=p, dmap=[rec])
    assert a[0].source == "dmap" and abs(np.linalg.norm(a[0].pos[:2] - C[:2]) - 21.0) < 1e-6
    sig_along = np.sqrt(np.max(np.linalg.eigvalsh(a[0].pos_cov)))
    assert abs(sig_along - 0.06 * 21.0) < 0.05                         # relative sigma floor
    for bad in (dict(rec, ok=False), dict(rec, r=12.0)):               # not offered / below dmap_min_range
        b = fuse([det], rf, CAL, params=p, dmap=[bad])
        assert b[0].source != "dmap"
    off = fuse([det], rf, CAL, params=FusionParams(), dmap=[rec])
    assert off[0].source != "dmap"
    pb = FusionParams(dmap_fallback=True, dmap_fallback_bias_only=True)
    assert fuse([det], rf, CAL, params=pb, dmap=[rec])[0].source == "dmap"
    assert fuse([det], rf, CAL, params=pb, dmap=[dict(rec, bias=None)])[0].source != "dmap"   # no track history
    tr = fuse([det], rf, CAL, params=p, dmap=[rec], range_priors=[(19.0, 0.3)])
    r_tr = np.linalg.norm(tr[0].pos[:2] - C[:2])
    assert tr[0].source == "dmap" and 19.0 < r_tr < 21.0                # combined with the track range like mono+track


def test_radar_backed_det_reports_dmap_ratio():
    rf = _frame(WORKER)
    det = Det2D("worker", 0.9, _box(WORKER))
    rec = {"r": 19.0, "r_org": 19.0, "sig": 0.3, "ok": True, "bias": None, "sig_rel": 0.06, "age": 0}
    a = fuse([det], rf, CAL, params=FusionParams(), dmap=[rec])
    assert a[0].source.startswith("radar") and abs(a[0].dmap_ratio - np.linalg.norm(a[0].pos[:2] - CAL.camera_center()[:2]) / 19.0) < 0.05
    b = fuse([det], rf, CAL, params=FusionParams(), dmap=[dict(rec, ok=False)])
    assert getattr(b[0], "dmap_ratio", None) is None


# ------------------------------------------------------------------ depth stage
def _stage(every=3, rel=0.05, split_u=640, z_left=10.0, z_right=30.0):
    st = DepthStage(CAL, DepthStageParams(every=every))
    st.md.model = object()                                             # no network: update() is replaced below
    D = np.full((720, 1280), z_right, np.float32)
    D[:, :split_u] = z_left
    calls = []

    def fake_update(img, xyz, world=None, stamp=0.0, disparity=None, fit_R=None):
        st.md.push_static(xyz)
        fit = DepthFit(0.05, 0.0, 100, 90, 0.01, 0.001, np.eye(2) * 1e-8, rel)
        st.md.last = DepthMapResult(stamp, np.zeros((720, 1280), np.float32), D, fit, 0.05, 0.0, 0.0, True, len(xyz))
        st.md._rel = rel
        calls.append((stamp, fit_R))
        return st.md.last

    st.md.update = fake_update
    return st, calls


def _scan(t):
    rf = _frame(WORKER, CLUTTER)
    rf.stamp = t
    n = len(rf.xyz)
    rf.stats = SimpleNamespace(static_inlier=np.ones(n, bool), keep=np.ones(n, bool), below_ground=None, p_dyn=np.zeros(n))
    return rf


def test_stage_every_third_frame_and_quality():
    st, calls = _stage(every=3)
    flags = [st.step(np.zeros((720, 1280, 3), np.uint8), _scan(0.066 * i), 0.066, np.zeros(3), 0.0, False) for i in range(7)]
    assert flags == [True, False, False, True, False, False, True] and len(calls) == 3
    assert len(st.md.static_points()) == 3 * 6                        # last n_frames scans
    det = Det2D("worker", 0.9, _box(np.array([10.0, 2.0, 0.0])))
    rec = st.boxes([det])[0]
    assert rec["ok"] and abs(rec["r"] - 10.0) < 0.6 and rec["age"] == 0
    st2, _ = _stage(rel=0.2)                                           # poor fit: map not offered
    st2.step(np.zeros((720, 1280, 3), np.uint8), _scan(0.0), 0.066, np.zeros(3), 0.0, False)
    assert not st2.boxes([det])[0]["ok"]
    far = Det2D("worker", 0.9, _box(np.array([30.0, -6.0, 0.0])))
    assert not st.boxes([far])[0]["ok"]                                # beyond max_range


def test_stage_stale_map_read_in_its_own_frame():
    """Map: 10 m left of column 640, 30 m right of it. A box at column ~690: 30 m on the map's frame. The base then
    turns left 6 deg before the next frame: the same static object was 100 px further left in the map's image ->
    the stale map must be read there (10 m) -- also when this frame already counts as static (swing reversal, review
    11-02 #3); without motion since the map it is read as it is."""
    st, _ = _stage(every=3)
    img = np.zeros((720, 1280, 3), np.uint8)
    st.step(img, _scan(0.0), 0.066, np.zeros(3), 0.0, True)
    det = Det2D("worker", 0.9, np.array([675.0, 300.0, 705.0, 400.0]))
    assert abs(st.boxes([det], moving=True)[0]["r"] - 30.0) < 3.0
    st.step(img, _scan(0.066), 0.066, np.zeros(3), np.radians(6) / 0.066, True)   # 6 deg in one frame
    moved = st.boxes([det], moving=True)[0]
    still = st.boxes([det], moving=False)[0]
    assert moved["age"] == 1 and moved["r"] < 15.0 and still["r"] < 15.0
    st2, _ = _stage(every=3)                                           # static, no motion since the map: as it is
    st2.step(img, _scan(0.0), 0.066, np.zeros(3), 0.0, False)
    st2.step(img, _scan(0.066), 0.066, np.zeros(3), 0.0, False)
    assert abs(st2.boxes([det], moving=False)[0]["r"] - 30.0) < 3.0


def test_stage_rejected_fit_not_offered_and_filter_restarted():
    st, _ = _stage(every=1)
    img = np.zeros((720, 1280, 3), np.uint8)
    upd0 = st.md.update

    def rejected(*a, **k):
        r = upd0(*a, **k)
        st.md.last = r = r.__class__(*[(False if f == "updated" else getattr(r, f)) for f in r.__dataclass_fields__])
        return r
    st.md.update = rejected
    st.md.kf.x, st.md.kf.P = np.array([0.05, 0.0]), np.eye(2)
    det = Det2D("worker", 0.9, _box(np.array([10.0, 2.0, 0.0])))
    for i in range(st.p.max_rejects - 1):
        st.step(img, _scan(0.066 * i), 0.066, np.zeros(3), 0.0, False)
        assert not st.boxes([det])[0]["ok"] and st.md.kf.x is not None
    st.step(img, _scan(1.0), 0.066, np.zeros(3), 0.0, False)
    assert st.md.kf.x is None and st.n_rej == 0                       # restarted: the next fit initialises it
    st.md.update = upd0
    st.step(img, _scan(1.1), 0.066, np.zeros(3), 0.0, False)
    assert st.boxes([det])[0]["ok"]


class _Trk:                                                            # hashable like rcmot Track objects
    def __init__(self, pos, box):
        self.kf, self.cls, self.tsu, self.box2d = SimpleNamespace(pos=pos.copy(), vel=np.zeros(3)), "worker", 0.0, box


def test_stage_track_bias_learned_and_looked_up():
    st, _ = _stage()
    st.step(np.zeros((720, 1280, 3), np.uint8), _scan(0.0), 0.066, np.zeros(3), 0.0, False)
    pos = np.array([18.0, 1.0, 0.0])
    box = _box(pos)
    trk = _Trk(pos, box)
    for i in range(3):
        d3 = SimpleNamespace(box2d=box, dmap_ratio=1.1)
        st.observe([d3], [trk], 0.2 * i)
    det = Det2D("worker", 0.9, box.copy())
    assert abs(st.boxes([det], [trk], t_now=0.5)[0]["bias"] - 1.1) < 1e-9
    assert st.boxes([det], [trk], t_now=20.0)[0]["bias"] is None       # older than bias_window_s
    assert st.boxes([det], [], t_now=0.5)[0]["bias"] is None           # track gone


def test_stage_track_bias_one_to_one_and_range():
    """A biased track lends its bias to its own box only: a neighbour's box overlapping the track's predicted box
    (IoU >= 0.3) gets none (review 11-02 #2); a bias outside bias_range is not used."""
    st, _ = _stage()
    st.step(np.zeros((720, 1280, 3), np.uint8), _scan(0.0), 0.066, np.zeros(3), 0.0, False)
    pos = np.array([20.0, 1.0, 0.0])
    box = _box(pos)
    trk = _Trk(pos, box)
    for i in range(3):
        st.observe([SimpleNamespace(box2d=box, dmap_ratio=1.25)], [trk], 0.2 * i)
    own = Det2D("worker", 0.9, box.copy())
    w = box[2] - box[0]
    other = Det2D("worker", 0.9, box + np.array([0.5 * w, 0.0, 0.5 * w, 0.0]))   # IoU 1/3 with the track's box
    from rcmot.fusion.depth_stage import _iou
    assert _iou(other.box, box) >= st.p.track_iou
    assert [r["bias"] for r in st.boxes([other], [trk], t_now=0.5)] == [1.25]     # alone it is the track's box
    recs = st.boxes([other, own], [trk], t_now=0.5)
    assert recs[1]["bias"] == 1.25 and recs[0]["bias"] is None
    st.ratios[trk].extend([(0.7, 1.8)] * 10)                           # median now 1.8: outside bias_range
    assert st.boxes([own], [trk], t_now=0.8)[0]["bias"] is None


def test_pipeline_runs_stage_before_fusion():
    cfg = {"recovery": {"enable": False}, "movers": {"enable": False}, "world": {"enable": False},
           "fusion": {"range_model": "legacy", "dmap_fallback": True}, "depth_map": {"every": 3}}
    assert make_depth_stage({}, CAL, FusionParams()) is None
    pipe = Pipeline(_NoDet(), CAL, cfg)
    assert pipe.dstage is not None
    st, calls = _stage(z_left=18.0, z_right=18.0)
    pipe.dstage = st
    for i in range(4):
        res = pipe.step(np.zeros((720, 1280, 3), np.uint8), _scan(0.066 * i), 0.0)
        assert "depth_map" in res.timing_ms
    assert len(calls) == 2
    det = Det2D("worker", 0.9, _box(np.array([18.0, 1.0, 0.0])))
    out = pipe.fuse_boxes([det], _frame(), 0.0)                       # no radar points in the box: B-a range
    assert out and out[0].source == "dmap"


def test_vectorized_ransac_matches_loop():
    rng = np.random.default_rng(3)
    z = rng.uniform(3, 60, 400)
    dn = (1 / z - 0.002) / 0.03 * (1 + rng.normal(0, 0.01, 400))
    dn[:60] = rng.uniform(dn.min(), dn.max(), 60)                     # 15 % outliers
    a = fit_disparity(dn, z, DepthMapParams(), np.random.default_rng(0))
    b = fit_disparity(dn, z, DepthMapParams(ransac_vectorized=True), np.random.default_rng(0))
    assert abs(a.s - 0.03) / 0.03 < 0.02 and abs(b.s - a.s) / a.s < 0.01 and abs(b.t - a.t) < 2e-4
    assert abs(a.n_in - b.n_in) <= 0.05 * a.n_in and b.rel_err < 0.05


def test_stage_phase_offsets_first_map():
    st, calls = _stage(every=3)
    st.p.phase = 1
    flags = [st.step(np.zeros((720, 1280, 3), np.uint8), _scan(0.066 * i), 0.066, np.zeros(3), 0.0, False) for i in range(5)]
    assert flags == [False, True, False, False, True]
    det = Det2D("worker", 0.9, _box(np.array([10.0, 2.0, 0.0])))
    assert st.boxes([det])[0]["age"] == 0
