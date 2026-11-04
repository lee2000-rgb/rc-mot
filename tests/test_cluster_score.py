"""Cluster-level candidate scoring and the 2-D track gate of fuse() (docs/30): a worker at 20 m with background
clutter 8 m behind inside the same box. The layout scorer prefers the cluster consistent with the box height; the
track gate removes the clutter when the track predicts the worker; the mono fallback catches a broken ground ray."""
import numpy as np

from rcmot.fusion.radar_camera import FusionParams as _FusionParams, cluster_candidates, fuse, gate_clusters, load_cluster_model, range_clusters
from rcmot.types import Det2D, RadarFrame
from rcmot.utils.calib import RigCalib


def FusionParams(**kw):  # noqa: N802  (these tests cover the legacy point-range fusion; lik is the default since docs/38)
    return _FusionParams(**{"range_model": "legacy", **kw})

R_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
CAL = RigCalib(np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], float), np.zeros(5), (1280, 720),
               np.eye(3), np.zeros(3), R_OPT, -R_OPT @ np.array([0, 0, 1.6]))


def _box(p, w=0.6, h=1.75):
    c = np.array([[p[0] + dx, p[1] + dy, z] for dx in (-w / 2, w / 2) for dy in (-w / 2, w / 2) for z in (0, h)])
    uv, _ = CAL.project(c)
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def _frame(worker, clutter):
    rng = np.random.default_rng(1)
    pts = [worker + rng.normal(0, [0.1, 0.1, 0.2], (3, 3)) + [0, 0, 0.9]]           # torso returns
    pts.append(clutter + rng.normal(0, [0.1, 0.4, 0.3], (6, 3)) + [0, 0, 0.9])       # a wall behind, wider
    xyz = np.vstack(pts)
    vr = np.zeros(len(xyz))
    rf = RadarFrame(0.0, xyz, vr, np.zeros(len(xyz)), np.zeros(len(xyz)), xyz)
    rf.v_r_comp = vr.copy()
    return rf


WORKER, CLUTTER = np.array([20.0, 1.0, 0.0]), np.array([28.0, 1.4, 0.0])
NEW = dict(cluster_score="lr", point_bearing=True, ground_gate_mono_fallback=True)
OLD = dict(cluster_score="off", point_bearing=False, ground_gate_mono_fallback=False)


def test_range_clusters_single_linkage():
    lab = range_clusters(np.array([10.0, 10.5, 28.0, 11.2, 27.5]), 1.0)
    assert lab.tolist() == [0, 0, 1, 0, 1]


def test_scorer_prefers_height_consistent_cluster():
    rf = _frame(WORKER, CLUTTER)
    box = _box(WORKER)
    uv, _ = CAL.project(rf.xyz)
    cc = cluster_candidates(np.arange(len(rf.xyz)), rf.xyz, uv, rf.v_r, box, CAL.K[0, 0], load_cluster_model(FusionParams().cluster_score_model), 1.0)
    assert len(cc) == 2
    near, far = sorted(cc, key=lambda c: c[1])
    assert abs(near[1] - 20) < 0.5 and abs(far[1] - 28) < 0.8
    assert near[3] > far[3]                                           # the 20 m cluster matches the box height


def test_gate_rejects_far_cluster_and_weights_near():
    cc = [(np.array([0]), 20.0, np.array([20.0, 1.0]), 0.5), (np.array([1]), 28.0, np.array([28.0, 1.4]), 0.5)]
    w = gate_clusters(cc, (np.array([20.3, 1.1]), np.eye(2) * 0.3 ** 2, 0.2), 9.21, 0.5, 2.5, 0.5)
    assert w[0] > 0.5 and w[1] == 0.0


def test_fuse_new_path_picks_worker_and_point_bearing():
    rf = _frame(WORKER, CLUTTER)
    d = Det2D("worker", 0.9, _box(WORKER))
    out = fuse([d], rf, CAL, params=FusionParams(**NEW))
    assert out and out[0].source.startswith("radar")
    assert abs(np.linalg.norm(out[0].pos[:2] - CAL.camera_center()[:2]) - 20.0) < 0.6
    assert abs(out[0].pos[1] - 1.0) < 0.4                             # lateral position from the points' centroid


def test_gate_falls_back_to_camera_when_no_cluster_passes():
    rf = _frame(WORKER, CLUTTER)
    d = Det2D("worker", 0.9, _box(WORKER))
    prior = (np.array([14.0, -3.0]), np.eye(2) * 0.3 ** 2, 0.1)         # the track predicts somewhere else
    kw = dict(range_priors=[(14.3, 0.4)], xy_priors=[prior])
    out = fuse([d], rf, CAL, params=FusionParams(cluster_gate_mono_override=0.0, **NEW), **kw)
    assert out and out[0].source == "mono+track" and out[0].n_radar == 0
    out_kde = fuse([d], rf, CAL, params=FusionParams(cluster_gate_fail="kde", cluster_gate_mono_override=0.0, **NEW), **kw)
    assert out_kde and out_kde[0].source.startswith("radar")
    out_ov = fuse([d], rf, CAL, params=FusionParams(**NEW), **kw)      # the 20 m cluster agrees with the box height:
    assert out_ov and out_ov[0].source.startswith("radar") and out_ov[0].dbg.get("override")   # override the track


def test_old_path_unchanged_with_switches_off():
    rf = _frame(WORKER, CLUTTER)
    d = Det2D("worker", 0.9, _box(WORKER))
    a = fuse([d], rf, CAL, params=FusionParams(**OLD))
    b = fuse([d], rf, CAL, params=FusionParams(**OLD), xy_priors=[(np.array([14.0, -3.0]), np.eye(2), 0.1)])
    assert a and b and np.allclose(a[0].pos, b[0].pos) and a[0].source == b[0].source and a[0].dbg is None


def test_ground_gate_mono_fallback_keeps_true_points():
    """xpanner calibration (docs/30 §4): the level frame puts the camera 0.27 m above the origin while the real
    ground is 1.33 m lower (feet at z = -1.33 in that frame). The ground ray of a 20 m worker then lands at ~3.4 m
    and the old gate |r - 3.4| <= 4.7 m drops every true return; the mono fallback keeps them."""
    cal = RigCalib(CAL.K, np.zeros(5), (1280, 720), np.eye(3), np.zeros(3), R_OPT, -R_OPT @ np.array([0, 0, 0.27]))
    feet_z = -1.33
    rng = np.random.default_rng(2)
    xyz = WORKER + rng.normal(0, [0.1, 0.1, 0.2], (4, 3)) + [0, 0, 0.9 + feet_z]
    rf = RadarFrame(0.0, xyz, np.zeros(4), np.zeros(4), np.zeros(4), xyz)
    rf.v_r_comp = np.zeros(4)
    c = np.array([[WORKER[0] + dx, WORKER[1] + dy, z] for dx in (-.3, .3) for dy in (-.3, .3) for z in (feet_z, feet_z + 1.75)])
    uv, _ = cal.project(c)
    box = np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])
    g = cal.pixel_ray_to_ground(0.5 * (box[0] + box[2]), box[3])
    assert g is not None and np.linalg.norm(g[:2]) < 5.0                # the broken ground ray
    d = Det2D("worker", 0.9, box)
    old = fuse([d], rf, cal, params=FusionParams(**OLD))
    new = fuse([d], rf, cal, params=FusionParams(cluster_score="off", point_bearing=False, ground_gate_mono_fallback=True))
    assert old and old[0].n_radar == 0
    assert new and new[0].n_radar == 4 and abs(np.linalg.norm(new[0].pos[:2] - cal.camera_center()[:2]) - 20.0) < 0.6
