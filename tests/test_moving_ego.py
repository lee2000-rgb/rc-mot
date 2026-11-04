"""Moving-ego fusion (docs/30 §6-8, impl2): box-time rotation of the radar points, image-time propagation and
ego-noise inflation of the 2-D prior, lateral gate widening while turning, the old-pick safety net -- and the
static-ego path staying bit-identical with everything on."""
import dataclasses

import numpy as np

from rcmot.fusion.radar_camera import FusionParams as _FusionParams, fuse, gate_clusters, turn_az_sigma
from rcmot.pipeline import Pipeline, _rz, radar_type_of
from rcmot.types import Det2D, Det3D, RadarFrame
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


def _frame(worker, clutter=None, img_dt=0.0):
    rng = np.random.default_rng(1)
    pts = [worker + rng.normal(0, [0.1, 0.1, 0.2], (3, 3)) + [0, 0, 0.9]]
    if clutter is not None:
        pts.append(clutter + rng.normal(0, [0.1, 0.4, 0.3], (6, 3)) + [0, 0, 0.9])
    xyz = np.vstack(pts)
    vr = np.zeros(len(xyz))
    rf = RadarFrame(0.0, xyz, vr, np.zeros(len(xyz)), np.zeros(len(xyz)), xyz, img_dt=img_dt)
    rf.v_r_comp = vr.copy()
    return rf


def _pipe(**fusion):
    cfg = {"recovery": {"enable": False}, "topics": {"radar": "/radar/radar_detection_image"}, "fusion": {"range_model": "legacy", **fusion},
           "movers": {"enable": False}, "world": {"enable": False}}
    return Pipeline(_NoDet(), CAL, cfg)


def test_radar_type_from_config_or_topic():
    assert radar_type_of({"radar_type": "conti_ars548"}) == "conti"
    assert radar_type_of({"radar_type": "bosch_premium"}) == "bosch"
    assert radar_type_of({"topics": {"radar": "/radar/radar_detection_image"}}) == "conti"
    assert radar_type_of({"topics": {"radar": "/off_highway_premium_radar_sample_driver/locations"}}) == "bosch"


def test_box_time_offset_calib_vs_lag():
    rf = _frame(WORKER, img_dt=0.04)
    pipe = _pipe(radar_lag_s={"conti": -0.067})
    pipe.calib = dataclasses.replace(CAL, radar_time_offset_s=-0.067)
    assert abs(pipe._box_time_offset(rf) - 0.04) < 1e-9                  # calib == lag: img_dt only
    pipe.fusion_p = FusionParams(radar_lag_s={"conti": -0.03})
    assert abs(pipe._box_time_offset(rf) - (0.04 - 0.067 + 0.03)) < 1e-9  # the difference to the calib value
    pipe.fusion_p = FusionParams(radar_lag_s={"conti": "calib"})
    assert abs(pipe._box_time_offset(rf) - 0.04) < 1e-9


def test_gate_widened_laterally_while_turning():
    """A cluster 2 m beside the prior at 25 m: rejected by the static gate (chi2 and reach), accepted when the
    gate is widened by sigma_az x r across the line of sight; a cluster 2 m along the ray stays rejected."""
    prior = (np.array([25.0, 0.0]), np.eye(2) * 0.3 ** 2, 0.066)
    beside = [(np.array([0]), 25.0, np.array([25.0, 2.0]), 0.5)]
    along = [(np.array([0]), 27.0, np.array([27.0, 0.0]), 0.5)]
    assert gate_clusters(beside, prior, 9.21, 0.5, 2.5, 1.0)[0] == 0.0
    s_lat = turn_az_sigma(FusionParams(), np.radians(60), 0.066) * 25.0      # ~0.5 m at 25 m
    assert 0.3 < s_lat < 1.0
    assert gate_clusters(beside, prior, 9.21, 0.5, 2.5, 1.0, lat_sigma=s_lat, origin=np.zeros(2))[0] > 0.0
    assert gate_clusters(along, prior, 9.21, 0.5, 2.5, 1.0, lat_sigma=s_lat, origin=np.zeros(2))[0] == 0.0


def test_safety_net_keeps_old_pick_on_moving_ego():
    """A wrong track prediction (3 m beside both clusters, 24 m) gates out every cluster; the old KDE x camera prior
    picks the worker's 20 m cluster, so with the safety net that cluster stays available and is chosen instead of
    the camera fallback. Static ego: the net is off by default (camera fallback as in impl1)."""
    rf = _frame(WORKER, CLUTTER)
    det = Det2D("worker", 0.9, _box(WORKER))
    xy = [(np.array([24.0, 4.0]), np.eye(2) * 0.3 ** 2, 0.066)]
    rp = [(24.0, 0.5)]
    on = fuse([det], rf, CAL, params=FusionParams(), xy_priors=xy, range_priors=rp, ego_static=False)
    off = fuse([det], rf, CAL, params=FusionParams(cluster_gate_safety_m=0.0), xy_priors=xy, range_priors=rp, ego_static=False)
    static = fuse([det], rf, CAL, params=FusionParams(), xy_priors=xy, range_priors=rp, ego_static=True)
    assert on[0].source.startswith("radar") and abs(np.linalg.norm(on[0].pos[:2]) - 20) < 1.0
    assert on[0].dbg["safe"] == 1 and on[0].dbg["npass"] == 1
    assert off[0].dbg["npass"] == 0 and not off[0].source.startswith("radar")                 # camera fallback
    assert "safe" not in static[0].dbg and static[0].dbg.get("override")      # impl1's static-ego mono override instead


def test_box_time_rotation_brings_point_into_box():
    """Base turning at 60 deg/s, image 50 ms after the scan: the worker's return (scan frame) sits 1 m beside the
    box the camera drew at image time. Static ego (no rotation) misses it; with box_time the point is inside and
    the fused position comes back in the scan frame."""
    w, img_dt = np.radians(60), 0.05
    rf = _frame(WORKER, img_dt=img_dt)
    box = Det2D("worker", 0.9, _box(_rz(-w * img_dt) @ WORKER))      # what the camera sees at t_img
    pipe = _pipe()
    pipe.calib = dataclasses.replace(CAL, radar_time_offset_s=-0.067)
    v0 = np.zeros(3)
    miss = pipe._fuse([box], rf, w, v0, ego_static=True)
    hit = pipe._fuse([box], rf, w, v0, ego_static=False)
    assert miss[0].source in ("mono", "ground_ray") and miss[0].n_radar == 0
    assert hit[0].source.startswith("radar") and hit[0].n_radar == 3
    assert np.linalg.norm(hit[0].pos[:2] - WORKER[:2]) < 0.3            # scan-frame position restored
    assert pipe.fuse_boxes([box], rf, w)[0].n_radar == 0               # last step state: static (no step yet)


def test_prior_propagated_to_image_time_and_inflated():
    pipe = _pipe()
    trk = pipe.tracker
    det = Det3D("worker", 0.9, _box(WORKER), WORKER.copy(), np.eye(3) * 0.09)
    for _ in range(4):
        trk.update([det], 0.066)
    w, dt, img_dt = np.radians(30), 0.066, 0.05
    exp0 = (_rz(-w * dt) @ WORKER)[:2]
    exp1 = (_rz(-w * (dt + img_dt)) @ WORKER)[:2]
    box0 = Det2D("worker", 0.9, _box(WORKER))                          # old: track box projected unrotated
    box1 = Det2D("worker", 0.9, _box(np.r_[exp1, 0.0]))                # what the camera sees at image time
    _, _, _, xy0 = pipe._track_range_priors([box0], dt, np.zeros(3), w)
    _, _, _, xy1 = pipe._track_range_priors([box1], dt, np.zeros(3), w, img_dt=img_dt, inflate=True)
    _, _, _, miss = pipe._track_range_priors([box1], dt, np.zeros(3), w)
    assert xy0[0] is not None and xy1[0] is not None and miss[0] is None   # the old projection misses the moved box
    assert np.linalg.norm(xy0[0][0] - exp0) < 0.05 and np.linalg.norm(xy1[0][0] - exp1) < 0.05
    perp = np.array([-WORKER[1], WORKER[0]]) / np.linalg.norm(WORKER[:2])
    assert perp @ xy1[0][1] @ perp > perp @ xy0[0][1] @ perp            # lateral variance grew with sigma_w x r
    assert xy1[0][2] > xy0[0][2]


def test_static_ego_path_bit_identical_with_everything_on():
    """ego_static True / yaw rate below the turn threshold: the moving-ego machinery (box time, prior propagation,
    turn gate, safety net) must leave fuse() and the pipeline's fusion untouched."""
    rf = _frame(WORKER, CLUTTER, img_dt=0.04)
    det = Det2D("worker", 0.9, _box(WORKER))
    xy = [(np.array([20.3, 1.1]), np.eye(2) * 0.3 ** 2, 0.2)]
    off = FusionParams(box_time=False, prior_img_time=False, cluster_gate_safety_m=0.0, prior_sigma_w_deg=0.0,
                       prior_sigma_v_ego=0.0, gate_turn_yaw_deg=1e9)
    a = fuse([det], rf, CAL, params=FusionParams(), xy_priors=xy, ego_static=True, ego_yaw_rate=np.radians(2))
    b = fuse([det], rf, CAL, params=off, xy_priors=xy, ego_static=True, ego_yaw_rate=np.radians(2))
    assert np.array_equal(a[0].pos, b[0].pos) and np.array_equal(a[0].pos_cov, b[0].pos_cov) and a[0].source == b[0].source
    pa, pb = _pipe(), _pipe(**dataclasses.asdict(off))
    da = pa._fuse([det], rf, np.radians(2), np.zeros(3), ego_static=True, priors=(None, None, None, xy))
    db = pb._fuse([det], rf, np.radians(2), np.zeros(3), ego_static=True, priors=(None, None, None, xy))
    assert np.array_equal(da[0].pos, db[0].pos) and da[0].source == db[0].source
