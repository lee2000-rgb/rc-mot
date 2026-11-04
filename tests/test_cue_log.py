"""docs/37 phase 1: fuse(cue_log=...) is read-only (identical detections) and records every box, including boxes that
get no 3-D output; the Pipeline snapshot of the prior's track is taken before the update."""
import json

import numpy as np

from rcmot.fusion.radar_camera import FusionParams as _FusionParams, RangeReliability, fuse
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


def _frame(*clusters, n=3, seed=1):
    rng = np.random.default_rng(seed)
    pts = [c + rng.normal(0, [0.1, 0.05, 0.2], (n, 3)) + [0, 0, 0.9] for c in clusters]
    xyz = np.vstack(pts) if pts else np.zeros((0, 3))
    vr = np.zeros(len(xyz))
    rf = RadarFrame(0.0, xyz, vr, np.zeros(len(xyz)), np.zeros(len(xyz)), xyz)
    rf.v_r_comp = vr.copy()
    return rf


def _same(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.source == y.source and np.array_equal(x.pos, y.pos) and np.array_equal(x.pos_cov, y.pos_cov)
        assert x.n_radar == y.n_radar and x.dbg == y.dbg


def test_cue_log_read_only_and_complete():
    worker, clutter = np.array([20.0, 1.0, 0.0]), np.array([28.0, 1.4, 0.0])
    dets = [Det2D("worker", 0.9, _box(worker)), Det2D("worker", 0.9, _box(np.array([12.0, -4.0, 0.0])))]
    rf = _frame(worker, clutter)
    pri = [((20.0, 0.4)), None]
    xyp = [(worker[:2].copy(), np.eye(2) * 0.2, 0.1), None]
    p = FusionParams()
    a = fuse(dets, rf, CAL, params=p, range_priors=pri, xy_priors=xyp, reliability=RangeReliability())
    log = []
    b = fuse(dets, rf, CAL, params=p, range_priors=pri, xy_priors=xyp, reliability=RangeReliability(), cue_log=log)
    _same(a, b)
    assert [c["k"] for c in log] == [0, 1]
    json.dumps(log)                                               # JSON-ready
    c0 = log[0]
    assert c0["cc"] and c0["ch"] is not None and c0["rad"]["n"] >= 1 and c0["out"]["src"].startswith("radar")
    assert abs(1.8 / (c0["tan_top"] - c0["tan_bot"]) - c0["mono"][0]) < 1e-2 * c0["mono"][0]
    assert len(c0["cc"][0]["f"]) == 11 and c0["tp"] == [20.0, 0.4] and len(c0["xyp"]) == 6
    assert "pts" not in log[1] and log[1]["out"]["src"] in ("mono", "radar+mono", "ground_ray")
