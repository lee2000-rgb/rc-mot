import json
from pathlib import Path

import numpy as np

from rcmot.utils.rig_extrinsics import R_OPT, RigExtrinsics, sensor_to_level

ROOT = Path(__file__).resolve().parents[1]


def test_sensor_to_level_hdx_layout_is_rig_extrinsics_level():
    ex = RigExtrinsics(cam_rpy_deg=[0.1, -1.5, 0.9], ground_normal=[-0.01, 0.003, 0.9999], sensor_height=1.66)
    d = ex.to_calib_json(np.eye(3), np.zeros(5), (1280, 720))
    R, t = sensor_to_level(d)
    R0, t0 = ex.level()
    assert np.allclose(R, R0) and np.allclose(t, t0)


def test_sensor_to_level_xpanner_layout_matches_json_chain():
    """construction_field.json has no `extrinsics`: os_sensor -> level must reproduce the radar chain the json
    carries (T_base_level_radar = R_level^T T_base_radar) and the camera chain (T_cam_level includes dR)."""
    d = json.load(open(ROOT / "configs/calib/construction_field.json"))
    assert "extrinsics" not in d
    R, t = sensor_to_level(d)
    # the radar sits at a different base offset than os_sensor: compose radar -> os_sensor -> level and compare
    R_bs, t_bs = np.asarray(d["T_base_sensor"]["R"]), np.asarray(d["T_base_sensor"]["t"])
    R_br, t_br = np.asarray(d["T_base_radar"]["R"]), np.asarray(d["T_base_radar"]["t"])
    R_sr, t_sr = R_bs.T @ R_br, R_bs.T @ (t_br - t_bs)           # radar -> os_sensor
    assert np.allclose(R @ R_sr, d["T_base_level_radar"]["R"], atol=1e-9)
    assert np.allclose(R @ t_sr + t, d["T_base_level_radar"]["t"], atol=1e-9)
    # camera: p_cam = R_OPT (dR p_s - t_cam) with p_s = R^T (p_level - t)
    dR, t_cam = np.asarray(d["dR"]["R"]), np.asarray(d["t_cam_in_os_sensor"])
    R_cl = R_OPT @ dR @ R.T
    t_cl = R_OPT @ (-dR @ R.T @ t - t_cam)
    assert np.allclose(R_cl, d["T_cam_level"]["R"], atol=1e-9)
    assert np.allclose(t_cl, d["T_cam_level"]["t"], atol=1e-9)


def test_sensor_to_level_unknown_layout_raises():
    try:
        sensor_to_level({"K": []})
    except KeyError:
        return
    raise AssertionError("expected KeyError")
