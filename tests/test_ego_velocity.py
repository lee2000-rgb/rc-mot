import numpy as np

from rcmot.radar.ego_velocity import compensate_doppler, estimate_ego_velocity, estimate_object_velocity


def _scene(rng, v_s, n_static=200):
    xyz = rng.uniform([2, -15, -1], [40, 15, 3], (n_static, 3))
    vr = -(xyz / np.linalg.norm(xyz, axis=1, keepdims=True)) @ v_s + rng.normal(0, 0.03, n_static)
    return xyz, vr


def test_ego_velocity_with_moving_object():
    rng = np.random.default_rng(1)
    v_s = np.array([2.0, 0.3, 0.0])
    xyz, vr = _scene(rng, v_s)
    obj = rng.normal([15, 6, 0.5], [0.8, 1.5, 0.3], (40, 3))           # moving truck, 3 m/s along -y
    v_o = np.array([0.0, -3.0, 0.0])
    vr_o = (obj / np.linalg.norm(obj, axis=1, keepdims=True)) @ (v_o - v_s)
    res = estimate_ego_velocity(np.vstack([xyz, obj]), np.concatenate([vr, vr_o]), planar=True)
    assert res.valid and np.allclose(res.v_sensor[:2], v_s[:2], atol=0.05)
    assert res.inlier_mask[-40:].mean() < 0.2
    comp = compensate_doppler(obj, vr_o, res.v_sensor)
    ov = estimate_object_velocity(obj, comp)
    assert not ov.radial_only and np.allclose(ov.v[:2], v_o[:2], atol=0.3)
