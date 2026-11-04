"""docs/21 E6-4: image-shake estimate and the camera covariance terms (box-size pixel sigma, shake)."""
import numpy as np

from rcmot.fusion.image_shake import ImageShake
from rcmot.fusion.radar_camera import FusionParams, bearing_sigma_rad, mono_range, pitch_shake_range_sigma
from rcmot.utils.calib import RigCalib

R_OPT = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)
CAL = RigCalib(np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], float), np.zeros(5), (1280, 720),
               np.eye(3), np.zeros(3), R_OPT, -R_OPT @ np.array([0, 0, 1.5]))


def _scene(rng):
    big = rng.uniform(0, 255, (900, 1700)).astype(np.float32)
    import cv2
    big = cv2.GaussianBlur(big, (0, 0), 3)
    return np.clip(big, 0, 255).astype(np.uint8)


def _frame(big, x, y):
    x, y = int(round(x)), int(round(y))
    g = big[90 + y: 90 + y + 720, 200 + x: 200 + x + 1280]
    return np.dstack([g, g, g])


def test_steady_pan_is_not_shake_but_a_jolt_is():
    rng = np.random.default_rng(0)
    big = _scene(rng)
    sh = ImageShake()
    out = []
    for k in range(12):                       # steady pan 8 px/frame (a swing), jolt of +12 px vertical at k=8
        out.append(sh.update(_frame(big, 8 * k, 12 if k == 8 else 0)))
    assert out[0] is None
    steady = np.array(out[3:8])
    assert steady[:, 0].max() < 2.0 and steady[:, 1].max() < 2.0
    assert out[8][1] > 8.0                    # the jolt shows up vertically


def test_covariance_terms_off_by_default_and_grow_when_on():
    box = np.array([600.0, 300.0, 720.0, 500.0])              # 120 x 200 px (near worker)
    off = FusionParams()
    assert np.isclose(bearing_sigma_rad(off, box, CAL, (5.0, 5.0)), np.radians(off.bearing_sigma_deg))
    assert pitch_shake_range_sigma(off, 20.0, CAL, (5.0, 5.0)) == 0.0
    on = FusionParams(px_sigma_rel=0.05, shake_gain=1.0)
    b_on = bearing_sigma_rad(on, box, CAL, (5.0, 0.0))
    assert b_on > np.radians(off.bearing_sigma_deg)            # 0.05 * 120 px = 6 px (0.43 deg) > 0.3 deg floor
    assert np.isclose(b_on, np.hypot(0.05 * 120 / 800, 5.0 / 800))
    small = np.array([600.0, 300.0, 640.0, 400.0])            # 40 px wide: the 0.3 deg floor wins
    assert np.isclose(bearing_sigma_rad(on, small, CAL, None), np.radians(on.bearing_sigma_deg))
    # pitch shake: 2 px at f=800 = 2.5 mrad, camera 1.5 m high, 20 m -> 20^2/1.5*0.0025 = 0.67 m
    assert np.isclose(pitch_shake_range_sigma(on, 20.0, CAL, (0.0, 2.0)), 400 / 1.5 * 2.0 / 800)
    # box-size pixel sigma raises the monocular range sigma only above the fixed floor
    r_off, s_off = mono_range(box, 1.8, CAL, off)
    r_on, s_on = mono_range(box, 1.8, CAL, on)
    assert np.isclose(r_off, r_on) and s_on > s_off


def test_rewind_rotation_roundtrip_and_sign():
    """docs/21 E2-c: a static point at bearing phi at scan time was at phi + a at image time when the sensor
    turned by a = yaw_rate * (t_scan - t_img) in between (z up, counter-clockwise positive)."""
    from rcmot.pipeline import _rotate_det, _rz
    from rcmot.types import Det3D
    a = np.radians(1.5)
    p_scan = np.array([20.0, 0.0, 0.0])
    p_img = _rz(a) @ p_scan
    assert np.isclose(np.degrees(np.arctan2(p_img[1], p_img[0])), 1.5)
    d = Det3D("worker", 0.9, np.zeros(4), p_img.copy(), np.diag([1.0, 0.01, 0.1]), vel=np.array([0.0, 1.0, 0.0]),
              vel_cov=np.eye(3) * 0.1)
    _rotate_det(d, _rz(-a))
    assert np.allclose(d.pos, p_scan) and np.isclose(np.trace(d.pos_cov), 1.11)
    assert np.isclose(np.linalg.norm(d.vel), 1.0)
