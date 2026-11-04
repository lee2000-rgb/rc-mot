"""Online drive reconstruction (rcmot.work.recon, docs/26 §6): keyframe / chunk scheduling, the multi-view
accumulation and the ground refiner, without the VGGT model."""
import numpy as np

from rcmot.work import recon as R
from rcmot.world import WorldParams, WorldPose


class _Calib:
    R_cl = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)      # level x fwd -> camera z
    t_cl = np.zeros(3)
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    D = np.zeros(5)
    image_size = (1280, 720)

    def camera_center(self):
        return -self.R_cl.T @ self.t_cl


def _recon(**kw):
    p = R.ReconParams(enable=True, sync=True, **kw)
    r = R.DriveRecon(p, _Calib(), ".")
    r.runs = []
    r._run = lambda s, i0, i1, final: r.runs.append((i0, i1, final))   # no model in the tests
    return r


def test_model_shape_matches_vggt_balanced():
    assert R.model_shape(720, 1280, 512) == (384, 688)
    assert R.model_shape(1512, 2016, 512) == (448, 592)


def test_keyframes_and_chunks_while_driving():
    r = _recon(chunk=16, overlap=4, kf_dist=1.0)
    wp = WorldPose(WorldParams())
    img = np.zeros((720, 1280, 3), np.uint8)
    t, n = 0.0, 0
    while n < 46:                                          # 1 m/s straight, 15 Hz
        st = wp.update(t, 1 / 15, 0.0, [1.0, 0.0])
        rs = r.update(t, img, st, np.zeros((0, 3)))
        t += 1 / 15
        n = rs.n_keyframes
    assert rs.n_keyframes == 46
    assert r.runs[:3] == [(0, 16, False), (12, 28, False), (24, 40, False)]
    assert r.s.kf[1].T_W_lvl[0, 3] >= 1.0 - 1e-6           # one keyframe per metre
    # standing still afterwards: a keyframe every kf_max_dt = 8 s but at most 2 in a row; once >= 10 views wait with
    # >= 3 m of travel and the last moving keyframe is 12 s old, they are mapped as a short, final chunk
    for k in range(450):                                   # 30 s
        st = wp.update(t, 1 / 15, 0.0, [0.0, 0.0])
        rs = r.update(t, img, st, np.zeros((0, 3)))
        t += 1 / 15
    assert r.runs[-1] == (36, 47, True)                    # 46 moving + the first standstill keyframe (8 s)
    assert rs.n_keyframes == 48 and r.s.next_start == 43   # the second standstill keyframe (16 s); no third


def test_turning_in_place_makes_keyframes_by_angle():
    r = _recon(kf_deg=8.0)
    wp = WorldPose(WorldParams(lever_arm=[1.17, -0.89]), np.array([0.1, 0.0, 1.5]))
    img = np.zeros((720, 1280, 3), np.uint8)
    rr = wp.lever
    w = np.radians(30)
    n = None
    for k in range(60):                                    # 30 deg/s for 4 s = 120 deg -> 15 keyframes + the first
        st = wp.update(k / 15, 1 / 15, w, w * np.array([-rr[1], rr[0]]))
        n = r.update(k / 15, img, st, np.zeros((0, 3))).n_keyframes
    assert 14 <= n <= 17


def test_people_mask_from_detector():
    from rcmot.types import Det2D

    r = _recon()
    img = np.zeros((720, 1280, 3), np.uint8)
    m = np.ones((100, 50), bool)
    d = Det2D("worker", 0.9, np.array([600.0, 300.0, 650.0, 400.0]), mask=m)
    wp = WorldPose(WorldParams())
    st = wp.update(0.0, 1 / 15, 0.0, [0.0, 0.0])
    r.update(0.0, img, st, np.zeros((0, 3)), [d])
    pm = r.s.kf[0].pmask
    assert pm is not None and pm.shape == (384, 688)
    assert pm[int(350 * 384 / 720), int(625 * 688 / 1280)] and not pm[10, 10]


def test_accumulate_keeps_multiview_points_and_rechecks_pending():
    r = _recon(min_views=2, voxel=0.1)
    s = r.s
    rng = np.random.default_rng(0)
    ground = np.c_[rng.uniform(0, 10, 2000), rng.uniform(-3, 3, 2000), np.zeros(2000)]
    far = np.c_[rng.uniform(50, 60, 500), rng.uniform(-3, 3, 500), np.zeros(500)]   # seen once only
    C = np.full((2500, 3), 128, np.uint8)
    # chunk 1: keyframe 0 sees the ground + the far patch, keyframe 1 sees the ground again (shifted copy)
    P = np.concatenate([ground, far, ground + [0.02, 0.0, 0.0]])
    K = np.r_[np.zeros(2500, np.int64), np.ones(2000, np.int64)]
    r._accumulate(s, P.astype(np.float32), np.concatenate([C, C[:2000]]), K, 1.0)
    assert len(s.P) > 0 and s.P[:, 0].max() < 11.0          # the far patch is pending, not in the map
    assert len(s.pending) == 1 and 500 <= len(s.pending[0][0]) <= 520    # + ground points on 0.3 m voxel edges
    # chunk 2: keyframe 2 sees the far patch -> the pending points enter the map
    r._accumulate(s, (far + [0.02, 0, 0]).astype(np.float32), C[:500], np.full(500, 2, np.int64), 2.0)
    assert s.P[:, 0].max() > 50.0 and sum(len(x[0]) for x in s.pending) <= 60   # only 0.3 m voxel-edge points stay
    assert len(np.unique(R.voxel_keys(s.P, 0.1))) == len(s.P)     # one point per 0.1 m voxel
    assert s.snapshot is not None and len(s.snapshot.xyz) == len(s.P)


def test_ground_refiner_levels_a_tilted_keyframe():
    p = R.ReconParams(gr_max_deg=0.5, gr_max_dz=0.1, gr_dist=1.0)
    g = R._GroundRefiner(p)
    rng = np.random.default_rng(1)
    T_cam = np.linalg.inv(R.se3(_Calib.R_cl, _Calib.t_cl))                   # camera at the level origin
    T_cam[2, 3] = 1.6
    pts = np.c_[rng.uniform(2, 18, 3000), rng.uniform(-6, 6, 3000), np.zeros(3000)]   # ground, level frame
    local = R.apply(np.linalg.inv(T_cam), pts)                                # camera-frame points
    g.step(0, local, T_cam)                                                   # the map
    tilt = R.se3(R.Rotation.from_rotvec([0, np.radians(0.3), 0]).as_matrix(), [0, 0, 0.04])
    T_bad = tilt @ T_cam                                                      # keyframe 1: 0.3 deg pitch, 4 cm high
    T_ref = g.step(1, local, T_bad)
    err_before = np.degrees(R.Rotation.from_matrix(T_bad[:3, :3] @ T_cam[:3, :3].T).magnitude())
    err_after = np.degrees(R.Rotation.from_matrix(T_ref[:3, :3] @ T_cam[:3, :3].T).magnitude())
    assert err_after < 0.4 * err_before and abs(T_ref[2, 3] - 1.6) < 0.02
    # the correction is carried to the next keyframe (chain)
    T_next = g.corr @ T_bad
    assert np.allclose(T_next, T_ref, atol=1e-9)
