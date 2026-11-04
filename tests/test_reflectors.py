"""Synthetic field scene: ground + wall + 4 corner reflectors on poles. The radar is rotated by 3 deg yaw /
1 deg pitch and shifted (0.2, -0.1, 0.05) m relative to the LiDAR; reflector anchors + ICP must recover it."""
import numpy as np
from scipy.spatial import cKDTree

from rcmot.calib import radar_lidar as rl
from rcmot.calib import reflectors as rf
from rcmot.utils.rig_extrinsics import RigExtrinsics, rot_rpy

H = 1.65
REFL = np.array([[8.0, -4.0, 0.6 - H], [12.0, 3.0, 0.6 - H], [18.0, -7.0, 0.6 - H], [22.0, 6.0, 0.6 - H]])


def _lidar(rng):
    g = np.c_[rng.uniform(2, 30, 20000), rng.uniform(-15, 15, 20000), np.full(20000, -H)]
    wall = np.c_[np.full(6000, 28.0), rng.uniform(-15, 15, 6000), rng.uniform(-H, 3, 6000)]
    side = np.c_[rng.uniform(5, 28, 4000), np.full(4000, 14.0), rng.uniform(-H, 2, 4000)]
    poles = [c + np.c_[rng.normal(0, .05, 150), rng.normal(0, .05, 150), rng.uniform(-0.6, 0.15, 150)] for c in REFL]
    return np.vstack([g, wall, side] + poles) + rng.normal(0, 0.01, (1, 3))


def _window(R_true, t_true, rng, n_frames=60):
    L = _lidar(rng)
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L)).voxel_down_sample(0.1)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    q, nq = np.asarray(pc.points), np.asarray(pc.normals)
    pts, rcs = [], []
    for _ in range(n_frames):
        clutter = L[rng.choice(len(L), 150)] + rng.normal(0, [0.1, 0.1, 0.4], (150, 3))   # radar elevation noise
        refl = REFL + rng.normal(0, [0.05, 0.05, 0.4], REFL.shape)
        pts += [clutter, refl]
        rcs += [rng.normal(0, 4, 150), rng.normal(20, 2, len(REFL))]
    S = np.vstack(pts)
    P = (S - t_true) @ R_true                      # os_sensor -> radar frame: p_r = R^T (p_s - t)
    w = rl.Window("synthetic", q, nq, cKDTree(q), P)
    w.rcs, w.n_frames, w.lidar_raw = np.concatenate(rcs), n_frames, L
    return w


def test_reflector_anchors_recover_radar_extrinsic():
    rng = np.random.default_rng(0)
    rpy_true, t_true = np.array([0.0, 1.0, 3.0]), np.array([0.2, -0.1, 0.05])
    w = _window(rot_rpy(rpy_true), t_true, rng)
    ex = RigExtrinsics(ground_normal=[0, 0, 1], sensor_height=H)          # initial: identity
    refs = rf.detect(w, ex, np.array([0, 0, 1.0]), H)
    anc = rf.anchors(refs)
    assert len(anc) == 4, [r.note for r in refs]
    R, t, _ = rl.joint_icp([w], anchors=anc)
    for _ in range(1):                                                     # re-match with the refined estimate
        ex.radar_rpy_deg, ex.radar_t = rl.rpy_deg(R).tolist(), t.tolist()
        anc = rf.anchors(rf.detect(w, ex, np.array([0, 0, 1.0]), H))
        R, t, _ = rl.joint_icp([w], R, t, anchors=anc)
    err = rl.rpy_deg(R) - rpy_true
    assert abs(err[2]) < 0.2 and abs(err[1]) < 0.4, err
    assert np.linalg.norm((t - t_true)[:2]) < 0.05, t
