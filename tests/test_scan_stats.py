import numpy as np

from rcmot.radar.scan_stats import ScanStatsParams, ScanStatistician, combine_contrast, local_contrast_polar, snr_saturated


def _scene(rng, v_s, sigma, n=250):
    xyz = np.c_[rng.uniform(3, 45, n), rng.uniform(-20, 20, n), rng.uniform(-1.5, 2, n)]
    d = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    vr = -d @ v_s + rng.normal(0, sigma, n)
    r = np.linalg.norm(xyz, axis=1)
    snr = np.clip(35 - 0.4 * r + rng.normal(0, 3, n), 0, 31)          # range dependent + saturation
    rcs = rng.normal(-10, 4, n)
    return xyz, vr, snr, rcs


def test_noise_scale_ego_and_dynamic_labels():
    rng = np.random.default_rng(0)
    v_s = np.array([0.8, 0.05, 0.0])
    for sigma in (0.012, 0.04):
        xyz, vr, snr, rcs = _scene(rng, v_s, sigma)
        obj = np.array([[12.0, 3.0, 0.5], [12.2, 3.1, 1.0], [12.1, 2.9, 0.2], [11.9, 3.0, 0.8]])
        d = obj / np.linalg.norm(obj, axis=1, keepdims=True)
        vr_o = d @ (np.array([0.0, 1.2, 0.0]) - v_s)                     # worker walking 1.2 m/s across
        st = ScanStatistician().compute(0.0, np.vstack([xyz, obj]), np.r_[vr, vr_o], np.r_[snr, [25] * 4],
                                        np.r_[rcs, [-5] * 4])
        assert st.ego_valid and np.allclose(st.v_sensor[:2], v_s[:2], atol=0.02)
        assert 0.6 * sigma < st.sigma_v < 1.6 * sigma, (sigma, st.sigma_v)
        assert st.dynamic[-4:].all()
        assert st.dynamic[:-4].mean() < 0.03                              # ~ 3-sigma tail only


def test_range_z_is_gain_invariant():
    rng = np.random.default_rng(1)
    xyz, vr, snr, rcs = _scene(rng, np.zeros(3), 0.01)
    a = ScanStatistician().compute(0.0, xyz, vr, snr, rcs)
    b = ScanStatistician().compute(0.0, xyz, vr, snr - 8.0, rcs + 6.0)    # other radar gain / setup
    assert np.allclose(a.z_snr, b.z_snr) and np.allclose(a.z_rcs, b.z_rcs)
    assert np.allclose(a.alpha, b.alpha)


def test_ambiguity_pair_keeps_stronger():
    """Default: the member with the lower RCS goes, also when both sit at the 31 dB SNR cap; ambig_source='snr' keeps
    the old rule (lower range-binned SNR z goes)."""
    rng = np.random.default_rng(2)
    xyz, vr, snr, rcs = _scene(rng, np.zeros(3), 0.01, n=120)
    peer = np.full(120, 65535.0)
    peer[5], peer[6] = 6, 5
    snr[5], snr[6] = 31.0, 31.0
    rcs[5], rcs[6] = -5.0, 3.0
    st = ScanStatistician().compute(0.0, xyz, vr, snr, rcs, peer=peer)
    assert not st.keep[5] and st.keep[6]
    snr[5], snr[6] = 10.0, 28.0
    rcs[5], rcs[6] = 3.0, -5.0
    old = ScanStatistician(ScanStatsParams(ambig_source="snr")).compute(0.0, xyz, vr, snr, rcs, peer=peer)
    assert not old.keep[5] and old.keep[6]


def test_contrast_ignores_saturated_snr_branch():
    """A corner reflector (+12 dBsm) among -15 dBsm clutter: its SNR sits at the sensor cap (31 dB) like 10 % of
    the clutter, so the SNR local contrast is ~0; the combined contrast must still flag it (indoor capture
    2026-10-24, where the plain min() found 0 reflectors)."""
    rng = np.random.default_rng(1)
    n = 300
    r, az = rng.uniform(3, 30, n), rng.uniform(-0.6, 0.6, n)
    rcs = rng.normal(-15, 3, n)
    snr = np.clip(rng.normal(27, 4, n), 0, 31)                     # ~16 % of the clutter at the cap
    rcs[0], snr[0] = 12.0, 31.0
    c_rcs, c_snr = local_contrast_polar(r, az, rcs), local_contrast_polar(r, az, snr)
    assert c_snr[0] < 3.0 < c_rcs[0]                               # 3.0 = reflectors.detect contrast_min
    assert np.minimum(c_rcs, c_snr)[0] < 3.0                       # the old rule hides it
    assert combine_contrast(c_rcs, c_snr, snr)[0] > 3.0
    assert snr_saturated(snr)[0] and snr_saturated(snr).mean() < 0.5
    # a sensor without a cap (no ties at the maximum): plain min, nothing marked saturated
    snr2 = rng.normal(24, 4, n)
    assert not snr_saturated(snr2).any()
    np.testing.assert_allclose(combine_contrast(c_rcs, c_snr, snr2), np.minimum(c_rcs, c_snr))


def test_ego_fit_in_level_frame_on_tilted_radar():
    """docs/23 §3.3: a radar pitched 26 deg down on a machine driving 1 m/s straight ahead. The radar-frame planar fit
    misses the motion along the radar z axis and reads the speed low; the level-frame fit recovers it."""
    rng = np.random.default_rng(3)
    th = np.radians(26.0)
    R_lr = np.array([[np.cos(th), 0.0, np.sin(th)], [0.0, 1.0, 0.0], [-np.sin(th), 0.0, np.cos(th)]])   # radar -> level
    n = 300
    xyz_l = np.c_[rng.uniform(3, 30, n), rng.uniform(-15, 15, n), rng.uniform(-3.3, 1.0, n)]          # level frame
    xyz_r = xyz_l @ R_lr                                                                               # = R_lr.T @ x
    v_l = np.array([1.0, 0.0, 0.0])
    d_r = xyz_r / np.linalg.norm(xyz_r, axis=1, keepdims=True)
    vr = -d_r @ (R_lr.T @ v_l) + rng.normal(0, 0.01, n)
    snr, rcs = np.full(n, 20.0), rng.normal(-10, 4, n)
    new = ScanStatistician().compute(0.0, xyz_r, vr, snr, rcs, R_level=R_lr)
    old = ScanStatistician(ScanStatsParams(ego_frame="radar")).compute(0.0, xyz_r, vr, snr, rcs, R_level=R_lr)
    sp_new = np.linalg.norm((R_lr @ new.v_sensor)[:2])
    sp_old = np.linalg.norm((R_lr @ old.v_sensor)[:2])
    assert abs(sp_new - 1.0) < 0.02
    assert sp_old < 0.97                                     # the old fit reads low
