"""World pose (swing-axis model, standstill bias) and the soil-pile work monitor (docs/24)."""
import numpy as np

from rcmot.work.monitor import WorkMonitor, WorkParams
from rcmot.world import WorldParams, WorldPose, WorldState


class _Calib:                                            # only what WorkMonitor needs without stereo
    R_cl = np.eye(3)
    t_cl = np.zeros(3)
    K = np.eye(3)


def test_inplace_swing_keeps_axis():
    radar_t = np.array([0.14, -0.05, 1.56])
    wp = WorldPose(WorldParams(lever_arm=[1.17, -0.89]), radar_t)
    r = wp.lever
    w, dt = np.radians(60), 1 / 15
    for k in range(int(round(3 / dt))):                  # 180 deg in 3 s, Doppler = w x r
        st = wp.update(k * dt, dt, w, w * np.array([-r[1], r[0]]))
    assert np.allclose(st.axis, -r, atol=1e-6)
    assert abs(np.degrees(st.theta) - 180) < 1e-6
    assert np.allclose(st.origin, -2 * r, atol=1e-6)    # sensor on the opposite side of the axis
    assert np.allclose(st.to_world([[0, 0, 0]])[0], st.origin)


def test_standstill_learns_bias_and_holds_yaw():
    wp = WorldPose(WorldParams(bias_tau_s=2.0))
    for k in range(300):
        st = wp.update(k * 0.1, 0.1, np.radians(0.7), [0.0, 0.0])
    assert st.still and abs(st.theta) < 1e-9
    assert abs(np.degrees(st.bias) - 0.7) < 0.01


def test_drive_without_lever_arm():
    wp = WorldPose(WorldParams())
    for k in range(20):
        st = wp.update(k * 0.1, 0.1, 0.0, [1.0, 0.0])
    assert np.allclose(st.origin, [2.0, 0.0])
    assert len(st.trail()) > 10


def _still_world(t):
    return WorldState(t, 0.0, np.zeros(2), np.zeros(2), np.zeros(2), True, 0.0, 0.0, [np.zeros(2)], 1, None, 0)


def _cone(cx, h, R, step=0.05):
    g = np.arange(-R, R + step, step)
    X, Y = np.meshgrid(g, g)
    Z = h * np.clip(1 - np.hypot(X, Y) / R, 0, None)
    return np.c_[X.ravel() + cx, Y.ravel(), Z.ravel()]


def test_pile_volume_and_cut():
    mon = WorkMonitor(WorkParams(enable=True, alpha_radar=0.5, ref_min_obs=3, r_min=0.0), _Calib())
    pile = _cone(5.0, 1.0, 2.0)
    for k in range(10):
        st = mon.update(k * 0.1, 0.1, _still_world(k * 0.1), pile, None, [])
    v_true = np.pi * 2.0 ** 2 * 1.0 / 3
    assert abs(st.volume - v_true) / v_true < 0.05     # "filled": footprint incl. the thin toe, no threshold
    flat = pile.copy()
    flat[:, 2] = 0.0
    for k in range(10, 40):                              # the pile is dug away
        st = mon.update(k * 0.1, 0.1, _still_world(k * 0.1), flat, None, [])
    assert st.volume < 0.05 * v_true
    assert abs(st.cut - v_true) / v_true < 0.3 and st.fill < 0.1


def _site(gz=0.0, tilt_deg=0.0, holes=False, half=6.0, step=0.05):
    """Ground +-half m around a 1.4 m / R 2.5 m cone at x = 6; ground (and cone) raised gz and tilted along x;
    holes: 0.6 m wide empty stripes every 2 m across the pile (LiDAR rings / boom shadow)."""
    g = np.arange(-half, half + step, step)
    X, Y = np.meshgrid(g + 6.0, g)
    Z = 1.4 * np.clip(1 - np.hypot(X - 6.0, Y) / 2.5, 0, None)
    P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    if holes:
        P = P[~((np.mod(P[:, 1] + 0.3, 2.0) < 0.6) & (np.hypot(P[:, 0] - 6, P[:, 1]) < 2.2))]
    P[:, 2] += gz + np.tan(np.radians(tilt_deg)) * (P[:, 0] - 6.0)
    return P, np.pi * 2.5 ** 2 * 1.4 / 3


def _run(P, **kw):
    mon = WorkMonitor(WorkParams(enable=True, alpha_radar=1.0, r_min=0.0, reach_m=0, **kw), _Calib())
    for k in range(8):
        st = mon.update(k * 0.6, 0.6, _still_world(k * 0.6), P, None, [])
    return st


def test_filled_volume_local_ground_and_holes():
    P, v = _site()
    assert abs(_run(P).volume / v - 1) < 0.04                                   # full cone: (near) exact
    P, v = _site(gz=0.08, tilt_deg=1.5, holes=True)                            # the 155530 situation (docs/34 §4.1)
    st = _run(P)
    assert abs(st.volume / v - 1) < 0.08 and st.fill_frac > 0.1                # own ground + interpolated stripes
    old = _run(P, volume_method="threshold").volume
    assert abs(old / v - 1) > 0.15                                              # the old rule is far off here


def test_ground_radar_shift():
    """ground_radar: the pile's ring plane is shifted to the radar heights on its ring (docs/40)."""
    P, v = _site()
    mon = WorkMonitor(WorkParams(enable=True, alpha_radar=1.0, r_min=0.0, reach_m=0, ground_radar=True, ground_radar_min=50),
                      _Calib())
    for k in range(8):
        st = mon.update(k * 0.6, 0.6, _still_world(k * 0.6), P, None, [])
    assert abs(st.volume / v - 1) < 0.04 and abs(mon.pa["piles"][0]["g_off"]) < 0.01   # radar = the map: no shift
    g = np.arange(-6.0, 6.0, 0.1)
    gx, gy = np.meshgrid(g + 6.0, g)
    far = np.hypot(gx - 6.0, gy) > 2.6
    mon.rad_buf = [(10.0, np.c_[gx[far], gy[far]].astype(np.float32), np.full(far.sum(), 0.08, np.float32))]
    pa = mon._pile_analysis(mon.H, 1000.0)               # radar: the ground is 8 cm above the map's ring (long after:
                                                         # the time smoothing has converged)
    q = pa["piles"][0]
    assert abs(q["g_off"] - 0.08) < 0.01
    assert pa["volume"] < st.volume - 0.06 * q["area"]
    q2 = mon._pile_analysis(mon.H, 1000.5)["piles"][0]  # 0.5 s later, radar gone: the shift is held, not dropped
    mon.rad_buf = []
    q3 = mon._pile_analysis(mon.H, 1001.0)["piles"][0]
    assert abs(q2["g_off"] - 0.08) < 0.01 and abs(q3["g_off"] - 0.08) < 0.01


def test_separate_piles_in_snapshot():
    mon = WorkMonitor(WorkParams(enable=True, alpha_radar=0.5, r_min=0.0), _Calib())
    a, b = _cone(4.0, 1.2, 2.0), _cone(4.0, 0.8, 1.5)
    b[:, 1] += 5.0                                       # second, smaller pile 5 m to the side
    pts = np.r_[a, b]
    for k in range(30):
        st = mon.update(k * 0.1, 0.1, _still_world(k * 0.1), pts, None, [])
    piles = st.snapshot.piles
    assert len(piles) == 2
    (x0, y0, z0, v0, _, h0, _), (x1, y1, _, v1, _, _, _) = piles
    assert v0 > v1 and abs(v0 + v1 - st.volume) < 1e-3   # largest first, together = the total
    assert abs(x0 - 4.0) < 0.3 and abs(y0) < 0.3 and abs(y1 - 5.0) < 0.3
    assert abs(h0 - 1.2) < 0.15 and abs(z0 - 1.2) < 0.15
    assert np.isnan(st.t_map)                            # radar only: no dense-depth update yet


def test_work_cycle_ledger():
    mon = WorkMonitor(WorkParams(enable=True, alpha_radar=0.5, r_min=0.0), _Calib())
    for k in range(30):
        st = mon.update(k * 0.1, 0.1, _still_world(k * 0.1), _cone(4.0, 1.2, 2.0), None, [])
    v_pile = mon._site_volume(mon.H, (4.0, 0.0))
    assert abs(v_pile - st.volume) < 1e-3 and mon._site_volume(mon.H, (-6.0, 0.0)) == 0.0
    nan = float("nan")
    #            t   kind          cut  fill  hdg  t_down x    y   off  before after
    mon.events = [(10, "dig",       0.6, 0.0, 0, 9, 4.0, 0.0, 0, 5.0, 4.4),
                  (12, "dig+sweep", 0.2, 0.0, 0, 9, 4.3, 0.2, 0, 4.4, 4.2),   # second look, same site
                  (20, "dump",      0.0, 0.9, 0, 19, 0.0, 6.0, 0, 0.0, 0.9),
                  (30, "dig",       0.7, 0.0, 0, 29, 4.0, 0.0, 0, 4.2, 3.5),
                  (40, "dump",      0.0, 0.5, 0, 39, 0.0, 6.0, 0, nan, nan)]   # older event tuple: no site volumes
    c = mon.work_cycles()
    assert len(c) == 2
    assert c[0]["dig_before"] == 5.0 and c[0]["dig_after"] == 4.2 and abs(c[0]["cut"] - 0.8) < 1e-9
    assert c[0]["fill"] == 0.9 and c[0]["dump_after"] == 0.9
    assert c[1]["cut"] == 0.7 and c[1]["fill"] == 0.5


def test_swing_cycles():
    mon = WorkMonitor(WorkParams(enable=True), _Calib())
    t, n = 0.0, 0
    for w_dps, dur in ((0, 1), (60, 1.5), (0, 2), (-60, 1.5), (0, 2), (60, 1.5), (0, 2), (-60, 1.5), (0, 2)):
        for _ in range(int(dur / 0.05)):
            t += 0.05
            w = WorldState(t, 0.0, np.zeros(2), np.zeros(2), np.zeros(2), False, 0.0, np.radians(w_dps), [np.zeros(2)], 1, None, 0)
            st = mon.update(t, 0.05, w, np.zeros((0, 3)), None, [])
    assert st.n_cycles == 2
    assert abs(st.cycle_s - 7.0) < 0.2


def test_swing_axis_tilt_keeps_ground_level_across_headings():
    """Level frame = the ground ahead, 2 deg off the swing axis (docs/25 §3.1): the same sloped ground seen 90 deg
    to the side must land on the calib plane (no false pile), which it does not without the axis correction."""
    k = np.radians([2.0, -0.3])
    lever = np.array([1.0, 0.5])                              # level origin from the swing axis
    c = np.r_[-lever, 0.0]

    class Calib(_Calib):
        swing_axis_level = np.r_[k, 1.0] / np.linalg.norm(np.r_[k, 1.0])

    def world(t, th):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return WorldState(t, th, R @ lever, np.zeros(2), lever, True, 0.0, 0.0, [np.zeros(2)], 1, None, 0)

    g = np.arange(3.0, 9.0, 0.1)
    X, Y = np.meshgrid(g, g - 6.0)
    ahead = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]   # level frame, ground in front at heading 0
    R_ax = WorkMonitor(WorkParams(), Calib()).R_axis
    to_axis = lambda P: (P - c) @ R_ax.T + c                  # noqa: E731
    gA = to_axis(ahead)                                       # true ground plane in the axis frame (heading 0 = world)
    coef = np.linalg.lstsq(np.c_[world(0, 0.0).to_world(gA), np.ones(len(gA))], gA[:, 2], rcond=None)[0]

    def seen_at(th):                                          # the same world ground, cab turned by th
        P = ahead.copy()
        for _ in range(4):
            A = to_axis(P)
            w = world(0, th).to_world(A)
            P[:, 2] += np.c_[w, np.ones(len(w))] @ coef - A[:, 2]
        return P

    side = seen_at(np.radians(90))
    for tilt in (True, False):
        mon = WorkMonitor(WorkParams(enable=True, alpha_radar=1.0, axis_tilt=tilt, r_min=0.0), Calib())
        for i in range(5):
            mon.update(i * 0.1, 0.1, world(i * 0.1, 0.0), ahead, None, [])
        for i in range(5, 10):
            st = mon.update(i * 0.1, 0.1, world(i * 0.1, np.radians(90)), side, None, [])
        fin = np.isfinite(mon.H)
        dev = np.abs(mon.H - mon._ground_grid())[fin]
        if tilt:
            assert dev.max() < 0.03 and st.volume < 0.05
        else:
            assert dev.max() > 0.15
