import numpy as np

from rcmot.fusion.radar_camera import RangeReliability, camera_range


def test_conflict_follows_the_source_the_radar_backed():
    """A crouching worker: the height prior says 12 m, the ground contact 6.7 m. After the radar has sided with
    the ground contact in past conflicts, the camera range follows it (no new track 5 m behind the person)."""
    rel = RangeReliability(min_n=5)
    for _ in range(30):
        rel.observe(6.7, 12.0, 6.75, conflict=True)
    r, src, conflict = camera_range((12.0, 0.6), 6.7, rel)
    assert abs(r[0] - 6.7) < 0.3                         # the height prior has learned to be unreliable here


def test_ground_contact_dropped_when_learned_unreliable():
    """While driving the ground contact scatters (pitch); once its learned sigma is large it is ignored."""
    rel = RangeReliability(min_n=5)
    rng = np.random.default_rng(0)
    for _ in range(40):
        rel.observe(10.0, 10.0 + rng.normal(0, 0.3), 10.0 * (1 + rng.normal(0, 0.3)), conflict=False)
    r, src, _ = camera_range((10.2, 0.4), 13.0, rel)
    assert src == "mono" and abs(r[0] - 10.2) < 1e-6


def test_consistent_sources_are_fused():
    r, src, conflict = camera_range((10.0, 0.5), 10.4, RangeReliability())
    assert not conflict and 10.0 < r[0] < 10.4 and r[1] < 0.83          # tighter than either source (prior 10 % / 8 %)


def test_mask_helpers():
    from rcmot.fusion.radar_camera import bottom_occlusion, mask_hits, mask_mass_centre

    full = np.zeros((100, 100), bool)
    full[20:80, 40:60] = True
    assert mask_hits(full, np.array([[50, 50], [10, 10], [-5, 50]])).tolist() == [True, False, False]
    crouch = np.zeros((40, 40), bool)
    crouch[20:, :] = True
    assert mask_mass_centre(crouch) > 0.6 and mask_mass_centre(full[20:80, 40:60]) < 0.55
    other = np.zeros((100, 100), bool)
    other[70:100, 30:70] = True                                   # someone in front covering the feet
    assert bottom_occlusion(np.array([40, 20, 60, 80.0]), full, [other], 0.15) == 0.0   # own mask covers it
    own = np.zeros((100, 100), bool)
    own[20:70, 40:60] = True
    assert bottom_occlusion(np.array([40, 20, 60, 80.0]), own, [other], 0.15) > 0.5


def test_range_proportional_ground_bias_is_learned_and_removed():
    """Pitch / slope error: the ground contact reads rel = k r too far (halt-mid: +10 % at 20 m)."""
    rel = RangeReliability(min_n=10)
    k = 0.005
    for r in np.linspace(8, 22, 60):
        rel.observe(r, None, r * (1 + k * r), conflict=False)
    assert abs(rel.ground_k() - k) < 1e-4
    assert abs(rel.correct_ground(20 * (1 + k * 20)) - 20) < 0.05
    assert rel.sigma_rel("ground") <= 0.031                      # spread after the bias is removed


def test_radar_occluder_in_the_lower_box_marks_the_legs_hidden():
    from rcmot.fusion.radar_camera import FusionParams, occluded_bottom
    box = np.array([100.0, 100.0, 140.0, 200.0])
    uv = np.array([[120.0, 180.0], [120.0, 180.0], [120.0, 120.0]])
    front = np.ones(3, bool)
    p = FusionParams()
    # a barrier 12.7 m away, 0.6 m above ground, in front of a worker at 19 m -> hidden legs
    assert occluded_bottom(box, uv[:1], front[:1], np.array([12.7]), np.array([0.6]), np.array([0.9]), 19.0, p)
    # a ground return (z ~ 0) or a return in the upper half does not count
    assert not occluded_bottom(box, uv[1:2], front[:1], np.array([12.7]), np.array([0.05]), np.array([0.9]), 19.0, p)
    assert not occluded_bottom(box, uv[2:], front[:1], np.array([12.7]), np.array([0.6]), np.array([0.9]), 19.0, p)
