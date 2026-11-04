"""docs/41 §10: BEV cell memory of static radar returns (stationary mode)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.radar.cell_memory import CellMemory, CellMemoryParams, learn_mask  # noqa: E402


def test_background_score_learns_persistent_cells_and_clears_on_motion():
    m = CellMemory(CellMemoryParams(enable=True, window_s=5.0, min_learn_s=1.0))
    post = np.array([[20.0, 3.0, 1.0]])
    person = np.array([[15.0, 0.0, 1.0]])
    t = 0.0
    for _ in range(30):                              # 3 s at 10 Hz: the post every scan, the person never learned
        m.step(t, stationary=True)
        m.learn(t, post, np.array([True]))
        t += 0.1
    m.step(t, stationary=True)
    assert m.valid(t)
    s = m.score(np.vstack([post, person, [[20.2, 3.1, 1.0]]]))
    assert s[0] == 1.0 and s[1] == 0.0 and s[2] == 1.0   # neighbouring cell of the post is background too
    assert m.factor(1.0) == 0.05 and m.factor(0.0) == 1.0 and 0.1 < m.factor(0.5) < 0.2
    m.step(t + 0.1, stationary=False)                # the machine moves: memory cleared, not valid
    assert not m.valid(t + 0.1) and m.score(post)[0] == 0.0


def test_learn_mask_excludes_tracks_and_person_wedge():
    p = CellMemoryParams(enable=True)
    xyz = np.array([[10.0, 0.0, 1.0],                # on a worker track
                    [15.0, 0.0, 1.0],                # in the person box's wedge, inside its range window
                    [25.0, 0.0, 1.0],                # same wedge, far behind the window -> learned
                    [20.0, 8.0, 1.0]])               # elsewhere -> learned
    stats = SimpleNamespace(keep=np.ones(4, bool), v_comp=np.zeros(4), sigma_v=0.02)
    trk = [SimpleNamespace(cls="worker", state=SimpleNamespace(name="CONFIRMED"), pos=np.array([10.0, 0.0, 1.0]))]
    rl = SimpleNamespace(summ={"r": 15.0, "sig": 0.3, "sig_full": 0.5})
    det = SimpleNamespace(cls="worker", pos=np.array([15.0, 0.0, 1.0]), box2d=np.array([600.0, 300.0, 680.0, 500.0]), rlik=rl)
    calib = SimpleNamespace(pixel_ray=lambda u, v: np.array([1.0, (640.0 - u) / 1000.0, 0.0]))   # az ~ (640-u)/1000 rad
    m = learn_mask(stats, xyz, np.zeros(3), trk, [det], p, calib)
    assert list(m) == [False, False, True, True]
