import numpy as np

from rcmot.labeling.metrics import Frame, evaluate


def _f(gt, pr, ign=None):
    gi = np.array([g[0] for g in gt]); gb = np.array([g[1] for g in gt], float).reshape(-1, 4)
    pi = np.array([p[0] for p in pr]); pb = np.array([p[1] for p in pr], float).reshape(-1, 4)
    return Frame(gi, gb, pi, pb, None if ign is None else np.array(ign))


A, B = [0, 0, 10, 20], [50, 0, 60, 20]


def test_perfect_tracking_scores_one():
    fr = [_f([(1, A), (2, B)], [(7, A), (9, B)]) for _ in range(10)]
    r = evaluate(fr)
    assert r["MOTA"] == 1 and r["IDF1"] == 1 and r["IDsw"] == 0
    assert abs(r["HOTA"] - 1) < 1e-9 and abs(r["AssA"] - 1) < 1e-9


def test_id_switch_halves_association():
    fr = [_f([(1, A)], [(7 if k < 5 else 8, A)]) for k in range(10)]
    r = evaluate(fr)
    assert r["IDsw"] == 1 and r["MOTA"] == 0.9 and abs(r["IDF1"] - 0.5) < 1e-9
    assert abs(r["DetA"] - 1) < 1e-9 and abs(r["AssA"] - 0.5) < 1e-9       # each half: 5/(10+5-5)
    assert abs(r["HOTA"] - np.sqrt(0.5)) < 1e-9


def test_duplicate_ghost_and_ignore():
    fr = [_f([(1, A), (2, B)], [(7, A), (8, [1, 0, 11, 20]), (9, [200, 0, 210, 20]), (5, B)], ign=[False, True])]
    r = evaluate(fr)
    assert r["TP"] == 1 and r["FP"] == 2 and r["pred_in_ignore"] == 1
    assert r["FP_breakdown"] == {"duplicate": 1, "mislocated": 0, "ghost": 1}
