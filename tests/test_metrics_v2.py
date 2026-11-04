"""Schema-v2 evaluation (docs/13 "평가 지표 v2"): IDSWR / FNR / FPR, MOTP[m] on ref3d, BEV 2 m matching, recall by
ref3d range and by occlusion, ego-state frame split, and the segment helpers of scripts/label_auto_meta.py."""
import importlib.util
from pathlib import Path

import numpy as np

from rcmot.labeling.gt import GT
from rcmot.labeling.metrics import Frame, bev_sim, evaluate, evaluate_bev, motp_range, rates

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_mot_gt")
am = _load("label_auto_meta")

A, B = [0, 0, 10, 20], [50, 0, 60, 20]


def _f(gt, pr, ign=None):
    """gt: [(id, box, range, xy, quality)], pr: [(id, box, range, xy)]"""
    return Frame(np.array([g[0] for g in gt]), np.array([g[1] for g in gt], float).reshape(-1, 4),
                 np.array([p[0] for p in pr]), np.array([p[1] for p in pr], float).reshape(-1, 4),
                 None if ign is None else np.array(ign),
                 gt_range=np.array([g[2] for g in gt], float), gt_xy=np.array([g[3] for g in gt], float).reshape(-1, 2),
                 gt_quality=np.array([g[4] for g in gt], object),
                 pr_range=np.array([p[2] for p in pr], float), pr_xy=np.array([p[3] for p in pr], float).reshape(-1, 2))


def test_rates_and_idswr():
    fr = [_f([(1, A, 10.0, [10, 0], "dense")], [(7 if k < 5 else 8, A, 10.3, [10.3, 0])]) for k in range(10)]
    r = evaluate(fr)
    r.update(rates(r))
    assert r["IDsw"] == 1 and abs(r["IDSWR"] - 0.1) < 1e-9
    assert r["FNR"] == 0 and r["FPR_gt"] == 0 and r["FPR_pred"] == 0
    fr += [_f([(1, A, 10.0, [10, 0], "dense")], [])] * 2 + [_f([], [(9, B, 5.0, [5, 0])])] * 4   # 2 FN, 4 FP
    r = evaluate(fr)
    r.update(rates(r))
    assert r["FN"] == 2 and r["FP"] == 4 and abs(r["FNR"] - 2 / 12) < 1e-9
    assert abs(r["FPR_gt"] - 4 / 12) < 1e-9 and abs(r["FPR_pred"] - 4 / 14) < 1e-9
    r = evaluate([_f([], [(9, B, 5.0, [5, 0])])] * 3)                 # no regular GT of the class: ratios undefined
    r.update(rates(r))
    assert r["FP"] == 3 and r["IDSWR"] is None and r["FPR_gt"] is None and r["FPR_pred"] == 1.0


def test_motp_m_by_range_and_quality():
    fr = []
    for k in range(10):
        fr.append(_f([(1, A, 8.0, [8, 0], "dense"), (2, B, 25.0, [25, 3], "sparse"), (3, [100, 0, 110, 20], 15.0, [15, -2], "none")],
                     [(7, A, 8.5, [8.5, 0]), (8, B, 24.0, [24, 3]), (9, [100, 0, 110, 20], 15.2, [15.2, -2])]))
    m = motp_range(fr, 0.5)
    assert m["n"] == 20 and m["n_tp"] == 30 and m["n_tp_without_ref"] == 10      # id 3 has no reference
    assert abs(m["mean"] - 0.75) < 1e-9 and abs(m["median"] - 0.75) < 1e-9 and abs(m["p90"] - 1.0) < 1e-9
    assert abs(m["bias"] - 0.5 * (0.5 - 1.0)) < 1e-9
    assert m["by_range"]["0-10"] == {"n": 10, "mean": 0.5, "median": 0.5, "p90": 0.5}
    assert m["by_range"]["20-30"]["n"] == 10 and m["by_range"]["10-20"]["n"] == 0 and m["by_range"]["30-inf"]["n"] == 0
    assert m["by_quality"]["dense"]["mean"] == 0.5 and m["by_quality"]["sparse"]["mean"] == 1.0
    # matching is 2D: a prediction 3 m off in range but on the box still pairs (the range error is measured, not gated)
    fr2 = [_f([(1, A, 10.0, [10, 0], "dense")], [(7, A, 13.0, [13, 0])])]
    assert motp_range(fr2, 0.5)["mean"] == 3.0
    # frames without the 3D columns contribute nothing
    plain = [Frame(np.array([1]), np.array([A], float), np.array([7]), np.array([A], float))]
    assert motp_range(plain, 0.5)["n"] == 0 and motp_range(plain, 0.5)["n_tp"] == 1


def test_bev2m_matching():
    S = bev_sim(np.array([[10.0, 0.0], [np.nan, np.nan]]), np.array([[11.0, 0.0], [10.0, 2.5]]))
    assert abs(S[0, 0] - 0.5) < 1e-9 and S[0, 1] == 0 and S[1].sum() == 0
    # box IoU says "miss" (shifted box) but the BEV centre is 1 m away -> TP under bev2m, FN+FP under IoU
    shifted = [30, 0, 40, 20]
    fr = [_f([(1, A, 10.0, [10, 0], "dense"), (3, B, 20.0, [np.nan, np.nan], "none")],
             [(7, shifted, 11.0, [11, 0]), (8, B, 20.0, [20, 5])]) for _ in range(5)]
    r2d = evaluate(fr)
    assert r2d["TP"] == 5 and r2d["FN"] == 5 and r2d["FP"] == 5
    rb = evaluate_bev(fr)
    assert rb["match"] == "bev2m" and rb["n_gt_boxes"] == 5          # GT without a reference is left out ...
    assert rb["TP"] == 5 and rb["FN"] == 0 and rb["FP"] == 0 and rb["MOTA"] == 1.0 and rb["pred_on_noref"] == 5
    assert rb["MOTP_m"]["n"] == 5 and rb["MOTP_m"]["mean"] == 1.0   # ... together with the prediction on it
    # a prediction that overlaps no GT at all stays an FP under bev2m
    ghost = [_f([(1, A, 10.0, [10, 0], "dense")], [(7, A, 10.5, [10.5, 0]), (9, [300, 300, 310, 320], 30.0, [30, 0])])]
    assert evaluate_bev(ghost)["FP"] == 1
    # beyond 2 m: no match
    far = [_f([(1, A, 10.0, [10, 0], "dense")], [(7, A, 12.5, [12.5, 0])])]
    assert evaluate_bev(far)["TP"] == 0 and evaluate(far)["TP"] == 1
    # an ignore GT still absorbs by 2D IoU before the BEV rule
    ign = [_f([(1, A, 10.0, [10, 0], "dense"), (2, B, 20.0, [20, 5], "dense")], [(7, A, 10.5, [10.5, 0]), (8, B, 20.0, [20, 5])],
              ign=[False, True])]
    rb = evaluate_bev(ign)
    assert rb["TP"] == 1 and rb["FP"] == 0 and rb["pred_in_ignore"] == 1


def _gt(tmp_path, n=20):
    frames = []
    for k in range(n):
        boxes = [{"id": 1, "cls": "worker", "xyxy": [100, 100, 140, 200], "flags": [], "src": "det", "occ": "none",
                  "ref3d": {"range_m": 8.0 + 0.5 * k, "xy_level": [8.0 + 0.5 * k, 0.0], "n_lidar": 12, "quality": "dense", "src": "lidar"}},
                 {"id": 2, "cls": "worker", "xyxy": [400, 100, 440, 200], "flags": [], "src": "det", "occ": "partial",
                  "ref3d": {"range_m": None, "xy_level": None, "n_lidar": 0, "quality": "none", "src": "lidar"}}]
        frames.append({"stamp": 100.0 + 0.2 * k, "bag_t": 0.2 * k, "file": f"frames/{k:04d}.jpg", "boxes": boxes})
    d = {"meta": {"start": 100.0, "end": 100.0 + 0.2 * (n - 1), "ego_state": [[0, 9, "static"], [10, 19, "swing"]]},
         "ids": {"1": {"cls": "worker", "desc": ""}, "2": {"cls": "worker", "desc": ""}}, "frames": frames, "review": {}, "notes": []}
    import json
    (tmp_path / "gt.json").write_text(json.dumps(d))
    return GT(tmp_path / "gt.json")


def _pred(gt, miss_far=True):
    """id 1 tracked (range +0.4 m) except beyond 15 m when miss_far; id 2 always tracked."""
    pred = {}
    for k, f in enumerate(gt.frames):
        r = f["boxes"][0]["ref3d"]["range_m"]
        pb = [(22, "worker", "CONFIRMED", np.array([400, 100, 440, 200], float), 0.0, True, np.nan, np.full(2, np.nan))]
        if not (miss_far and r >= 15.0):
            pb.append((11, "worker", "CONFIRMED", np.array([100, 100, 140, 200], float), 0.0, True, r + 0.4, np.array([r + 0.4, 0.0])))
        pred[k] = pb
    return pred


def test_recall_by_range_and_occ(tmp_path):
    gt = _gt(tmp_path)
    pred = _pred(gt)
    rr = ev.recall_by_range(gt, pred)
    assert rr["r0-10"] == {"n": 4, "recall": 1.0} and rr["r10-20"]["n"] == 16 and abs(rr["r10-20"]["recall"] - 10 / 16) < 1e-9
    assert rr["r20-30"]["n"] == 0 and rr["no_ref"] == {"n": 20, "recall": 1.0}
    ro = ev.recall_by_occ(gt, pred)
    assert ro["none"] == {"n": 20, "recall": 0.7} and ro["partial"] == {"n": 20, "recall": 1.0} and ro["full"]["n"] == 0
    for f in gt.frames:
        for b in f["boxes"]:
            b.pop("occ")
    assert ev.recall_by_occ(gt, pred) is None


def test_frames_with_ref3d_columns_and_ego_split(tmp_path):
    gt = _gt(tmp_path)
    pred = _pred(gt, miss_far=False)
    fr = ev.class_frames(gt, pred, "worker")
    assert fr[0].gt_range[0] == 8.0 and np.isnan(fr[0].gt_range[1]) and list(fr[0].gt_quality) == ["dense", "none"]
    assert np.isnan(fr[0].pr_range[0]) and fr[0].pr_range[1] == 8.4
    r = evaluate(fr)
    assert r["MOTA"] == 1.0
    m = ev._motp_clean(fr, 0.5)
    assert m["n"] == 20 and abs(m["mean"] - 0.4) < 1e-9 and m["n_tp_without_ref"] == 20
    ego = ev.ego_state_frames(gt)
    assert ego["static"] == list(range(10)) and ego["swing"] == list(range(10, 20)) and ego["moving"] == ego["swing"]
    sub = ev.class_frames(gt, _pred(gt), "worker", ego["swing"])
    e = evaluate(sub)
    assert e["FN"] == 6 and e["TP"] == 14                      # id 1 missed at >= 15 m (k = 14..19), id 2 always tracked
    assert ev.has_ref3d(gt)


def test_auto_meta_runs():
    stamps = np.arange(0, 4.0, 0.2)                                  # 20 frames at 5 Hz
    lab = ["static"] * 8 + ["swing"] * 2 + ["static"] * 10           # 0.4 s blip
    runs = am.merge_short(am.runs_of(lab), stamps, 1.0)
    assert runs == [[0, 19, "static"]]
    lab = ["static"] * 8 + ["swing"] * 7 + ["drive"] * 5
    assert am.merge_short(am.runs_of(lab), stamps, 1.0) == [[0, 7, "static"], [8, 14, "swing"], [15, 19, "drive"]]
    assert am.fill_unknown([None, "a", None, None, "b", None], "x") == ["a", "a", "a", "a", "b", "b"]
    assert am.fill_unknown([None, None], "x") == ["x", "x"]
    # motion draft: a worker walking 1 m/s for the first 2 s, then standing
    frames = [{"boxes": [{"id": 1, "cls": "worker", "xyxy": [0, 0, 1, 1], "flags": []}]} for _ in stamps]
    xy = {}
    for k, s in enumerate(stamps):
        xy[(k, 1)] = np.array([min(s, 2.0), 0.0])
    segs, n_known, n_all = am.motion_draft(frames, stamps, 1, xy, 0.3, 0.5, 1.0)
    assert n_all == 20 and n_known == 20
    assert segs[0][2] == "walk" and segs[-1][2] == "static" and segs[0][0] == 0 and segs[-1][1] == 19
    # no reference at all -> one mixed run
    assert am.motion_draft(frames, stamps, 1, {}, 0.3, 0.5, 1.0)[0] == [[0, 19, "mixed"]]


def test_ref3d_smoothing_and_jump_stats():
    efd = _load("eval_fusion_depth")

    def box(i, r, q="sparse", ign=False):
        return {"id": i, "cls": "worker", "xyxy": [0, 0, 10, 20], "flags": ["ignore"] if ign else [],
                "ref3d": {"range_m": r, "xy_level": None if r is None else [r, 0.0], "n_lidar": 4, "quality": "none" if r is None else q, "src": "lidar"}}
    # id 1: one outlier in the middle, id 2: outlier at the start (two consistent followers), id 3: genuine walk
    r1 = [25.0, 25.1, 32.0, 25.2, 25.3]
    r2 = [19.7, 13.7, 13.8, 13.7, 13.6]
    r3 = [10.0, 10.2, 10.4, 10.6, 10.8]
    frames = [{"boxes": [box(1, a), box(2, b), box(3, c, ign=True)]} for a, b, c in zip(r1, r2, r3)]
    before = efd.jump_stats(frames)
    assert before["all"] == {"pairs": 12, "jumps": 3, "jump_frac": 0.25} and before["regular"]["pairs"] == 8
    sm = efd.smooth_refs(frames, thr=2.0)
    assert sm["outliers"] == 2
    assert frames[2]["boxes"][0]["ref3d"]["quality"] == "none" and frames[2]["boxes"][0]["ref3d"]["range_m"] is None
    assert frames[0]["boxes"][1]["ref3d"]["quality"] == "none"
    after = efd.jump_stats(frames)
    assert after["all"]["jumps"] == 0
    got = [f["boxes"][2]["ref3d"]["range_m"] for f in frames]
    assert got == r3                                               # monotonic series: median of 3 changes nothing
    assert frames[3]["boxes"][0]["ref3d"]["range_m"] == 25.2       # neighbours of the outlier keep their value
