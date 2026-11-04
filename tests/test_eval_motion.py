"""docs/21 metrics of scripts/eval_mot_gt.py: motion-state speeds, standstill entry delay, recall by box height."""
import importlib.util
import json
from pathlib import Path

import numpy as np

from rcmot.labeling.gt import GT

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("eval_mot_gt", ROOT / "scripts/eval_mot_gt.py")
ev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ev)


def _gt(tmp_path, n=20):
    frames = []
    for k in range(n):
        frames.append({"stamp": 100.0 + 0.2 * k, "bag_t": 0.2 * k, "file": f"frames/{k:04d}.jpg",
                       "boxes": [{"id": 1, "cls": "worker", "xyxy": [100, 100, 140, 200], "flags": [], "src": "det"},
                                 {"id": 2, "cls": "worker", "xyxy": [400, 300, 420, 340], "flags": [], "src": "det"}]})
    d = {"meta": {"start": 0, "end": 4}, "frames": frames, "notes": [], "review": {"ids": {}, "frames": {}},
         "ids": {"1": {"cls": "worker", "desc": "", "motion": [[0, 19, "static"]]},
                 "2": {"cls": "worker", "desc": "", "motion": [[0, 19, "walk"]]}}}
    p = tmp_path / "gt.json"
    p.write_text(json.dumps(d))
    return GT(p)


def test_motion_stats_and_entry_delay(tmp_path):
    g = _gt(tmp_path)
    pred = {}
    for k in range(20):
        static_flag = k >= 5                                     # standstill entered at frame 5 = 1.0 s
        speed1 = 0.5 if k < 5 else 0.0
        pred[k] = [(7, "worker", "CONFIRMED", np.array([101, 101, 141, 199.0]), speed1, static_flag),
                   (8, "worker", "CONFIRMED", np.array([400, 300, 420, 340.0]), 1.2, k == 10)]
    mo = ev.motion_stats(g, pred)
    assert mo["static"]["n"] == 20 and mo["static"]["frac_gt0.3"] == 0.25
    assert mo["static"]["frac_static_flag"] == 0.75
    assert mo["walk"]["frac_static_flag"] == 0.05 and mo["walk"]["p50"] == 1.2
    assert mo["static_entry"] == {"segments": 1, "never": 0, "delay_p50_s": 1.0, "delay_p90_s": 1.0}


def test_motion_stats_none_without_labels(tmp_path):
    g = _gt(tmp_path)
    for d in g.ids.values():
        d.pop("motion")
    assert ev.motion_stats(g, {}) is None


def test_recall_by_height(tmp_path):
    g = _gt(tmp_path, n=4)
    pred = {k: [(7, "worker", "CONFIRMED", np.array([100, 100, 140, 200.0]), 0.0, False)] for k in range(4)}
    rh = ev.recall_by_height(g, pred)
    assert rh["h0-60"] == {"n": 4, "recall": 0.0}             # 40 px box (id 2) never found
    assert rh["h60-120"] == {"n": 4, "recall": 1.0}           # 100 px box (id 1) always found
    assert rh["h120-inf"]["n"] == 0
