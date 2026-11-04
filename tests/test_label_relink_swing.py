"""scripts/label_relink_swing.py on a synthetic two-sweep segment (docs/13 "선회 구간 재연결"): standing people
get their sweep-1 ids back on sweep 2 only when azimuth AND ref3d range agree, boxes without ref3d keep their id,
and a second pass changes nothing."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("label_relink_swing", ROOT / "scripts/label_relink_swing.py")
rl = importlib.util.module_from_spec(spec)
sys.modules["label_relink_swing"] = rl                # dataclass resolution needs the module registered
spec.loader.exec_module(rl)

F = 800.0                                           # synthetic pinhole: az_level = (u - 640) / F


def az_of_pixel(u, v):
    return (u - 640.0) / F


PEOPLE = {                                          # old id per sweep, world azimuth [rad], range [m]
    "A": ((1, 4), 0.20, 15.0),
    "B": ((2, 5), 0.30, 15.0),
    "C": ((3, 6), 0.20, 25.0),                      # same azimuth as A, 10 m farther: never confused with A
    "D": ((None, 7), 0.20, 20.0),                   # sweep 2 only, A's azimuth, 5 m off A and C: stays its own id
}


def _doc():
    """Sweep 1 = frames 0-9 (heading 0 -> 0.5 rad), gap of 10 frames, sweep 2 = frames 20-29 (0.5 -> 0)."""
    frames, theta = [], []
    for k in range(30):
        if 10 <= k < 20:
            continue
        th = 0.5 * k / 9 if k < 10 else 0.5 * (29 - k) / 9
        boxes = []
        for name, (ids, azw, rng) in PEOPLE.items():
            old = ids[0] if k < 10 else ids[1]
            if old is None:
                continue
            u = 640.0 + (azw - th) * F
            h = 100.0 * 15.0 / rng
            b = {"id": old, "cls": "worker", "xyxy": [u - 10, 300.0, u + 10, 300.0 + h], "flags": [], "src": "det",
                 "conf": 0.9, "occ": "none", "ignore_reason": None,
                 "ref3d": {"range_m": rng, "xy_level": [rng * math.cos(azw - th), rng * math.sin(azw - th)],
                           "n_lidar": 10, "quality": "dense", "src": "lidar"}}
            if name == "A" and k == 25:             # one box without a LiDAR reference inside A's sweep-2 chain
                b["ref3d"] = {"range_m": None, "xy_level": None, "n_lidar": 0, "quality": "none", "src": "lidar"}
            boxes.append(b)
        frames.append({"stamp": 100.0 + 0.2 * k, "bag_t": 100.1 + 0.2 * k, "file": f"frames/{k:04d}.jpg", "boxes": boxes})
        theta.append(th)
    ids = {str(i): {"cls": "worker", "desc": "", "role": "target", "motion": [[0, 9, "static"]]} for i in range(1, 8)}
    return {"meta": {"schema_version": 2, "stamps": [f["stamp"] for f in frames], "provenance": []}, "ids": ids,
            "frames": frames, "review": {"ids": {k: "checked" for k in ids}, "frames": {}}, "notes": []}, np.array(theta)


def _ids_of(doc, old_or_new):
    out = {}
    for f in doc["frames"]:
        for b in f["boxes"]:
            out.setdefault(b["id"], []).append(f["stamp"])
    return out


def test_two_sweeps_merge_on_azimuth_and_range():
    doc, theta = _doc()
    plan = rl.relink(doc, theta, az_of_pixel, rl.Params())
    assert plan["n_no_az"] == 0
    assert plan["merges"] == 3 and plan["splits"] == 0                  # 4->1, 5->2, 6->3
    assert plan["old_of_new"][1] == {1, 4} and plan["old_of_new"][2] == {2, 5} and plan["old_of_new"][3] == {3, 6}
    assert plan["old_of_new"][7] == {7}                                 # D: azimuth agrees with A, range does not
    assert len(plan["changes"]) == 9 + 10 + 10                          # ref3d boxes of ids 4 (9: one has no ref3d), 5, 6
    assert [(k, old, new) for k, _, old, new in plan["kept"]] == [(15, 4, 1)]  # frame 25 is the 16th frame (index 15)
    assert plan["conflicts"] == []                                      # the kept id 4 meets no other id-4 box


def test_apply_keeps_no_ref3d_box_and_is_idempotent():
    doc, theta = _doc()
    plan = rl.relink(doc, theta, az_of_pixel, rl.Params())
    done = rl.apply(doc, plan)
    ids = _ids_of(doc, "new")
    assert set(ids) == {1, 2, 3, 4, 7}                                  # 4 survives only on the no-ref3d box
    assert len(ids[1]) == 19 and len(ids[4]) == 1 and len(ids[7]) == 10
    kept_box = [b for f in doc["frames"] for b in f["boxes"] if b["id"] == 4][0]
    assert kept_box["ref3d"]["quality"] == "none"
    assert done["dropped_ids"] == [5, 6] and set(doc["ids"]) == {"1", "2", "3", "4", "7"}
    assert "motion" not in doc["ids"]["1"] and doc["ids"]["7"]["motion"]   # changed ids lose the draft, others keep it
    assert doc["review"]["ids"]["1"] == "draft" and doc["review"]["ids"]["7"] == "checked"
    # second pass: nothing to renumber
    plan2 = rl.relink(doc, theta, az_of_pixel, rl.Params())
    assert plan2["merges"] == 0 and plan2["splits"] == 0 and plan2["changes"] == []
    assert [(old, new) for _, _, old, new in plan2["kept"]] == [(4, 1)]


def test_no_cross_sweep_merge_without_ref3d():
    doc, theta = _doc()
    for f in doc["frames"]:                                             # sweep 2 of B loses every LiDAR reference
        for b in f["boxes"]:
            if b["id"] == 5:
                b["ref3d"] = {"range_m": None, "xy_level": None, "n_lidar": 0, "quality": "none", "src": "lidar"}
    plan = rl.relink(doc, theta, az_of_pixel, rl.Params())
    assert plan["merges"] == 2 and 5 not in plan["old_of_new"].get(2, set())
    assert all(old != 5 for _, _, old, _ in plan["changes"])


def test_heading_gap_leaves_boxes_unlinked():
    doc, theta = _doc()
    theta = theta.copy()
    theta[10:] = np.nan                                                 # no pipeline pose on sweep 2
    plan = rl.relink(doc, theta, az_of_pixel, rl.Params())
    assert plan["n_no_az"] > 0 and plan["merges"] == 0 and plan["changes"] == []


def test_cli_dry_run_writes_nothing(tmp_path, monkeypatch):
    doc, theta = _doc()
    seg = tmp_path / "seg"
    seg.mkdir()
    json.dump(doc, open(seg / "gt.json", "w"))
    monkeypatch.setattr(rl, "pipeline_headings", lambda *a, **k: (theta, {"n_poses": len(theta)}))

    class _Cal:
        def pixel_ray(self, u, v):
            a = az_of_pixel(u, v)
            return np.array([math.cos(a), math.sin(a), 0.0])

    import rcmot.utils.calib as calib_mod
    monkeypatch.setattr(calib_mod.RigCalib, "load", classmethod(lambda cls, p: _Cal()))
    doc["meta"]["calib"] = "x"
    json.dump(doc, open(seg / "gt.json", "w"))
    before = (seg / "gt.json").read_text()
    monkeypatch.setattr("sys.argv", ["label_relink_swing.py", str(seg), "--dry-run"])
    rl.main()
    assert (seg / "gt.json").read_text() == before
    monkeypatch.setattr("sys.argv", ["label_relink_swing.py", str(seg)])
    rl.main()
    out = json.load(open(seg / "gt.json"))
    prov = out["meta"]["provenance"][-1]
    assert prov["stage"] == "relink" and prov["who"] == "label_relink_swing"
    assert (seg / "backup").is_dir()
