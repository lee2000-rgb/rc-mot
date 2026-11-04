"""Schema v2 label tools (docs/28 §3.3, gt_design §C/§D.5): GT.upgrade, the v2 setters, scripts/label_check.py and a
headless pass over the review GUI."""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rcmot.labeling.gt import GT, auto_flags

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _v1(tmp_path, n=40, rate=5.0, gaps=()):
    """v1 file: two workers + one all-ignore parked loader; id 2 has holes at `gaps` (frame indices)."""
    frames, stamps = [], []
    for k in range(n):
        st = 100.0 + k / rate
        stamps.append(st)
        boxes = [{"id": 1, "cls": "worker", "xyxy": [100 + k, 100, 140 + k, 200], "flags": [], "src": "det", "conf": .9},
                 {"id": 3, "cls": "heavy_equipment", "xyxy": [600, 300, 700, 360], "flags": ["ignore"], "src": "det",
                  "conf": .8}]
        if k not in gaps:
            boxes.append({"id": 2, "cls": "worker", "xyxy": [400, 300, 420, 340], "flags": ["occluded"], "src": "det",
                          "conf": .7})
        if k == 5:
            boxes.append({"id": 4, "cls": "worker", "xyxy": [900, 300, 910, 320], "flags": ["ignore"], "src": "det",
                          "conf": .3})
        frames.append({"stamp": st, "bag_t": st, "file": f"frames/{k:04d}.jpg", "boxes": boxes})
    d = {"meta": {"bag": "data/bags/hdx_boryeong_pg/20251125_4R_bosch_halt-short_LC",
                  "calib": "configs/calib/hdx_boryeong_pg_day1125.json", "config": "configs/hdx_boryeong.yaml",
                  "start": 0, "end": n / rate, "rate_hz": rate, "stamps": stamps, "stage": "agent-reviewed",
                  "split": "validation", "detector": "yolo11x.pt"},
         "ids": {"1": {"cls": "worker", "desc": "hi-vis", "motion": [[0, n - 1, "static"]]},
                 "2": {"cls": "worker", "desc": "black jacket", "motion": [[0, n - 1, "walk"]]},
                 "3": {"cls": "heavy_equipment", "desc": "parked wheel loader row", "motion": [[0, n - 1, "static"]]},
                 "4": {"cls": "worker", "desc": "tiny far person", "motion": [[5, 5, "mixed"]]}},
         "frames": frames, "review": {"ids": {"1": "checked"}, "frames": {}},
         "notes": [{"frame": 3, "stamp": stamps[3], "id": 2, "text": "merge candidate with draft id 7"}]}
    seg = tmp_path / "seg_halt_test"
    seg.mkdir(parents=True)
    (seg / "gt.json").write_text(json.dumps(d))
    return seg


# ---------------------------------------------------------------------------------------------------- upgrade
def test_upgrade_fills_defaults_and_is_idempotent(tmp_path):
    seg = _v1(tmp_path)
    g = GT(seg / "gt.json")
    ch = g.upgrade(who="test")
    assert ch and g.schema_version == 2
    m = g.meta
    assert m["split"] == "dev" and m["site"] == "boryeong_pg" and m["radar"] == "bosch_premium"
    assert m["lighting"] == "day" and m["ego_state"] == [[0, 39, "static"]] and m["image_size"] == [1280, 720]
    assert m["imu_topic"] == "/ouster/imu" and m["mount"].startswith("rig_1.66")
    assert [p["stage"] for p in m["provenance"]] == ["draft", "upgrade"]
    assert g.ids[3]["role"] == "background" and g.ids[3]["equipment_type"] == "loader"
    assert g.ids[1]["role"] == "target" and "equipment_type" not in g.ids[1]
    b2, b3, b4 = g.box_of(0, 2), g.box_of(0, 3), g.box_of(5, 4)
    assert b2["occ"] == "partial" and b2["ignore_reason"] is None and b2["ref3d"] is None
    assert b3["occ"] == "none" and b3["ignore_reason"] == "unidentifiable"
    assert b4["ignore_reason"] == "small"
    assert g.notes[0]["kind"] == "id" and g.notes[0]["status"] == "open"
    assert "agent" in g.review["passes"]
    g.save(backup=False)
    g2 = GT(seg / "gt.json")
    before = json.dumps(g2.to_dict(), sort_keys=True)
    assert g2.upgrade() == []
    assert json.dumps(g2.to_dict(), sort_keys=True) == before          # nothing changed, no provenance added
    assert len(g2.meta["provenance"]) == 2


def test_v1_file_still_readable_and_roundtrips(tmp_path):
    seg = _v1(tmp_path)
    raw = json.loads((seg / "gt.json").read_text())
    raw["custom_top_level"] = {"keep": 1}
    (seg / "gt.json").write_text(json.dumps(raw))
    g = GT(seg / "gt.json")
    assert g.schema_version == 1 and g.ego_state_at(0) is None
    g.save(backup=False)
    out = json.loads((seg / "gt.json").read_text())
    assert out["custom_top_level"] == {"keep": 1} and "passes" not in out["review"]
    assert out["frames"] == raw["frames"] and out["meta"] == raw["meta"]


# ---------------------------------------------------------------------------------------------------- setters
def test_setters(tmp_path):
    seg = _v1(tmp_path)
    g = GT(seg / "gt.json")
    g.upgrade()
    b = g.set_occ(0, 1, "full")
    assert b["occ"] == "full" and "occluded" in b["flags"]
    g.set_occ(0, 1, "none")
    assert "occluded" not in g.box_of(0, 1)["flags"]
    g.toggle_flag(g.box_of(0, 1), "occluded")                          # flag -> occ stays in step
    assert g.box_of(0, 1)["occ"] == "partial"
    b = g.set_ignore_reason(0, 1, "edge")
    assert "ignore" in b["flags"] and b["ignore_reason"] == "edge"
    g.set_ignore_reason(0, 1, "none")
    assert "ignore" not in g.box_of(0, 1)["flags"] and g.box_of(0, 1)["ignore_reason"] is None
    with pytest.raises(ValueError):
        g.set_ignore_reason(0, 1, "bogus")
    with pytest.raises(KeyError):
        g.set_occ(0, 99, "none")
    g.set_ref3d(0, 1, {"range_m": 12.345, "xy_level": [12.0, -2.0], "n_lidar": 9, "quality": "dense"})
    assert g.box_of(0, 1)["ref3d"] == {"range_m": 12.35, "xy_level": [12.0, -2.0], "n_lidar": 9, "quality": "dense",
                                       "src": "lidar"}
    g.set_ego_state([[0, 19, "static"], [20, 39, "swing"]])
    assert g.ego_state_at(25) == "swing"
    g.set_ego_segment(10, 29, "drive")
    assert g.meta["ego_state"] == [[0, 9, "static"], [10, 29, "drive"], [30, 39, "swing"]]
    with pytest.raises(ValueError):
        g.set_ego_state([[0, 10, "static"], [5, 20, "drive"]])
    with pytest.raises(ValueError):
        g.set_split("validation")
    g.set_split("test")
    g.set_meta(lighting="day_backlight", ego_rate={"yaw_rate_max_dps": 67.0, "speed_max_mps": 0.8})
    assert g.meta["split"] == "test" and g.meta["lighting"] == "day_backlight"
    with pytest.raises(ValueError):
        g.set_meta(site="mars")
    e = g.add_provenance("agent", "agent-A", "scripts/label_ops.py", "checked ids")
    assert e["stage"] == "agent" and e["time"] and "git" in e
    g.set_role(2, "background")
    g.set_equipment_type(3, "excavator")
    assert g.ids[2]["role"] == "background" and g.ids[3]["equipment_type"] == "excavator"
    n = g.note(4, 2, "which id?", kind="id", by="agent-A")
    assert n["status"] == "open" and n["stamp"] == g.frames[4]["stamp"]
    g.resolve_note(len(g.notes) - 1, by="user")
    assert g.notes[-1]["status"] == "resolved" and g.notes[-1]["resolved_by"] == "user"
    g.resolve_note(len(g.notes) - 1)
    assert g.notes[-1]["status"] == "open"
    p = g.bump_pass("user", "user", "frames_stepped")
    g.bump_pass("user", "user", "frames_stepped")
    assert p["frames_stepped"] == 2 and p["who"] == "user" and p["time"]
    nb = g.add_box(3, 7, [10, 10, 50, 120])
    assert nb["occ"] == "none" and nb["ref3d"] is None and g.ids[7]["role"] == "target"
    g.save(backup=False)
    g2 = GT(seg / "gt.json")
    assert g2.review["passes"]["user"]["frames_stepped"] == 2 and g2.meta["ego_state"][1] == [10, 29, "drive"]


def test_interpolate_gap_rule(tmp_path):
    seg = _v1(tmp_path, n=60, gaps=set(range(10, 13)) | set(range(20, 30)) | set(range(40, 58)))
    g = GT(seg / "gt.json")
    g.upgrade()
    n = g.interpolate(2)
    assert n == 3 + 10                                                   # 18-frame hole left empty
    assert g.box_of(11, 2)["src"] == "interp" and "ignore" not in g.box_of(11, 2)["flags"]
    assert g.box_of(25, 2)["ignore_reason"] == "interp_gap" and "ignore" in g.box_of(25, 2)["flags"]
    assert g.box_of(45, 2) is None


def test_auto_flags_keeps_v2_keys(tmp_path):
    seg = _v1(tmp_path)
    g = GT(seg / "gt.json")
    g.upgrade()
    g.set_occ(0, 2, "full")
    auto_flags(g)
    assert g.box_of(0, 2)["occ"] == "full" and "occluded" in g.box_of(0, 2)["flags"]
    assert g.box_of(5, 4)["ignore_reason"] == "small" and "ignore" in g.box_of(5, 4)["flags"]


# ---------------------------------------------------------------------------------------------------- checker
def test_checker_clean_and_findings(tmp_path):
    lc = _load("label_check")
    seg = _v1(tmp_path)
    g = GT(seg / "gt.json")
    g.upgrade()
    rep = lc.check(g)
    assert rep.errors == [], rep.errors
    assert any("no ref3d" in w for w in rep.warnings)
    assert len(rep.warnings) == 1, rep.warnings                          # contiguous ids: no hole / continuity noise
    seg2 = _v1(tmp_path / "b", n=40, gaps={10, 11, 20, 21, 22, 23, 24, 25, 26, 27, 28})
    rep2 = lc.check(GT(seg2 / "gt.json"))
    W = "\n".join(rep2.warnings)
    assert "id 2: 1 short holes" in W and "id 2: 9 empty frames f20..f28" in W
    # inject every kind of problem
    g.frames[0]["boxes"].append(dict(g.box_of(0, 1)))                   # duplicate id in a frame
    g.box_of(1, 2)["cls"] = "heavy_equipment"                            # class mismatch
    g.box_of(2, 1)["xyxy"] = [900, 100, 940, 200]                        # continuity jump (800 px)
    g.ids[1]["motion"] = [[0, 10, "static"]]                             # motion hole
    g.meta["ego_state"] = [[0, 2, "static"], [3, 39, "swing"]]           # 3 frames < 1 s at 5 Hz
    g.meta["stamps"][7] += 0.05                                          # stamp without pipeline frame
    g.frames[4]["boxes"][0]["occ"] = "half"                              # bad enum
    for k in range(20, 30):                                              # 10-frame interp run without ignore
        g.add_box(k, 5, [50, 50, 80, 130], src="interp")
    g.ids[5]["motion"] = [[20, 29, "walk"]]
    for k in range(10):
        g.set_ref3d(k, 1, {"range_m": 10 + (5 if k % 2 else 0), "xy_level": [10, 0], "n_lidar": 10, "quality": "dense"})
    rep = lc.check(g)
    E = "\n".join(rep.errors)
    for key in ("appears 2 times", "box cls heavy_equipment != ids cls worker", "not in any motion segment",
                "< 1 s", "without a pipeline stamp", "occ 'half' invalid", "needs ignore(interp_gap)",
                "LiDAR-box coverage", "range jumps > 2 m"):
        assert key in E, (key, E)
    W = "\n".join(rep.warnings)
    assert "centre moves" in W and "vs mono" in W
    d = rep.to_dict()
    assert set(d) == {"errors", "warnings", "info", "stats"} and d["stats"]["ref3d_coverage"] is not None


def test_checker_cli_strict_exit(tmp_path):
    seg = _v1(tmp_path)
    g = GT(seg / "gt.json")
    g.upgrade()
    g.ids[1]["motion"] = []
    g.save(backup=False)
    out = tmp_path / "rep.json"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts/label_check.py"), str(seg), "--strict", "--json", str(out)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "not in any motion segment" in r.stdout
    assert json.loads(out.read_text())["seg"] == seg.name
    r = subprocess.run([sys.executable, str(ROOT / "scripts/label_check.py"), str(seg)], capture_output=True, text=True,
                       env=env)
    assert r.returncode == 0


# ---------------------------------------------------------------------------------------------------- label_ops
def test_label_ops_v2_commands(tmp_path):
    seg = _v1(tmp_path)
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}

    def ops(*args):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/label_ops.py"), str(seg), *args], capture_output=True,
                           text=True, env=env)
        assert r.returncode == 0, r.stderr
        return r.stdout

    ops("upgrade", "--by", "tester")
    ops("occ", "0", "1", "full")
    ops("ignore-reason", "0", "2", "out_fov")
    ops("ego", "10", "19", "swing")
    ops("meta", "lighting=day_overexposed", 'ego_rate={"yaw_rate_max_dps": 30}', "gnss=null")
    ops("provenance", "agent", "agent-A", "ids checked")
    ops("note", "2", "1", "is this the same person", "--kind", "id", "--by", "agent-A")
    ops("resolve", "0")
    ops("split", "test")
    ops("role", "2", "background")
    ops("equipment", "3", "truck")
    ops("motion", "1", "walk", "0", "5")                                 # v1 command still works
    g = GT(seg / "gt.json")
    assert g.box_of(0, 1)["occ"] == "full" and g.box_of(0, 2)["ignore_reason"] == "out_fov"
    assert g.meta["ego_state"] == [[0, 9, "static"], [10, 19, "swing"], [20, 39, "static"]]
    assert g.meta["lighting"] == "day_overexposed" and g.meta["ego_rate"] == {"yaw_rate_max_dps": 30}
    assert g.meta["provenance"][-1]["who"] == "agent-A" and g.meta["split"] == "test"
    assert g.notes[-1]["kind"] == "id" and g.notes[0]["status"] == "resolved"
    assert g.ids[2]["role"] == "background" and g.ids[3]["equipment_type"] == "truck"
    assert g.ids[1]["motion"][0] == [0, 5, "walk"]


# ---------------------------------------------------------------------------------------------------- GUI
def test_gui_headless(tmp_path):
    pytest.importorskip("PyQt5")
    import cv2
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    seg = _v1(tmp_path, n=8)
    (seg / "frames").mkdir()
    for k in range(8):
        cv2.imwrite(str(seg / f"frames/{k:04d}.jpg"), np.zeros((720, 1280, 3), np.uint8))
    g = GT(seg / "gt.json")
    g.upgrade()
    g.set_ref3d(0, 1, {"range_m": 8.2, "xy_level": [8, 1], "n_lidar": 12, "quality": "dense"})
    g.save(backup=False)
    lr = _load("label_review")
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = lr.Review(seg)
    w.show()
    app.processEvents()
    for _ in range(3):
        w.check_frame()                                                  # C three times
    assert w.i == 3
    w.goto(0, 1)
    w.apply_flag_action("occ full")
    w.apply_flag_action("truncated")
    w.notes.setCurrentRow(0)
    w.toggle_note()
    w.view.grab()                                                        # paint with ref3d + ego band
    app.processEvents()
    g2 = GT(seg / "gt.json")
    assert g2.review["passes"]["user"]["frames_stepped"] == 3
    assert sum(v == "checked" for v in g2.review["frames"].values()) == 3
    b = g2.box_of(0, 1)
    assert b["occ"] == "full" and {"occluded", "truncated"} <= set(b["flags"])
    assert g2.notes[0]["status"] == "resolved"
    assert "ego=" in w.lbl.text() and w.notes.item(0).text().startswith("[id/ok]")
    w.close()
