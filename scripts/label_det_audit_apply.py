#!/usr/bin/env python3
"""GT audit, step 3: apply reviewed findings of label_audit_compare.py to gt.json (docs/13 §audit).

  PYTHONPATH=src python3 scripts/label_det_audit_apply.py data/labels/<seg> [options] [--apply]

Automatic rules (each one is off unless its flag is given):
  --fill-gaps      MISSED consensus detection -> the worker id whose track predicts a box there (linear
                   interpolation between the nearest labelled frames on both sides, or constant-velocity
                   extrapolation within 3 frames of a track end). The box is added as src det_audit with
                   ref3d copied from the nearest box of the id (src neighbor). Remaining holes of touched ids
                   are then interpolated by the usual rule (GT.interpolate).
  --snap-proj      boxes with src lidar_proj (LiDAR projection, label_lidar_audit_apply --add-missed) are
                   replaced by the detection that covers them (centre inside, vertical overlap > 0.5):
                   consensus first, else the single most confident model at conf >= --snap-conf.
  --loose          LOOSE findings (consensus IoU 0.3..0.6, conf >= 0.5) replace the GT geometry, unless the
                   GT box is occluded or the proposed box overlaps another GT box (IoU > 0.1, likely a merge).
  --add-ignore     MISSED runs still unmatched after --fill-gaps become new ids with ignore(edge) when the
                   box touches the image border, else ignore(unidentifiable) (far cluster rule, docs/28 §7).
                   Runs tagged in_vehicle (cab / UTV operator) and frames in --skip-frames are never added.
  --lidar-ignore P8[=id]  a LiDAR person of lidar_audit.json (per_frame) that is never labelled becomes an
                   ignore(unidentifiable) track over its in-FOV frames: projection corrected by the median
                   image offset of the other matched persons of the frame, then snapped to a detection
                   (any model, conf >= --snap-conf) when one covers it. For people beyond the LiDAR / 35 m rule.
  --far-ignore IDS  ids of people beyond the 35 m rule (no LiDAR reference, docs/28 §7 decision 2): every box
                   ignore(unidentifiable), role background. --add-ignore ids get the same role.
Explicit edits (ID:F0-F1, repeatable):
  --assign F:ID    the MISSED detection of frame F becomes a box of ID (reviewer's call when the track
                   prediction fails, e.g. the first frame of a fast swing)
  --occ-full       interpolated boxes hidden behind a closer worker / machine: occ = full (kept as GT)
  --edge           ignore(edge): sliver at the image border
  --delete         remove the boxes (wrong interpolation towards another object)
  --newid          move the boxes to a fresh id (wrong link after a swing)
  --static-ignore IDS=x0,y0,x1,y1   merge the ids (a fragmented static ignore object), one box per frame of the
                   segment, ignore(unidentifiable), src static_fill; IDS = new creates a fresh id.
Touched ids lose their motion draft (label_auto_meta.py refills it). Nothing is written without --apply.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rcmot.labeling.gt import GT  # noqa: E402
from label_audit_compare import consensus, iou, load_dets, person_dets  # noqa: E402

MODELS = ("yolo11x", "rtdetrx", "gdino", "owlv2")
IMG_W, IMG_H = 1280, 720


def cxcy(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def covers(det, box, vmin: float = 0.5) -> bool:
    """det covers box: box centre inside det horizontally (slack 0.3 w) and vertical overlap > vmin of box h."""
    cx, _ = cxcy(box)
    w = box[2] - box[0]
    if not (det[0] - 0.3 * w <= cx <= det[2] + 0.3 * w):
        return False
    ov = min(det[3], box[3]) - max(det[1], box[1])
    return ov > vmin * (box[3] - box[1])


def frame_range(spec: str):
    tid, fr = spec.split(":")
    f0, f1 = (fr.split("-") + [None])[:2]
    f0 = int(f0)
    return int(tid), f0, int(f1) if f1 else f0


def ref_from(box: dict | None):
    r = (box or {}).get("ref3d") or {}
    if r.get("range_m") is None:
        return None
    return {"range_m": r["range_m"], "xy_level": r.get("xy_level"), "n_lidar": 0, "quality": "sparse", "src": "neighbor"}


class Apply:
    def __init__(self, seg: Path, by: str, verbose: bool):
        self.seg, self.by, self.verbose = seg, by, verbose
        self.gt = GT(seg / "gt.json")
        self.audit = json.load(open(seg / "audit" / "audit.json"))
        self.dets = load_dets(seg, list(MODELS))
        self.touched: set[int] = set()
        self.log: list[str] = []
        p = seg / "lidar_audit.json"
        self.lidar = json.load(open(p)) if p.exists() else {}

    def say(self, s):
        self.log.append(s)
        if self.verbose:
            print(" ", s)

    def frame_dets(self, i: int) -> dict:
        return {m: d[i] for m, d in self.dets.items()}

    def worker_ids(self) -> list[int]:
        return [int(t) for t, d in self.gt.ids.items() if d.get("cls", "worker") == "worker"]

    # ------------------------------------------------------------------------------------------ predictions
    def predict(self, tid: int, f: int, frames: list[int]):
        """Predicted box of tid at frame f from its labelled frames (None when the track gives no prediction)."""
        before = [k for k in frames if k < f]
        after = [k for k in frames if k > f]
        bo = lambda k: np.array(self.gt.box_of(k, tid)["xyxy"], float)
        if before and after:
            a, b = before[-1], after[0]
            if b - a > 30:
                return None
            w = (f - a) / (b - a)
            return (1 - w) * bo(a) + w * bo(b)
        side = before[::-1] if before else after
        a = side[0]
        if abs(f - a) > 3:
            return None
        if len(side) > 1 and abs(side[1] - a) <= 3:
            v = (bo(a) - bo(side[1])) / (a - side[1])
            return bo(a) + v * (f - a)
        return bo(a)

    def match_id(self, f: int, det) -> int | None:
        best = None
        for tid in self.worker_ids():
            frames = self.gt.id_frames(tid)
            if not frames or f in frames:
                continue
            p = self.predict(tid, f, frames)
            if p is None:
                continue
            (px, py), (dx, dy) = cxcy(p), cxcy(det)
            w, h = det[2] - det[0], det[3] - det[1]
            hp = p[3] - p[1]
            if abs(px - dx) < max(25, 0.8 * w) and abs(py - dy) < 0.5 * h and 0.75 < hp / h < 1.33:
                d = abs(px - dx) + abs(py - dy)
                if best is None or d < best[0]:
                    best = (d, tid)
        return best[1] if best else None

    # ------------------------------------------------------------------------------------------ rules
    def fill_gaps(self):
        n, left = 0, []
        for run in self.audit["missed"]:
            for fe in run["frames"]:
                f, det = fe["frame"], fe["xyxy"]
                if any(iou(det, b["xyxy"])[0, 0] >= 0.3 for b in self.gt.frames[f]["boxes"]):
                    continue
                tid = self.match_id(f, det)
                if tid is None:
                    left.append((run, fe))
                    continue
                frames = self.gt.id_frames(tid)
                near = min(frames, key=lambda k: abs(k - f))
                box = self.gt.add_box(f, tid, det, src="det_audit")
                box["conf"] = fe["conf"]
                self.gt.set_ref3d(f, tid, ref_from(self.gt.box_of(near, tid)))
                self.touched.add(tid)
                n += 1
                self.say(f"fill f{f} id{tid} {[round(v) for v in det]} conf {fe['conf']:.2f} models {list(fe['models'])} (track f{near})")
        for tid in sorted(self.touched):
            k = self.gt.interpolate(tid)
            if k:
                self.say(f"interp id{tid}: {k} boxes")
        print(f"fill-gaps: {n} boxes added, {len(left)} detections unmatched")
        return left

    def snap_proj(self):
        n, miss = 0, 0
        for i, fr in enumerate(self.gt.frames):
            for b in list(fr["boxes"]):
                if b.get("src") != "lidar_proj":
                    continue
                fd = self.frame_dets(i)
                cand = [(c["conf"] + 1, c["xyxy"], list(c["models"])) for c in consensus(fd, self.conf) if covers(c["xyxy"], b["xyxy"], 0.25)]
                if not cand:
                    for m, dets in fd.items():
                        for d in person_dets(m, dets):
                            if d[4] >= self.conf and covers(d[:4], b["xyxy"], 0.25):
                                cand.append((d[4], [round(float(v), 1) for v in d[:4]], [m]))
                if not cand:
                    miss += 1
                    self.say(f"snap f{i} id{b['id']}: no detection covers the projection, kept")
                    continue
                cand.sort(key=lambda c: -c[0])
                old = b["xyxy"]
                b["xyxy"], b["src"], b["conf"] = cand[0][1], "det_audit", round(cand[0][0] % 1, 3)
                self.touched.add(b["id"])
                n += 1
                self.say(f"snap f{i} id{b['id']} {[round(v) for v in old]} -> {[round(v) for v in cand[0][1]]} ({', '.join(cand[0][2])})")
        print(f"snap-proj: {n} boxes snapped, {miss} kept")

    def loose(self):
        n = 0
        for run in self.audit["loose"]:
            for fe in run["frames"]:
                f, tid = fe["frame"], fe["id"]
                b = self.gt.box_of(f, tid)
                if b is None or fe["conf"] < 0.5 or b.get("occ", "none") != "none":
                    self.say(f"loose f{f} id{tid}: skipped (conf {fe['conf']:.2f}, occ {b and b.get('occ')})")
                    continue
                others = [o for o in self.gt.frames[f]["boxes"] if o["id"] != tid and iou(fe["proposed"], o["xyxy"])[0, 0] > 0.1]
                if others:
                    self.say(f"loose f{f} id{tid}: skipped, proposal overlaps id{others[0]['id']}")
                    continue
                self.say(f"loose f{f} id{tid} {[round(v) for v in b['xyxy']]} -> {[round(v) for v in fe['proposed']]} conf {fe['conf']:.2f}")
                b["xyxy"], b["src"], b["conf"] = fe["proposed"], "det_audit", fe["conf"]
                self.touched.add(tid)
                n += 1
        print(f"loose: {n} boxes replaced")

    def add_ignore(self, left, skip: set[int]):
        n = 0
        by_run: dict[int, list] = {}
        for run, fe in left:
            by_run.setdefault(id(run), (run, []))[1].append(fe)
        for run, fes in by_run.values():
            fes = [fe for fe in fes if fe["frame"] not in skip]
            if not fes or "in_vehicle" in run["tags"] or run["h"] < 25:
                continue
            tid = self.gt.new_id()
            for fe in fes:
                d = fe["xyxy"]
                reason = "edge" if d[0] <= 2 or d[2] >= IMG_W - 2 or d[1] <= 2 or d[3] >= IMG_H - 2 else "unidentifiable"
                self.gt.add_box(fe["frame"], tid, d, src="det_audit")
                self.gt.set_ignore_reason(fe["frame"], tid, reason)
                n += 1
            self.gt.note(fes[0]["frame"], tid, f"detector consensus with no track: ignore({reason}), {len(fes)} frames", kind="flag", by=self.by, status="resolved")
            self.gt.set_role(tid, "background")
            self.touched.add(tid)
            self.say(f"ignore id{tid} f{fes[0]['frame']}-{fes[-1]['frame']} n{len(fes)} h{run['h']:.0f} {reason}")
        print(f"add-ignore: {n} boxes")

    def lidar_ignore(self, name: str, tid: int | None):
        pf = self.lidar.get("per_frame", {})
        tid = tid or self.gt.new_id()
        n, snapped = 0, 0
        for fs, rec in pf.items():
            f = int(fs)
            o = rec.get("objects", {}).get(name)
            if not o or not o.get("box") or not o.get("in_fov"):
                continue
            proj = np.array(o["box"], float)
            # offset of the other matched persons: GT box vs projection
            dx, dy, sc = [], [], []
            for on, oo in rec["objects"].items():
                if on == name or not oo.get("box") or not oo.get("ids"):
                    continue
                gb = self.gt.box_of(f, oo["ids"][0][0])
                if gb is None:
                    continue
                pb = oo["box"]
                dx.append(cxcy(gb["xyxy"])[0] - cxcy(pb)[0])
                dy.append(gb["xyxy"][3] - pb[3])
                sc.append((gb["xyxy"][3] - gb["xyxy"][1]) / max(pb[3] - pb[1], 1))
            box = proj.copy()
            if dx:
                mx, my, ms = np.median(dx), np.median(dy), np.median(sc)
                h = (proj[3] - proj[1]) * ms
                w = (proj[2] - proj[0]) * ms
                cx = cxcy(proj)[0] + mx
                y2 = proj[3] + my
                box = np.array([cx - w / 2, y2 - h, cx + w / 2, y2])
            best = None
            for m, dets in self.frame_dets(f).items():
                for d in person_dets(m, dets):
                    if d[4] >= self.conf and covers(d[:4], box) and 0.5 < (d[3] - d[1]) / (box[3] - box[1]) < 2.0:
                        if best is None or d[4] > best[0]:
                            best = (d[4], d[:4], m)
            src = "lidar_proj"
            if best:
                box, src, snapped = np.array(best[1], float), "det_audit", snapped + 1
            box = np.clip(box, [0, 0, 0, 0], [IMG_W, IMG_H, IMG_W, IMG_H])
            if box[3] - box[1] < 8 or box[2] - box[0] < 3:
                continue
            b = self.gt.add_box(f, tid, box, src=src)
            if best:
                b["conf"] = round(float(best[0]), 3)
            self.gt.set_ignore_reason(f, tid, "unidentifiable")
            self.gt.set_ref3d(f, tid, {"range_m": o.get("range"), "xy_level": o.get("xy_level"), "n_lidar": 0, "quality": "sparse", "src": "lidar_world_audit"})
            n += 1
        self.gt.interpolate(tid)
        frames = self.gt.id_frames(tid)
        if frames:
            self.gt.note(frames[0], tid, f"LiDAR person {name} ({pf[str(frames[0])]['objects'][name].get('range')} m) beyond the 35 m rule: ignore(unidentifiable); {n} frames, {snapped} snapped to a detection, rest projected", kind="flag", by=self.by, status="resolved")
        self.touched.add(tid)
        print(f"lidar-ignore {name} -> id{tid}: {n} boxes ({snapped} from detections)")

    def static_ignore(self, spec: str):
        ids, box = spec.split("=")
        box = [float(v) for v in box.split(",")]
        ids = [int(v) for v in ids.split(",")] if ids != "new" else []
        tid = ids[0] if ids else self.gt.new_id()
        self.gt.ensure_id(tid)
        for o in ids[1:]:
            self.gt.merge(o, tid)
        for f in range(len(self.gt.frames)):
            if self.gt.box_of(f, tid) is None:
                self.gt.add_box(f, tid, box, src="static_fill")
            self.gt.set_ignore_reason(f, tid, "unidentifiable")
        self.gt.note(0, tid, f"static ignore object (merged {ids[1:] or 'new'}), one box per frame", kind="flag", by=self.by, status="resolved")
        self.touched.add(tid)
        self.say(f"static-ignore id{tid} <- {ids[1:]} box {box}")

    def assign(self, specs: list[str]):
        for sp in specs:
            f, tid = (int(v) for v in sp.split(":"))
            fes = [fe for r in self.audit["missed"] for fe in r["frames"] if fe["frame"] == f]
            if not fes:
                raise SystemExit(f"--assign {sp}: no MISSED detection in frame {f}")
            fe = max(fes, key=lambda x: x["conf"])
            self.gt.ensure_id(tid)
            frames = self.gt.id_frames(tid)
            near = min(frames, key=lambda k: abs(k - f)) if frames else None
            b = self.gt.add_box(f, tid, fe["xyxy"], src="det_audit")
            b["conf"] = fe["conf"]
            if near is not None:
                self.gt.set_ref3d(f, tid, ref_from(self.gt.box_of(near, tid)))
            self.touched.add(tid)
            self.say(f"assign f{f} id{tid} {[round(v) for v in fe['xyxy']]} conf {fe['conf']:.2f}")

    def far_ignore(self, ids: list[int]):
        for tid in ids:
            n = 0
            for f in self.gt.id_frames(tid):
                self.gt.set_ignore_reason(f, tid, "unidentifiable")
                n += 1
            self.gt.set_role(tid, "background")
            self.gt.note(self.gt.id_frames(tid)[0], tid, "beyond the 35 m rule (no LiDAR reference): ignore(unidentifiable), role background", kind="flag", by=self.by, status="resolved")
            self.touched.add(tid)
            self.say(f"far-ignore id{tid}: {n} boxes")

    def explicit(self, kind: str, specs: list[str]):
        for s in specs:
            tid, f0, f1 = frame_range(s)
            new = self.gt.new_id() if kind == "newid" else None
            n = 0
            for f in range(f0, f1 + 1):
                b = self.gt.box_of(f, tid)
                if b is None:
                    continue
                if kind == "occ-full":
                    self.gt.set_occ(f, tid, "full")
                elif kind == "edge":
                    self.gt.set_ignore_reason(f, tid, "edge")
                elif kind == "delete":
                    self.gt.frames[f]["boxes"] = [o for o in self.gt.frames[f]["boxes"] if o["id"] != tid]
                elif kind == "newid":
                    self.gt.set_id(f, b, new)
                n += 1
            if new is not None:
                self.gt.ids[new]["cls"] = self.gt.ids[tid]["cls"]
                self.touched.add(new)
            self.touched.add(tid)
            self.say(f"{kind} id{tid} f{f0}-{f1}: {n} boxes" + (f" -> id{new}" if new else ""))

    def finish(self, apply: bool, cmd: str):
        for tid in list(self.touched):
            if tid in self.gt.ids:
                if not self.gt.id_frames(tid):
                    self.gt.delete_id(tid)
                    self.say(f"id{tid} now empty: removed")
                else:
                    self.gt.ids[tid]["motion"] = None
        self.gt.add_provenance("audit", self.by, "label_det_audit_apply", note="; ".join(self.log)[:2000])
        c = self.gt.counts()
        print(f"{'APPLIED' if apply else 'DRY RUN'}: ids {c['ids']} boxes {c['boxes']} touched {sorted(self.touched)}")
        if apply:
            self.gt.save()
            with open(self.seg / "commands.log", "a") as fh:
                fh.write(cmd + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg", type=Path)
    ap.add_argument("--fill-gaps", action="store_true")
    ap.add_argument("--snap-proj", action="store_true")
    ap.add_argument("--loose", action="store_true")
    ap.add_argument("--add-ignore", action="store_true")
    ap.add_argument("--skip-frames", default="", help="frames never added by --add-ignore (cones etc.), comma list")
    ap.add_argument("--lidar-ignore", nargs="*", default=[], metavar="P8[=id]")
    ap.add_argument("--assign", nargs="*", default=[], metavar="F:ID")
    ap.add_argument("--far-ignore", default="", help="comma list of ids beyond the 35 m rule")
    ap.add_argument("--occ-full", nargs="*", default=[], metavar="ID:F0-F1")
    ap.add_argument("--edge", nargs="*", default=[], metavar="ID:F0-F1")
    ap.add_argument("--delete", nargs="*", default=[], metavar="ID:F0-F1")
    ap.add_argument("--newid", nargs="*", default=[], metavar="ID:F0-F1")
    ap.add_argument("--static-ignore", nargs="*", default=[], metavar="IDS=x0,y0,x1,y1")
    ap.add_argument("--snap-conf", type=float, default=0.3)
    ap.add_argument("--by", default="agent")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    seg = a.seg if a.seg.is_absolute() else ROOT / a.seg
    A = Apply(seg, a.by, a.verbose)
    A.conf = a.snap_conf
    A.explicit("delete", a.delete)
    A.explicit("newid", a.newid)
    A.explicit("occ-full", a.occ_full)
    A.explicit("edge", a.edge)
    for s in a.static_ignore:
        A.static_ignore(s)
    A.assign(a.assign)
    if a.snap_proj:
        A.snap_proj()
    if a.loose:
        A.loose()
    left = A.fill_gaps() if a.fill_gaps else []
    if a.add_ignore:
        A.add_ignore(left, {int(v) for v in a.skip_frames.split(",") if v})
    elif left:
        for run, fe in left:
            print(f"  unmatched f{fe['frame']} {[round(v) for v in fe['xyxy']]} h{run['h']:.0f} conf {fe['conf']:.2f} tags {run['tags']}")
    for s in a.lidar_ignore:
        name, _, tid = s.partition("=")
        A.lidar_ignore(name, int(tid) if tid else None)
    if a.far_ignore:
        A.far_ignore([int(v) for v in a.far_ignore.split(",")])
    A.finish(a.apply, "PYTHONPATH=src python3 scripts/label_det_audit_apply.py " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()
