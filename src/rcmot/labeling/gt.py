"""gt.json: {meta, ids: {id: {cls, desc}}, frames: [{stamp, bag_t, file, boxes: [{id, cls, xyxy, flags, src, conf}]}],
review: {ids: {id: draft|checked}, frames: {stamp: draft|checked}}, notes: [{stamp, id, text}]}.

Schema v2 (docs/28 §3.3, pres/framework_deck/ref/gt_design.md §C) adds keys only; readers ignore unknown keys:
  meta:    schema_version, split (dev|test), site, radar, mount, imu_topic, lidar_topic, gnss, image_size, lighting,
           ego_state [[f0, f1, state]], ego_rate, ref3d (how the per-box 3D reference was made), provenance []
  ids:     equipment_type (equipment only), role (target|background)
  boxes:   occ (none|partial|full), ignore_reason, ref3d {range_m, xy_level, n_lidar, quality, src} | null
  review:  passes {agent|user|second: {who, time, ...}}
  notes:   kind (id|box|flag|motion|ref3d|ego), status (open|resolved), by

Editing operations shared by the review GUI and scripts/label_ops.py.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

FLAGS = ("occluded", "truncated", "crouch", "ignore")
SCHEMA_VERSION = 2
SPLITS = ("dev", "test")
SITES = ("boryeong_pg", "arizona_xpanner")
RADARS = ("bosch_premium", "conti_ars548")
LIGHTINGS = ("day", "day_backlight", "day_overexposed", "night")
EGO_STATES = ("static", "drive", "swing", "drive+swing")
OCC = ("none", "partial", "full")
IGNORE_REASONS = ("small", "out_fov", "unidentifiable", "interp_gap", "edge")
REF3D_QUALITY = ("dense", "sparse", "none")
ROLES = ("target", "background")
EQUIPMENT_TYPES = ("excavator", "loader", "truck", "utv", "unknown")
NOTE_KINDS = ("id", "box", "flag", "motion", "ref3d", "ego")
NOTE_STATUS = ("open", "resolved")
PASSES = ("agent", "user", "second")
MIN_BOX_H = 25.0            # px; shorter boxes are ignore(small) (auto_flags)
INTERP_IGNORE_GAP = 7       # interpolated runs of 7..15 frames are ignore(interp_gap); > 15 are not interpolated
INTERP_MAX_GAP = 15

_ROOT = Path(__file__).resolve().parents[3]


def git_sha(root: Path | str = _ROOT) -> str | None:
    """Short sha of the label tools' checkout, or None when not in a git repository."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(root), capture_output=True, text=True,
                           timeout=5)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _check(value, allowed, what: str):
    if value not in allowed:
        raise ValueError(f"{what}: {value!r} not in {allowed}")
    return value


class GT:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        d = json.load(open(self.path))
        self.meta, self.frames = d["meta"], d["frames"]
        self.ids = {int(k): v for k, v in d.get("ids", {}).items()}
        rv = d.get("review", {})
        self.review = {"ids": {int(k): v for k, v in rv.get("ids", {}).items()}, "frames": dict(rv.get("frames", {})),
                       "passes": dict(rv.get("passes", {}))}
        self.notes = d.get("notes", [])
        self.extra = {k: v for k, v in d.items() if k not in ("meta", "ids", "frames", "review", "notes")}

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def schema_version(self) -> int:
        return int(self.meta.get("schema_version", 1))

    def to_dict(self) -> dict:
        review = {"ids": {str(k): v for k, v in sorted(self.review["ids"].items())}, "frames": self.review["frames"]}
        if self.review.get("passes"):
            review["passes"] = self.review["passes"]
        return {**self.extra, "meta": self.meta, "ids": {str(k): v for k, v in sorted(self.ids.items())},
                "frames": self.frames, "review": review, "notes": self.notes}

    def save(self, backup: bool = True):
        if backup and self.path.exists():
            bdir = self.root / "backup"
            bdir.mkdir(exist_ok=True)
            shutil.copy(self.path, bdir / f"gt_{time.strftime('%Y%m%d_%H%M%S')}.json")
            olds = sorted(bdir.glob("gt_*.json"))
            for f in olds[:-30]:
                f.unlink()
        tmp = self.path.with_suffix(".tmp")
        json.dump(self.to_dict(), open(tmp, "w"), indent=1)
        tmp.replace(self.path)

    # ------------------------------------------------------------------ queries
    def key(self, i: int) -> str:
        return f"{self.frames[i]['stamp']:.6f}"

    def review_key(self, i: int) -> str:
        """Key of frame i in review.frames: the stamp string, or the frame index when the file already uses that
        (day_halt_short_085 was checked with index keys)."""
        k = self.key(i)
        return str(i) if k not in self.review["frames"] and str(i) in self.review["frames"] else k

    def frame_index(self, stamp: float) -> int:
        return int(np.argmin([abs(f["stamp"] - stamp) for f in self.frames]))

    def id_frames(self, tid: int) -> list[int]:
        return [i for i, f in enumerate(self.frames) if any(b["id"] == tid for b in f["boxes"])]

    def box_of(self, i: int, tid: int) -> dict | None:
        return next((b for b in self.frames[i]["boxes"] if b["id"] == tid), None)

    def new_id(self) -> int:
        used = set(self.ids) | {b["id"] for f in self.frames for b in f["boxes"]}
        return max(used, default=0) + 1

    def counts(self) -> dict:
        return {"frames": len(self.frames), "ids": len(self.ids), "boxes": sum(len(f["boxes"]) for f in self.frames)}

    def ego_state_at(self, i: int) -> str | None:
        for f0, f1, st in self.meta.get("ego_state") or []:
            if f0 <= i <= f1:
                return st
        return None

    # ------------------------------------------------------------------ edits
    def ensure_id(self, tid: int, cls: str = "worker", desc: str = ""):
        if tid not in self.ids:
            self.ids[tid] = {"cls": cls, "desc": desc}
            if self.schema_version >= 2:
                self.ids[tid]["role"] = "target"
            self.review["ids"][tid] = "draft"

    def merge(self, src: int, dst: int):
        """All boxes of src become dst (where both exist in a frame, the dst box is kept)."""
        for f in self.frames:
            s = [b for b in f["boxes"] if b["id"] == src]
            if not s:
                continue
            if any(b["id"] == dst for b in f["boxes"]):
                f["boxes"] = [b for b in f["boxes"] if b["id"] != src]
            else:
                s[0]["id"] = dst
        if src in self.ids:
            if dst not in self.ids:
                self.ids[dst] = self.ids[src]
            self.ids.pop(src)
        self.review["ids"].pop(src, None)

    def split(self, tid: int, from_index: int) -> int:
        nid = self.new_id()
        self.ensure_id(nid, self.ids.get(tid, {}).get("cls", "worker"), f"split of {tid}")
        for f in self.frames[from_index:]:
            for b in f["boxes"]:
                if b["id"] == tid:
                    b["id"] = nid
        return nid

    def set_id(self, i: int, box: dict, tid: int):
        """Relabel one box; a box of tid already in this frame swaps to the old id."""
        old = box["id"]
        other = self.box_of(i, tid)
        if other is not None and other is not box:
            other["id"] = old
        box["id"] = tid
        self.ensure_id(tid, box.get("cls", "worker"))

    def delete_id(self, tid: int):
        for f in self.frames:
            f["boxes"] = [b for b in f["boxes"] if b["id"] != tid]
        self.ids.pop(tid, None)
        self.review["ids"].pop(tid, None)

    def add_box(self, i: int, tid: int, xyxy, cls: str | None = None, src: str = "manual", flags=()):
        cls = cls or self.ids.get(tid, {}).get("cls", "worker")
        self.ensure_id(tid, cls)
        self.frames[i]["boxes"] = [b for b in self.frames[i]["boxes"] if b["id"] != tid]
        box = {"id": tid, "cls": cls, "xyxy": [round(float(v), 1) for v in xyxy], "flags": list(flags), "src": src,
               "conf": 1.0}
        if self.schema_version >= 2:
            box.update({"occ": "partial" if "occluded" in flags else "none", "ignore_reason": None, "ref3d": None})
        self.frames[i]["boxes"].append(box)
        return box

    def interpolate(self, tid: int, max_gap: int | None = None) -> int:
        """Linear boxes for tid in frames between two labelled frames. Gap = number of missing frames: up to 6 is
        normal GT, 7..15 is interpolated as ignore(interp_gap), more is left empty (docs/13 §2, gt_design §C.3).
        max_gap (frames between the labelled frames) narrows this further when given."""
        idx = self.id_frames(tid)
        n = 0
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if gap < 1 or gap > INTERP_MAX_GAP or (max_gap is not None and b - a > max_gap):
                continue
            ba, bb = np.array(self.box_of(a, tid)["xyxy"]), np.array(self.box_of(b, tid)["xyxy"])
            fl = sorted(set(self.box_of(a, tid)["flags"]) & set(self.box_of(b, tid)["flags"]))
            long = gap >= INTERP_IGNORE_GAP
            if long and "ignore" not in fl:
                fl.append("ignore")
            for k in range(a + 1, b):
                w = (k - a) / (b - a)
                box = self.add_box(k, tid, (1 - w) * ba + w * bb, src="interp", flags=fl)
                if long:
                    box["ignore_reason"] = "interp_gap"
                n += 1
        return n

    def toggle_flag(self, box: dict, flag: str):
        if flag in box["flags"]:
            box["flags"].remove(flag)
        else:
            box["flags"].append(flag)
        if "occ" in box and flag == "occluded":                  # v2: flags.occluded <=> occ != none
            box["occ"] = "partial" if "occluded" in box["flags"] else "none"
        if "ignore_reason" in box and flag == "ignore" and "ignore" not in box["flags"]:
            box["ignore_reason"] = None

    def renumber(self):
        """Ids 1..N in order of first appearance."""
        order = []
        for f in self.frames:
            for b in sorted(f["boxes"], key=lambda b: b["xyxy"][0]):
                if b["id"] not in order:
                    order.append(b["id"])
        m = {o: k + 1 for k, o in enumerate(order)}
        for f in self.frames:
            for b in f["boxes"]:
                b["id"] = m[b["id"]]
        self.ids = {m[k]: v for k, v in self.ids.items() if k in m}
        self.review["ids"] = {m[k]: v for k, v in self.review["ids"].items() if k in m}
        for n in self.notes:
            if n.get("id") in m:
                n["id"] = m[n["id"]]
        return m

    # ------------------------------------------------------------------ v2 setters
    def _box(self, i: int, tid: int) -> dict:
        b = self.box_of(i, tid)
        if b is None:
            raise KeyError(f"no box of id {tid} in frame {i}")
        return b

    def set_occ(self, i: int, tid: int, level: str) -> dict:
        """Occlusion level of one box; keeps flags.occluded in step (occluded <=> occ != none)."""
        b = self._box(i, tid)
        b["occ"] = _check(level, OCC, "occ")
        if level == "none":
            b["flags"] = [f for f in b["flags"] if f != "occluded"]
        elif "occluded" not in b["flags"]:
            b["flags"].append("occluded")
        return b

    def set_ignore_reason(self, i: int, tid: int, reason: str | None) -> dict:
        """reason None clears both the reason and the ignore flag; a reason sets the ignore flag."""
        b = self._box(i, tid)
        if reason in (None, "none", "null", ""):
            b["ignore_reason"] = None
            b["flags"] = [f for f in b["flags"] if f != "ignore"]
        else:
            b["ignore_reason"] = _check(reason, IGNORE_REASONS, "ignore_reason")
            if "ignore" not in b["flags"]:
                b["flags"].append("ignore")
        return b

    def set_ref3d(self, i: int, tid: int, ref: dict | None) -> dict:
        """ref = {range_m, xy_level [x, y], n_lidar, quality dense|sparse|none, src} or None (no reference)."""
        b = self._box(i, tid)
        if ref is not None:
            _check(ref.get("quality", "none"), REF3D_QUALITY, "ref3d.quality")
            ref = {"range_m": None if ref.get("range_m") is None else round(float(ref["range_m"]), 2),
                   "xy_level": None if ref.get("xy_level") is None else [round(float(v), 2) for v in ref["xy_level"]],
                   "n_lidar": int(ref.get("n_lidar", 0)), "quality": ref.get("quality", "none"),
                   "src": ref.get("src", "lidar")}
        b["ref3d"] = ref
        return b

    def set_ego_state(self, segs: list) -> list:
        """[[f0, f1, state], ...] sorted, non-overlapping; the checker verifies full 0..N-1 coverage."""
        out = []
        for f0, f1, st in segs:
            f0, f1 = int(f0), int(f1)
            _check(st, EGO_STATES, "ego_state")
            if not (0 <= f0 <= f1 < max(len(self.frames), 1)):
                raise ValueError(f"ego_state: bad frame range {f0}..{f1} (N={len(self.frames)})")
            out.append([f0, f1, st])
        out.sort()
        for a, b in zip(out[:-1], out[1:]):
            if b[0] <= a[1]:
                raise ValueError(f"ego_state: overlapping segments {a} and {b}")
        self.meta["ego_state"] = out
        return out

    def set_ego_segment(self, f0: int, f1: int, state: str) -> list:
        """Insert one segment, trimming / replacing whatever overlaps it (same semantics as motion in label_ops)."""
        _check(state, EGO_STATES, "ego_state")
        keep = []
        for a, b, st in self.meta.get("ego_state") or []:
            if b < f0 or a > f1:
                keep.append([a, b, st])
            else:                                                # overlap: keep the non-overlapping remainders
                if a < f0:
                    keep.append([a, f0 - 1, st])
                if b > f1:
                    keep.append([f1 + 1, b, st])
        return self.set_ego_state(keep + [[f0, f1, state]])

    def set_split(self, split: str):
        self.meta["split"] = _check(split, SPLITS, "split")

    def set_meta(self, **kw):
        """Validated meta keys (site, radar, lighting, split, ...); other keys are stored as given."""
        checks = {"split": SPLITS, "site": SITES, "radar": RADARS, "lighting": LIGHTINGS}
        for k, v in kw.items():
            if k in checks and v is not None:
                _check(v, checks[k], k)
            if k == "ego_state":
                self.set_ego_state(v)
            else:
                self.meta[k] = v

    def add_provenance(self, stage: str, who: str, tool: str = "", note: str = "", git: str | None = None) -> dict:
        e = {"stage": stage, "who": who, "tool": tool, "git": git_sha() if git is None else git, "time": now_iso(),
             "note": note}
        self.meta.setdefault("provenance", []).append(e)
        return e

    def set_role(self, tid: int, role: str):
        self.ensure_id(tid)
        self.ids[tid]["role"] = _check(role, ROLES, "role")

    def set_equipment_type(self, tid: int, etype: str):
        self.ensure_id(tid, "heavy_equipment")
        self.ids[tid]["equipment_type"] = _check(etype, EQUIPMENT_TYPES, "equipment_type")

    def note(self, i: int, tid: int | None, text: str, kind: str = "box", by: str = "agent",
             status: str = "open") -> dict:
        n = {"frame": int(i), "stamp": self.frames[i]["stamp"], "id": tid, "text": text,
             "kind": _check(kind, NOTE_KINDS, "note.kind"), "status": _check(status, NOTE_STATUS, "note.status"),
             "by": by}
        self.notes.append(n)
        return n

    def resolve_note(self, k: int, status: str | None = None, by: str | None = None) -> dict:
        """Toggle open <-> resolved (or set status) of notes[k]."""
        n = self.notes[k]
        if status is None:
            status = "open" if n.get("status", "open") == "resolved" else "resolved"
        n["status"] = _check(status, NOTE_STATUS, "note.status")
        if by:
            n["resolved_by" if status == "resolved" else "by"] = by
        return n

    def bump_pass(self, which: str = "user", who: str = "user", field: str = "frames_stepped", n: int = 1) -> dict:
        """review.passes[which][field] += n and stamp the time (C in the GUI: user.frames_stepped)."""
        _check(which, PASSES, "pass")
        p = self.review.setdefault("passes", {}).setdefault(which, {"who": who, "time": None, field: 0})
        p["who"] = p.get("who") or who
        p[field] = int(p.get(field, 0) or 0) + n
        p["time"] = now_iso()
        return p

    def set_pass(self, which: str, who: str, **fields) -> dict:
        _check(which, PASSES, "pass")
        p = self.review.setdefault("passes", {}).setdefault(which, {})
        p.update({"who": who, "time": now_iso(), **fields})
        return p

    # ------------------------------------------------------------------ v1 -> v2
    def upgrade(self, who: str = "upgrade", root: Path | str = _ROOT) -> list[str]:
        """Fill the v2 keys that are missing (never overwrites present values) and return the list of changes.
        Idempotent: a second call changes nothing and adds no provenance entry."""
        ch: list[str] = []
        m = self.meta

        def setdefault(d, k, v, label):
            if k not in d:
                d[k] = v
                ch.append(label)

        old_split = m.get("split")
        if old_split not in SPLITS:
            m["split"] = "dev"                                   # validation / eval21 / missing -> dev (docs/28 §3.2)
            ch.append(f"meta.split {old_split!r} -> 'dev'")
        inf = infer_meta(m, self.root.name, root)
        for k in ("site", "radar", "mount", "imu_topic", "lidar_topic", "gnss", "image_size", "lighting", "ego_rate",
                  "ref3d"):
            setdefault(m, k, inf.get(k), f"meta.{k} = {inf.get(k)!r}")
        if "ego_state" not in m:
            st = inf.get("ego_state_default", "static")
            m["ego_state"] = [[0, len(self.frames) - 1, st]] if self.frames else []
            ch.append(f"meta.ego_state = whole segment {st!r} (from the bag name; review the boundaries)")
        m.setdefault("provenance", [])
        if not any(p.get("stage") == "draft" for p in m["provenance"]) and m.get("detector"):
            m["provenance"].append({"stage": "draft", "who": "label_extract", "tool": "scripts/label_extract.py",
                                    "git": None, "time": None,
                                    "note": f"{m.get('detector')} imgsz={m.get('imgsz')} conf={m.get('conf')} "
                                            f"(reconstructed at upgrade)"})
            ch.append("meta.provenance += draft (reconstructed)")
        roles, etypes = {}, {}
        for tid, v in self.ids.items():
            boxes = [b for f in self.frames for b in f["boxes"] if b["id"] == tid]
            all_ign = bool(boxes) and all("ignore" in b["flags"] for b in boxes)
            if "role" not in v:
                v["role"] = "background" if all_ign and v.get("cls") == "heavy_equipment" else "target"
                roles[tid] = v["role"]
            if v.get("cls") == "heavy_equipment" and "equipment_type" not in v:
                v["equipment_type"] = etypes[tid] = infer_equipment_type(v.get("desc", ""))
        if roles:
            bg = sorted(t for t, r in roles.items() if r == "background")
            ch.append(f"ids.role filled on {len(roles)} ids ({len(bg)} background: {bg})")
        if etypes:
            ch.append(f"ids.equipment_type filled on {len(etypes)} ids: "
                      + ", ".join(f"{t}={e}" for t, e in sorted(etypes.items())))
        nb = {"occ": 0, "ignore_reason": 0, "ref3d": 0}
        for f in self.frames:
            for b in f["boxes"]:
                if "occ" not in b:
                    b["occ"] = "partial" if "occluded" in b["flags"] else "none"
                    nb["occ"] += 1
                if "ignore_reason" not in b:
                    b["ignore_reason"] = infer_ignore_reason(b, self.ids.get(b["id"], {}).get("role"))
                    nb["ignore_reason"] += 1
                if "ref3d" not in b:
                    b["ref3d"] = None
                    nb["ref3d"] += 1
        ch += [f"boxes.{k} filled on {n} boxes" for k, n in nb.items() if n]
        self.review.setdefault("passes", {})
        if m.get("stage") in ("agent-reviewed", "vision-reviewed", "user-checked") and "agent" not in self.review["passes"]:
            self.review["passes"]["agent"] = {"who": "agent", "time": None, "frames_viewed": None}
            ch.append("review.passes.agent (reconstructed from stage)")
        if m.get("stage") == "user-checked" and "user" not in self.review["passes"]:
            nchk = sum(v == "checked" for v in self.review["frames"].values())
            self.review["passes"]["user"] = {"who": "user", "time": None, "frames_stepped": nchk}
            ch.append(f"review.passes.user (reconstructed: {nchk} checked frames)")
        nn = 0
        for n in self.notes:
            if "kind" not in n or "status" not in n:
                n.setdefault("kind", infer_note_kind(n.get("text", "")))
                n.setdefault("status", "open")
                n.setdefault("by", "agent")
                nn += 1
        if nn:
            ch.append(f"notes kind/status/by filled on {nn} notes")
        if self.schema_version < SCHEMA_VERSION:
            ch.append(f"meta.schema_version {self.schema_version} -> {SCHEMA_VERSION}")
            m["schema_version"] = SCHEMA_VERSION
        if ch:
            self.add_provenance("upgrade", who, "rcmot.labeling.gt.GT.upgrade", note="; ".join(ch)[:500])
        return ch


# ---------------------------------------------------------------------- upgrade heuristics
def infer_meta(meta: dict, seg_name: str = "", root: Path | str = _ROOT) -> dict:
    """Best-effort v2 meta from the bag / config / calib paths of a v1 file (the user reviews the result)."""
    root = Path(root)
    bag, cfg, cal = str(meta.get("bag", "")), str(meta.get("config", "")), str(meta.get("calib", ""))
    low = (bag + " " + cfg + " " + cal + " " + seg_name).lower()
    out: dict = {"gnss": None, "ego_rate": None, "ref3d": None, "image_size": [1280, 720], "lidar_topic": "/ouster/points"}
    if "xpanner" in low or "construction_field" in low or "arizona" in low:
        out["site"] = "arizona_xpanner"
    elif "boryeong" in low or "hdx" in low or "conti" in low:
        out["site"] = "boryeong_pg"
    else:
        out["site"] = None
    out["radar"] = "conti_ars548" if "conti" in low else "bosch_premium"
    out["lighting"] = "night" if "night" in low else "day"
    if "halt" in low or "cf_vib" in low or "construction_field" in low:
        out["ego_state_default"] = "static"
    elif "swing" in low:
        out["ego_state_default"] = "swing"
    elif "drive" in low:
        out["ego_state_default"] = "drive"
    else:
        out["ego_state_default"] = "static"
    out["imu_topic"] = None
    try:
        import yaml
        topics = yaml.safe_load(open(root / cfg)).get("topics", {})
        out["imu_topic"] = topics.get("imu")
    except Exception:
        pass
    out["mount"] = None
    try:
        c = json.load(open(root / cal))
        if "image_size" in c:
            out["image_size"] = [int(c["image_size"]["width"]), int(c["image_size"]["height"])]
        h = c.get("extrinsics", {}).get("sensor_height")
        if h is not None:
            out["mount"] = f"rig_{float(h):.2f}m"
    except Exception:
        pass
    return out


def infer_equipment_type(desc: str) -> str:
    d = desc.lower()
    for k in ("excavator", "loader", "truck", "utv"):
        if k in d:
            return k
    if "digger" in d:
        return "excavator"
    return "unknown"


def infer_ignore_reason(box: dict, role: str | None = None) -> str | None:
    """Reason for an existing ignore flag: small (< MIN_BOX_H px, the auto_flags rule), interp_gap for interpolated
    boxes, unidentifiable for boxes of a background id (parked / floodlit equipment rows, gt_design §C.3),
    otherwise unknown (None -> the checker lists it for the reviewer)."""
    if "ignore" not in box["flags"]:
        return None
    x1, y1, x2, y2 = box["xyxy"]
    if y2 - y1 < MIN_BOX_H:
        return "small"
    if box.get("src") == "interp":
        return "interp_gap"
    if role == "background":
        return "unidentifiable"
    return None


def infer_note_kind(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("merge", "split", "same person", "same worker", "identity", "new id", "draft id")):
        return "id"
    if any(k in t for k in ("ignore", "flag", "occlu", "truncat", "hidden", "crouch")):
        return "flag"
    if any(k in t for k in ("static", "walk", "motion", "stands", "standing")):
        return "motion"
    if any(k in t for k in ("swing", "ego", "yaw")):
        return "ego"
    if any(k in t for k in ("range", "lidar", "distance", "ref3d")):
        return "ref3d"
    return "box"


def auto_flags(gt: GT, img_wh=(1280, 720), min_h: float = MIN_BOX_H, crouch_aspect: float = 1.35) -> None:
    """Geometric flags (the reviewer checks them): truncated (touches the border), crouch (worker h/w small),
    occluded (> 50 % of the box covered by a box whose bottom edge is lower = closer), ignore (< min_h px tall).
    On v2 boxes occ (partial) and ignore_reason (small) are filled alongside; reviewer-set full / other reasons stay."""
    W, H = img_wh
    for f in gt.frames:
        bs = f["boxes"]
        for b in bs:
            fl = set(b["flags"]) - {"truncated", "crouch", "occluded"}
            x1, y1, x2, y2 = b["xyxy"]
            if x1 <= 2 or y1 <= 2 or x2 >= W - 2 or y2 >= H - 2:
                fl.add("truncated")
            if b["cls"] == "worker" and (y2 - y1) / max(x2 - x1, 1) < crouch_aspect and "truncated" not in fl:
                fl.add("crouch")
            area = max((x2 - x1) * (y2 - y1), 1)
            for o in bs:
                if o is b or o["cls"] != "worker" and b["cls"] != "worker":
                    continue
                ox1, oy1, ox2, oy2 = o["xyxy"]
                inter = max(min(x2, ox2) - max(x1, ox1), 0) * max(min(y2, oy2) - max(y1, oy1), 0)
                if inter / area > 0.5 and oy2 > y2 + 5 and o["cls"] == "worker":
                    fl.add("occluded")
            small = y2 - y1 < min_h
            if small:
                fl.add("ignore")
            b["flags"] = sorted(fl)
            if "occ" in b:
                if "occluded" in fl and b["occ"] == "none":
                    b["occ"] = "partial"
                elif "occluded" not in fl and b["occ"] == "partial":
                    b["occ"] = "none"
                if b.get("occ") == "full" and "occluded" not in b["flags"]:
                    b["flags"] = sorted(fl | {"occluded"})
            if "ignore_reason" in b:
                if small:
                    b["ignore_reason"] = b["ignore_reason"] or "small"
                elif b["ignore_reason"] == "small":
                    b["ignore_reason"] = None
                    b["flags"] = [x for x in b["flags"] if x != "ignore"]
