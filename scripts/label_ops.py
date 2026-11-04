#!/usr/bin/env python3
"""Batch edits of a gt.json (same operations as the review GUI).

  python3 scripts/label_ops.py <seg> merge SRC DST | delete ID | delete-box FRAME ID | split ID FRAME
        | interp ID [MAXGAP] | interp-all [MAXGAP] | autoflags | renumber | desc ID "text" [CLS]
        | note FRAME ID "text" [--kind id|box|flag|motion|ref3d|ego] [--by WHO] | resolve N
        | flag FRAME ID FLAG | add FRAME ID x1 y1 x2 y2 | stage NAME | info
        | motion ID static|walk|mixed F0 F1 | motion-clear ID
      v2 (docs/28 §3.3):
        | upgrade | occ FRAME ID none|partial|full | ignore-reason FRAME ID small|out_fov|unidentifiable|interp_gap|edge|none
        | ego F0 F1 static|drive|swing|drive+swing | ego-clear | meta key=value ... | split dev|test
        | provenance STAGE WHO [note] | role ID target|background | equipment ID excavator|loader|truck|utv|unknown

motion: per-id ground-truth motion state over the label frames F0..F1 (inclusive), stored as
ids[ID]["motion"] = [[F0, F1, state], ...]; used by the standstill / vibration metrics (docs/21).
"static" = feet planted (arm / torso movement allowed), "walk" = displacement, "mixed" = unsure.
Set motion AFTER merge / split: those ops do not carry the motion segments over.
ego: meta.ego_state segments (same replace-overlap semantics as motion). meta: values are parsed as JSON when
possible (meta ego_rate='{"yaw_rate_max_dps": 67}' lighting=day_backlight gnss=null).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.gt import GT, auto_flags  # noqa: E402


def pop_opts(args: list[str], names=("--kind", "--by")) -> tuple[list[str], dict]:
    """Remove --name value pairs from args."""
    out, opts, i = [], {}, 0
    while i < len(args):
        if args[i] in names and i + 1 < len(args):
            opts[args[i][2:]] = args[i + 1]
            i += 2
        else:
            out.append(args[i])
            i += 1
    return out, opts


def parse_value(s: str):
    try:
        return json.loads(s)
    except ValueError:
        return s


def main():
    seg, op, *args = sys.argv[1:]
    args, opts = pop_opts(args)
    g = GT(Path(seg) / "gt.json")
    if op == "merge":
        g.merge(int(args[0]), int(args[1]))
    elif op == "delete":
        g.delete_id(int(args[0]))
    elif op == "delete-box":
        i, tid = int(args[0]), int(args[1])
        g.frames[i]["boxes"] = [b for b in g.frames[i]["boxes"] if b["id"] != tid]
    elif op == "split" and args and args[0] in ("dev", "test"):
        g.set_split(args[0])
    elif op == "split":
        print("new id", g.split(int(args[0]), int(args[1])))
    elif op == "interp":
        print(g.interpolate(int(args[0]), int(args[1]) if len(args) > 1 else None), "boxes")
    elif op == "interp-all":
        print(sum(g.interpolate(t, int(args[0]) if args else 10) for t in list(g.ids)), "boxes")
    elif op == "autoflags":
        import cv2
        im = cv2.imread(str(g.root / g.frames[0]["file"])) if g.frames else None
        auto_flags(g, img_wh=(im.shape[1], im.shape[0]) if im is not None else (1280, 720))
    elif op == "motion":
        tid, state, f0, f1 = int(args[0]), args[1], int(args[2]), int(args[3])
        if state not in ("static", "walk", "mixed") or f1 < f0 or not (0 <= f0 and f1 < len(g.frames)):
            raise SystemExit(f"motion: bad arguments {args}")
        g.ensure_id(tid)
        segs = [m for m in g.ids[tid].get("motion", []) if m[1] < f0 or m[0] > f1]   # replace overlaps
        g.ids[tid]["motion"] = sorted(segs + [[f0, f1, state]])
    elif op == "motion-clear":
        g.ids[int(args[0])].pop("motion", None)
    elif op == "renumber":
        print(g.renumber())
    elif op == "desc":
        g.ensure_id(int(args[0]))
        g.ids[int(args[0])]["desc"] = args[1]
        if len(args) > 2:
            g.ids[int(args[0])]["cls"] = args[2]
            for f in g.frames:
                for b in f["boxes"]:
                    if b["id"] == int(args[0]):
                        b["cls"] = args[2]
    elif op == "note":
        g.note(int(args[0]), int(args[1]), args[2], kind=opts.get("kind", "box"), by=opts.get("by", "agent"))
    elif op == "resolve":
        print(g.resolve_note(int(args[0]), by=opts.get("by")))
    elif op == "flag":
        g.toggle_flag(g.box_of(int(args[0]), int(args[1])), args[2])
    elif op == "add":
        g.add_box(int(args[0]), int(args[1]), [float(v) for v in args[2:6]])
    elif op == "stage":
        g.meta["stage"] = args[0]
    # ---------------------------------------------------------------- v2
    elif op == "upgrade":
        for c in g.upgrade(who=opts.get("by", "user")):
            print(" +", c)
    elif op == "occ":
        g.set_occ(int(args[0]), int(args[1]), args[2])
    elif op == "ignore-reason":
        g.set_ignore_reason(int(args[0]), int(args[1]), args[2])
    elif op == "ego":
        print("ego_state:", g.set_ego_segment(int(args[0]), int(args[1]), args[2]))
    elif op == "ego-clear":
        g.meta.pop("ego_state", None)
    elif op == "meta":
        kv = dict(a.split("=", 1) for a in args)
        g.set_meta(**{k: parse_value(v) for k, v in kv.items()})
    elif op == "provenance":
        print(g.add_provenance(args[0], args[1], tool="scripts/label_ops.py", note=args[2] if len(args) > 2 else ""))
    elif op == "role":
        g.set_role(int(args[0]), args[1])
    elif op == "equipment":
        g.set_equipment_type(int(args[0]), args[1])
    elif op != "info":
        raise SystemExit(f"unknown op {op}")
    g.save()
    from collections import Counter
    c = Counter(b["id"] for f in g.frames for b in f["boxes"])
    print(g.counts(), {k: (c[k], g.ids[k]["cls"][:6], g.ids[k]["desc"][:30]) for k in sorted(g.ids)})
    mo = {k: g.ids[k]["motion"] for k in sorted(g.ids) if g.ids[k].get("motion")}
    if mo:
        print("motion:", mo)
    if g.schema_version >= 2:
        m = g.meta
        print(f"v{g.schema_version} split={m.get('split')} site={m.get('site')} radar={m.get('radar')} "
              f"lighting={m.get('lighting')} ego_state={m.get('ego_state')} provenance={len(m.get('provenance', []))}")
        op_notes = [n for n in g.notes if n.get("status", "open") == "open"]
        print(f"notes open {len(op_notes)}/{len(g.notes)}")


if __name__ == "__main__":
    main()
