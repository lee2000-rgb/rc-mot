#!/usr/bin/env python3
"""Apply the LiDAR world-map audit (label_lidar_world_audit.py -> <seg>/lidar_audit.json) to gt.json.

For the ids listed in --ids (fragmented ids of the same static LiDAR person), every box is re-assigned per frame
to the LiDAR static person (P*) whose projected box it overlaps, and that person's canonical id (--canon P1=2 ...).
Boxes of one id that overlap no P keep their id, unless --canon maps the id itself (fallback id).
Then, per frame, two boxes of the same id are reduced to the one closest to the LiDAR box (a duplicate box from the
detector), ref3d of every box on a P (or on a mover M from ref3d_wrong) is replaced by the LiDAR position when it
is missing or off by > 5 % + 0.5 m (the sparse ref3d of the fast-swing frames picked the background), explicit
--set overrides are applied (frame:id->id), empty ids are removed and a resolved note + provenance are written.

  PYTHONPATH=src python3 scripts/label_lidar_audit_apply.py data/labels/<seg> --ids 2,53 --canon P1=2 --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rcmot.labeling.gt import GT  # noqa: E402


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def overlap_score(box, pbox):
    """IoU, or a centre test for the swing frames where the projection lags by up to one box width."""
    v = iou(box, pbox)
    cx = 0.5 * (box[0] + box[2])
    pcx = 0.5 * (pbox[0] + pbox[2])
    w = max(box[2] - box[0], pbox[2] - pbox[0])
    vy = min(box[3], pbox[3]) - max(box[1], pbox[1])
    hy = min(box[3] - box[1], pbox[3] - pbox[1])
    if abs(cx - pcx) < 0.9 * w and vy > 0.5 * hy:
        v = max(v, 0.3 + 0.1 * (1 - abs(cx - pcx) / (0.9 * w)))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("--ids", default="", help="ids to re-assign per frame by the LiDAR person they overlap")
    ap.add_argument("--canon", nargs="*", default=[], help="P1=2 P2=3 ... canonical id per LiDAR person; also id=id fallbacks")
    ap.add_argument("--set", nargs="*", default=[], help="explicit frame:id=newid overrides (applied first)")
    ap.add_argument("--merge", nargs="*", default=[], help="plain merges src=dst (movers etc.)")
    ap.add_argument("--min-score", type=float, default=0.3)
    ap.add_argument("--ref-fix", action="store_true", help="also fix ref3d on boxes over a LiDAR person / listed in ref3d_wrong")
    ap.add_argument("--add-missed", nargs="*", default=[], help="frame:P:id boxes to add (only this, when given)")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--by", default="label_lidar_audit_apply")
    a = ap.parse_args()

    seg = Path(a.seg)
    if a.add_missed:
        return add_missed(seg, [(int(x.split(":")[0]), x.split(":")[1], int(x.split(":")[2])) for x in a.add_missed],
                          a.by, a.apply, a.min_score)
    g = GT(seg / "gt.json")
    audit = json.load(open(seg / "lidar_audit.json"))
    pf = audit["per_frame"]
    ids = {int(x) for x in a.ids.split(",") if x}
    canon_obj, canon_id = {}, {}
    for s in a.canon:
        k, v = s.split("=")
        (canon_obj if k.startswith(("P", "M")) else canon_id)[k if k.startswith(("P", "M")) else int(k)] = int(v)
    log = []

    # explicit overrides
    for s in a.set:
        fi, rest = s.split(":")
        src, dst = (int(x) for x in rest.split("="))
        b = g.box_of(int(fi), src)
        if b is None:
            log.append(f"set f{fi} {src}->{dst}: no box")
            continue
        other = g.box_of(int(fi), dst)
        if other is not None:
            g.frames[int(fi)]["boxes"].remove(other)
            log.append(f"set f{fi}: dropped existing {dst} box")
        b["id"] = dst
        g.ensure_id(dst)
        log.append(f"set f{fi} {src}->{dst}")

    # per-frame re-assignment
    changes = defaultdict(lambda: defaultdict(int))
    on_person = {}  # (frame, id) -> object key
    for i, f in enumerate(g.frames):
        objs = pf.get(str(i), {}).get("objects", {})
        pers = {k: o for k, o in objs.items() if k in canon_obj and o.get("in_fov") and "box" in o}
        for b in f["boxes"]:
            if b["id"] not in ids:
                continue
            best, bs = None, a.min_score
            for k, o in pers.items():
                s = overlap_score(b["xyxy"], o["box"])
                if s > bs:
                    best, bs = k, s
            if best is None:
                new = canon_id.get(b["id"], b["id"])
            else:
                new = canon_obj[best]
                on_person[(i, new)] = best
            h = b["xyxy"][3] - b["xyxy"][1]
            if a.verbose and (best is not None or b["id"] in ids):
                ph = pers[best]["h_px"] if best else None
                print(f"  f{i} id {b['id']} -> {new} via {best} score {bs:.2f} h {h:.0f} vs lidar {ph}"
                      + ("  !!" if ph and abs(h - ph) > 0.35 * ph else ""))
            if new != b["id"]:
                changes[b["id"]][new] += 1
                b["id"] = new
                g.ensure_id(new)
    for s in a.merge:
        src, dst = (int(x) for x in s.split("="))
        g.merge(src, dst)
        log.append(f"merge {src}->{dst}")

    # duplicates: same id twice in a frame
    ndup = 0
    for i, f in enumerate(g.frames):
        seen = defaultdict(list)
        for b in f["boxes"]:
            seen[b["id"]].append(b)
        for t, bs in seen.items():
            if len(bs) < 2:
                continue
            k = on_person.get((i, t))
            pbox = pf.get(str(i), {}).get("objects", {}).get(k, {}).get("box") if k else None
            if pbox:
                bs.sort(key=lambda b: -overlap_score(b["xyxy"], pbox))
            else:
                bs.sort(key=lambda b: -(b["xyxy"][3] - b["xyxy"][1]))
            for b in bs[1:]:
                f["boxes"].remove(b)
                ndup += 1
                log.append(f"f{i} id {t}: duplicate box dropped {[round(v) for v in b['xyxy']]}")

    # ref3d from the LiDAR person
    nref = 0
    if a.ref_fix:
        wrong = {(w["frame"], w["id"]): w["object"] for w in audit.get("ref3d_wrong", [])}
        for i, f in enumerate(g.frames):
            objs = pf.get(str(i), {}).get("objects", {})
            for b in f["boxes"]:
                k = on_person.get((i, b["id"]))
                if k is None:
                    # a box of a canonical id that overlaps its own person (not re-assigned this run)
                    for kk, cid in canon_obj.items():
                        o = objs.get(kk)
                        if cid == b["id"] and o and o.get("in_fov") and "box" in o and overlap_score(b["xyxy"], o["box"]) >= a.min_score:
                            k = kk
                            break
                if k is None:
                    k = wrong.get((i, b["id"]))
                o = objs.get(k) if k else None
                if not o or o.get("range") is None:
                    continue
                r = (b.get("ref3d") or {}).get("range_m")
                if r is not None and abs(r - o["range"]) <= 0.05 * o["range"] + 0.5:
                    continue
                g.set_ref3d(i, b["id"], {"range_m": o["range"], "xy_level": o["xy_level"], "n_lidar": 0,
                                         "quality": "sparse", "src": "lidar_world_audit"})
                nref += 1

    # empty ids
    alive = {b["id"] for f in g.frames for b in f["boxes"]}
    gone = [t for t in list(g.ids) if t not in alive]
    for t in gone:
        g.ids.pop(t, None)
        g.review["ids"].pop(t, None)
    touched = set(changes) | {d for v in changes.values() for d in v} | {int(s.split("=")[1]) for s in a.merge}
    for t in touched:
        if t in g.ids and "motion" in g.ids[t]:
            g.ids[t]["motion"] = None

    print("re-assignments:")
    for s, v in sorted(changes.items()):
        print(f"  id {s} -> {dict(v)}")
    print(f"duplicates dropped {ndup}, ref3d fixed {nref}, ids removed {gone}")
    for s in log:
        print(" ", s)
    print("counts", g.counts())
    if a.apply:
        summary = (f"LiDAR world audit applied: {', '.join(f'{s}->{dict(v)}' for s, v in sorted(changes.items()))}; "
                   f"{ndup} duplicate boxes dropped; {nref} ref3d replaced by the LiDAR person position; ids removed {gone}; "
                   f"overrides {a.set}; merges {a.merge}; see lidar_audit.md")
        for t in sorted(touched):
            if t in g.ids and g.id_frames(t):
                g.note(g.id_frames(t)[0], t, summary, kind="id", by=a.by, status="resolved")
                break
        g.add_provenance("agent", a.by, "scripts/label_lidar_audit_apply.py", note=summary)
        g.save()
        print("saved")



def add_missed(seg: Path, items: list[tuple[int, str, int]], by: str, apply: bool, min_score: float = 0.3):
    """Add a box for (frame, LiDAR person, id) where the GT has none: the LiDAR projection shifted by the median
    image offset of the other LiDAR persons in that frame (the projection lags the image during a fast swing), or
    the mean of the id's own boxes in the neighbouring frames when no other person is matched. src=lidar_proj."""
    g = GT(seg / "gt.json")
    pf = json.load(open(seg / "lidar_audit.json"))["per_frame"]
    added = []
    for i, k, tid in items:
        if g.box_of(i, tid) is not None:
            print(f"  f{i} id {tid}: box exists, skipped")
            continue
        objs = pf[str(i)]["objects"]
        o = objs[k]
        pb = o["box"]
        dx, dy, sh = [], [], []
        for kk, oo in objs.items():
            if kk == k or not oo.get("in_fov") or "box" not in oo:
                continue
            for b in g.frames[i]["boxes"]:
                s = overlap_score(b["xyxy"], oo["box"])
                if s >= min_score:
                    bb = b["xyxy"]
                    dx.append(0.5 * (bb[0] + bb[2]) - 0.5 * (oo["box"][0] + oo["box"][2]))
                    dy.append(bb[3] - oo["box"][3])
                    sh.append((bb[3] - bb[1]) / max(oo["box"][3] - oo["box"][1], 1))
                    break
        how = ""
        if dx:
            import statistics as st
            mdx, mdy, msh = st.median(dx), st.median(dy), st.median(sh)
            h = (pb[3] - pb[1]) * msh
            w = (pb[2] - pb[0]) * msh
            cx = 0.5 * (pb[0] + pb[2]) + mdx
            y2 = pb[3] + mdy
            box = [cx - w / 2, y2 - h, cx + w / 2, y2]
            how = f"LiDAR projection + median offset of {len(dx)} other persons ({mdx:+.0f},{mdy:+.0f}) px, h x{msh:.2f}"
        else:
            nb = [g.box_of(j, tid) for j in (i - 1, i + 1) if 0 <= j < len(g.frames) and g.box_of(j, tid)]
            if len(nb) == 2:
                box = [0.5 * (p + q) for p, q in zip(nb[0]["xyxy"], nb[1]["xyxy"])]
                how = "mean of the neighbouring frames"
            else:
                box = list(pb)
                how = "raw LiDAR projection (no reference in the frame; check)"
        box = [round(max(0.0, min(v, lim)), 1) for v, lim in zip(box, (1279, 719, 1279, 719))]
        added.append((i, tid, box, how))
        print(f"  f{i} id {tid} {k} {o['range']} m: {box} ({how})")
        if apply:
            b = g.add_box(i, tid, box, src="lidar_proj")
            g.set_ref3d(i, tid, {"range_m": o["range"], "xy_level": o["xy_level"], "n_lidar": 0,
                                 "quality": "sparse", "src": "lidar_world_audit"})
            g.note(i, tid, f"box added from the LiDAR static person {k} ({how}); verify the box edges", kind="box",
                   by=by, status="open")
    if apply and added:
        g.add_provenance("agent", by, "scripts/label_lidar_audit_apply.py",
                         note=f"{len(added)} missed boxes added from LiDAR persons: " + ", ".join(f"f{i}:{t}" for i, t, _, _ in added))
        g.save()
        print("saved", g.counts())


if __name__ == "__main__":
    sys.exit(main())
