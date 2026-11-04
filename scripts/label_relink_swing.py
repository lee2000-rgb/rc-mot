#!/usr/bin/env python3
"""Re-link the boxes of a swing segment in the WORLD frame (ego yaw removed) - docs/13 "선회 구간 재연결".

The draft linker (rcmot.labeling.draft.link) matches boxes by image position; at 60 deg/s the scene moves ~170 px per
5 Hz frame, so inside a sweep it chains different people and every sweep starts new ids. Here every box gets a world
azimuth = azimuth of the camera ray through the box centre (level frame) + the pipeline's world heading at the frame,
and, when it carries ref3d, a range. Boxes are then linked

  (a) frame to frame inside a sweep: same class, |d azimuth| <= az_gate + az_rate * gap, |d range| <= range_gate +
      0.4 m * gap when both have ref3d, box-height ratio <= 1.8 (Hungarian per frame, gap <= gap_short frames);
  (b) tracklet to tracklet across sweeps (gap_short < gap <= gap_long frames): only when the azimuth (<= --az-deg)
      AND the ref3d range (<= --range-m) agree at the junction, i.e. a standing person seen again on the next sweep
      (position + ref3d continuity = 2 of the 3 identity criteria of gt_design D.3). Tracklets without ref3d on
      the junction side are never merged across sweeps.

Heading source: `Pipeline.step(...).world.theta` (rcmot.world: gyro-integrated yaw with bias / lever-arm model), run
with a null detector over meta.start-1 s .. meta.end of meta.bag (scripts/label_auto_meta.ego_samples - the same
poses label_auto_meta uses for the motion drafts). That pass takes ~1 min per minute of bag, so the per-frame
heading is cached in <seg>/cache/heading.json keyed by bag / config / calib / start / end / stamps; --no-cache
recomputes it.

Rules: boxes WITHOUT ref3d (quality none) take part in the frame-to-frame chain (azimuth only) but their `id` is never
rewritten; when their chain-mates get another id they are listed as "kept" so the reviewer can decide. A group keeps
the id most of its boxes already have (so a reviewed file is near-idempotent); split-offs get fresh ids. ids whose
membership changed lose their `motion` draft (label_auto_meta re-drafts it) and go back to review state "draft".

  PYTHONPATH=src python3 scripts/label_relink_swing.py data/labels/<seg> [--az-deg 3 --range-m 2.5 --dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rcmot.labeling.provenance import add_provenance, load_doc, ref3d_of, save_doc  # noqa: E402

TOOL = "scripts/label_relink_swing.py"
WHO = "label_relink_swing"


@dataclass
class Params:
    gap_short: int = 3          # frame-to-frame max gap inside a sweep [frames]
    az_gate: float = 2.5        # deg, + az_rate * gap
    az_rate: float = 1.5        # deg per frame of gap (walking at 15 m ~ 1.1 deg / 0.2 s)
    range_gate: float = 2.0     # m, + 0.4 * gap (both boxes with ref3d)
    height_ratio: float = 1.8   # max box height ratio between linked boxes
    gap_long: int = 30          # tracklet-to-tracklet max gap [frames]
    az_long: float = 3.0        # --az-deg: azimuth agreement at the junction [deg]
    range_long: float = 2.5     # --range-m: ref3d range agreement at the junction [m]


# ------------------------------------------------------------------------------------------------ heading
def heading_key(m: dict, n_frames: int, stamps) -> str:
    s = json.dumps([m.get("bag"), m.get("config"), m.get("calib"), m.get("start"), m.get("end"), n_frames,
                    [round(float(x), 4) for x in stamps[:3]], [round(float(x), 4) for x in stamps[-3:]]])
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def pipeline_headings(seg: Path, doc: dict, cfg_path: Path | None, calib_path: Path | None, bag_path: Path | None,
                      use_cache: bool = True) -> tuple[np.ndarray, dict]:
    """World heading theta [rad] of the level frame at every GT frame (np.interp over the pipeline's world poses),
    plus the cache record. Frames farther than 0.3 s from any pose get NaN (no heading)."""
    import yaml
    from label_auto_meta import ego_samples
    from rcmot.utils.calib import RigCalib

    m, frames = doc["meta"], doc["frames"]
    stamps = np.array([f["stamp"] for f in frames], float)
    key = heading_key(m, len(frames), stamps)
    cpath = seg / "cache" / "heading.json"
    if use_cache and cpath.exists():
        rec = json.load(open(cpath))
        if rec.get("key") == key:
            return np.array(rec["theta"], float), rec
    cfg = yaml.safe_load(open(cfg_path or ROOT / m["config"]))
    cal = RigCalib.load(str(calib_path or ROOT / m["calib"]))
    samp, worlds = ego_samples(m, cfg, cal, bag_path or ROOT / m["bag"])
    t = samp[:, 0] if len(samp) else np.zeros(0)
    ok = np.array([w is not None for w in worlds], bool)
    theta = np.full(len(frames), np.nan)
    if ok.any():
        tw = t[ok]
        th = np.unwrap([w.theta for w, o in zip(worlds, ok) if o])
        theta = np.interp(stamps, tw, th)
        near = np.min(np.abs(stamps[:, None] - tw[None, :]), axis=1)
        theta[near > 0.3] = np.nan
    rec = {"key": key, "source": "rcmot.pipeline.Pipeline.step().world.theta via label_auto_meta.ego_samples",
           "bag": m.get("bag"), "config": m.get("config"), "calib": m.get("calib"), "start": m.get("start"),
           "end": m.get("end"), "n_poses": int(ok.sum()), "theta": [None if np.isnan(x) else round(float(x), 6) for x in theta]}
    if use_cache:
        cpath.parent.mkdir(exist_ok=True)
        json.dump(rec, open(cpath, "w"))
    return theta, rec


# ------------------------------------------------------------------------------------------------ linking
def _daz(a1: float, a2: float) -> float:
    return abs((a1 - a2 + 180.0) % 360.0 - 180.0)


def observations(frames: list, theta: np.ndarray, az_of_pixel) -> list[dict]:
    """One record per box: frame index k, box, world azimuth [deg] (None without heading), ref3d range, height.
    az_of_pixel(u, v) -> level-frame azimuth [rad] of the camera ray (RigCalib.pixel_ray)."""
    obs = []
    for k, f in enumerate(frames):
        th = theta[k] if k < len(theta) else np.nan
        for b in f["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            az = None
            if not np.isnan(th):
                az = (np.degrees(az_of_pixel(0.5 * (x1 + x2), 0.5 * (y1 + y2)) + th) + 180.0) % 360.0 - 180.0
            r = ref3d_of(b)
            obs.append({"k": k, "b": b, "az": az, "rng": None if r is None else float(r["range_m"]),
                        "h": max(float(y2 - y1), 1.0), "ref": r is not None,
                        "lock": r is None or az is None})       # id never rewritten (no ref3d / no heading)
    return obs


def link_frames(obs: list[dict], n_frames: int, p: Params) -> list[dict]:
    """(a) frame-to-frame tracklets. Boxes without a world azimuth (no heading at the frame) stay singletons."""
    by_frame: dict[int, list] = defaultdict(list)
    for o in obs:
        by_frame[o["k"]].append(o)
    tracks: list[dict] = []
    for k in range(n_frames):
        cur = by_frame.get(k, [])
        live = [tr for tr in tracks if 0 < k - tr["last"] <= p.gap_short and tr["az"] is not None]
        C = np.full((len(live), len(cur)), 1e6)
        for i, tr in enumerate(live):
            gap = k - tr["last"]
            for j, o in enumerate(cur):
                if o["az"] is None or o["b"]["cls"] != tr["cls"]:
                    continue
                dz = _daz(tr["az"], o["az"])
                lim = p.az_gate + p.az_rate * gap
                if dz > lim:
                    continue
                c = dz / lim
                if tr["rng"] is not None and o["rng"] is not None:
                    dr = abs(tr["rng"] - o["rng"])
                    rl = p.range_gate + 0.4 * gap
                    if dr > rl:
                        continue
                    c += dr / rl
                hr = max(tr["h"], o["h"]) / min(tr["h"], o["h"])
                if hr > p.height_ratio:
                    continue
                C[i, j] = c + 0.3 * (hr - 1) + 0.05 * (gap - 1)
        rows, cols = linear_sum_assignment(C) if C.size else ([], [])
        used = set()
        for i, j in zip(rows, cols):
            if C[i, j] < 1e5:
                tr, o = live[i], cur[j]
                tr.update(last=k, az=o["az"], rng=o["rng"] if o["rng"] is not None else tr["rng"], h=o["h"])
                tr["obs"].append(o)
                used.add(j)
        for j, o in enumerate(cur):
            if j not in used:
                tracks.append({"id": len(tracks) + 1, "cls": o["b"]["cls"], "first": k, "last": k, "az": o["az"],
                               "rng": o["rng"], "h": o["h"], "obs": [o]})
    return tracks


def link_tracklets(tracks: list[dict], p: Params) -> list[tuple]:
    """(b) cross-sweep links (pred_tracklet, next_tracklet, gap, daz, drange): azimuth AND ref3d range must agree
    at the junction; each tracklet takes at most one predecessor / successor (earliest first, best cost)."""
    for tr in tracks:
        az = [o["az"] for o in tr["obs"] if o["az"] is not None]
        rg = [o["rng"] for o in tr["obs"] if o["rng"] is not None]
        tr["az0"], tr["az1"] = (az[0], az[-1]) if az else (None, None)
        tr["r0"] = float(np.median(rg[:3])) if rg else None
        tr["r1"] = float(np.median(rg[-3:])) if rg else None
        tr["next"] = tr["prev"] = None
    links = []
    for tr in sorted(tracks, key=lambda t: t["first"]):
        if tr["az0"] is None or tr["r0"] is None:
            continue
        best = None
        for pr in tracks:
            if pr is tr or pr["cls"] != tr["cls"] or pr["next"] is not None or pr["az1"] is None or pr["r1"] is None:
                continue
            gap = tr["first"] - pr["last"]
            if not (p.gap_short < gap <= p.gap_long):
                continue
            dz = _daz(pr["az1"], tr["az0"])
            dr = abs(pr["r1"] - tr["r0"])
            if dz > p.az_long or dr > p.range_long:
                continue
            c = dz / p.az_long + dr / p.range_long + 0.02 * gap
            if best is None or c < best[0]:
                best = (c, pr, dz, dr, gap)
        if best is not None:
            _, pr, dz, dr, gap = best
            pr["next"], tr["prev"] = tr["id"], pr["id"]
            links.append((pr["id"], tr["id"], gap, dz, dr))
    return links


def groups_of(tracks: list[dict], links: list[tuple]) -> list[list[dict]]:
    """Tracklets joined by the links -> groups (lists of observations), ordered by first frame."""
    parent = {tr["id"]: tr["id"] for tr in tracks}

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for src, dst, *_ in links:
        parent[root(dst)] = root(src)
    g: dict[int, list] = defaultdict(list)
    for tr in tracks:
        g[root(tr["id"])].extend(tr["obs"])
    return sorted((sorted(v, key=lambda o: o["k"]) for v in g.values()), key=lambda v: v[0]["k"])


def assign_ids(groups: list[list[dict]], all_ids: set[int]) -> dict:
    """Group -> id: the id most of the group's boxes already carry, larger groups first; a taken id (split-off) gets
    a fresh one above every id in the file. Groups made only of locked boxes keep their id and claim nothing (those
    boxes are not rewritten, so another group may still take that id by majority). Returns {group index: id}."""
    nxt = max(all_ids, default=0) + 1
    taken: set[int] = set()
    out = {}
    for gi in sorted(range(len(groups)), key=lambda i: (-len(groups[i]), groups[i][0]["k"])):
        cnt = Counter(int(o["b"]["id"]) for o in groups[gi])
        if all(o["lock"] for o in groups[gi]):
            out[gi] = cnt.most_common(1)[0][0]
            continue
        tid = None
        for cand, _ in cnt.most_common():
            if cand not in taken:
                tid = cand
                break
        if tid is None:
            tid = nxt
            nxt += 1
        taken.add(tid)
        out[gi] = tid
    return out


def relink(doc: dict, theta: np.ndarray, az_of_pixel, p: Params) -> dict:
    """Plan the re-link (nothing is written): {groups, ids, links, tracks, changes, kept, merges, splits}."""
    frames = doc["frames"]
    obs = observations(frames, theta, az_of_pixel)
    tracks = link_frames(obs, len(frames), p)
    links = link_tracklets(tracks, p)
    groups = groups_of(tracks, links)
    all_ids = {int(b["id"]) for f in frames for b in f["boxes"]}
    gid = assign_ids(groups, all_ids)
    changes, kept = [], []                                 # (k, box, old, new) for ref3d boxes / no-ref3d boxes
    new_of_old: dict[int, set] = defaultdict(set)          # old id -> new ids of its ref3d boxes
    old_of_new: dict[int, set] = defaultdict(set)
    for gi, grp in enumerate(groups):
        for o in grp:
            old, new = int(o["b"]["id"]), gid[gi]
            if o["lock"]:
                if old != new:
                    kept.append((o["k"], o["b"], old, new))
                continue
            new_of_old[old].add(new)
            old_of_new[new].add(old)
            if old != new:
                changes.append((o["k"], o["b"], old, new))
    merges = sum(len(v) - 1 for v in old_of_new.values())
    splits = sum(len(v) - 1 for v in new_of_old.values())
    final: dict[int, list] = defaultdict(list)             # frame -> ids after the plan (locked boxes keep theirs)
    for gi, grp in enumerate(groups):
        for o in grp:
            final[o["k"]].append(int(o["b"]["id"]) if o["lock"] else gid[gi])
    conflicts = sorted((k, i) for k, ids in final.items() for i, n in Counter(ids).items() if n > 1)
    return {"groups": groups, "ids": gid, "links": links, "tracks": tracks, "changes": changes, "kept": kept,
            "merges": merges, "splits": splits, "old_of_new": old_of_new, "new_of_old": new_of_old,
            "conflicts": conflicts, "n_obs": len(obs), "n_no_az": sum(o["az"] is None for o in obs)}


def apply(doc: dict, plan: dict) -> dict:
    """Write the plan into doc (ids of ref3d boxes; ids table; review state; motion of changed ids)."""
    changed_ids: set[int] = set()
    for k, b, old, new in plan["changes"]:
        b["id"] = new
        changed_ids.update((old, new))
    for new, olds in plan["old_of_new"].items():
        if olds != {new}:
            changed_ids.add(new)
    for old, news in plan["new_of_old"].items():
        if news != {old}:
            changed_ids.add(old)
    used: dict[str, str] = {}
    for f in doc["frames"]:
        for b in f["boxes"]:
            used.setdefault(str(b["id"]), b["cls"])
    ids = doc.setdefault("ids", {})
    for sid, cls in used.items():
        ids.setdefault(sid, {"cls": cls, "desc": "", "role": "target"})
    dropped = [sid for sid in ids if sid not in used]
    for sid in dropped:
        ids.pop(sid)
    for tid in changed_ids:
        if str(tid) in ids:
            ids[str(tid)].pop("motion", None)
    doc["ids"] = dict(sorted(ids.items(), key=lambda kv: int(kv[0])))
    rv = doc.setdefault("review", {}).setdefault("ids", {})
    for sid in dropped:
        rv.pop(sid, None)
    for tid in changed_ids:
        if str(tid) in ids:
            rv[str(tid)] = "draft"
    for sid in used:
        rv.setdefault(sid, "draft")
    doc["review"]["ids"] = dict(sorted(rv.items(), key=lambda kv: int(kv[0])))
    return {"changed_ids": sorted(changed_ids), "dropped_ids": sorted(int(s) for s in dropped)}


def report(plan: dict, frames: list, verbose: bool = False) -> None:
    print(f"boxes {plan['n_obs']} (no heading: {plan['n_no_az']}); tracklets {len(plan['tracks'])} -> groups "
          f"{len(plan['groups'])}; cross-sweep links {len(plan['links'])}")
    print(f"proposed: {plan['merges']} merges, {plan['splits']} splits; ref3d boxes re-numbered {len(plan['changes'])}; "
          f"boxes without ref3d kept with another id than their chain {len(plan['kept'])}")
    for new, olds in sorted(plan["old_of_new"].items()):
        if len(olds) > 1:
            print(f"  merge {sorted(olds)} -> id {new}")
    for old, news in sorted(plan["new_of_old"].items()):
        if len(news) > 1:
            print(f"  split id {old} -> {sorted(news)}")
    for k, i in plan["conflicts"]:
        print(f"  CONFLICT f{k}: id {i} twice (a kept no-ref3d box and a re-linked box) - resolve in the GUI")
    for k, b, old, new in plan["kept"][:20]:
        print(f"  kept  f{k} id {old} (no ref3d; chain is id {new})")
    if len(plan["kept"]) > 20:
        print(f"  ... {len(plan['kept']) - 20} more kept")
    if verbose:
        for src, dst, gap, dz, dr in plan["links"]:
            print(f"  link tracklet {src}->{dst} gap {gap} f  daz {dz:.1f} deg  dr {dr:.1f} m")
        for gi, grp in enumerate(plan["groups"]):
            ks = [o["k"] for o in grp]
            az = [o["az"] for o in grp if o["az"] is not None]
            rg = [o["rng"] for o in grp if o["rng"] is not None]
            print(f"  id {plan['ids'][gi]:3d} {grp[0]['b']['cls'][:6]} f{ks[0]}-{ks[-1]} n={len(ks)} "
                  f"az {np.median(az) if az else float('nan'):6.1f} rng {np.median(rg) if rg else float('nan'):5.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("seg", help="data/labels/<seg> (v2 gt.json with ref3d + meta.ego_state)")
    ap.add_argument("--az-deg", type=float, default=3.0, help="cross-sweep azimuth agreement [deg]")
    ap.add_argument("--range-m", type=float, default=2.5, help="cross-sweep ref3d range agreement [m]")
    ap.add_argument("--gap-long", type=int, default=30, help="cross-sweep max gap [frames]")
    ap.add_argument("--gap-short", type=int, default=3, help="frame-to-frame max gap [frames]")
    ap.add_argument("--az-gate", type=float, default=2.5, help="frame-to-frame azimuth gate [deg] (+ --az-rate per gap frame)")
    ap.add_argument("--az-rate", type=float, default=1.5)
    ap.add_argument("--range-gate", type=float, default=2.0, help="frame-to-frame range gate [m] (+0.4 per gap frame)")
    ap.add_argument("--bag", help="override meta.bag")
    ap.add_argument("--config", help="override meta.config")
    ap.add_argument("--calib", help="override meta.calib")
    ap.add_argument("--no-cache", action="store_true", help="recompute the heading cache (<seg>/cache/heading.json)")
    ap.add_argument("--dry-run", action="store_true", help="print the proposal, do not write gt.json")
    ap.add_argument("-v", "--verbose", action="store_true", help="list links and groups")
    a = ap.parse_args()
    from rcmot.utils.calib import RigCalib

    seg = Path(a.seg)
    gt_path = seg / "gt.json"
    doc = load_doc(gt_path)
    m = doc["meta"]
    if m.get("schema_version", 1) < 2:
        sys.exit("v1 gt.json: run `label_ops.py <seg> upgrade` and eval_fusion_depth.py --write-ref first")
    n_ref = sum(ref3d_of(b) is not None for f in doc["frames"] for b in f["boxes"])
    if n_ref == 0:
        sys.exit("no box carries ref3d - run scripts/eval_fusion_depth.py --write-ref first")
    cal = RigCalib.load(str(a.calib or ROOT / m["calib"]))
    theta, rec = pipeline_headings(seg, doc, Path(a.config) if a.config else None, Path(a.calib) if a.calib else None,
                                   Path(a.bag) if a.bag else None, use_cache=not a.no_cache)
    print(f"heading: {rec['n_poses']} pipeline poses, {int(np.isnan(theta).sum())}/{len(theta)} GT frames without one "
          f"(cache {seg / 'cache' / 'heading.json'})")
    p = Params(gap_short=a.gap_short, az_gate=a.az_gate, az_rate=a.az_rate, range_gate=a.range_gate,
               gap_long=a.gap_long, az_long=a.az_deg, range_long=a.range_m)

    def az_of_pixel(u, v):
        d = cal.pixel_ray(u, v)
        return float(np.arctan2(d[1], d[0]))

    plan = relink(doc, theta, az_of_pixel, p)
    report(plan, doc["frames"], a.verbose)
    if a.dry_run:
        print("dry run: not written")
        return
    if not plan["changes"]:
        print("nothing to change: not written")
        return
    done = apply(doc, plan)
    add_provenance(doc, "relink", WHO, TOOL, ROOT,
                   note=f"world-frame re-link (ego yaw removed): {len(plan['tracks'])} tracklets -> {len(plan['groups'])} "
                        f"groups, {len(plan['links'])} cross-sweep links (az<={a.az_deg} deg & range<={a.range_m} m, "
                        f"gap<={a.gap_long} f); {plan['merges']} merges, {plan['splits']} splits, {len(plan['changes'])} "
                        f"ref3d boxes re-numbered, {len(plan['kept'])} no-ref3d boxes kept, {len(plan['conflicts'])} "
                        f"duplicate-id frames to resolve; ids changed {done['changed_ids']}; "
                        f"ids dropped {done['dropped_ids']}; motion cleared on changed ids")
    save_doc(doc, gt_path)
    print(f"written {gt_path}: ids changed {done['changed_ids']}, dropped {done['dropped_ids']}")


if __name__ == "__main__":
    main()
