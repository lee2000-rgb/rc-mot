#!/usr/bin/env python3
"""Merge the ids of STATIC workers across a whole swing segment by their WORLD position (docs/13 "정지 작업자 월드 묶음").

The draft linker and label_relink_swing.py only join boxes a few seconds apart, so a standing worker (or mannequin)
seen once per sweep gets a new id every sweep. A static worker has one world position, so every box with a LiDAR
reference (ref3d) is put in the world frame

    p_world = R(theta_i) · (xy_level + lever_arm)          theta_i = cached world heading (cache/heading.json)

and clustered with DBSCAN whose radius grows with range, max(--eps, --eps-per-m · range) (the heading error of a
66 deg/s swing is ~4 deg = 1.5 m at 22 m). An id is attached to a cluster when >= --frac of its ref3d boxes fall in
it. All ids of one cluster are merged into the id with most boxes unless
  * two of them have boxes in the same frame with low IoU (two people at the same spot -> reported, not merged);
    overlapping same-frame boxes (IoU >= --dup-iou) are duplicates of one person: the minor id's box is dropped;
  * the appearance check fails: OSNet (models/reid) embeddings of the crops, mean cosine inside the cluster >= --sim
    and each id's mean cosine to the rest of the cluster >= --sim-id.
Pass 2 attaches ids WITHOUT any ref3d (far single-frame boxes) to a cluster when the world azimuth of the box centre
is within --az-deg of the cluster, the box height is within 35 % of the cluster's median height and the OSNet
similarity to the cluster is >= --sim-id. Every decision is written to <seg>/cluster_report.md, cluster_bev.png and
cluster_<k>.jpg (crop montage per cluster); --apply performs the merges (provenance + a resolved Note per cluster).

  PYTHONPATH=src python3 scripts/label_static_cluster.py data/labels/<seg> [--lever 1.17 -0.89] [--apply]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcmot.labeling.gt import GT  # noqa: E402

LEVER = {"excavator_front_1.6m": (1.17, -0.89), "excavator_cab_3.2m_pitch26": (0.96, 0.88)}


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def dbscan(X: np.ndarray, eps: float, min_pts: int, scale: np.ndarray | None = None) -> np.ndarray:
    """DBSCAN; with `scale` (per point, metres) the distance between i and j is |xi-xj| / max(scale_i, scale_j)."""
    n = len(X)
    lab = -np.ones(n, int)
    D = np.linalg.norm(X[:, None] - X[None], axis=2)
    if scale is not None:
        D = D / np.maximum(scale[:, None], scale[None, :])
    nb = [np.where(D[i] <= eps)[0] for i in range(n)]
    k = 0
    for i in range(n):
        if lab[i] != -1 or len(nb[i]) < min_pts:
            continue
        lab[i] = k
        stack = list(nb[i])
        while stack:
            j = stack.pop()
            if lab[j] != -1:
                continue
            lab[j] = k
            if len(nb[j]) >= min_pts:
                stack.extend(q for q in nb[j] if lab[q] == -1)
        k += 1
    return lab


def daz(a1: float, a2: float) -> float:
    return abs((a1 - a2 + 180.0) % 360.0 - 180.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--lever", type=float, nargs=2, default=None)
    ap.add_argument("--eps", type=float, default=0.7, help="cluster radius at short range [m]")
    ap.add_argument("--eps-per-m", type=float, default=0.08, help="radius = max(eps, eps_per_m * range)")
    ap.add_argument("--min-pts", type=int, default=3)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--dup-iou", type=float, default=0.4)
    ap.add_argument("--az-deg", type=float, default=1.5)
    ap.add_argument("--sim", type=float, default=0.55)
    ap.add_argument("--sim-id", type=float, default=0.50)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    seg = Path(a.seg)
    seg = seg if seg.is_dir() else seg.parent
    g = GT(seg / "gt.json")
    m = g.meta
    hp = seg / "cache" / "heading.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        th = np.asarray([np.nan if x is None else x for x in h["theta"]], float)
    else:
        states = {st for _, _, st in (m.get("ego_state") or [])} or {"static"}
        assert states == {"static"}, f"no heading cache and ego_state {states}: run label_relink_swing.py --dry-run first"
        th = np.zeros(len(g.frames))
    assert len(th) == len(g.frames), "heading cache does not match the frames"
    lever = np.array(a.lever if a.lever else LEVER.get(m.get("mount"), (0.0, 0.0)))
    from rcmot.utils.calib import RigCalib
    cal = RigCalib.load(str(ROOT / m["calib"]))

    def world_az(i, b):
        x1, y1, x2, y2 = b["xyxy"]
        d = cal.pixel_ray(0.5 * (x1 + x2), 0.5 * (y1 + y2))
        return (np.degrees(np.arctan2(d[1], d[0]) + th[i]) + 180.0) % 360.0 - 180.0

    def usable(b):
        return b.get("cls") == "worker" and not b.get("ignore_reason") and "ignore" not in b["flags"]

    rows = []  # (frame, id, wx, wy, range)
    for i, f in enumerate(g.frames):
        if np.isnan(th[i]):
            continue
        c, s = np.cos(th[i]), np.sin(th[i])
        for b in f["boxes"]:
            r = b.get("ref3d") or {}
            if not usable(b) or r.get("range_m") is None:
                continue
            p = np.asarray(r["xy_level"], float) + lever
            rows.append((i, b["id"], c * p[0] - s * p[1], s * p[0] + c * p[1], r["range_m"]))
    A = np.array(rows)
    lab = dbscan(A[:, 2:4], 1.0, a.min_pts, scale=np.maximum(a.eps, a.eps_per_m * A[:, 4]))

    id_boxes = Counter(int(t) for t in A[:, 1])
    id_cl = defaultdict(Counter)
    for (i, t, *_), k in zip(rows, lab):
        id_cl[int(t)][int(k)] += 1
    assign = {}
    for t, cnt in id_cl.items():
        k, n = max(((k, n) for k, n in cnt.items() if k >= 0), key=lambda x: x[1], default=(-1, 0))
        if k >= 0 and n / id_boxes[t] >= a.frac:
            assign[t] = k
    members = defaultdict(list)
    for t, k in assign.items():
        members[k].append(t)

    # ---- appearance (OSNet) of every id that has boxes
    from rcmot.reid.extractor import ReID
    reid = ReID(str(ROOT / "models" / "reid" / "osnet_ain_x1_0_msmt17.pth"))
    imgs, emb, crops = {}, {}, {}

    def image(i):
        if i not in imgs:
            imgs[i] = cv2.imread(str(seg / g.frames[i]["file"]))
        return imgs[i]

    def embed(t):
        if t in emb:
            return emb[t]
        fr = g.id_frames(t)
        pick = [fr[int(q)] for q in np.linspace(0, len(fr) - 1, min(6, len(fr)))]
        cs = []
        for i in pick:
            x1, y1, x2, y2 = np.array(g.box_of(i, t)["xyxy"]).astype(int)
            c = image(i)[max(y1, 0):max(y2, y1 + 2), max(x1, 0):max(x2, x1 + 2)]
            if c.size:
                cs.append(c)
        crops[t] = cs
        e = np.array(reid(cs)) if cs else np.zeros((0, 512))
        emb[t] = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9) if len(e) else e
        return emb[t]

    # ---- pass 2: ids without any ref3d -> cluster by azimuth + height + appearance
    all_ids = {b["id"] for f in g.frames for b in f["boxes"] if usable(b)}
    noref = sorted(all_ids - set(id_boxes))
    cl_az, cl_h = {}, {}
    for k, ids in members.items():
        azs, hs = [], []
        for t in ids:
            for i in g.id_frames(t):
                if np.isnan(th[i]):
                    continue
                b = g.box_of(i, t)
                azs.append(world_az(i, b)), hs.append(b["xyxy"][3] - b["xyxy"][1])
        cl_az[k], cl_h[k] = float(np.median(azs)), float(np.median(hs))
    pass2 = defaultdict(list)
    pass2_log = []
    for t in noref:
        fr = [i for i in g.id_frames(t) if not np.isnan(th[i])]
        if not fr:
            continue
        az = float(np.median([world_az(i, g.box_of(i, t)) for i in fr]))
        hh = float(np.median([g.box_of(i, t)["xyxy"][3] - g.box_of(i, t)["xyxy"][1] for i in fr]))
        cand = [(daz(az, cl_az[k]), k) for k in members if daz(az, cl_az[k]) <= a.az_deg and abs(hh - cl_h[k]) / cl_h[k] <= 0.35]
        if not cand:
            continue
        d, k = min(cand)
        e = embed(t)
        ref = np.concatenate([embed(u) for u in members[k] if len(embed(u))])
        sim = float((e @ ref.T).mean()) if len(e) and len(ref) else 0.0
        pass2_log.append(f"  id {t} (no ref3d, {len(fr)} f, az {az:.1f}, h {hh:.0f}px) -> cluster {k} (daz {d:.1f}, h {cl_h[k]:.0f}px) sim {sim:.2f} "
                         + ("ATTACH" if sim >= a.sim_id else "no"))
        if sim >= a.sim_id:
            pass2[k].append(t)

    # ---- same-frame constraint: ids that appear together with NON-overlapping boxes are different people ->
    #      split such a cluster into sub-groups (greedy, by world-xy closeness, never joining a cannot-link pair).
    #      Overlapping same-frame boxes (centre inside the other box, or IoU >= dup_iou) are one person with two boxes.
    id_xy = {int(t): np.median(A[A[:, 1] == t, 2:4], 0) for t in id_boxes}

    def same_person(b1, b2):
        """Two boxes in one frame on one person: IoU >= dup_iou, or the centres are closer than 0.8 box width
        horizontally and the vertical extents overlap >= 60 % (the detector draws a body box and an edge box)."""
        if iou(b1["xyxy"], b2["xyxy"]) >= a.dup_iou:
            return True
        (x1, y1, x2, y2), (u1, v1, u2, v2) = b1["xyxy"], b2["xyxy"]
        dx = abs(0.5 * (x1 + x2) - 0.5 * (u1 + u2))
        vo = max(0.0, min(y2, v2) - max(y1, v1)) / max(1.0, min(y2 - y1, v2 - v1))
        return dx < 0.8 * max(x2 - x1, u2 - u1) and vo >= 0.6

    def cannot_link(t1, t2):
        both = set(g.id_frames(t1)) & set(g.id_frames(t2))
        return any(not same_person(g.box_of(i, t1), g.box_of(i, t2)) for i in both)

    clusters, split_log = {}, []
    nxt = max(members) + 1 if members else 0
    for k, v in members.items():
        ids = sorted(v + pass2.get(k, []), key=lambda t: -len(g.id_frames(t)))
        groups: list[list[int]] = []
        for t in ids:
            cand = [gi for gi, grp in enumerate(groups) if not any(cannot_link(t, u) for u in grp)]
            if cand and t in id_xy:
                gi = min(cand, key=lambda gi: np.linalg.norm(np.median([id_xy[u] for u in groups[gi] if u in id_xy] or [id_xy[t]], 0) - id_xy[t]))
                groups[gi].append(t)
            elif cand:
                groups[cand[0]].append(t)
            else:
                groups.append([t])
        if len(groups) > 1:
            split_log.append(f"  cluster {k} split by same-frame constraint into {[sorted(gr) for gr in groups]}")
        for gi, grp in enumerate(groups):
            if len(grp) >= 2:
                kk = k if gi == 0 else nxt
                nxt += gi > 0
                clusters[kk] = sorted(grp)
                if gi > 0:
                    lab[np.isin(A[:, 1], grp) & (lab == k)] = kk
                    cl_az[kk], cl_h[kk] = cl_az[k], cl_h[k]
    if split_log:
        pass2_log += [""] + split_log
    report = [f"# static-worker world clustering: {seg.name}", "",
              f"lever {lever.tolist()} eps max({a.eps}, {a.eps_per_m}*r) m frac {a.frac} sim {a.sim}/{a.sim_id} az {a.az_deg} deg; "
              f"ref3d boxes {len(A)}, ids with ref3d {len(id_boxes)}, ids without ref3d {len(noref)}, clusters with >= 2 ids: {len(clusters)}", ""]
    if pass2_log:
        report += ["pass 2 (ids without ref3d):"] + pass2_log + [""]
    merges, refused = [], []
    for k, ids in sorted(clusters.items()):
        pts = A[lab == k]
        cen = np.median(pts[:, 2:4], 0)
        frames_of = {t: set(g.id_frames(t)) for t in ids}
        conflict, dups = [], []
        for i1, t1 in enumerate(ids):
            for t2 in ids[i1 + 1:]:
                both = sorted(frames_of[t1] & frames_of[t2])
                if not both:
                    continue
                ious = [iou(g.box_of(i, t1)["xyxy"], g.box_of(i, t2)["xyxy"]) for i in both]
                if all(same_person(g.box_of(i, t1), g.box_of(i, t2)) for i in both):
                    rmed = float(np.median(pts[:, 4]))
                    for i in both:
                        b1, b2 = g.box_of(i, t1), g.box_of(i, t2)
                        dev = [abs(((b.get("ref3d") or {}).get("range_m") or rmed) - rmed) for b in (b1, b2)]
                        worse = (t1, t2)[int(np.argmax(dev))] if abs(dev[0] - dev[1]) > 0.5 else \
                                (t2 if b1.get("conf", 0) >= b2.get("conf", 0) else t1)
                        dups.append((i, worse))
                else:
                    conflict.append((t1, t2, [round(v, 2) for v in ious]))
        E = {t: embed(t) for t in ids if len(embed(t))}
        allE = np.concatenate(list(E.values())) if E else np.zeros((0, 512))
        S = allE @ allE.T if len(allE) else np.zeros((0, 0))
        inside = float((S.sum() - np.trace(S)) / max(1, S.size - len(S))) if len(S) > 1 else 1.0
        sim_id = {}
        for t, e in E.items():
            others = np.concatenate([v for u, v in E.items() if u != t]) if len(E) > 1 else e
            sim_id[t] = float((e @ others.T).mean())
        main_id = max(ids, key=lambda t: len(g.id_frames(t)))
        bad_ids = [t for t, v in sim_id.items() if v < a.sim_id]
        ok = not conflict and inside >= a.sim and not bad_ids
        line = (f"## cluster {k}: world ({cen[0]:.1f}, {cen[1]:.1f}) m, range {np.median(pts[:, 4]):.1f} m, az {cl_az[k]:.1f} deg, "
                f"ref3d boxes {len(pts)}, ids {ids} -> {main_id}; appearance inside {inside:.2f}, per id "
                + ", ".join(f"{t}:{v:.2f}" for t, v in sim_id.items()))
        if dups:
            line += f"\n  duplicate boxes dropped (one person, two boxes): {sorted(set(dups))}"
        if conflict:
            line += f"\n  REFUSED: same-frame pairs with low IoU {conflict} (two people at this spot?)"
        if bad_ids:
            line += f"\n  REFUSED: appearance of ids {bad_ids} below {a.sim_id}"
        if ok:
            merges.append((k, main_id, [t for t in ids if t != main_id], dups))
            line += "\n  -> MERGE"
        else:
            refused.append(k)
        report.append(line)
        tiles = []
        for t in ids:
            for c in crops.get(t, [])[:4]:
                c = cv2.resize(c, (64, 128))
                cv2.putText(c, str(t), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                tiles.append(c)
        if tiles:
            while len(tiles) % 8:
                tiles.append(np.zeros((128, 64, 3), np.uint8))
            cv2.imwrite(str(seg / f"cluster_{k}.jpg"), np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)]))
            report.append(f"  montage: cluster_{k}.jpg")
        report.append("")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 9))
    noise = lab < 0
    ax.scatter(A[noise, 3], A[noise, 2], s=6, c="#bbbbbb", label="noise")
    for k in sorted(set(lab[lab >= 0])):
        p = A[lab == k]
        ax.scatter(p[:, 3], p[:, 2], s=8, label=f"c{k} ids {clusters.get(k, members.get(k, '-'))}"[:48])
        ax.annotate(str(k), (np.median(p[:, 3]), np.median(p[:, 2])), fontsize=9)
    ax.invert_xaxis()
    ax.set_xlabel("world y (left) [m]"), ax.set_ylabel("world x (fwd) [m]"), ax.set_aspect("equal")
    ax.set_title(f"{seg.name}: ref3d boxes in the world frame"), ax.grid(alpha=0.3)
    ax.legend(fontsize=6, loc="lower right")
    fig.savefig(seg / "cluster_bev.png", dpi=110, bbox_inches="tight")

    report.append(f"merges proposed: {len(merges)}, refused: {len(refused)}")
    (seg / "cluster_report.md").write_text("\n".join(report))
    print("\n".join(report))
    if a.apply and merges:
        for k, main_id, others, dups in merges:
            for i, t in dups:
                g.frames[i]["boxes"] = [b for b in g.frames[i]["boxes"] if b["id"] != t]
            for t in others:
                g.merge(t, main_id)
            g.ids[main_id]["motion"] = None if "motion" in g.ids.get(main_id, {}) else g.ids.get(main_id, {}).get("motion")
            g.note(g.id_frames(main_id)[0], main_id,
                   f"static cluster {k}: ids {others} merged into {main_id} by world position + OSNet appearance "
                   f"(dropped duplicate boxes {sorted(set(dups))}); see cluster_report.md",
                   kind="id", by="label_static_cluster", status="resolved")
        g.add_provenance("agent", "label_static_cluster", "scripts/label_static_cluster.py",
                         note=f"{len(merges)} static clusters merged (eps max({a.eps},{a.eps_per_m}r), frac {a.frac}, sim {a.sim}/{a.sim_id})")
        g.save()
        print("applied", len(merges), "merges")


if __name__ == "__main__":
    sys.exit(main())
