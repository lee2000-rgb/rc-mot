#!/usr/bin/env python3
"""ref3d from the sensor points INSIDE each image box, chained over frames (no teleports) - docs/13.

The first --write-ref pass picked a LiDAR range per box independently of the other frames and gave up when fewer
than three returns fell in the box (xpanner05: 52 % of the boxes without a reference), so the GT had holes and
per-frame picks on the background.  This tool does what a person does in the review GUI: look at the LiDAR and radar
points that project inside the box, find the one place in the top-down view that matches them, and keep the track
continuous.

  per box      candidates = BEV clusters (0.7 m link) of the cached LiDAR points above ground whose image projection
               falls inside the box (10 % x / 5 % y margin), plus the radar points inside it (clustered at 1 m);
               weight = log(1 + n_points) [radar: -1] - 0.5 (ln(range / range_from_box_height) / 0.35)^2
  per id       Viterbi over frames: state = a candidate or "no reference" (fixed penalty); the move between frames is
               scored in the pipeline WORLD frame (cache/ego_poses.npz, so the ego motion is taken out) with
               sigma = --max-step m per frame (0.3 m = 1.5 m/s) and a hard wall at 3 sigma (0.9 m/frame); a "no reference" frame remembers the last
               position so a track cannot teleport across a hole; boxes the user placed by hand (ref3d.src manual)
               are fixed states
  afterwards   holes of <= --max-gap frames between chosen candidates are interpolated in the world frame
               (src track_interp), longer holes stay without reference
  output       ref3d {range_m, xy_level (level frame of the frame), n_lidar, quality dense (>= 8 LiDAR pts) | sparse,
               src track_lidar | track_radar | track_interp | manual}

Dry run prints the per-id table and the comparison with the existing references (and with the manual ones);
--apply writes gt.json (backup), <seg>/ref_track.md, provenance and the command log.

  PYTHONPATH=src python3 scripts/label_ref_track.py data/labels/<seg> [--apply] [--ids 2,4] [--max-step 0.5]
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
from rcmot.utils.calib import RigCalib  # noqa: E402

from label_static_fix import Bev, ego_poses, rot  # noqa: E402

PERSON_H = 1.7
NONE_PEN = -2.5          # log-weight of "no reference" in a frame
SWITCH_PEN = -1.0        # entering / leaving "no reference"
MONO_SIG = 0.35          # log-range sigma of the box-height prior (crouching, children, hats...)


def cluster_xy(xy: np.ndarray, link: float):
    """Greedy single-linkage components in the plane (small point sets)."""
    n = len(xy)
    lab = -np.ones(n, int)
    k = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        lab[i] = k
        stack = [i]
        while stack:
            j = stack.pop()
            d = np.hypot(xy[:, 0] - xy[j, 0], xy[:, 1] - xy[j, 1])
            for q in np.flatnonzero((d < link) & (lab < 0)):
                lab[q] = k
                stack.append(q)
        k += 1
    return lab, k


class Projector:
    def __init__(self, cal: RigCalib, bev: Bev, fx: float):
        self.cal, self.bev, self.fx = cal, bev, fx

    def inbox(self, i, box):
        """(lidar xy (n,2), radar xy (m,2), radar v) of the cache points projecting inside the box."""
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        inside = lambda uv: (uv[:, 0] > x1 + 0.1 * w) & (uv[:, 0] < x2 - 0.1 * w) & (uv[:, 1] > y1 + 0.05 * h) & (uv[:, 1] < y2 - 0.05 * h)
        z = self.bev.z
        xy, hh = self.bev.above(i)
        L = np.zeros((0, 2))
        if len(xy):
            keep = hh >= 0.4
            xy, hh = xy[keep], hh[keep]
            if len(xy):
                uv, fr = self.cal.project(np.column_stack([xy, self.bev.zg(i) + hh]))
                ok = fr & np.isfinite(uv).all(1)
                ok[ok] &= inside(uv[ok])
                L = xy[ok]
        R, Rv = np.zeros((0, 2)), np.zeros(0)
        if "roff" in z.files:
            ra, rb = int(z["roff"][i]), int(z["roff"][i + 1])
            if rb > ra:
                rxy = z["rxy"][ra:rb].astype(float) / 100.0
                rz = z["rz"][ra:rb].astype(float) if "rz" in z.files else np.full(rb - ra, 1.0)
                uv, fr = self.cal.project(np.column_stack([rxy, rz]))
                ok = fr & np.isfinite(uv).all(1)
                ok[ok] &= inside(uv[ok])
                R, Rv = rxy[ok], z["rv"][ra:rb].astype(float)[ok]
        return L, R, Rv

    def candidates(self, i, box, cam, radar_w):
        """[{xy, n, n_r, w, src}] in the level frame of frame i."""
        L, R, _ = self.inbox(i, box)
        x1, y1, x2, y2 = box
        r_h = self.fx * PERSON_H / max(y2 - y1, 1.0)                 # box-height range prior
        cands = []
        if len(L):
            lab, k = cluster_xy(L, 0.7)
            for c in range(k):
                P = L[lab == c]
                cands.append({"xy": P.mean(0), "n": int(len(P)), "n_r": 0, "src": "track_lidar"})
        if len(R):
            lab, k = cluster_xy(R, 1.0)
            for c in range(k):
                P = R[lab == c]
                xy = P.mean(0)
                near = [q for q in cands if np.hypot(*(q["xy"] - xy)) < 1.0]
                if near:
                    near[0]["n_r"] += int(len(P))
                else:
                    cands.append({"xy": xy, "n": 0, "n_r": int(len(P)), "src": "track_radar"})
        for q in cands:
            r = float(np.hypot(*(q["xy"] - cam)))
            mono = -0.5 * (np.log(max(r, 0.5) / r_h) / MONO_SIG) ** 2
            base = np.log1p(q["n"]) + (np.log1p(q["n_r"]) + radar_w) * (1.0 if q["n"] == 0 else 0.5)
            q["w"] = float(base + mono)
            q["r"] = r
        return cands


def viterbi(frames_c: list, pos_w: list, fixed: list, sigma: float):
    """frames_c[t] = candidate list (dicts with 'w'), pos_w[t] = world xy per candidate (list of arrays),
    fixed[t] = index of the fixed (manual) candidate or None.  Returns chosen candidate index per frame (-1 = none)."""
    T = len(frames_c)
    # states per frame: ("c", j) or ("n", t0, j): none, last concrete position = candidate j of frame t0
    best = []           # list of dict state -> (score, prev_state)
    for t in range(T):
        cur = {}
        cands = frames_c[t]
        prev = best[t - 1] if t else {}

        def trans(ps, j_new):
            """score of moving from previous state ps to candidate j_new of frame t."""
            if ps[0] == "c":
                p0, t0 = pos_w[t - 1][ps[1]], t - 1
                pen = 0.0
            elif ps[1] < 0:                                           # never had a position yet
                return SWITCH_PEN
            else:
                p0, t0 = pos_w[ps[1]][ps[2]], ps[1]
                pen = SWITCH_PEN
            d = float(np.hypot(*(pos_w[t][j_new] - p0)))
            s = sigma * max(t - t0, 1)
            return (-0.5 * (d / s) ** 2 if d <= 3 * s else -50.0) + pen

        allowed = range(len(cands)) if fixed[t] is None else [fixed[t]]
        for j in allowed:
            w = cands[j]["w"] if fixed[t] is None else 20.0
            if not prev:
                cur[("c", j)] = (w, None)
                continue
            sc, arg = max(((v[0] + trans(ps, j), ps) for ps, v in prev.items()), key=lambda x: x[0])
            cur[("c", j)] = (sc + w, arg)
        if fixed[t] is None:
            # none states: carry every previous concrete / none position forward
            none_states = []
            for ps, v in prev.items():
                if ps[0] == "c":
                    none_states.append((("n", t - 1, ps[1]), v[0] + NONE_PEN + SWITCH_PEN, ps))
                else:
                    none_states.append((ps, v[0] + NONE_PEN, ps))
            if not prev:
                none_states.append((("n", -1, -1), NONE_PEN, None))
            none_states.sort(key=lambda x: -x[1])
            for st, sc, arg in none_states[:8]:
                if st not in cur or cur[st][0] < sc:
                    cur[st] = (sc, arg)
        best.append(cur)
    # backtrack
    out = [-1] * T
    st = max(best[-1].items(), key=lambda kv: kv[1][0])[0]
    for t in range(T - 1, -1, -1):
        out[t] = st[1] if st[0] == "c" else -1
        st = best[t][st][1]
        if st is None:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("--ids", default="")
    ap.add_argument("--max-step", type=float, default=0.3, help="sigma of the move per frame in the world frame [m]; wall at 3 sigma (0.9 m/frame = 4.5 m/s)")
    ap.add_argument("--max-gap", type=int, default=15, help="interpolate holes up to this many frames")
    ap.add_argument("--radar-w", type=float, default=-1.0, help="log-weight offset of a radar-only candidate")
    ap.add_argument("--range-prior", nargs="*", default=[], help="ID=R[:sigma_frac]: reviewer's range hint for an id (Gaussian on ln r, default sigma 0.2); "
                    "use when clutter in front (a berm, a pile top) inside the box outscores the far worker")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--by", default="label_ref_track")
    a = ap.parse_args()
    seg = Path(a.seg)
    seg = seg if seg.is_dir() else seg.parent
    g = GT(seg / "gt.json")
    cal = RigCalib.load(str(ROOT / g.meta["calib"]))
    bev = Bev(seg / "cache" / "bev_points.npz")
    if bev.z is None:
        sys.exit("no cache/bev_points.npz: run scripts/label_bev_cache.py first")
    yaw, org, valid = ego_poses(seg, g, cal)
    cam = cal.camera_center()[:2]
    proj = Projector(cal, bev, float(cal.K[0][0]))
    want = {int(v) for v in a.ids.split(",") if v}
    rprior = {}
    for it in a.range_prior:
        k, v = it.split("=")
        r0, _, sg = v.partition(":")
        rprior[int(k)] = (float(r0), float(sg) if sg else 0.2)
    report = [f"# ref3d by in-box points + continuity: {seg.name}", ""]
    tot = {"lidar": 0, "radar": 0, "interp": 0, "none": 0, "manual": 0}
    cmp_all, cmp_man = [], []
    for tid in sorted(g.ids):
        fr = g.id_frames(tid)
        if not fr or (want and tid not in want):
            continue
        boxes = [g.box_of(i, tid) for i in fr]
        frames_c, pos_w, fixed = [], [], []
        for i, b in zip(fr, boxes):
            r = b.get("ref3d") or {}
            if r.get("src") == "manual" and r.get("xy_level") is not None:
                c = [{"xy": np.asarray(r["xy_level"], float), "n": int(r.get("n_lidar", 0)), "n_r": 0, "w": 20.0, "src": "manual",
                      "r": float(np.hypot(*r["xy_level"]))}]
                fixed.append(0)
            else:
                c = proj.candidates(i, b["xyxy"], cam, a.radar_w)
                if tid in rprior:
                    r0, sg = rprior[tid]
                    for q in c:
                        q["w"] += -0.5 * (np.log(max(q["r"], 0.5) / r0) / sg) ** 2
                fixed.append(None)
            frames_c.append(c)
            pos_w.append([rot(yaw[i]) @ q["xy"] + org[i] for q in c])
        choice = viterbi(frames_c, pos_w, fixed, a.max_step)
        # interpolate holes in the world frame
        new = {}
        for k, (i, j) in enumerate(zip(fr, choice)):
            if j >= 0:
                q = frames_c[k][j]
                new[i] = (q["xy"], q["n"], q["src"], pos_w[k][j])
        known = sorted(new)
        for k0, k1 in zip(known, known[1:]):
            gap = k1 - k0 - 1
            if 0 < gap <= a.max_gap:
                p0, p1 = new[k0][3], new[k1][3]
                for i in range(k0 + 1, k1):
                    if g.box_of(i, tid) is None:
                        continue
                    t = (i - k0) / (k1 - k0)
                    pw = (1 - t) * p0 + t * p1
                    xy_i = rot(yaw[i]).T @ (pw - org[i])
                    bx = g.box_of(i, tid)["xyxy"]
                    uv, fr_ = cal.project(np.array([[xy_i[0], xy_i[1], bev.zg(i) + 1.0]]))
                    wbx = bx[2] - bx[0]
                    if fr_[0] and np.isfinite(uv[0, 0]) and bx[0] - 0.3 * wbx <= uv[0, 0] <= bx[2] + 0.3 * wbx:   # inside its own box only
                        new[i] = (xy_i, 0, "track_interp", pw)
        cnt = {"lidar": 0, "radar": 0, "interp": 0, "none": 0, "manual": 0}
        dr, dr_m, jumps = [], [], 0
        prev_r = None
        for i, b in zip(fr, boxes):
            old = b.get("ref3d") or {}
            if i in new:
                xy, n, src, _ = new[i]
                rng = float(np.hypot(*xy))
                cnt[{"track_lidar": "lidar", "track_radar": "radar", "track_interp": "interp", "manual": "manual"}[src]] += 1
                if old.get("range_m") is not None:
                    (dr_m if old.get("src") == "manual" else dr).append(rng - float(old["range_m"]))
                if prev_r is not None and abs(rng - prev_r) > 2.0:
                    jumps += 1
                prev_r = rng
                if a.apply and src != "manual":
                    g.set_ref3d(i, tid, {"range_m": rng, "xy_level": [float(xy[0]), float(xy[1])], "n_lidar": int(n),
                                         "quality": "dense" if n >= 8 else "sparse", "src": src})
            else:
                cnt["none"] += 1
                prev_r = None
                if a.apply and old.get("src") != "manual" and old.get("range_m") is not None:
                    g.set_ref3d(i, tid, {"range_m": None, "xy_level": None, "n_lidar": 0, "quality": "none", "src": "track_none"})
        for k in tot:
            tot[k] += cnt[k]
        cmp_all += dr
        cmp_man += dr_m
        dr = np.array(dr)
        line = (f"id {tid}: {len(fr)} boxes f{fr[0]}-{fr[-1]}: lidar {cnt['lidar']}, radar {cnt['radar']}, interp {cnt['interp']}, "
                f"manual {cnt['manual']}, none {cnt['none']}; range jumps > 2 m: {jumps}; vs old refs: n {len(dr)}"
                + (f", median |d| {np.median(np.abs(dr)):.2f} m, > 2 m {np.mean(np.abs(dr) > 2) * 100:.0f} %" if len(dr) else ""))
        print(line)
        report.append("- " + line)
        rs = " ".join(f"{i}:{np.hypot(*new[i][0]):.0f}{'' if new[i][2] == 'track_lidar' else new[i][2][6:7]}" if i in new else f"{i}:-" for i in fr)
        report.append(f"  `{rs}`")
    cov = 1 - tot["none"] / max(sum(tot.values()), 1)
    s = (f"total: lidar {tot['lidar']}, radar {tot['radar']}, interp {tot['interp']}, manual {tot['manual']}, none {tot['none']} "
         f"-> coverage {cov * 100:.1f} %; vs old refs median |d| {np.median(np.abs(cmp_all)) if cmp_all else 0:.2f} m, "
         f"vs manual refs median |d| {np.median(np.abs(cmp_man)) if cmp_man else 0:.2f} m (n {len(cmp_man)})")
    print(s)
    report += ["", s]
    if a.apply:
        g.add_provenance("ref3d", "agent", "scripts/label_ref_track.py",
                         note=f"in-box points + Viterbi continuity: {s}; max_step {a.max_step} max_gap {a.max_gap}")
        g.save(backup=True)
        (seg / "ref_track.md").write_text("\n".join(report) + "\n")
        with open(seg / "commands.log", "a") as fh:
            fh.write("PYTHONPATH=src python3 scripts/label_ref_track.py " + " ".join(sys.argv[1:]) + "\n")
        print(f"written {seg / 'gt.json'}")
    else:
        print("(dry run)")


if __name__ == "__main__":
    main()
