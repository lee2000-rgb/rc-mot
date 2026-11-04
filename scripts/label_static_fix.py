#!/usr/bin/env python3
"""Static objects (mannequins, standing workers) of a GT segment: find them, fix their distance, flag box jumps.

An object that does not move has ONE world position.  Its per-frame ref3d should therefore be that position seen
from the sensor at that moment, not the noisy sparse LiDAR return the --write-ref pass happened to pick, and a box of
the same id that lands somewhere else (on a person walking past, on the background) is an id error.  This script:

  1. ego pose per GT frame from the pipeline's world state (radar Doppler speed + gyro heading; label_auto_meta's
     ego_samples), cached in cache/ego_poses.npz;
  2. per id: world xy of every box's ref3d.xy_level, robust centre (median), fraction of referenced frames within the
     gate (0.5 m + 3 % of range) of the centre.  Static = the longest window of in-gate frames (blips <= 3 frames tolerated) holds >= --min-frac of the
     referenced frames and >= --min-frames; frames outside the window (walk-in / walk-out) are left alone (ignore ids are listed but not touched unless given with --ids);
  3. for every static id and every frame of its span:
       - expected level xy = world centre seen from that frame's pose; LiDAR points of cache/bev_points.npz
         (scripts/label_bev_cache.py) within --snap m of it -> centroid = ref3d (src static_lidar, quality dense when
         >= 10 points else sparse); no points -> the expected position itself (src static_world);
       - a box whose own ref3d is > gate from the expected position AND whose image box is far from the projected
         column (centre error > max(60 px, 1.0 x projected height)) is a JUMP: left untouched, Note (kind id) with
         --apply;
       - frames of the span without a box where the column projects inside the image and no LiDAR points stand
         in front of it (occlusion): listed as gaps (one Note);
     motion -> [[f0, f1, static]], desc prefixed "static object (auto)".
  Dry run prints the table; --apply writes gt.json (backup) + <seg>/static_fix.md + provenance.

  PYTHONPATH=src python3 scripts/label_static_fix.py data/labels/<seg> [--apply] [--ids 9,24] [--min-frac 0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rcmot.labeling.gt import GT  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

from label_auto_meta import ego_samples  # noqa: E402

PERSON_H = 1.7


def ego_poses(seg: Path, g: GT, cal: RigCalib) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(yaw [rad], origin xy [m], valid) of the level frame in the pipeline world at every GT frame (cached)."""
    m = g.meta
    cpath = seg / "cache" / "ego_poses.npz"
    key = json.dumps([m["bag"], m["start"], m["end"], len(g.frames), m.get("ego_state"), "hold-static"])
    if cpath.exists():
        z = np.load(cpath, allow_pickle=True)
        if str(z["key"]) == key:
            return z["yaw"], z["origin"], z["valid"]
    cfg = yaml.safe_load(open(ROOT / m["config"]))
    bag = Path(m["bag"]) if str(m["bag"]).startswith("/") else ROOT / m["bag"]
    samp, worlds = ego_samples(m, cfg, cal, bag)
    t = samp[:, 0]
    stamps = np.array([f["stamp"] for f in g.frames])
    yaw, org, valid = np.zeros(len(stamps)), np.zeros((len(stamps), 2)), np.zeros(len(stamps), bool)
    for k, s in enumerate(stamps):
        j = int(np.argmin(np.abs(t - s)))
        w = worlds[j] if abs(t[j] - s) <= 0.15 else None
        if w is not None:
            yaw[k], org[k], valid[k] = float(w.theta), np.asarray(w.origin, float)[:2], True
    yaw, org, valid = hold_static(g, yaw, org, valid)
    cpath.parent.mkdir(exist_ok=True)
    np.savez(cpath, key=key, yaw=yaw, origin=org, valid=valid)
    return yaw, org, valid


def hold_static(g: GT, yaw, org, valid):
    """ego_state static => the pose cannot change: every frame of a static run takes the run's first valid pose.
    (xpanner pile-driver rigs: the pipeline heading jitters 20-40 deg while the machine stands still.)"""
    yaw, org, valid = yaw.copy(), org.copy(), valid.copy()
    for f0, f1, st in (g.meta.get("ego_state") or []):
        if st != "static":
            continue
        ks = [k for k in range(f0, min(f1, len(yaw) - 1) + 1) if valid[k]]
        if not ks:
            continue
        y0, o0 = yaw[ks[0]], org[ks[0]].copy()
        for k in range(f0, min(f1, len(yaw) - 1) + 1):
            yaw[k], org[k], valid[k] = y0, o0, True
    return yaw, org, valid


def rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def to_world(k, p, yaw, org):
    return rot(yaw[k]) @ np.asarray(p, float) + org[k]


def to_level(k, w, yaw, org):
    return rot(yaw[k]).T @ (np.asarray(w, float) - org[k])


def gate_of(r):
    return 0.5 + 0.03 * r


class Bev:
    def __init__(self, path: Path):
        self.z = np.load(path) if path.exists() else None

    def above(self, i):
        z = self.z
        a, b, na = int(z["off"][i]), int(z["off"][i + 1]), int(z["n_above"][i])
        return z["xy"][a:b][:na].astype(float) / 100, z["h"][a:b][:na].astype(float) * 0.05

    def occluded(self, i, xy0, lateral=1.0, nmin=5):
        """LiDAR person-height points between the sensor and xy0 (same bearing, >= 1.5 m nearer): something stands in front."""
        if self.z is None:
            return False
        xy, h = self.above(i)
        r0, az0 = float(np.hypot(*xy0)), float(np.arctan2(xy0[1], xy0[0]))
        r, az = np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])
        daz = np.abs((az - az0 + np.pi) % (2 * np.pi) - np.pi)
        sel = (h >= 0.5) & (h <= 2.5) & (r < r0 - 1.5) & (r > 1.5) & (daz < lateral / np.maximum(r, 1.0))
        return int(sel.sum()) >= nmin

    def zg(self, i):
        return float(self.z["zg"][i]) if self.z is not None else 0.0

    def snap_inbox(self, i, xy0, r, cal, box, zg, hmin=0.5, hmax=2.3, nmin=2, cluster_r=0.7):
        """Like snap, but only LiDAR points whose image projection falls inside the box (10 % x / 5 % y margin)."""
        if self.z is None:
            return None, 0
        xy, h = self.above(i)
        ok = (h >= hmin) & (h <= hmax)
        if not ok.any():
            return None, 0
        uv, fr = cal.project(np.column_stack([xy[ok], zg + h[ok]]))
        x1, y1, x2, y2 = box
        w, hh = x2 - x1, y2 - y1
        ins = fr & np.isfinite(uv).all(1)
        ins[ins] &= (uv[ins, 0] > x1 + 0.1 * w) & (uv[ins, 0] < x2 - 0.1 * w) & (uv[ins, 1] > y1 + 0.05 * hh) & (uv[ins, 1] < y2 - 0.05 * hh)
        P = xy[ok][ins]
        if len(P) < nmin:
            return None, int(len(P))
        d = np.hypot(P[:, 0] - xy0[0], P[:, 1] - xy0[1])
        if d.min() > r:
            return None, int(len(P))
        seed = P[np.argmin(d)]
        near = np.hypot(P[:, 0] - seed[0], P[:, 1] - seed[1]) < cluster_r
        return P[near].mean(0), int(near.sum())

    def snap(self, i, xy0, r, hmin=0.5, hmax=2.3, nmin=3, cluster_r=0.7):
        """Centroid of the LiDAR person-height points around the point nearest to xy0 within r (seeded cluster, so a
        wide search radius does not pull the centroid towards a neighbour)."""
        if self.z is None:
            return None, 0
        xy, h = self.above(i)
        ok = (h >= hmin) & (h <= hmax)
        d = np.hypot(xy[:, 0] - xy0[0], xy[:, 1] - xy0[1])
        cand = ok & (d < r)
        if cand.sum() < nmin:
            return None, int(cand.sum())
        seed = xy[cand][np.argmin(d[cand])]
        near = ok & (np.hypot(xy[:, 0] - seed[0], xy[:, 1] - seed[1]) < cluster_r)
        if near.sum() < nmin:
            return None, int(near.sum())
        return xy[near].mean(0), int(near.sum())


def column_box(cal: RigCalib, xy, zg, h=PERSON_H, half=0.35):
    c = np.array([[xy[0] + dx, xy[1] + dy, z] for dx in (-half, half) for dy in (-half, half) for z in (zg + 0.05, zg + h)])
    uv, fr = cal.project(c)
    if not fr.all():
        return None
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("seg")
    ap.add_argument("--ids", default="", help="treat these ids as static whatever the statistics say (comma list)")
    ap.add_argument("--min-frac", type=float, default=0.6, help="referenced frames inside the static window / all")
    ap.add_argument("--min-frames", type=int, default=10)
    ap.add_argument("--snap", type=float, default=0.8, help="LiDAR points within this radius of the expected position [m]")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--by", default="label_static_fix")
    a = ap.parse_args()
    seg = Path(a.seg)
    seg = seg if seg.is_dir() else seg.parent
    g = GT(seg / "gt.json")
    cal = RigCalib.load(str(ROOT / g.meta["calib"]))
    W, H = cal.image_size
    tic = time.time()
    yaw, org, valid = ego_poses(seg, g, cal)
    bev = Bev(seg / "cache" / "bev_points.npz")
    print(f"{seg.name}: ego poses valid {valid.sum()}/{len(valid)}, BEV cache {'yes' if bev.z is not None else 'NO'} ({time.time() - tic:.0f} s)")
    forced = {int(v) for v in a.ids.split(",") if v}

    # ---- per-id world statistics
    rows, static_ids = [], []
    for tid in sorted(g.ids):
        fr = g.id_frames(tid)
        if not fr:
            continue
        pts = []
        for i in fr:
            b = g.box_of(i, tid)
            r = b.get("ref3d") or {}
            if r.get("xy_level") is None or not valid[i]:
                continue
            pts.append((i, to_world(i, r["xy_level"], yaw, org), float(np.hypot(*r["xy_level"]))))
        n_ign = sum("ignore" in g.box_of(i, tid)["flags"] for i in fr)
        if len(pts) < 3:
            rows.append((tid, len(fr), n_ign, len(pts), 0.0, None, None, "no ref"))
            continue
        Wp = np.array([p[1] for p in pts])
        centre = np.median(Wp, 0)
        d = np.hypot(*(Wp - centre).T)
        rng = np.array([p[2] for p in pts])
        inl = d <= gate_of(rng)
        if inl.sum() >= 3:                                            # refine the centre on the inliers
            centre = Wp[inl].mean(0)
            d = np.hypot(*(Wp - centre).T)
            inl = d <= gate_of(rng)
        # static window: longest run of in-gate referenced frames, out-of-gate blips of <= 3 frames inside it tolerated
        # (a sparse return on the background, a box on a passer-by); a walk-in / walk-out stays outside the window
        fi = np.array([p[0] for p in pts])
        lab = inl.copy()
        k = 0
        while k < len(lab):
            if not lab[k]:
                j = k
                while j < len(lab) and not lab[j]:
                    j += 1
                if 0 < k and j < len(lab) and j - k <= 3:
                    lab[k:j] = True
                k = j
            else:
                k += 1
        best, cur = (0, 0, 0), None
        for k, v in enumerate(lab):
            if v and cur is None:
                cur = k
            if (not v or k == len(lab) - 1) and cur is not None:
                end = k if v else k - 1
                if end - cur + 1 > best[0]:
                    best = (end - cur + 1, cur, end)
                cur = None
        n_win, w0, w1 = best[0], (int(fi[best[1]]) if best[0] else None), (int(fi[best[2]]) if best[0] else None)
        frac = n_win / len(pts)
        is_static = tid in forced or (n_win >= a.min_frames and frac >= a.min_frac and n_ign < len(fr) / 2
                                      and g.ids[tid].get("cls", "worker") == "worker")      # equipment refs: not a column
        verdict = "STATIC" if is_static else ("ignore" if n_ign >= len(fr) / 2 else "moving")
        if is_static:
            if w0 is None:
                w0, w1 = fr[0], fr[-1]
            if tid in forced and n_win < a.min_frames:
                w0, w1 = fr[0], fr[-1]
            c_win = Wp[(fi >= w0) & (fi <= w1) & inl]
            if len(c_win):
                centre = c_win.mean(0)
            static_ids.append((tid, centre, float(np.median(rng)), w0, w1))
        rows.append((tid, len(fr), n_ign, len(pts), frac, centre, float(np.median(rng)), verdict + (f" f{w0}-{w1}" if is_static else "")))
    print(f"\n{'id':>4} {'n':>4} {'ign':>4} {'ref':>4} {'in-win':>8} {'range':>6}  verdict           desc")
    for tid, n, ni, nr, frac, c, rng, v in rows:
        print(f"{tid:4d} {n:4d} {ni:4d} {nr:4d} {frac:8.2f} {rng if rng is not None else -1:6.1f}  {v:16s}  {g.ids[tid].get('desc', '')[:48]}")

    # ---- fix the static ids
    report = [f"# static objects: {seg.name}", "", f"ego poses valid {valid.sum()}/{len(valid)}; gate 0.5 m + 3 % range; snap {a.snap} m", ""]
    n_fix = n_snap = 0
    jumps, gaps = {}, {}
    for tid, centre, rng_med, w0, w1 in static_ids:
        fr = g.id_frames(tid)
        have = set(fr)
        jl, gl, nsn, nfx, nkeep, nout = [], [], 0, 0, 0, 0
        for i in range(w0, w1 + 1):
            if not valid[i]:
                continue
            exp = to_level(i, centre, yaw, org)
            r_exp = float(np.hypot(*exp))
            zg = bev.zg(i)
            pb = column_box(cal, exp, zg)
            if i not in have:
                if pb is not None and pb[0] > 10 and pb[2] < W - 10 and pb[3] < H and pb[1] > 0 and not bev.occluded(i, exp):
                    gl.append(i)
                continue
            b = g.box_of(i, tid)
            r = b.get("ref3d") or {}
            turning = g.ego_state_at(i) in ("swing", "drive+swing")      # gyro-heading pose: a few deg off -> metres at 25 m
            # the static position must lie inside THIS box's viewing wedge, else the object has left (walk-out) -> leave the frame
            x1, y1, x2, y2 = b["xyxy"]
            uv_e, fr_e = cal.project(np.array([[exp[0], exp[1], zg + 1.0]]))
            if not fr_e[0] or not np.isfinite(uv_e[0, 0]) or uv_e[0, 0] < x1 - 0.3 * (x2 - x1) or uv_e[0, 0] > x2 + 0.3 * (x2 - x1):
                nout += 1
                continue
            ref_ok = False
            if r.get("xy_level") is not None:
                d = float(np.hypot(*(np.asarray(r["xy_level"]) - exp)))
                dr = abs(float(np.hypot(*r["xy_level"])) - r_exp)
                ref_ok = dr <= gate_of(r_exp) and (turning or d <= gate_of(r_exp))
                if (dr > gate_of(r_exp) or (not turning and d > gate_of(r_exp))) and pb is not None:
                    x1, y1, x2, y2 = b["xyxy"]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    px, py, ph = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2, pb[3] - pb[1]
                    if np.hypot(cx - px, cy - py) > (max(120.0, 2 * ph) if turning else max(60.0, ph)):   # LiDAR lags the image while turning
                        jl.append((i, round(d, 1), round(float(np.hypot(*r["xy_level"])), 1), round(r_exp, 1)))
                        continue
            s, npts = bev.snap_inbox(i, exp, a.snap + (0.06 * r_exp if turning else 0.0), cal, b["xyxy"], zg)
            if s is not None and abs(float(np.hypot(*s)) - r_exp) <= gate_of(r_exp):
                ref = {"range_m": float(np.hypot(*s)), "xy_level": [float(s[0]), float(s[1])], "n_lidar": npts,
                       "quality": "dense" if npts >= 10 else "sparse", "src": "static_lidar"}
                nsn += 1
            elif ref_ok:
                nkeep += 1
                continue                                                  # the existing reference is consistent: keep it
            else:
                ref = {"range_m": r_exp, "xy_level": [float(exp[0]), float(exp[1])], "n_lidar": 0, "quality": "sparse", "src": "static_world"}
            if a.apply:
                g.set_ref3d(i, tid, ref)
            nfx += 1
        n_fix += nfx
        n_snap += nsn
        jumps[tid], gaps[tid] = jl, gl
        line = (f"id {tid}: {len(fr)} boxes f{fr[0]}-{fr[-1]}, static window f{w0}-{w1}, world ({centre[0]:.1f}, {centre[1]:.1f}) range ~{rng_med:.1f} m: "
                f"ref3d fixed {nfx} (LiDAR snap {nsn}, pose only {nfx - nsn}), kept {nkeep}, outside own box {nout}, jumps {len(jl)}, in-view gaps {len(gl)}")
        print(line)
        report.append("- " + line)
        if jl:
            report.append(f"  jumps (frame, off by m, box range, expected): {jl[:40]}")
            print("   jumps:", jl[:20], "..." if len(jl) > 20 else "")
        if gl:
            report.append(f"  gaps: {gl[:60]}")
            print("   gaps:", gl[:30], "..." if len(gl) > 30 else "")
        if a.apply:
            v = g.ids[tid]
            keep = []
            for s0, s1, st in (v.get("motion") or []):                 # existing motion outside the window survives
                if s1 < w0 or s0 > w1:
                    keep.append([s0, s1, st])
                else:
                    if s0 < w0:
                        keep.append([s0, w0 - 1, st])
                    if s1 > w1:
                        keep.append([w1 + 1, s1, st])
            v["motion"] = sorted(keep + [[w0, w1, "static"]])
            if not str(v.get("desc", "")).startswith("static object"):
                v["desc"] = f"static object (auto, LiDAR world {rng_med:.0f} m, f{w0}-{w1})" + (" | " + v["desc"] if v.get("desc") else "")
            if jl:
                g.note(jl[0][0], tid, f"static object id {tid}: {len(jl)} box(es) at f{[j[0] for j in jl[:12]]} sit {jl[0][1]}+ m off its "
                       f"LiDAR position (box range {jl[0][2]} m vs {jl[0][3]} m) - on another person / background? re-id or delete",
                       kind="id", by=a.by, status="open")
            if gl:
                g.note(gl[0], tid, f"static object id {tid}: no box in {len(gl)} in-view frame(s) {gl[:12]}{'...' if len(gl) > 12 else ''} - "
                       f"occluded (ok) or missed?", kind="box", by=a.by, status="open")
    if not static_ids:
        print("no static id")
    if a.apply:
        g.add_provenance("audit", "agent", "scripts/label_static_fix.py",
                         note=f"static ids {[t for t, *_ in static_ids]}: ref3d fixed {n_fix} ({n_snap} LiDAR snaps), motion static; jumps/gaps noted")
        g.save(backup=True)
        (seg / "static_fix.md").write_text("\n".join(report) + "\n")
        with open(seg / "commands.log", "a") as fh:
            fh.write("PYTHONPATH=src python3 scripts/label_static_fix.py " + " ".join(sys.argv[1:]) + "\n")
        print(f"written {seg / 'gt.json'}; report {seg / 'static_fix.md'}")
    else:
        print("(dry run: --apply to write)")


if __name__ == "__main__":
    main()
