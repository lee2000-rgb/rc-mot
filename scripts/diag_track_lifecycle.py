#!/usr/bin/env python3
"""Track lifecycle diagnostics without ground truth (docs/12).

Runs the pipeline over a bag window and logs, per new ID, why it was born next to an existing track
(the existing track's gate / reachability / Doppler / class), per merge which ID survived, and per frame
the displayed boxes that have no measurement ("ghosts") split into duplicate-ghosts (next to a track that
was measured) and isolated ghosts. Writes a JSON summary and a crop montage of every new ID born near
an existing one.

  python3 scripts/diag_track_lifecycle.py --bag <bag> --calib <json> --start 85 --duration 100 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.pipeline import build_pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.tracking import tracker as T  # noqa: E402
from rcmot.types import TrackState  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

LOG = {"births": [], "merges": [], "deaths": []}
NEAR_M = 3.0                                                   # --near
CTX = {"trk": None, "frame": 0, "stamp": 0.0, "img": None}


def why_not(t, d, p) -> str:
    """Which gate kept detection d away from track t."""
    if t.cls != d.cls:
        return "class"
    m = t.kf.mahalanobis_pos(d.pos, t.pos_R(d, p))
    if m > p.gate_pos:
        return f"pos_gate(m={m:.1f})"
    if not t.reachable(d, p):
        return "reach"
    if t.hits >= 2 and d.vel is None and d.vr is not None:
        dd = t.kf.radial_d2(d.vr_u, d.vr, d.vr_var + t.doppler_var(p))
        if dd > p.gate_dop:
            return f"doppler(d2={dd:.1f})"
    return "lost_in_assignment" if t.tsu > 0 else "already_matched"


_init = T.Track.__init__


def _track_init(self, det, p, t, tid=None):
    _init(self, det, p, t, tid)
    trk = CTX["trk"]
    if trk is None or tid is not None:
        if tid is not None:
            LOG["births"].append({"frame": CTX["frame"], "t": CTX["stamp"], "tid": self.tid, "relinked": True})
        return
    near = []
    for o in trk.tracks:
        if o is self or o.state == TrackState.TENTATIVE:
            continue
        dist = float(np.linalg.norm(o.kf.pos[:2] - det.pos[:2]))
        if dist < NEAR_M:
            near.append({"tid": o.tid, "state": o.state.name, "dist": round(dist, 2), "tsu": round(o.tsu, 2),
                         "range_new": round(float(np.linalg.norm(det.pos[:2])), 2),
                         "range_old": round(float(np.linalg.norm(o.kf.pos[:2])), 2),
                         "why": why_not(o, det, p), "src_new": det.source, "src_old": o.last_source})
    near.sort(key=lambda x: x["dist"])
    grave = []
    for td, o in trk.graveyard:
        if o.cls != det.cls:
            continue
        cos = None if o.emb is None or getattr(det, "emb", None) is None else round(1.0 - float(o.emb @ det.emb), 3)
        grave.append({"tid": o.tid, "dead_s": round(trk.t - td, 2), "dist": round(float(np.linalg.norm(o.kf.pos[:2] - det.pos[:2])), 2),
                      "cam_dist": round(float(np.linalg.norm(o.cam_pos[:2] - det.pos[:2])), 2) if hasattr(o, "cam_pos") else None,
                      "cos": cos})
    grave.sort(key=lambda x: x["cam_dist"] if x["cam_dist"] is not None else x["dist"])
    LOG["births"].append({"frame": CTX["frame"], "t": CTX["stamp"], "tid": self.tid, "cls": det.cls,
                          "score": round(det.score, 2), "src": det.source, "near": near[:2], "grave": grave[:3], "relinked": False,
                          "box": det.box2d.tolist()})


T.Track.__init__ = _track_init
_merge = T.RadarCameraTracker._merge_duplicates


def _merge_dup(self, tracks):
    before = {t.tid: t for t in tracks}
    out = _merge(self, tracks)
    kept = {t.tid for t in out}
    for tid in set(before) - kept:
        a = before[tid]
        partner = min((t for t in out if t.cls == a.cls), key=lambda t: np.linalg.norm(t.kf.pos[:2] - a.kf.pos[:2]),
                      default=None)
        LOG["merges"].append({"frame": CTX["frame"], "dropped": tid, "dropped_age_hits": a.hits,
                              "kept": partner.tid if partner else None, "kept_hits": partner.hits if partner else None,
                              "kept_is_newer": bool(partner and partner.tid > tid)})
    return out


T.RadarCameraTracker._merge_duplicates = _merge_dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/hdx_boryeong.yaml"))
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float)
    ap.add_argument("--out", required=True)
    ap.add_argument("--near", type=float, default=3.0, help="radius [m] for 'born next to an existing track'")
    a = ap.parse_args()
    global NEAR_M
    NEAR_M = a.near
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    cal = RigCalib.load(a.calib)
    pipe = build_pipeline(cfg, cal, ROOT)
    CTX["trk"] = pipe.tracker
    tp = cfg["topics"]
    shown = Counter()
    ghosts = Counter()
    life = defaultdict(lambda: {"first": None, "last": None, "frames": 0, "meas": 0, "cls": None})
    crops = {}
    n = 0
    for k, (img, rf, yaw) in enumerate(iterate_bag(a.bag, cal, tp["radar"], tp["image"], tp.get("imu"),
                                                   start=a.start, duration=a.duration)):
        CTX["frame"], CTX["stamp"] = k, rf.stamp
        nb = len(LOG["births"])
        res = pipe.step(img, rf, yaw)
        n += 1
        for b in LOG["births"][nb:]:
            if b.get("near") and "box" in b:
                x1, y1, x2, y2 = np.array(b["box"]).astype(int)
                crops[b["tid"]] = img[max(y1, 0):y2, max(x1, 0):x2].copy()
        measured = [t for t in res.tracks if t.time_since_update == 0]
        for t in res.tracks:
            L = life[t.tid]
            L["first"] = L["first"] if L["first"] is not None else k
            L["last"], L["cls"] = k, t.cls
            L["frames"] += 1
            L["meas"] += int(t.time_since_update == 0 and t.source != "radar_only")
            shown[t.state.name] += 1
            if t.time_since_update > 0 or t.source == "radar_only":
                dup = any(m.tid != t.tid and np.linalg.norm(m.pos[:2] - t.pos[:2]) < 2.0 and m.source != "radar_only"
                          for m in measured)
                ghosts["dup_ghost" if dup else "isolated_ghost"] += 1
                ghosts[f"{'radar_only' if t.source == 'radar_only' else 'coasting'}"] += 1
    births = [b for b in LOG["births"] if not b["relinked"]]
    near = [b for b in births if b["near"]]
    summ = {
        "frames": n, "ids_shown": len(life), "relinked": sum(b["relinked"] for b in LOG["births"]),
        "births_new_id": len(births), "births_within_3m_of_track": len(near),
        "why_not_matched": Counter(b["near"][0]["why"].split("(")[0] for b in near),
        "old_track_state": Counter(b["near"][0]["state"] for b in near),
        "src_pairs": Counter(f'{b["src"]}<-{b["near"][0]["src_old"]}' for b in near),
        "merges": len(LOG["merges"]), "merges_kept_newer_id": sum(m["kept_is_newer"] for m in LOG["merges"]),
        "shown_boxes_by_state": dict(shown), "ghost_boxes": dict(ghosts),
        "id_lifetimes_s": sorted([round((L["last"] - L["first"] + 1) / 15, 1) for L in life.values()]),
        "id_spans": {tid: [L["first"], L["last"], L["cls"]] for tid, L in sorted(life.items())},
        "ids_with_<20pct_measured": [tid for tid, L in life.items() if L["meas"] < 0.2 * L["frames"]],
    }
    json.dump({"summary": summ, **LOG}, open(out / "lifecycle.json", "w"), indent=1, default=str)
    print(json.dumps(summ, indent=1, default=str))
    if crops:                                                       # montage: new IDs born near existing tracks
        tiles = []
        for tid, c in list(crops.items())[:40]:
            if not c.size:
                continue
            c = cv2.resize(c, (64, 128))
            cv2.putText(c, str(tid), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            tiles.append(c)
        rows = [np.hstack(tiles[i:i + 10] + [np.zeros_like(tiles[0])] * (10 - len(tiles[i:i + 10])))
                for i in range(0, len(tiles), 10)]
        cv2.imwrite(str(out / "births_near_montage.jpg"), np.vstack(rows))


if __name__ == "__main__":
    main()
