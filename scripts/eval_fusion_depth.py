#!/usr/bin/env python3
"""Per-box 3D range of the radar-camera fusion vs a LiDAR reference, on GT boxes (docs/14).

For every non-ignored GT box of a labelled segment (data/labels/<seg>/gt.json) the fusion is run on the GT
boxes of that frame (so detector misses do not interfere). The LiDAR reference range is the LiDAR mode inside
the central part of the box that is closest to the monocular / ground-ray range (the object, not the
occluder in front or the background behind). Each box is classified:

  ok            |fused - lidar| <= tol
  wrong_pick    a radar point near the LiDAR range was inside the box, but another range was chosen
  no_point      no radar point near the LiDAR range inside the box (radar missed the object or projection)
  no_ref        LiDAR reference ambiguous / missing (excluded)

and the in-box radar points are split into target / foreground (closer) / background (farther), with their
standardised features (robust z of RCS and SNR, |z_v|, normalised distance to the box centre) saved for
designing the point selection.

  python3 scripts/eval_fusion_depth.py data/labels/day_halt_short_085 [--set fusion.x=y] [--tag name]

--depth-map (docs/27): the metric depth map of rcmot.fusion.depth_map (DA-V2 small + static radar points,
disparity-domain RANSAC + (s,t) KF) is run on every GT frame and each box also gets its depth-map range
(bottom-third median, and the instance-mask median when --seg-weights gives a mask) with sigma, inlier count and
(s,t); --dm-ablate adds the N=1 / no-KF variants from the same disparity map. The reference and the fusion
ranges are untouched.

--write-ref (docs/13 v2, gt_design §D.2 step 3): the LiDAR reference of EVERY GT box (ignore boxes included) is
written into gt.json as box["ref3d"] = {range_m, xy_level, n_lidar, quality, src} with quality "dense" (central
rule, >= 8 points), "sparse" (--ref-sparse fallback, >= 3 points) or "none"; xy_level = camera centre + range along
the box-centre pixel ray azimuth (base_link_level). meta.ref3d records the rule / coverage and a provenance entry
{stage: ref3d, who: eval_fusion_depth} is appended. The depth statistics above are printed as usual.
  --ref-ego-comp  de-skew the LiDAR scan for the ego rotation (per-point IMU yaw x (t_point - t_frame), LidarStream.at):
                  on swing segments the nearest scan is up to +-60 ms off and a 100 ms sweep smears 5.6 deg at 56 deg/s,
                  so the central box window catches background returns (conti_swing_065: 3 of 8 worker jumps > 2 m).
--cues (docs/37): every box record also carries "cue" (fuse(cue_log=...): box geometry, height-prior / ground-contact /
width terms, RangeReliability state, track priors, candidate radar points and scored clusters, chosen radar range,
output range and sigma), "trk" (snapshot of the track whose prior the box got, before this frame's update; GT-prior
protocol), "fr" (frame: base velocity, yaw rate, scan dt, scan sigma_v, world pose theta / origin) and "ref3d" (gt.json).
  --ref-smooth    temporal consistency per ID (smooth_refs): a box whose range is > 2 m from both adjacent-frame
                  neighbours of the same ID becomes quality none, the others take the median over +-1 frame
                  (dense equipment boxes entering the view pick another density peak of a 5 m machine; 3-4 point
                  sparse picks). The summary reports the adjacent-frame jump fraction (ref3d.jump_frac).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_velocity_lidar import LidarStream  # noqa: E402
from rcmot.fusion.depth_map import DepthMapParams, MetricDepth  # noqa: E402
from rcmot.fusion.radar_camera import FusionParams, fuse, mono_range  # noqa: E402
from rcmot.labeling.provenance import add_provenance, load_doc, now_str, save_doc, git_sha  # noqa: E402
from rcmot.pipeline import Pipeline  # noqa: E402
from rcmot.sensors.bag_source import iterate_bag  # noqa: E402
from rcmot.tracking.tracker import iou_xyxy  # noqa: E402
from rcmot.types import Det2D  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


class _NoDet:
    def __call__(self, img):
        return []


class _GtDet:
    """Detector stand-in returning the GT boxes of the current frame (set by the loop): the tracker then runs on
    GT boxes and fuse() receives the online track priors (--track-prior gt, docs/30). GT frames are every 3rd-4th
    scan; on the scans in between the boxes are interpolated per ID between the neighbouring GT frames (held
    for <= hold_s when the ID is in one of them only), so tracks confirm and coast as with a full-rate detector."""

    def __init__(self, gt_frames: list, hold_s: float = 0.25):
        self.dets: list = []
        self.frames = [(f["stamp"], {b["id"]: (b["cls"], np.array(b["xyxy"], float)) for b in f["boxes"] if "ignore" not in b["flags"]})
                       for f in gt_frames]
        self.st = np.array([f[0] for f in self.frames])
        self.hold = hold_s

    def at(self, t: float) -> list:
        k1 = int(np.searchsorted(self.st, t))
        k0 = k1 - 1
        a = self.frames[k0] if k0 >= 0 else None
        b = self.frames[k1] if k1 < len(self.frames) else None
        out = []
        if a is not None and b is not None and b[0] - a[0] <= 1.0:
            w = (t - a[0]) / max(b[0] - a[0], 1e-6)
            for i, (cls, box) in a[1].items():
                if i in b[1]:
                    out.append(Det2D(cls, 1.0, (1 - w) * box + w * b[1][i][1]))
                elif t - a[0] <= self.hold:
                    out.append(Det2D(cls, 1.0, box.copy()))
            for i, (cls, box) in b[1].items():
                if i not in a[1] and b[0] - t <= self.hold:
                    out.append(Det2D(cls, 1.0, box.copy()))
            return out
        near = min((x for x in (a, b) if x is not None), key=lambda x: abs(x[0] - t), default=None)
        if near is not None and abs(near[0] - t) <= self.hold:
            out = [Det2D(cls, 1.0, box.copy()) for cls, box in near[1].values()]
        return out

    def __call__(self, img):
        return list(self.dets)


def set_cfg(cfg, kv):
    k, v = kv.split("=", 1)
    d = cfg
    for part in k.split(".")[:-1]:
        d = d.setdefault(part, {})
    d[k.split(".")[-1]] = yaml.safe_load(v)


def _mono_any(box, H, cal, fp):
    """Height-prior range without the posture / border checks (diagnostics)."""
    import dataclasses
    q = dataclasses.replace(fp, crouch_aspect=99.0, border_px=-1)
    m = mono_range(box, H, cal, q)
    return None if m is None else round(m[0], 2)


def lidar_ref_full(lv, uv, front, rng, box, prior, bw=0.3, sparse=False):
    """LiDAR range mode inside the central box part closest to the prior range.
    Returns (range or None, n points used, quality) with quality "dense" (central part, >= 8 points), "sparse"
    (sparse mode: central part with 3-7 points, or the whole-box fallback with >= 3 points) or "none".
    sparse: for thin clouds (xpanner Ouster: 3-7 returns on a worker at 20 m) accept >= 3 points, falling back
    from the central part to the whole box; such references are coarser (docs/27 §2)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    m = front & (uv[:, 0] > x1 + 0.3 * w) & (uv[:, 0] < x2 - 0.3 * w) & (uv[:, 1] > y1 + 0.15 * h) & (uv[:, 1] < y2 - 0.1 * h)
    r = rng[m]
    quality = "dense" if len(r) >= 8 else "sparse"
    if sparse and len(r) < 3:
        m = front & (uv[:, 0] > x1 + 0.1 * w) & (uv[:, 0] < x2 - 0.1 * w) & (uv[:, 1] > y1 + 0.05 * h) & (uv[:, 1] < y2 - 0.05 * h)
        r = rng[m]
    if len(r) < (3 if sparse else 8):
        return None, int(len(r)), "none"
    grid = np.arange(r.min() - 1, r.max() + 1, 0.1)
    dens = np.exp(-0.5 * ((grid[:, None] - r[None]) / bw) ** 2).sum(1)
    peaks = [i for i in range(1, len(grid) - 1) if dens[i] >= dens[i - 1] and dens[i] >= dens[i + 1] and dens[i] >= 0.15 * dens.max()]
    if not peaks:
        return None, int(len(r)), "none"
    if prior is None:
        i = max(peaks, key=lambda i: dens[i])
    else:
        i = min(peaks, key=lambda i: abs(grid[i] - prior))
    return float(grid[i]), int(len(r)), quality


def lidar_ref(lv, uv, front, rng, box, prior, bw=0.3, sparse=False):
    """Range only (None if too few points); see lidar_ref_full."""
    return lidar_ref_full(lv, uv, front, rng, box, prior, bw, sparse)[0]


def box_prior(cal, fp, box, cls):
    """Monocular (height prior) range of the box, else the ground-ray range of its bottom centre, else None."""
    H = fp.class_height.get(cls)
    mono = mono_range(box, H, cal, fp) if H else None
    g = cal.pixel_ray_to_ground(0.5 * (box[0] + box[2]), box[3])
    C = cal.camera_center()
    return mono, g, (mono[0] if mono else (float(np.linalg.norm(g[:2] - C[:2])) if g is not None else None))


def ref3d_record(cal, rng, n, quality, box) -> dict:
    """Schema-v2 ref3d of a box: horizontal range from the camera centre and the level-frame xy along the azimuth of
    the box-centre pixel ray (docs/13 v2 contract: keys range_m, xy_level, n_lidar, quality, src)."""
    if rng is None or quality == "none":
        return {"range_m": None, "xy_level": None, "n_lidar": int(n), "quality": "none", "src": "lidar"}
    C = cal.camera_center()
    d = cal.pixel_ray(0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3]))
    az = float(np.arctan2(d[1], d[0]))
    xy = C[:2] + rng * np.array([np.cos(az), np.sin(az)])
    return {"range_m": round(float(rng), 2), "xy_level": [round(float(xy[0]), 2), round(float(xy[1]), 2)],
            "n_lidar": int(n), "quality": quality, "src": "lidar"}


def smooth_refs(frames: list, thr: float = 2.0, cal=None) -> dict:
    """Temporal consistency of box["ref3d"] per ID over adjacent GT frames (ignore boxes included): a range that is
    more than thr from both neighbours (for an end box: from its only neighbour and that one's next, when those two
    agree) becomes quality none; the rest is replaced by the median over +-1 frame (xy_level moved along the same
    azimuth when the range changes). Returns {"outliers": n, "smoothed": n}."""
    series: dict = {}
    for k, f in enumerate(frames):
        for b in f["boxes"]:
            if b.get("ref3d", {}).get("range_m") is not None:
                series.setdefault(b["id"], {})[k] = b
    n_out = n_sm = 0
    for tid, boxes in series.items():
        ks = sorted(boxes)
        rng = {k: boxes[k]["ref3d"]["range_m"] for k in ks}
        new = {}
        for k in ks:                                       # pass 1: outliers (judged on the raw neighbours)
            r = rng[k]
            prev, nxt = rng.get(k - 1), rng.get(k + 1)
            nb = [v for v in (prev, nxt) if v is not None]
            if len(nb) == 2 and all(abs(r - v) > thr for v in nb):
                new[k] = None
            elif len(nb) == 1:
                far = rng.get(k + 2) if prev is None else rng.get(k - 2)
                if far is not None and abs(r - nb[0]) > thr and abs(r - far) > thr and abs(nb[0] - far) <= thr:
                    new[k] = None
        for k in ks:                                       # pass 2: median over the non-outlier +-1 neighbours
            if k in new:
                continue
            nb = [rng[j] for j in (k - 1, k + 1) if j in rng and new.get(j, rng[j]) is not None]
            new[k] = float(np.median([nb[0], rng[k], nb[1]])) if len(nb) == 2 else rng[k]
        for k, v in new.items():
            b = boxes[k]
            r3 = b["ref3d"]
            if v is None:
                n_out += 1
                r3["range_m"], r3["xy_level"], r3["quality"] = None, None, "none"
            elif abs(v - r3["range_m"]) > 0.005:
                n_sm += 1
                if r3.get("xy_level") is not None and cal is not None:
                    r3.update(ref3d_record(cal, v, r3["n_lidar"], r3["quality"], np.array(b["xyxy"], float)))
                else:
                    r3["range_m"] = round(float(v), 2)
    return {"outliers": n_out, "smoothed": n_sm}


def jump_stats(frames: list, thr: float = 2.0) -> dict:
    """Adjacent-frame same-ID pairs with both references whose range differs by more than thr (all boxes, and the
    non-ignore boxes only)."""
    out = {}
    for name, keep_ign in (("all", True), ("regular", False)):
        rs: dict = {}
        for k, f in enumerate(frames):
            for b in f["boxes"]:
                if (keep_ign or "ignore" not in b["flags"]) and b.get("ref3d", {}).get("range_m") is not None:
                    rs[(k, b["id"])] = b["ref3d"]["range_m"]
        n = j = 0
        for (k, i), r in rs.items():
            r1 = rs.get((k + 1, i))
            if r1 is not None:
                n += 1
                j += abs(r1 - r) > thr
        out[name] = {"pairs": n, "jumps": j, "jump_frac": round(j / n, 3) if n else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump", help="write per-box / per-point records (json) here")
    ap.add_argument("--seg-weights", help="instance-segmentation weights (e.g. models/yolo11s-seg.pt): masks for the GT boxes")
    ap.add_argument("--seg-imgsz", type=int, default=1280, help="input size of the --seg-weights model (online configs: 960)")
    ap.add_argument("--depth-map", action="store_true", help="also the metric depth-map range per box (docs/27)")
    ap.add_argument("--dm-size", default="518x924", help="DA-V2 input HxW")
    ap.add_argument("--dm-frames", type=int, default=3, help="static-point accumulation N")
    ap.add_argument("--dm-encoder", default="vits")
    ap.add_argument("--dm-ablate", action="store_true", help="also N=1 and no-KF variants (same disparity map)")
    ap.add_argument("--ref-sparse", action="store_true", help="LiDAR reference from >= 3 returns (thin clouds, xpanner)")
    ap.add_argument("--write-ref", action="store_true", help="write the reference of every GT box into gt.json as ref3d (+ meta.ref3d, provenance)")
    ap.add_argument("--stamp-tol", type=float, default=0.02, help="max |pipeline - GT stamp| for a frame match [s]")
    ap.add_argument("--ref-ego-comp", action="store_true", help="de-skew the LiDAR scan for the IMU yaw rate (swing segments)")
    ap.add_argument("--ref-smooth", action="store_true", help="per-ID temporal consistency of the written ref3d (+-1 frame median, > 2 m outliers -> none)")
    ap.add_argument("--cues", action="store_true", help="docs/37: per-box range-cue records (fuse cue log, prior track, ego)")
    ap.add_argument("--track-prior", choices=["none", "gt"], default="none",
                    help="gt: the pipeline's tracker runs on the GT boxes and fuse() gets its track priors (online path); none: fuse() alone")
    ap.add_argument("--variant", action="append", default=[],
                    help='"NAME key=val key=val ...": evaluate several configurations in one pass over the bag (own Pipeline '
                         'per variant, shared masks / LiDAR references; --dump / --tag may contain {variant}); the token '
                         'prior=gt|none sets --track-prior for that variant')
    a = ap.parse_args()
    gt_path = Path(a.seg) / "gt.json"
    doc = load_doc(gt_path)
    m, gt_frames = doc["meta"], doc["frames"]
    cfg = yaml.safe_load(open(ROOT / m["config"]))
    for kv in a.set:
        set_cfg(cfg, kv)
    cfg.setdefault("radar_type", m.get("radar", ""))     # fusion.radar_lag_s key (bosch / conti), impl2
    calib_json = json.load(open(ROOT / m["calib"]))
    cal = RigCalib.load(str(ROOT / m["calib"]))
    fp = FusionParams(**cfg.get("fusion", {}))
    cfg.setdefault("recovery", {})["enable"] = False
    multi = bool(a.variant)
    if multi and (a.write_ref or a.depth_map):
        ap.error("--variant cannot be combined with --write-ref / --depth-map")
    V = []                                                 # per configuration: own Pipeline (+ tracker on GT boxes)
    for spec in (a.variant or [""]):
        name, *sets = spec.split() or [""]          # no --variant: one unnamed configuration
        tp = a.track_prior
        for kv in [kv for kv in sets if kv.startswith("prior=")]:
            tp = kv.split("=", 1)[1]
            sets.remove(kv)
        cfg_v = copy.deepcopy(cfg)
        for kv in sets:
            set_cfg(cfg_v, kv)
        gtdet_v = _GtDet(gt_frames) if tp == "gt" else None
        V.append(SimpleNamespace(name=name or "", sets=sets, gtdet=gtdet_v, pipe=Pipeline(gtdet_v or _NoDet(), cal, cfg_v),
                                 rows=[], pts=[], fuse_ms=[], stage_ms=[], track_prior=tp, last_stamp=None))
        V[-1].pipe.cue_log_on = bool(a.cues)
    lid = LidarStream(ROOT / m["bag"], calib_json, radar_topic=cfg["topics"]["radar"])
    gt_st = np.array([f["stamp"] for f in gt_frames])
    # sensor-clock radars (Continental): the stamp -> bag-time mapping of iterate_bag is one fixed offset per bag
    # (or meta.radar_clock_offset_s); still match on the radar message's bag receive time when the GT has it
    gt_bt = np.array([f.get("bag_t", np.nan) for f in gt_frames])
    use_bt = bool(len(gt_bt)) and not np.isnan(gt_bt).any()
    seen: set[int] = set()
    if a.write_ref:                                        # every box gets the contract keys, filled in below
        for f in gt_frames:
            for b in f["boxes"]:
                b["ref3d"] = {"range_m": None, "xy_level": None, "n_lidar": 0, "quality": "none", "src": "lidar"}
    seg = None
    if a.seg_weights:
        from rcmot.detection.yolo2d import Yolo2D
        seg = Yolo2D(str(ROOT / a.seg_weights), cfg["detector"]["class_map"], imgsz=a.seg_imgsz, conf=0.1)
    C = cal.camera_center()
    dm, dm_var = None, {}
    if a.depth_map:
        hw = tuple(int(v) for v in a.dm_size.lower().split("x"))
        dm = MetricDepth(cal, DepthMapParams(encoder=a.dm_encoder, input_hw=hw, n_frames=a.dm_frames), root=ROOT)
        if a.dm_ablate:
            dm_var = {"n3raw": MetricDepth(cal, DepthMapParams(encoder=a.dm_encoder, input_hw=hw, n_frames=a.dm_frames, kalman=False), root=ROOT),
                      "n1kf": MetricDepth(cal, DepthMapParams(encoder=a.dm_encoder, input_hw=hw, n_frames=1), root=ROOT),
                      "n1raw": MetricDepth(cal, DepthMapParams(encoder=a.dm_encoder, input_hw=hw, n_frames=1, kalman=False), root=ROOT)}
    dm_frames = []
    ego_of = {k: st for f0, f1, st in (m.get("ego_state") or []) for k in range(int(f0), int(f1) + 1)}   # GT frame -> ego state
    for img, rf0, yaw in iterate_bag(ROOT / m["bag"], cal, cfg["topics"]["radar"], cfg["topics"]["image"],
                                     cfg["topics"].get("imu"), start=max(m["start"] - 1.0, 0), duration=m["end"] - m["start"] + 1.5,
                                     radar_clock_offset_s=m.get("radar_clock_offset_s")):
        if use_bt and getattr(rf0, "bag_t", None) is not None:  # same radar message: exact bag receive time
            k = int(np.argmin(np.abs(gt_bt - rf0.bag_t)))
            f = gt_frames[k] if abs(gt_bt[k] - rf0.bag_t) <= 2e-3 and k not in seen else None
        else:
            k = int(np.argmin(np.abs(gt_st - rf0.stamp)))  # nearest GT frame (sensor-clock radars: a few ms off)
            f = gt_frames[k] if abs(gt_st[k] - rf0.stamp) <= a.stamp_tol and k not in seen else None
        if f is not None:
            seen.add(k)
        boxes = [b for b in f["boxes"] if "ignore" not in b["flags"]] if f is not None else []
        dets = [Det2D(b["cls"], 1.0, np.array(b["xyxy"], float)) for b in boxes]
        if seg is not None and dets:                       # mask of the best-overlapping instance
            sd = [x for x in seg(img) if x.mask is not None]
            for d in dets:
                cand = [(iou_xyxy(d.box[None], x.box[None])[0, 0], x) for x in sd if x.cls == d.cls]
                best = max(cand, key=lambda c: c[0], default=(0, None))
                if best[0] >= 0.5:
                    x = best[1]
                    full = np.zeros(img.shape[:2], bool)
                    x1, y1 = max(int(x.box[0]), 0), max(int(x.box[1]), 0)
                    full[y1:y1 + x.mask.shape[0], x1:x1 + x.mask.shape[1]] = x.mask
                    d.mask = full[max(int(d.box[1]), 0):int(np.ceil(d.box[3])), max(int(d.box[0]), 0):int(np.ceil(d.box[2]))]
        ref_frame = None                                   # (lidar scan ok, refs) of this GT frame, shared by the variants
        for v in V:
            rf = replace(rf0) if multi else rf0            # step() fills stats / v_r_comp: one copy per variant
            pipe, gtdet = v.pipe, v.gtdet
            if gtdet is not None:
                gtdet.dets = dets if f is not None else gtdet.at(rf.stamp)
            res = pipe.step(img, rf, yaw)                  # fills rf.stats / v_r_comp (and the world pose)
            scan_dt = None if v.last_stamp is None else round(float(rf.stamp - v.last_stamp), 4)
            v.last_stamp = rf.stamp
            if "depth_map" in res.timing_ms:               # docs/33 wave 3 online depth stage (every N-th frame a map)
                v.stage_ms.append((res.timing_ms["depth_map"], res.timing_ms.get("depth_map_new", 0.0)))
            if gtdet is not None and dets:
                v.fuse_ms.append(res.timing_ms.get("fusion", np.nan))   # fuse() + the track-prior lookup
            stat_xyz = None
            if dm is not None:
                stat = rf.stats.static_inlier & rf.stats.keep
                if rf.stats.below_ground is not None:
                    stat &= ~rf.stats.below_ground
                stat_xyz = rf.xyz[stat]
                if f is None:                              # every scan feeds the N-frame point history, inference on GT frames
                    dm.push_static(stat_xyz, res.world)
                    for dv in dm_var.values():
                        dv.push_static(stat_xyz, res.world)
            if f is None:
                continue
            boxes = [b for b in f["boxes"] if "ignore" not in b["flags"]]
            if not f["boxes"]:
                continue
            if ref_frame is None:
                lv, dts = lid.at(rf.stamp, yaw if a.ref_ego_comp else None)
                refs = {}                                  # box id -> (range, n, quality) of every box of the frame
                if lv is not None and abs(dts) <= 0.06:
                    luv, lfront = cal.project(lv)
                    lrng = np.linalg.norm(lv[:, :2] - C[:2], axis=1)
                    for b in f["boxes"]:
                        bx = np.array(b["xyxy"], float)
                        r_full = lidar_ref_full(lv, luv, lfront, lrng, bx, box_prior(cal, fp, bx, b["cls"])[2], sparse=a.ref_sparse)
                        refs[b["id"]] = r_full
                        if a.write_ref:
                            b["ref3d"] = ref3d_record(cal, r_full[0], r_full[1], r_full[2], bx)
                ref_frame = (lv is not None and abs(dts) <= 0.06, refs)
            if not ref_frame[0]:
                continue
            refs = ref_frame[1]
            if not boxes:
                continue
            if gtdet is not None:
                fused = pipe.last_dets3d                   # fused inside the step, with the tracker's priors
            else:
                t0 = time.perf_counter()
                fused = pipe.fuse_boxes(dets, rf, yaw)     # no track priors; box-time rotation of the moving ego (impl2)
                v.fuse_ms.append((time.perf_counter() - t0) * 1e3)
            by_box = {tuple(np.round(d.box2d, 1)): d for d in fused}
            cues, ptrk, fr = None, None, None
            if a.cues:                                     # docs/37: aligned with dets (GT-prior: the detector's list)
                cues = list(pipe.last_cues)
                ptrk = list(pipe.last_prior_tracks) if gtdet is not None else None
                w_ = res.world
                fr = {"v_lvl": [round(float(x), 3) for x in pipe.last_v_lvl[:2]], "yaw_rate": round(float(yaw), 4),
                      "dt": scan_dt, "sigma_v": None if rf.stats is None else round(float(rf.stats.sigma_v), 4),
                      "n_scan": int(len(rf.v_r)),
                      "world": None if w_ is None else [round(float(w_.theta), 5), round(float(w_.origin[0]), 3),
                                                        round(float(w_.origin[1]), 3)]}
            dm_res, dm_info = None, {}
            if dm is not None:
                dm_res = dm.update(img, stat_xyz, res.world, stamp=rf.stamp)
                for dv in dm_var.values():
                    dv.update(img, stat_xyz, res.world, stamp=rf.stamp, disparity=dm_res.disparity)
                ft = dm_res.fit
                dm_info = {"dmap_n": None if ft is None else ft.n, "dmap_n_in": None if ft is None else ft.n_in,
                           "dmap_rel": None if ft is None else round(ft.rel_err, 4), "dmap_s": dm_res.s, "dmap_t": dm_res.t,
                           "dmap_nis": None if dm_res.nis is None else round(dm_res.nis, 2), "dmap_upd": dm_res.updated,
                           "dmap_n_static": dm_res.n_static}
                dm_frames.append({"stamp": f["stamp"], **dm_info, "ms": {kk: round(vv, 1) for kk, vv in dm_res.ms.items()}})
            st = rf.stats
            uv, front = cal.project(rf.xyz)
            rr = np.linalg.norm(rf.xyz[:, :2] - C[:2], axis=1)
            for kb, (b, d2) in enumerate(zip(boxes, dets)):
                H = fp.class_height.get(b["cls"])
                mono, g, prior = box_prior(cal, fp, d2.box, b["cls"])
                ref = refs[b["id"]][0]
                d3 = by_box.get(tuple(np.round(d2.box, 1)))
                x1, y1, x2, y2 = d2.box
                inb = np.flatnonzero(front & (uv[:, 0] >= x1) & (uv[:, 0] <= x2) & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))
                rec = {"stamp": f["stamp"], "id": b["id"], "cls": b["cls"], "flags": b["flags"], "ref": ref,
                       "ego": ego_of.get(k, "static"), "yaw_dps": round(float(np.degrees(yaw)), 1), "img_dt": round(float(rf.img_dt), 3),
                       "ego_static": bool(pipe.last_ego_static),
                       "mono": None if mono is None else round(mono[0], 2), "n_inbox": int(len(inb)),
                       "mono_any": None if not H else _mono_any(d2.box, H, cal, fp),
                       "ground": None if g is None else round(float(np.linalg.norm(g[:2] - C[:2])), 2),
                       "aspect": round(float((d2.box[2] - d2.box[0]) / max(d2.box[3] - d2.box[1], 1)), 2),
                       "bottom_edge": bool(d2.box[3] >= cal.image_size[1] - 4)}
                if a.cues:
                    cq = cues[kb] if cues is not None and kb < len(cues) else None
                    if cq is not None and cq["box"] != [round(float(c), 1) for c in d2.box]:
                        cq = None                          # misaligned (should not happen): drop rather than mislabel
                    rec.update(cue=cq, fr=fr, ref3d=b.get("ref3d"),
                               trk=ptrk[kb] if ptrk is not None and kb < len(ptrk) else None)
                if d3 is None:
                    rec.update(fused=None, src="none", n_radar=0)
                else:
                    rec.update(fused=round(float(np.linalg.norm(d3.pos[:2] - C[:2])), 2), src=d3.source, n_radar=int(d3.n_radar),
                               pos=[round(float(vv), 2) for vv in d3.pos[:2]], dbg=d3.dbg)
                if dm is not None:
                    rec.update(dm_info, has_mask=d2.mask is not None)
                    for name, obj in [("dmap", dm)] + list(dm_var.items()):
                        r_b = obj.range_for_box(d2.box)
                        r_m = obj.range_for_box(d2.box, d2.mask, "mask") if d2.mask is not None else None
                        rec[name] = None if r_b is None else round(r_b[0], 2)
                        rec[name + "_mask"] = None if r_m is None or r_m[2]["how"] != "mask" else round(r_m[0], 2)
                        if name == "dmap":
                            rec["dmap_sigma"] = None if r_b is None else round(r_b[1], 2)
                            rec["dmap_z"] = None if r_b is None else round(r_b[2]["z"], 2)
                            rec["dmap_spread"] = None if r_b is None else round(r_b[2]["spread"], 2)
                            rec["dmap_mask_sigma"] = None if rec["dmap_mask"] is None else round(r_m[1], 2)
                if ref is None:
                    rec["cat"] = "no_ref"
                else:
                    near = np.abs(rr[inb] - ref) <= a.tol if len(inb) else np.zeros(0, bool)
                    if rec["fused"] is not None and abs(rec["fused"] - ref) <= a.tol:
                        rec["cat"] = "ok"
                    elif near.any():
                        rec["cat"] = "wrong_pick"
                    else:
                        rec["cat"] = "no_point"
                    cx, cy, w, h = 0.5 * (x1 + x2), 0.5 * (y1 + y2), x2 - x1, y2 - y1
                    for i in inb:
                        dr = rr[i] - ref
                        v.pts.append({"cls": b["cls"], "role": "target" if abs(dr) <= a.tol else ("fg" if dr < 0 else "bg"),
                                      "dr": round(float(dr), 2), "du": round(float((uv[i, 0] - cx) / w), 3),
                                      "dv": round(float((uv[i, 1] - cy) / h), 3),
                                      "z_rcs": None if st is None else round(float(st.z_rcs[i]), 2) if hasattr(st, "z_rcs") else None,
                                      "z_snr": None if st is None or not hasattr(st, "z_snr") else round(float(st.z_snr[i]), 2),
                                      "contrast": None if st is None or not hasattr(st, "contrast") else round(float(st.contrast[i]), 2),
                                      "z_v": None if st is None or getattr(st, "z_v", None) is None else round(float(st.z_v[i]), 2),
                                      "dyn": None if st is None or getattr(st, "dynamic", None) is None else bool(st.dynamic[i]),
                                      "used": bool(d3 is not None and d3.radar_idx is not None and i in set(d3.radar_idx.tolist()))})
                v.rows.append(rec)
    for v in V:
        rows, pts, fuse_ms, stage_ms = v.rows, v.pts, v.fuse_ms, v.stage_ms
        ev = [r for r in rows if r["cat"] != "no_ref"]
        summ = {"seg": Path(a.seg).name, "tag": a.tag.format(variant=v.name), "set": a.set + v.sets, "ref_sparse": a.ref_sparse,
                "boxes": len(rows), "with_ref": len(ev), "track_prior": v.track_prior}
        if multi:
            summ["variant"] = v.name
        if fuse_ms:
            fm = np.asarray(fuse_ms, float)
            fm = fm[np.isfinite(fm)]
            summ["fuse_ms"] = {"frames": int(len(fm)), "median": round(float(np.median(fm)), 3), "p90": round(float(np.percentile(fm, 90)), 3),
                               "max": round(float(fm.max()), 3)}
        if stage_ms:
            sm = np.asarray(stage_ms, float)
            new = sm[:, 1] > 0
            q = lambda x: {"n": int(len(x)), "median": round(float(np.median(x)), 2), "p90": round(float(np.percentile(x, 90)), 2)} if len(x) else None
            summ["depth_stage_ms"] = {"all": q(sm[:, 0]), "new_map": q(sm[new, 0]), "reuse": q(sm[~new, 0]),
                                      "mean_per_frame": round(float(sm[:, 0].mean()), 2), "maps": int(new.sum())}
        for cls in sorted({r["cls"] for r in ev}):
            rs = [r for r in ev if r["cls"] == cls]
            err = np.array([abs(r["fused"] - r["ref"]) for r in rs if r["fused"] is not None])
            summ[cls] = {"n": len(rs), "cats": dict(Counter(r["cat"] for r in rs)),
                         "by_src": {s: dict(Counter(r["cat"] for r in rs if r["src"] == s)) for s in sorted({r["src"] for r in rs})},
                         "abs_err_median": round(float(np.median(err)), 2) if len(err) else None,
                         "abs_err_p90": round(float(np.percentile(err, 90)), 2) if len(err) else None,
                         "inbox_point_roles": dict(Counter(p["role"] for p in pts if p["cls"] == cls))}
            if dm is not None:
                for key in ("dmap", "dmap_mask"):
                    e = np.array([r[key] - r["ref"] for r in rs if r.get(key) is not None])
                    summ[cls][key] = {"n": int(len(e)), "avail": round(len(e) / max(len(rs), 1), 3),
                                      "abs_err_median": round(float(np.median(np.abs(e))), 2) if len(e) else None,
                                      "abs_err_p90": round(float(np.percentile(np.abs(e), 90)), 2) if len(e) else None,
                                      "within_1m": round(float(np.mean(np.abs(e) <= 1.0)), 3) if len(e) else None,
                                      "bias": round(float(np.mean(e)), 2) if len(e) else None}
        if dm is not None:
            summ["depth_map"] = {"size": a.dm_size, "n_frames": a.dm_frames, "frames": len(dm_frames),
                                 "n_in_median": float(np.median([d["dmap_n_in"] for d in dm_frames if d["dmap_n_in"]])) if dm_frames else None,
                                 "nis_gated": int(sum(1 for d in dm_frames if d["dmap_nis"] is not None and not d["dmap_upd"])),
                                 "net_ms_median": float(np.median([d["ms"].get("net", np.nan) for d in dm_frames[1:]])) if len(dm_frames) > 1 else None}
        if a.write_ref:
            jumps_raw = jump_stats(gt_frames)
            sm = smooth_refs(gt_frames, cal=cal) if a.ref_smooth else None
            q = Counter(b["ref3d"]["quality"] for f in gt_frames for b in f["boxes"] if "ignore" not in b["flags"])
            n_reg = sum(q.values())
            cov = round((q["dense"] + q["sparse"]) / n_reg, 3) if n_reg else 0.0
            opts = "--write-ref" + (" --ref-sparse" if a.ref_sparse else "") + (" --ref-ego-comp" if a.ref_ego_comp else "") \
                + (" --ref-smooth" if a.ref_smooth else "")
            m["ref3d"] = {"rule": "docs14_s3", "sparse": bool(a.ref_sparse), "tool": "scripts/eval_fusion_depth.py",
                          "tol_m": float(a.tol), "coverage": cov, "git": git_sha(ROOT), "time": now_str()}
            jumps = jump_stats(gt_frames)
            add_provenance(doc, "ref3d", "eval_fusion_depth", "scripts/eval_fusion_depth.py", ROOT,
                           note=f"{opts}; frames matched {len(seen)}/{len(gt_frames)}; jump_frac(all) {jumps['all']['jump_frac']}"
                                + (f"; smooth outliers {sm['outliers']} smoothed {sm['smoothed']}" if sm else ""))
            save_doc(doc, gt_path)
            summ["ref3d"] = {"coverage": cov, "n_boxes": n_reg, "quality": dict(q), "frames_matched": len(seen),
                             "frames": len(gt_frames), "options": opts, "jumps": jumps, "jumps_before_smooth": jumps_raw, "smooth": sm}
            print(f"ref3d written to {gt_path} ({opts}): coverage {cov} over {n_reg} regular boxes {dict(q)}; "
                  f"frames matched {len(seen)}/{len(gt_frames)}; jumps {json.dumps(jumps)}"
                  + (f"; before smoothing {json.dumps(jumps_raw)} smooth {sm}" if sm else ""), file=sys.stderr)
        print(json.dumps(summ, indent=1))
        if a.dump:
            json.dump({"summary": summ, "boxes": rows, "points": pts, "depth_frames": dm_frames}, open(a.dump.format(variant=v.name), "w"))

if __name__ == "__main__":
    main()
