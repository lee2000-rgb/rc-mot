#!/usr/bin/env python3
"""GT audit, step 2: compare gt.json with the detector outputs of label_audit_detectors.py (docs/13 §audit).

  python3 scripts/label_audit_compare.py [--labels data/labels] [--segs seg ...] [--models yolo11x rtdetrx gdino owlv2]

Per segment (worker boxes; equipment is not audited):
  MISSED  a consensus detection (>= 2 models agree at IoU >= 0.5, each at conf >= --miss-conf 0.35, height >= 25 px)
          with IoU < 0.3 to every GT box of the frame, ignored ones and equipment included.
          Tagged in_vehicle when it lies inside a vehicle detection / equipment GT (operator in a cab is not GT).
  FALSE   a GT worker box that is not ignored and has IoU < 0.3 to every detection (any class) of every model at
          conf >= --conf 0.1.
  LOOSE   a GT worker box (not ignored) whose best consensus detection (conf >= --loose-conf 0.5, 2 models) has
          IoU 0.3..0.6 and prefers this GT box: the geometry is probably off, the consensus box is proposed.
Consecutive frames of the same finding are grouped into runs (MISSED: boxes linked at IoU >= 0.3 frame to frame;
FALSE / LOOSE: same id). Output per segment, nothing in gt.json is modified:
  data/labels/<seg>/audit/audit.json   runs with frames, boxes, confidences, proposed boxes
  data/labels/<seg>/audit/report.md    reviewer list
  data/labels/<seg>/audit/missed.jpg, false.jpg   crop montages (<= 48 crops, frame number on each)
  data/labels/audit_summary.md         all segments, ranked examples
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SKIP_SEGS = ("night_drivemid_front_000",)
PERSON = {"person", "worker", "mannequin", "construction worker"}
VEHICLE = {"car", "truck", "bus", "train", "boat", "airplane"}
OPEN_VOCAB = ("gdino", "owlv2")          # every label is a person prompt


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / np.maximum(ar(a)[:, None] + ar(b)[None] - inter, 1e-9)


def inside(a, b) -> float:
    """fraction of box a inside box b"""
    lt, rb = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.clip(rb - lt, 0, None).prod())
    return inter / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def is_ignored(b: dict) -> bool:
    return "ignore" in b.get("flags", []) or bool(b.get("ignore_reason"))


def load_dets(seg: Path, models: list[str]) -> dict[str, list[list]]:
    out = {}
    for m in models:
        p = seg / "audit" / f"dets_{m}.json"
        if p.exists():
            out[m] = [f["dets"] for f in json.load(open(p))["frames"]]
    return out


def person_dets(model: str, dets: list) -> list:
    return dets if model in OPEN_VOCAB else [d for d in dets if d[5] in PERSON]


def consensus(frame_dets: dict[str, list], conf: float, min_h: float = 0.0) -> list[dict]:
    """Cluster person detections of all models (IoU >= 0.5 to the cluster seed); keep clusters with >= 2 models.
    Returns [{xyxy, conf, models: {m: conf}}] sorted by conf, xyxy = box of the most confident member."""
    cand = []
    for m, dets in frame_dets.items():
        for d in person_dets(m, dets):
            if d[4] >= conf and d[3] - d[1] >= min_h:
                cand.append((d[4], m, np.array(d[:4], float)))
    cand.sort(key=lambda c: -c[0])
    used, out = [False] * len(cand), []
    for i, (c, m, box) in enumerate(cand):
        if used[i]:
            continue
        members = {m: c}
        used[i] = True
        for j in range(i + 1, len(cand)):
            if not used[j] and cand[j][1] not in members and iou(box, cand[j][2])[0, 0] >= 0.5:
                members[cand[j][1]] = cand[j][0]
                used[j] = True
        if len(members) >= 2:
            out.append({"xyxy": [round(float(v), 1) for v in box], "conf": round(float(c), 3), "models": members})
    return out


def vehicles(frame_dets: dict[str, list], conf: float = 0.3) -> list[np.ndarray]:
    return [np.array(d[:4], float) for m, dets in frame_dets.items() if m not in OPEN_VOCAB
            for d in dets if d[5] in VEHICLE and d[4] >= conf]


def analyse(seg: Path, models: list[str], conf: float, miss_conf: float, loose_conf: float, min_h: float) -> dict:
    gt = json.load(open(seg / "gt.json"))
    dets = load_dets(seg, models)
    N = len(gt["frames"])
    for m, d in dets.items():
        assert len(d) == N, f"{seg.name} {m}: {len(d)} det frames vs {N} GT frames"
    missed_raw, false_raw, loose_raw = [], [], []
    for i, fr in enumerate(gt["frames"]):
        fd = {m: d[i] for m, d in dets.items()}
        boxes = fr["boxes"]
        gboxes = np.array([b["xyxy"] for b in boxes], float).reshape(-1, 4)
        workers = [k for k, b in enumerate(boxes) if b["cls"] == "worker" and not is_ignored(b)]
        equip = [np.array(b["xyxy"], float) for b in boxes if b["cls"] != "worker"]
        cons = consensus(fd, miss_conf, min_h)
        veh = vehicles(fd) + equip
        # MISSED
        for c in cons:
            ious = iou(c["xyxy"], gboxes)
            if ious.size and ious.max() >= 0.3:
                continue
            best = float(ious.max()) if ious.size else 0.0
            tags = ["in_vehicle"] if any(inside(np.array(c["xyxy"]), v) >= 0.8 for v in veh) else []
            if ious.size:                       # a GT box close by (centre within 1 box height): geometry off, not a miss
                cx, cy, h = (c["xyxy"][0] + c["xyxy"][2]) / 2, (c["xyxy"][1] + c["xyxy"][3]) / 2, c["xyxy"][3] - c["xyxy"][1]
                gc = np.c_[(gboxes[:, 0] + gboxes[:, 2]) / 2, (gboxes[:, 1] + gboxes[:, 3]) / 2]
                dist = np.hypot(gc[:, 0] - cx, gc[:, 1] - cy)
                k = int(dist.argmin())
                if dist[k] < h and 0.5 < (gboxes[k, 3] - gboxes[k, 1]) / h < 2.0:
                    tags.append(f"near_gt:{boxes[k]['id']}")
            missed_raw.append({"frame": i, **c, "h": round(c["xyxy"][3] - c["xyxy"][1], 1), "best_gt_iou": round(best, 2),
                               "tags": tags})
        # FALSE
        alld = np.array([d[:4] for m, ds in fd.items() for d in ds if d[4] >= conf], float).reshape(-1, 4)
        for k in workers:
            b = boxes[k]
            ious = iou(b["xyxy"], alld)
            best = float(ious.max()) if ious.size else 0.0
            if best < 0.3:
                others = [o for o in boxes if o is not b and not is_ignored(o)]
                cover = max((inside(np.array(b["xyxy"], float), np.array(o["xyxy"], float)) for o in others), default=0.0)
                tags = ["hidden"] if cover >= 0.5 else []           # >= 50 % inside another GT box: occlusion interpolation
                false_raw.append({"frame": i, "id": b["id"], "xyxy": b["xyxy"], "h": round(b["xyxy"][3] - b["xyxy"][1], 1),
                                  "best_det_iou": round(best, 2), "src": b.get("src"), "gt_conf": b.get("conf"),
                                  "occ": b.get("occ"), "flags": b.get("flags", []), "cover": round(cover, 2), "tags": tags})
        # LOOSE
        strong = [c for c in consensus(fd, loose_conf)]
        if strong and workers:
            sb = np.array([c["xyxy"] for c in strong])
            wb = np.array([boxes[k]["xyxy"] for k in workers], float)
            M = iou(wb, sb)                                   # workers x strong
            for r, k in enumerate(workers):
                j = int(M[r].argmax())
                v = M[r, j]
                if 0.3 <= v < 0.6 and M[:, j].argmax() == r:
                    b = boxes[k]
                    loose_raw.append({"frame": i, "id": b["id"], "xyxy": b["xyxy"], "iou": round(float(v), 2),
                                      "proposed": strong[j]["xyxy"], "conf": strong[j]["conf"],
                                      "models": strong[j]["models"], "occ": b.get("occ"), "flags": b.get("flags", [])})
    return {"segment": seg.name, "n_frames": N, "models": sorted(dets), "worker_only": gt["meta"].get("worker_only"),
            "missed": group_missed(missed_raw), "false": group_by_id(false_raw), "loose": group_by_id(loose_raw)}


def group_missed(items: list[dict]) -> list[dict]:
    runs: list[dict] = []
    open_runs: list[dict] = []
    for it in items:
        hit = None
        for r in open_runs:
            if it["frame"] - r["f1"] <= 1 and it["frame"] != r["f1"] and iou(it["xyxy"], r["last"])[0, 0] >= 0.3:
                hit = r
                break
        if hit is None:
            hit = {"f0": it["frame"], "f1": it["frame"], "last": it["xyxy"], "items": []}
            open_runs.append(hit)
            runs.append(hit)
        hit["items"].append(it)
        hit["f1"], hit["last"] = it["frame"], it["xyxy"]
        open_runs = [r for r in open_runs if it["frame"] - r["f1"] <= 1]
    out = []
    for r in runs:
        its = r["items"]
        best = max(its, key=lambda x: x["conf"])
        n_models = max(len(x["models"]) for x in its)
        tags = sorted({t for x in its for t in x["tags"]})
        out.append({"f0": r["f0"], "f1": r["f1"], "n": len(its), "best_frame": best["frame"], "xyxy": best["xyxy"],
                    "h": best["h"], "conf": best["conf"], "n_models": n_models,
                    "mean_conf": round(float(np.mean([x["conf"] for x in its])), 3), "tags": tags,
                    "frames": [{"frame": x["frame"], "xyxy": x["xyxy"], "conf": x["conf"], "models": x["models"]}
                               for x in its]})
    # static sites: runs whose best boxes overlap (IoU >= 0.4) are the same image spot -> one site number, so a
    # repeated non-person (cone, dark object, mannequin) shows up as one entry with the total frame count
    sites: list[dict] = []
    for r in sorted(out, key=lambda r: -r["n"]):
        for s in sites:
            if iou(r["xyxy"], s["xyxy"])[0, 0] >= 0.4:
                r["site"] = s["site"]
                s["n"] += r["n"]
                s["runs"] += 1
                break
        else:
            r["site"] = len(sites) + 1
            sites.append({"site": r["site"], "xyxy": r["xyxy"], "n": r["n"], "runs": 1})
    for r in out:
        s = sites[r["site"] - 1]
        r["site_runs"], r["site_frames"] = s["runs"], s["n"]
    out.sort(key=lambda r: -score_missed(r))
    return out


def group_by_id(items: list[dict]) -> list[dict]:
    runs: dict[int, dict] = {}
    out = []
    for it in sorted(items, key=lambda x: (x["id"], x["frame"])):
        r = runs.get(it["id"])
        if r is None or it["frame"] - r["f1"] > 1:
            r = {"id": it["id"], "f0": it["frame"], "f1": it["frame"], "frames": []}
            runs[it["id"]] = r
            out.append(r)
        r["frames"].append(it)
        r["f1"] = it["frame"]
    for r in out:
        fs = r["frames"]
        r["n"] = len(fs)
        if "best_det_iou" in fs[0]:
            r["h"] = round(float(np.median([x["h"] for x in fs])), 1)
            r["max_det_iou"] = max(x["best_det_iou"] for x in fs)
            r["src"] = sorted({str(x["src"]) for x in fs})
            r["occ"] = sorted({str(x["occ"]) for x in fs})
            r["tags"] = sorted({t for x in fs for t in x["tags"]})
            r["best_frame"] = fs[len(fs) // 2]["frame"]
            r["xyxy"] = fs[len(fs) // 2]["xyxy"]
        else:
            r["mean_iou"] = round(float(np.mean([x["iou"] for x in fs])), 2)
            r["h"] = round(float(np.median([x["xyxy"][3] - x["xyxy"][1] for x in fs])), 1)
    key = score_false if out and "max_det_iou" in out[0] else (lambda r: r["n"])
    out.sort(key=lambda r: -key(r))
    return out


def score_missed(r: dict) -> float:
    pen = 0.3 if "in_vehicle" in r["tags"] else 1.0
    return pen * r["conf"] * min(r["n_models"], 3) * min(r["n"], 10) ** 0.5 * min(r["h"] / 60.0, 1.5)


def score_false(r: dict) -> float:
    pen = 0.4 if "hidden" in r.get("tags", []) else 1.0
    return pen * (1 - r["max_det_iou"]) * min(r["n"], 15) ** 0.5 * min(r["h"] / 60.0, 1.5)


# ------------------------------------------------------------------ montage
def crop(img, xyxy, margin=0.6, out_h=200, gt_boxes=()):
    """crop around xyxy (yellow); GT boxes of the frame in green (ignored: grey)"""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = xyxy
    w, h = max(x2 - x1, 10), max(y2 - y1, 10)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(w, h) * (1 + margin)
    s = max(s, 80)
    a, b = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    c, d = int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    patch = img[b:d, a:c].copy()
    for g in gt_boxes:
        gx1, gy1, gx2, gy2 = g["xyxy"]
        col = (160, 160, 160) if is_ignored(g) else (0, 200, 0)
        cv2.rectangle(patch, (int(gx1 - a), int(gy1 - b)), (int(gx2 - a), int(gy2 - b)), col, 1)
    cv2.rectangle(patch, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), (0, 255, 255), 1)
    sc = out_h / max(patch.shape[0], 1)
    patch = cv2.resize(patch, (max(int(patch.shape[1] * sc), 1), out_h))
    if patch.shape[1] > out_h:
        patch = patch[:, :out_h]
    pad = np.zeros((out_h, out_h, 3), np.uint8)
    pad[:, :patch.shape[1]] = patch
    return pad


def montage(seg: Path, gt_frames: list[dict], entries: list[tuple[int, list, str]], out: Path, cols=8, out_h=200):
    if not entries:
        if out.exists():
            out.unlink()
        return
    tiles = []
    for fi, xyxy, text in entries[:48]:
        img = cv2.imread(str(seg / gt_frames[fi]["file"]))
        if img is None:
            continue
        t = crop(img, xyxy, out_h=out_h, gt_boxes=gt_frames[fi]["boxes"])
        cv2.rectangle(t, (0, 0), (out_h, 16), (0, 0, 0), -1)
        cv2.putText(t, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * out_h, cols * out_h, 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * out_h:(r + 1) * out_h, c * out_h:(c + 1) * out_h] = t
    cv2.imwrite(str(out), canvas, [cv2.IMWRITE_JPEG_QUALITY, 85])


def montage_entries_missed(runs: list[dict]) -> list:
    ent = []
    for r in runs:
        tag = ("V" if "in_vehicle" in r["tags"] else "") + ("G" if any(t.startswith("near_gt") for t in r["tags"]) else "")
        ent.append((r["best_frame"], r["xyxy"],
                    f"f{r['best_frame']:04d} h{r['h']:.0f} c{r['conf']:.2f} m{r['n_models']}{tag} s{r['site']}"))
    if len(ent) < 48:                                  # fill with more frames of the longest runs
        for r in sorted(runs, key=lambda r: -r["n"]):
            for x in r["frames"][::max(1, r["n"] // 4)]:
                if x["frame"] != r["best_frame"] and len(ent) < 48:
                    ent.append((x["frame"], x["xyxy"], f"f{x['frame']:04d} h{x['xyxy'][3]-x['xyxy'][1]:.0f} c{x['conf']:.2f}"))
    return ent[:48]


def montage_entries_false(runs: list[dict]) -> list:
    ent = []
    for r in runs:
        ent.append((r["best_frame"], r["xyxy"], f"f{r['best_frame']:04d} id{r['id']} h{r['h']:.0f} iou{r['max_det_iou']:.2f}"))
    if len(ent) < 48:
        for r in sorted(runs, key=lambda r: -r["n"]):
            for x in r["frames"][::max(1, r["n"] // 4)]:
                if x["frame"] != r["best_frame"] and len(ent) < 48:
                    ent.append((x["frame"], x["xyxy"], f"f{x['frame']:04d} id{x['id']} h{x['h']:.0f}"))
    return ent[:48]


# ------------------------------------------------------------------ reports
def fr(r):
    return f"f{r['f0']:04d}" if r["f0"] == r["f1"] else f"f{r['f0']:04d}-{r['f1']:04d}"


def write_report(seg: Path, a: dict):
    L = [f"# GT audit: {a['segment']}", "",
         f"frames {a['n_frames']}, models {', '.join(a['models'])}, worker_only={a['worker_only']}. "
         f"MISSED runs {len(a['missed'])}, FALSE runs {len(a['false'])}, LOOSE boxes "
         f"{sum(r['n'] for r in a['loose'])} ({len(a['loose'])} runs). Montages: missed.jpg, false.jpg.", ""]
    L += ["## MISSED (consensus detection, no GT box)", "",
          "site = same image spot across runs (a static non-person repeats as one site); V = inside a vehicle / equipment box",
          "", "| run | site (runs/frames) | frames | n | best frame | xyxy | h px | conf | models | tags |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(a["missed"], 1):
        L.append(f"| {k} | s{r['site']} ({r['site_runs']}/{r['site_frames']}) | {fr(r)} | {r['n']} | f{r['best_frame']:04d} "
                 f"| {r['xyxy']} | {r['h']:.0f} | {r['conf']:.2f} (mean {r['mean_conf']:.2f}) | {r['n_models']} "
                 f"| {', '.join(r['tags'])} |")
    L += ["", "## FALSE (GT worker box, no detector agrees at conf >= 0.1)", "",
          "hidden = >= 50 % of the box lies inside another GT box (interpolated through an occlusion: occ / ignore question "
          "rather than a false box)", "",
          "| run | id | frames | n | h px | max det IoU | src | occ | tags |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in enumerate(a["false"], 1):
        L.append(f"| {k} | {r['id']} | {fr(r)} | {r['n']} | {r['h']:.0f} | {r['max_det_iou']:.2f} | {', '.join(r['src'])} "
                 f"| {', '.join(r['occ'])} | {', '.join(r['tags'])} |")
    L += ["", "## LOOSE (GT box vs consensus box IoU 0.3-0.6; proposed xyxy per frame in audit.json)", "",
          "| run | id | frames | n | h px | mean IoU | example frame: GT -> proposed |", "|---|---|---|---|---|---|---|"]
    for k, r in enumerate(a["loose"], 1):
        x = r["frames"][0]
        L.append(f"| {k} | {r['id']} | {fr(r)} | {r['n']} | {r['h']:.0f} | {r['mean_iou']:.2f} | f{x['frame']:04d}: "
                 f"{x['xyxy']} -> {x['proposed']} (c{x['conf']:.2f}) |")
    (seg / "audit" / "report.md").write_text("\n".join(L) + "\n")


def write_summary(labels: Path, results: list[dict], out: Path):
    L = ["# GT audit summary", "", "| segment | frames | models | MISSED runs (frames) | FALSE runs (frames) | LOOSE boxes |",
         "|---|---|---|---|---|---|"]
    for a in results:
        L.append(f"| {a['segment']} | {a['n_frames']} | {len(a['models'])} | {len(a['missed'])} "
                 f"({sum(r['n'] for r in a['missed'])}) | {len(a['false'])} ({sum(r['n'] for r in a['false'])}) | "
                 f"{sum(r['n'] for r in a['loose'])} |")
    allm = sorted(((score_missed(r), a["segment"], r) for a in results for r in a["missed"]), key=lambda t: -t[0])
    allf = sorted(((score_false(r), a["segment"], r) for a in results for r in a["false"]), key=lambda t: -t[0])
    L += ["", "## Top MISSED", "", "| segment | frames | n | site (runs/frames) | h px | conf | models | tags |",
          "|---|---|---|---|---|---|---|---|"]
    for s, seg, r in allm[:30]:
        L.append(f"| {seg} | {fr(r)} | {r['n']} | s{r['site']} ({r['site_runs']}/{r['site_frames']}) | {r['h']:.0f} "
                 f"| {r['conf']:.2f} | {r['n_models']} | {', '.join(r['tags'])} |")
    L += ["", "## Top FALSE", "", "| segment | id | frames | n | h px | max det IoU | src | occ | tags |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s, seg, r in allf[:25]:
        L.append(f"| {seg} | {r['id']} | {fr(r)} | {r['n']} | {r['h']:.0f} | {r['max_det_iou']:.2f} | {', '.join(r['src'])} "
                 f"| {', '.join(r['occ'])} | {', '.join(r['tags'])} |")
    out.write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--segs", nargs="*")
    ap.add_argument("--models", nargs="+", default=["yolo11x", "rtdetrx", "gdino", "owlv2"])
    ap.add_argument("--conf", type=float, default=0.1, help="FALSE: no detection above this")
    ap.add_argument("--miss-conf", type=float, default=0.35, help="MISSED: consensus members above this")
    ap.add_argument("--loose-conf", type=float, default=0.5, help="LOOSE: consensus members above this")
    ap.add_argument("--min-h", type=float, default=25.0, help="MISSED: minimum box height px")
    ap.add_argument("--no-montage", action="store_true")
    a = ap.parse_args()
    results = []
    for seg in sorted(a.labels.iterdir()):
        if not (seg / "gt.json").exists() or seg.name in SKIP_SEGS or (a.segs and seg.name not in a.segs):
            continue
        if not any((seg / "audit" / f"dets_{m}.json").exists() for m in a.models):
            print(f"{seg.name}: no detections, skipped")
            continue
        res = analyse(seg, a.models, a.conf, a.miss_conf, a.loose_conf, a.min_h)
        json.dump(res, open(seg / "audit" / "audit.json", "w"), indent=1)
        write_report(seg, res)
        if not a.no_montage:
            gtf = json.load(open(seg / "gt.json"))["frames"]
            montage(seg, gtf, montage_entries_missed(res["missed"]), seg / "audit" / "missed.jpg")
            montage(seg, gtf, montage_entries_false(res["false"]), seg / "audit" / "false.jpg")
        results.append(res)
        print(f"{seg.name}: models {len(res['models'])}, MISSED runs {len(res['missed'])} "
              f"({sum(r['n'] for r in res['missed'])} frames), FALSE runs {len(res['false'])} "
              f"({sum(r['n'] for r in res['false'])}), LOOSE boxes {sum(r['n'] for r in res['loose'])}")
    if results:
        write_summary(a.labels, results, a.labels / "audit_summary.md")


if __name__ == "__main__":
    main()
