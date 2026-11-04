#!/usr/bin/env python3
"""GT audit, step 1: run strong person detectors on every label frame (docs/13 §audit).

  python3 scripts/label_audit_detectors.py [--labels data/labels] [--segs seg ...] [--models yolo11x rtdetrx gdino owlv2]
                                           [--conf 0.1] [--device cuda:0]
  python3 scripts/label_audit_detectors.py --server <user>@<gpu-server> --remote /data/gt_audit [--segs ...] [--models ...]
      rsync frames + gt.json to the server, run this script there inside <remote>/venv, rsync audit/ back.

Writes data/labels/<seg>/audit/dets_<model>.json:
  {model, params, frames: [{file, dets: [[x1, y1, x2, y2, conf, label], ...]}]}
Models (all at conf >= --conf, every class kept so the comparison can pick person / vehicle classes):
  yolo11x   ultralytics YOLO11x, imgsz 1920 (frames are 1280 wide: 1.5x upscale), test-time augmentation
  rtdetrx   ultralytics RT-DETR-x, imgsz 1280 (full resolution; TTA not supported by the model)
  gdino     Grounding DINO base (HF transformers), prompt "person. worker. mannequin. construction worker."
  owlv2     OWLv2 large patch14 ensemble (HF transformers), text queries person / worker / mannequin / construction worker
Skips a segment+model whose output already exists unless --force. Nothing in gt.json is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_SEGS = ("night_drivemid_front_000",)        # dropped from the set (docs/29)
PROMPTS = ["person", "worker", "mannequin", "construction worker"]
WEIGHTS = {"yolo11x": "yolo11x.pt", "rtdetrx": "rtdetr-x.pt"}
HF = {"gdino": "IDEA-Research/grounding-dino-base", "owlv2": "google/owlv2-large-patch14-ensemble"}


def list_segs(labels: Path, segs: list[str] | None) -> list[Path]:
    out = []
    for d in sorted(labels.iterdir()):
        if not (d / "gt.json").exists() or d.name in SKIP_SEGS:
            continue
        if segs and d.name not in segs:
            continue
        out.append(d)
    return out


def frame_files(seg: Path) -> list[str]:
    gt = json.load(open(seg / "gt.json"))
    return [f["file"] for f in gt["frames"]]


def save(seg: Path, model: str, params: dict, frames: list[dict], t0: float):
    out = seg / "audit" / f"dets_{model}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"model": model, "params": params, "elapsed_s": round(time.time() - t0, 1), "frames": frames},
              open(out, "w"))
    n = sum(len(f["dets"]) for f in frames)
    print(f"  {seg.name} {model}: {len(frames)} frames, {n} dets, {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------ ultralytics
def run_ultralytics(model_name: str, segs: list[Path], conf: float, device: str, weights_dir: Path, force: bool,
                    batch: int = 8):
    from ultralytics import RTDETR, YOLO
    w = weights_dir / WEIGHTS[model_name]
    if model_name == "yolo11x":
        model, params = YOLO(str(w) if w.exists() else WEIGHTS[model_name]), dict(imgsz=1920, augment=True, conf=conf,
                                                                                  iou=0.6, max_det=300)
    else:
        model, params = RTDETR(str(w) if w.exists() else WEIGHTS[model_name]), dict(imgsz=1280, conf=conf, max_det=300)
    names = model.names
    for seg in segs:
        if (seg / "audit" / f"dets_{model_name}.json").exists() and not force:
            continue
        t0, files, frames = time.time(), frame_files(seg), []
        for i in range(0, len(files), batch):
            chunk = [str(seg / f) for f in files[i:i + batch]]
            res = model.predict(chunk, device=device, verbose=False, **params)
            for f, r in zip(files[i:i + batch], res):
                b = r.boxes
                xyxy, cf, cl = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
                frames.append({"file": f, "dets": [[round(float(v), 1) for v in xy] + [round(float(c), 3), names[k]]
                                                   for xy, c, k in zip(xyxy, cf, cl)]})
        save(seg, model_name, {**params, "weights": WEIGHTS[model_name]}, frames, t0)


# ------------------------------------------------------------------ HF open-vocabulary
def run_gdino(segs: list[Path], conf: float, device: str, force: bool):
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(HF["gdino"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(HF["gdino"]).to(device).eval()
    text = ". ".join(PROMPTS) + "."
    for seg in segs:
        if (seg / "audit" / "dets_gdino.json").exists() and not force:
            continue
        t0, frames = time.time(), []
        for f in frame_files(seg):
            img = Image.open(seg / f).convert("RGB")
            inp = proc(images=img, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            kw = {"threshold": conf, "text_threshold": conf, "target_sizes": [img.size[::-1]]}
            try:
                r = proc.post_process_grounded_object_detection(out, inp.input_ids, **kw)[0]
            except TypeError:                                   # older transformers: box_threshold
                kw["box_threshold"] = kw.pop("threshold")
                r = proc.post_process_grounded_object_detection(out, inp.input_ids, **kw)[0]
            labels = r.get("text_labels", r.get("labels"))
            dets = [[round(float(v), 1) for v in bx.tolist()] + [round(float(s), 3), str(lb) or "person"]
                    for bx, s, lb in zip(r["boxes"], r["scores"], labels)]
            frames.append({"file": f, "dets": dets})
        save(seg, "gdino", {"model": HF["gdino"], "prompt": text, "threshold": conf}, frames, t0)


def run_owlv2(segs: list[Path], conf: float, device: str, force: bool):
    import torch
    from PIL import Image
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained(HF["owlv2"])
    model = Owlv2ForObjectDetection.from_pretrained(HF["owlv2"]).to(device).eval()
    queries = [[f"a photo of a {p}" for p in PROMPTS]]
    for seg in segs:
        if (seg / "audit" / "dets_owlv2.json").exists() and not force:
            continue
        t0, frames = time.time(), []
        for f in frame_files(seg):
            img = Image.open(seg / f).convert("RGB")
            inp = proc(text=queries, images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inp)
            # OWLv2 pads the image to a square (bottom/right): boxes are relative to that square.
            s = max(img.size)
            pp = getattr(proc, "post_process_grounded_object_detection", None) or proc.post_process_object_detection
            r = pp(out, threshold=conf, target_sizes=torch.tensor([[s, s]]))[0]
            dets = [[round(float(v), 1) for v in bx.tolist()] + [round(float(sc), 3), PROMPTS[int(lb)]]
                    for bx, sc, lb in zip(r["boxes"], r["scores"], r["labels"])]
            frames.append({"file": f, "dets": dets})
        save(seg, "owlv2", {"model": HF["owlv2"], "queries": queries[0], "threshold": conf}, frames, t0)


# ------------------------------------------------------------------ server mode
def run_server(a):
    host, remote = a.server, a.remote
    labels = a.labels.resolve()
    sel = [s.name for s in list_segs(labels, a.segs)]
    rs = ["rsync", "-a", "--exclude=night_drivemid_front_000/", "--include=*/", "--include=frames/**",
          "--include=gt.json", "--exclude=*", f"{labels}/", f"{host}:{remote}/labels/"]
    subprocess.run(["ssh", host, f"mkdir -p {remote}/labels {remote}/weights"], check=True)
    subprocess.run(rs, check=True)
    subprocess.run(["scp", "-q", str(Path(__file__).resolve()), f"{host}:{remote}/label_audit_detectors.py"], check=True)
    cmd = (f"cd {remote} && CUDA_VISIBLE_DEVICES={a.gpu} {remote}/venv/bin/python label_audit_detectors.py "
           f"--labels {remote}/labels --weights {remote}/weights --conf {a.conf} --device cuda:0 "
           f"--models {' '.join(a.models)}" + (f" --segs {' '.join(sel)}" if a.segs else "") +
           (" --force" if a.force else ""))
    print(cmd, flush=True)
    subprocess.run(["ssh", host, cmd], check=True)
    subprocess.run(["rsync", "-a", "--include=*/", "--include=audit/dets_*.json", "--exclude=*",
                    f"{host}:{remote}/labels/", f"{labels}/"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=ROOT / "data/labels")
    ap.add_argument("--segs", nargs="*")
    ap.add_argument("--models", nargs="+", default=["yolo11x", "rtdetrx", "gdino", "owlv2"])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--weights", type=Path, default=ROOT / "models")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--server", help="user@host: run there (rsync in, run in <remote>/venv, rsync audit back)")
    ap.add_argument("--remote", default="/data/gt_audit")
    ap.add_argument("--gpu", default="0", help="server GPU index (CUDA_VISIBLE_DEVICES)")
    a = ap.parse_args()
    if a.server:
        return run_server(a)
    segs = list_segs(a.labels, a.segs)
    print(f"{len(segs)} segments, models {a.models}", flush=True)
    os.chdir(a.weights) if a.weights.exists() else None          # ultralytics downloads missing .pt into cwd
    for m in a.models:
        print(f"== {m}", flush=True)
        if m in WEIGHTS:
            run_ultralytics(m, segs, a.conf, a.device, a.weights, a.force)
        elif m == "gdino":
            run_gdino(segs, a.conf, a.device, a.force)
        elif m == "owlv2":
            run_owlv2(segs, a.conf, a.device, a.force)
        else:
            sys.exit(f"unknown model {m}")


if __name__ == "__main__":
    main()
