#!/usr/bin/env python3
"""Contact sheets for reviewing a label segment: overview grids (every --step frames, boxes + ids) and per-id
crop strips. Output: <segment>/sheets/.

  python3 scripts/label_sheets.py data/labels/day_halt_short_085 [--step 5] [--cols 3] [--ids 3,5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.gt import GT  # noqa: E402


def color(tid: int):
    h = (tid * 47) % 180
    c = cv2.cvtColor(np.uint8([[[h, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(v) for v in c)


def draw(img, boxes, scale=1.0, th=2):
    for b in boxes:
        x1, y1, x2, y2 = (np.array(b["xyxy"]) * scale).astype(int)
        c = color(b["id"])
        dashed = b["src"] != "det"
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 1 if dashed else th)
        tag = f"{b['id']}" + ("*" if dashed else "") + ("" if b["cls"] == "worker" else "M")
        cv2.putText(img, tag, (x1, max(y1 - 3, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg")
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--width", type=int, default=760)
    ap.add_argument("--ids")
    ap.add_argument("--range", help="frame range a:b for overviews")
    ap.add_argument("--brighten", action="store_true")
    a = ap.parse_args()
    gt = GT(Path(a.seg) / "gt.json")
    out = gt.root / "sheets"
    out.mkdir(exist_ok=True)
    for f in out.glob("*.jpg"):
        f.unlink()
    lo, hi = (map(int, a.range.split(":")) if a.range else (0, len(gt.frames)))
    idx = list(range(lo, min(hi, len(gt.frames)), a.step))
    per = a.cols * a.rows
    for s in range(0, len(idx), per):
        tiles = []
        for i in idx[s:s + per]:
            img = cv2.imread(str(gt.root / gt.frames[i]["file"]))
            if a.brighten:
                img = cv2.convertScaleAbs(img, alpha=2.2, beta=10)
            sc = a.width / img.shape[1]
            img = draw(cv2.resize(img, None, fx=sc, fy=sc), gt.frames[i]["boxes"], sc)
            cv2.putText(img, f"f{i} t={gt.frames[i]['bag_t'] - gt.frames[0]['bag_t']:.1f}s", (5, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(img)
        while len(tiles) % a.cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[k:k + a.cols]) for k in range(0, len(tiles), a.cols)]
        cv2.imwrite(str(out / f"overview_{idx[s]:04d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    ids = [int(x) for x in a.ids.split(",")] if a.ids else sorted(gt.ids)
    strips = []
    for tid in ids:
        fr = gt.id_frames(tid)
        if not fr:
            continue
        pick = [fr[int(round(k))] for k in np.linspace(0, len(fr) - 1, min(10, len(fr)))]
        crops = []
        for i in pick:
            img = cv2.imread(str(gt.root / gt.frames[i]["file"]))
            if a.brighten:
                img = cv2.convertScaleAbs(img, alpha=2.2, beta=10)
            x1, y1, x2, y2 = np.array(gt.box_of(i, tid)["xyxy"]).astype(int)
            c = img[max(y1, 0):max(y2, y1 + 2), max(x1, 0):max(x2, x1 + 2)]
            c = cv2.resize(c, (72, 144)) if c.size else np.zeros((144, 72, 3), np.uint8)
            cv2.putText(c, f"f{i}", (2, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            crops.append(c)
        crops += [np.zeros((144, 72, 3), np.uint8)] * (10 - len(crops))
        b0 = np.array(gt.box_of(fr[0], tid)["xyxy"])
        head = np.zeros((144, 170, 3), np.uint8)
        for k, txt in enumerate([f"id {tid} {gt.ids.get(tid, {}).get('cls', '?')[:6]}", f"f{fr[0]}-f{fr[-1]}",
                                 f"n={len(fr)}", f"x0={b0[0]:.0f} h={b0[3] - b0[1]:.0f}",
                                 gt.ids.get(tid, {}).get("desc", "")[:22]]):
            cv2.putText(head, txt, (3, 18 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color(tid), 1)
        strips.append(np.hstack([head] + crops))
    for s in range(0, len(strips), 12):
        cv2.imwrite(str(out / f"ids_{s // 12:02d}.jpg"), np.vstack(strips[s:s + 12]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"sheets in {out}: {len(list(out.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
