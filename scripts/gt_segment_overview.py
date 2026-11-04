#!/usr/bin/env python3
"""Build docs/_src/29_gt_segments.md + docs/img/gt_seg/*.jpg from data/labels/<seg>/gt.json (one card per segment).

Each card: condition line (site / radar / mount / lighting / split / window), ego-state band, worker stats
(ids, boxes, ignore, ref3d coverage and range bins, occlusion), a 2x2 contact sheet with GT boxes drawn
(ignored boxes dashed grey), and the open Notes. Re-run after labels change:

  PYTHONPATH=src python3 scripts/gt_segment_overview.py && python3 scripts/build_docs.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "data" / "labels"
IMG = ROOT / "docs" / "img" / "gt_seg"
OUT = ROOT / "docs" / "_src" / "29_gt_segments.md"

EXCLUDED = {"night_drivemid_front_000": "야간 원본 bag 소실, 세트에서 제외"}

# what each segment is for (hand-written, Korean)
PURPOSE = {
    "day_halt_short_085": "보령 Bosch 정지, 근거리(5~10 m) 작업자 3명. 기존 개발 구간, 파라미터 조정에 가장 많이 쓴 구간.",
    "val_day_haltshort_020": "보령 Bosch 정지, 같은 halt-short bag의 다른 창. 개발.",
    "val_day_haltmid_020": "보령 Bosch 정지, 15~20 m 중거리. 원거리 지면 거리 편향을 잡은 구간. 개발.",
    "conti_swing_025": "보령 Continental front 카메라, 리그 선회(18 s). 초기 선회 실험 구간. 개발.",
    "conti_swing_065": "보령 Continental front, 리그 선회 두 번째 창. 개발.",
    "conti_swing_105": "보령 Continental front, 리그 선회 세 번째 창. 개발.",
    "conti_halt_short_060": "보령 Continental 정지, Bosch halt-short와 **같은 장면**(37분 전). 레이더 비교용. 개발.",
    "conti_halt_mid_060": "보령 Continental 정지, Bosch halt-mid와 같은 장면, 15~20 m. 레이더 비교용. 시험.",
    "conti_side_swing_a": "보령 Continental side 카메라, 주행→정지→주행+선회(17 °/s)→정지→주행. 레이더 선회 + 주행 혼합 조건. 시험.",
    "conti_side_drive_b": "보령 Continental side, 정지 26 s → 옆으로 주행 0.85 m/s. 선회 없음(이름 변경 11-01). Continental 주행 조건. 시험.",
    "hdx1026_excA_142537_017": "보령 Bosch 굴착기 정면 장착, 제자리 선회 최대 67 °/s, 역광. 가장 어려운 선회 조건, 재식별 GT. 시험.",
    "hdx1026_excA_142951_000": "보령 Bosch 굴착기 정면, 정지 20 s → 선회. 선회 시작 경계 포함. 레버암 보정에 쓴 bag이라 개발.",
    "hdx1026_excA_135442_000": "보령 Bosch 굴착기 정면, 정지. 작업자 다수(28 ID), 1,000 박스. 정지 조건의 큰 구간. 시험.",
    "hdx1026_excB_154405_000": "보령 Bosch 굴착기 캐빈 3.3 m 높이, 아래 26° 기울임. 높은 장착 위치 조건. 시험.",
    "hdx1026_tripod_113917_040": "보령 Bosch 삼각대, 굴착기가 앞을 지나가며 작업자를 가림. 가림 조건. 시험.",
    "cf_vib_100": "애리조나 파일 드라이버 장착, 정지 → 주행(진동 포함). 기존 개발 구간.",
    "cf_vib_160": "애리조나 파일 드라이버, 같은 bag 다음 창. 개발.",
    "xpanner05_static_a": "애리조나 파일 드라이버, 정지, 작업자 15~30 m. 애리조나 시험 구간. LiDAR 빔 줄 사이라 기준 거리 43 %.",
    "xpanner03_mixed_a": "애리조나 파일 드라이버, 정지→주행→정지. 애리조나 주행 조건. 시험.",
}

SITE_KO = {"boryeong_pg": "보령 검증 현장", "arizona_xpanner": "애리조나"}
RADAR_KO = {"bosch_premium": "Bosch", "continental_ars548": "Continental", "conti_ars548": "Continental"}
SPLIT_KO = {"dev": "개발", "test": "시험"}
LIGHT_KO = {"day": "주간", "day_backlight": "주간 역광", "day_overexposed": "주간 과노출", "night": "야간"}
EGO_COL = {"static": (160, 160, 160), "drive": (91, 155, 213), "swing": (244, 162, 97), "drive+swing": (192, 0, 0)}
EGO_KO = {"static": "정지", "drive": "주행", "swing": "선회", "drive+swing": "주행+선회"}

try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except OSError:
    FONT = FONT_S = ImageFont.load_default()


def ego_band(seg: str, meta: dict, n: int) -> str:
    w, h = 1200, 44
    im = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    for f0, f1, st in meta.get("ego_state") or [[0, n - 1, "static"]]:
        x0, x1 = int(f0 / n * w), int((f1 + 1) / n * w)
        dr.rectangle([x0, 4, x1, 28], fill=EGO_COL.get(st, (200, 200, 200)))
        if x1 - x0 > 60:
            dr.text((x0 + 4, 7), EGO_KO.get(st, st), fill=(0, 0, 0), font=FONT_S)
    dur = float(meta.get("end", 0) - meta.get("start", 0)) or n / 5.0
    for s in range(0, int(dur) + 1, 10):
        x = int(s / dur * w)
        dr.line([x, 28, x, 34], fill=(0, 0, 0))
        dr.text((x + 2, 30), f"{s}s", fill=(80, 80, 80), font=FONT_S)
    p = IMG / f"{seg}_ego.png"
    im.save(p)
    return p.name


def contact_sheet(seg: str, d: dict) -> str:
    frames = d["frames"]
    n = len(frames)
    picks = [int(n * q) for q in (0.08, 0.36, 0.64, 0.92)]
    tile_w, tile_h = 640, 360
    sheet = Image.new("RGB", (tile_w * 2, tile_h * 2), (0, 0, 0))
    for k, fi in enumerate(picks):
        fr = frames[fi]
        try:
            im = Image.open(LAB / seg / fr["file"]).convert("RGB")
        except Exception:
            im = Image.new("RGB", (1280, 720), (40, 40, 40))
        sx, sy = tile_w / im.width, tile_h / im.height
        im = im.resize((tile_w, tile_h))
        dr = ImageDraw.Draw(im)
        for b in fr["boxes"]:
            if b.get("cls") != "worker":
                continue
            x0, y0, x1, y1 = b["xyxy"]
            x0, y0, x1, y1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
            ign = "ignore" in (b.get("flags") or []) or b.get("ignore_reason")
            col = (150, 150, 150) if ign else (255, 220, 0)
            dr.rectangle([x0, y0, x1, y1], outline=col, width=1 if ign else 2)
            if not ign:
                r = b.get("ref3d") or {}
                tag = f"{b['id']}" + (f" {r['range_m']:.0f}m" if r.get("range_m") else "")
                dr.text((x0, max(0, y0 - 16)), tag, fill=col, font=FONT_S)
        t = fr.get("bag_t") or fr["stamp"]
        lab = f"f{fi}  +{t - (frames[0].get('bag_t') or frames[0]['stamp']):.1f}s"
        dr.rectangle([0, 0, 12 + dr.textlength(lab, font=FONT), 28], fill=(0, 0, 0))
        dr.text((6, 4), lab, fill=(255, 255, 255), font=FONT)
        sheet.paste(im, ((k % 2) * tile_w, (k // 2) * tile_h))
    p = IMG / f"{seg}_sheet.jpg"
    sheet.save(p, quality=82)
    return p.name


def stats(d: dict) -> dict:
    boxes = [b for f in d["frames"] for b in f["boxes"] if b.get("cls") == "worker"]
    valid = [b for b in boxes if not ("ignore" in (b.get("flags") or []) or b.get("ignore_reason"))]
    bins = Counter()
    for b in valid:
        r = (b.get("ref3d") or {}).get("range_m")
        if r is None:
            bins["없음"] += 1
        else:
            bins["0-10" if r < 10 else "10-20" if r < 20 else "20-30" if r < 30 else "30+"] += 1
    occ = Counter(b.get("occ", "none") for b in valid)
    ign = Counter(b.get("ignore_reason") or "flag" for b in boxes if b not in valid)
    ids = {b["id"] for b in valid}
    cov = len([b for b in valid if (b.get("ref3d") or {}).get("range_m") is not None]) / max(1, len(valid))
    return dict(boxes=len(valid), ignored=len(boxes) - len(valid), ids=len(ids), bins=bins, occ=occ, ign=ign, cov=cov)


def card(seg: str) -> str:
    d = json.loads((LAB / seg / "gt.json").read_text())
    m = d["meta"]
    n = len(d["frames"])
    st = stats(d)
    band = ego_band(seg, m, n)
    sheet = contact_sheet(seg, d)
    notes = d.get("notes", [])
    open_notes = [x for x in notes if x.get("status", "open") == "open"]
    ego = ", ".join(f"{EGO_KO.get(s, s)} f{a}-{b}" for a, b, s in (m.get("ego_state") or []))
    rate = m.get("ego_rate") or {}
    bag = Path(m.get("bag", "")).name
    L = []
    L.append(f"### {seg}")
    L.append("")
    L.append(f"**용도**: {PURPOSE.get(seg, '')}")
    L.append("")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append(f"| 분할 / 단계 | {SPLIT_KO.get(m.get('split'), m.get('split'))} / {m.get('stage', '')} |")
    L.append(f"| 장소 / 레이더 / 장착 | {SITE_KO.get(m.get('site'), m.get('site'))} / {RADAR_KO.get(m.get('radar'), m.get('radar'))} / {m.get('mount')} |")
    L.append(f"| 조명 | {LIGHT_KO.get(m.get('lighting'), m.get('lighting'))} |")
    L.append(f"| bag / 창 | `{bag}` {m.get('start')}–{m.get('end')} s, {n}프레임 @ {m.get('rate_hz', 0):.1f} Hz |")
    L.append(f"| 자차 상태 | {ego or '정지'}" + (f" (최대 선회 {rate.get('yaw_rate_max_dps', 0):.0f} °/s, 최대 속도 {rate.get('speed_max_mps', 0):.2f} m/s)" if rate else "") + " |")
    L.append(f"| 작업자 | ID {st['ids']}개, 유효 박스 {st['boxes']:,}, 제외(ignore) {st['ignored']}" + (f" ({', '.join(f'{k} {v}' for k, v in st['ign'].most_common())})" if st['ignored'] else "") + " |")
    L.append(f"| 기준 거리(LiDAR) | 유효 박스의 {st['cov'] * 100:.0f} %, 규칙 {'희소(≥3점)' if (m.get('ref3d') or {}).get('sparse') else '조밀(≥8점)'} |")
    L.append(f"| 거리 분포(박스) | " + ", ".join(f"{k} m {st['bins'][k]}" for k in ("0-10", "10-20", "20-30", "30+") if st['bins'][k]) + (f", 기준 없음 {st['bins']['없음']}" if st['bins']['없음'] else "") + " |")
    L.append(f"| 가림 | " + ", ".join(f"{k} {v}" for k, v in st['occ'].most_common()) + " |")
    L.append(f"| Notes | 열림 {len(open_notes)} / 전체 {len(notes)} |")
    L.append("")
    L.append(f"![ego]({'img/gt_seg/' + band})")
    L.append("")
    L.append(f"![sheet]({'img/gt_seg/' + sheet})")
    L.append("")
    L.append("샘플 4프레임(8 / 36 / 64 / 92 % 지점). 노란 박스 = 유효 GT(ID, 기준 거리), 회색 박스 = 평가 제외.")
    if open_notes:
        L.append("")
        L.append("열린 Notes (사용자 판단 필요):")
        L.append("")
        for x in open_notes[:8]:
            txt = (x.get("text") or "").replace("|", "/")
            L.append(f"- f{x['frame']} [{x.get('kind')}] {txt[:170]}{'…' if len(txt) > 170 else ''}")
        if len(open_notes) > 8:
            L.append(f"- … 외 {len(open_notes) - 8}개")
    L.append("")
    return "\n".join(L), dict(seg=seg, split=m.get("split"), site=m.get("site"), radar=m.get("radar"), mount=m.get("mount"),
                              light=m.get("lighting"), ego=m.get("ego_state") or [], n=n, **st, open=len(open_notes))


def main():
    IMG.mkdir(parents=True, exist_ok=True)
    segs = sorted(p.name for p in LAB.iterdir() if (p / "gt.json").exists() and p.name not in EXCLUDED)
    order = [s for s in PURPOSE if s in segs] + [s for s in segs if s not in PURPOSE]
    cards, rows = [], []
    for s in order:
        c, r = card(s)
        cards.append(c)
        rows.append(r)
        print("ok", s, r["n"], r["ids"], r["boxes"])
    tot_f = sum(r["n"] for r in rows)
    tot_b = sum(r["boxes"] for r in rows)
    H = []
    H.append("# 29. GT 구간 설명 (판단용)")
    H.append("")
    H.append(f"`data/labels/` 아래 {len(rows)}개 구간, {tot_f:,}프레임, 유효 작업자 박스 {tot_b:,}개. 이 페이지는 `scripts/gt_segment_overview.py`가 gt.json에서 자동으로 만든다. "
             "세트에서 뺀 것: " + ", ".join(f"`{k}`({v})" for k, v in EXCLUDED.items()) + ". 계획과 결정 사항은 doc 28.")
    H.append("")
    H.append("## 1. 한눈에")
    H.append("")
    H.append("| 구간 | 분할 | 장소 | 레이더 | 장착 | 자차 | 프레임 | ID | 박스 | 제외 | 기준거리 | 열린 Notes |")
    H.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ego = "/".join(dict.fromkeys(EGO_KO.get(s, s) for _, _, s in r["ego"])) or "정지"
        H.append(f"| {r['seg']} | {SPLIT_KO.get(r['split'], r['split'])} | {SITE_KO.get(r['site'], r['site'])} | {RADAR_KO.get(r['radar'], r['radar'])} | {r['mount']} | {ego} | {r['n']} | {r['ids']} | {r['boxes']:,} | {r['ignored']} | {r['cov'] * 100:.0f} % | {r['open']} |")
    H.append("")
    H.append("## 2. 조건 셀 채움 (자차 상태 × 장소 × 레이더)")
    H.append("")
    cells = {}
    for r in rows:
        states = set(s for _, _, s in r["ego"]) or {"static"}
        key_state = "선회" if any("swing" in s for s in states) else "주행" if "drive" in states else "정지"
        cells.setdefault((key_state, SITE_KO.get(r["site"], r["site"]), RADAR_KO.get(r["radar"], r["radar"])), []).append(f"{r['seg']}({SPLIT_KO.get(r['split'], '?')[0]})")
    H.append("| 자차 | 장소 | 레이더 | 구간 (개=개발, 시=시험) |")
    H.append("|---|---|---|---|")
    for st_ in ("정지", "주행", "선회"):
        for site in ("보령 검증 현장", "애리조나"):
            for rad in ("Bosch", "Continental"):
                v = cells.get((st_, site, rad))
                if v or not (site == "애리조나" and rad == "Continental"):
                    H.append(f"| {st_} | {site} | {rad} | {', '.join(v) if v else '— (없음)'} |")
    H.append("")
    H.append("판단할 때 볼 것: (1) 셀이 비거나 한쪽 분할만 있는 곳, (2) 구간의 용도 설명이 실제 샘플 프레임과 맞는지, (3) 제외 박스가 너무 많은 구간, (4) 열린 Notes 수. "
             "구간을 빼거나 바꾸려면 구간 이름과 이유만 알려 주면 된다.")
    H.append("")
    H.append("## 3. 구간별 설명")
    H.append("")
    H.extend(cards)
    OUT.write_text("\n".join(H), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
