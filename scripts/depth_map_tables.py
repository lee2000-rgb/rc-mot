#!/usr/bin/env python3
"""Tables for docs/27 from the per-box dumps of `eval_fusion_depth.py --depth-map --dm-ablate --dump <json>`.

  python3 scripts/depth_map_tables.py main:<seg>=<json> [main:<seg>=<json> ...] [--sens 518x518:<seg>=<json> ...]

Per segment and LiDAR-range bin: N boxes; for each range source (fused, mono, ground_ray, dmap, dmap_mask, combined
"fused when radar-backed else dmap"): availability, median |err|, p90, within 0.5 / 1 m, mean signed bias (against the
LiDAR reference). The "radar-less" subset = boxes whose fusion did not use radar (source not radar / radar+mono).
Sensitivity (--sens): input size x {N=3/1} x {KF/raw} x {bottom third / mask}.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

BINS = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 999)]
WORKER = "worker"


def stats(rows, key):
    """(n_avail, median|e|, p90, within0.5, within1, bias) of source `key` over `rows` (all with a reference)."""
    e = np.array([r[key] - r["ref"] for r in rows if r.get(key) is not None], float)
    if len(e) == 0:
        return dict(n=0)
    a = np.abs(e)
    return dict(n=int(len(e)), avail=len(e) / len(rows), med=float(np.median(a)), p90=float(np.percentile(a, 90)),
                w05=float(np.mean(a <= 0.5)), w1=float(np.mean(a <= 1.0)), bias=float(np.mean(e)))


def combined(rows):
    for r in rows:
        r["combo"] = r["fused"] if str(r.get("src", "")).startswith("radar") else r.get("dmap")
    return rows


def fmt(s):
    if s.get("n", 0) == 0:
        return "—"
    return f"{s['avail']*100:.0f} % · {s['med']:.2f} / {s['p90']:.2f} · {s['w05']*100:.0f} / {s['w1']*100:.0f} % · {s['bias']:+.2f}"


def load(path):
    d = json.load(open(path))
    rows = [r for r in d["boxes"] if r["cls"] == WORKER]
    return d["summary"], combined(rows)


SOURCES = [("fused", "현재 융합"), ("mono", "키 기반 단안"), ("ground", "접지 광선"), ("dmap", "깊이 맵(아래 1/3)"),
           ("dmap_mask", "깊이 맵(마스크)"), ("combo", "융합, 레이더 없으면 깊이 맵")]


def seg_tables(name, summ, rows):
    out = []
    with_ref = [r for r in rows if r["ref"] is not None]
    out.append(f"\n### {name}\n")
    out.append(f"작업자 GT 박스 {len(rows)}개, LiDAR 기준 있음 {len(with_ref)}개({len(with_ref)/max(len(rows),1)*100:.0f} %), "
               f"레이더 없이 위치가 정해진 박스(기준 있음) {sum(1 for r in with_ref if not str(r['src']).startswith('radar'))}개"
               + (f"; 깊이 맵 프레임 {summ['depth_map']['frames']}, RANSAC inlier 중앙값 {summ['depth_map']['n_in_median']:.0f}, NIS 게이트 {summ['depth_map']['nis_gated']}회" if "depth_map" in summ else "") + "\n")
    out.append("| LiDAR 거리 | N | " + " | ".join(n for _, n in SOURCES) + " |")
    out.append("|---|---|" + "---|" * len(SOURCES))
    for lo, hi in BINS + [(0, 999)]:
        sel = [r for r in with_ref if lo <= r["ref"] < hi]
        if not sel:
            continue
        label = "전체" if hi == 999 and lo == 0 else (f"{lo}–{hi} m" if hi < 999 else f">{lo} m")
        out.append(f"| {label} | {len(sel)} | " + " | ".join(fmt(stats(sel, k)) for k, _ in SOURCES) + " |")
    # radar-less subset
    rl = [r for r in with_ref if not str(r["src"]).startswith("radar")]
    out.append(f"\n레이더 없는 박스만 (N={len(rl)}):\n")
    out.append("| LiDAR 거리 | N | " + " | ".join(n for _, n in SOURCES[:5]) + " |")
    out.append("|---|---|" + "---|" * 5)
    for lo, hi in BINS + [(0, 999)]:
        sel = [r for r in rl if lo <= r["ref"] < hi]
        if not sel:
            continue
        label = "전체" if hi == 999 and lo == 0 else (f"{lo}–{hi} m" if hi < 999 else f">{lo} m")
        out.append(f"| {label} | {len(sel)} | " + " | ".join(fmt(stats(sel, k)) for k, _ in SOURCES[:5]) + " |")
    return "\n".join(out)


def pooled_table(all_rows):
    out = ["\n### 전 구간 합산 (작업자, LiDAR 기준 있는 박스)\n", "| LiDAR 거리 | N | " + " | ".join(n for _, n in SOURCES) + " |",
           "|---|---|" + "---|" * len(SOURCES)]
    with_ref = [r for r in all_rows if r["ref"] is not None]
    for lo, hi in BINS + [(0, 999)]:
        sel = [r for r in with_ref if lo <= r["ref"] < hi]
        if sel:
            label = "전체" if hi == 999 and lo == 0 else (f"{lo}–{hi} m" if hi < 999 else f">{lo} m")
            out.append(f"| {label} | {len(sel)} | " + " | ".join(fmt(stats(sel, k)) for k, _ in SOURCES) + " |")
    rl = [r for r in with_ref if not str(r["src"]).startswith("radar")]
    out += [f"\n레이더 없는 박스만, 전 구간 합산 (N={len(rl)}):\n", "| LiDAR 거리 | N | " + " | ".join(n for _, n in SOURCES[:5]) + " |",
            "|---|---|" + "---|" * 5]
    for lo, hi in BINS + [(0, 999)]:
        sel = [r for r in rl if lo <= r["ref"] < hi]
        if sel:
            label = "전체" if hi == 999 and lo == 0 else (f"{lo}–{hi} m" if hi < 999 else f">{lo} m")
            out.append(f"| {label} | {len(sel)} | " + " | ".join(fmt(stats(sel, k)) for k, _ in SOURCES[:5]) + " |")
    return "\n".join(out)


def sens_table(entries):
    """entries: list of (size, seg, rows). Variants from the ablation columns of each dump."""
    variants = [("dmap", "N=3, KF, 아래 1/3"), ("dmap_mask", "N=3, KF, 마스크"), ("n3raw", "N=3, KF 없음, 아래 1/3"),
                ("n3raw_mask", "N=3, KF 없음, 마스크"), ("n1kf", "N=1, KF, 아래 1/3"), ("n1kf_mask", "N=1, KF, 마스크"),
                ("n1raw", "N=1, KF 없음, 아래 1/3")]
    out = ["\n| 입력 크기 | 구간 | 변형 | N | 중앙값 \\|오차\\| / p90 [m] | 0.5 / 1 m 이내 | 치우침 |", "|---|---|---|---|---|---|---|"]
    for size, seg, rows in entries:
        with_ref = [r for r in rows if r["ref"] is not None]
        for key, name in variants:
            if not any(r.get(key) is not None for r in with_ref):
                continue
            s = stats(with_ref, key)
            out.append(f"| {size} | {seg} | {name} | {s['n']} | {s['med']:.2f} / {s['p90']:.2f} | {s['w05']*100:.0f} / {s['w1']*100:.0f} % | {s['bias']:+.2f} |")
    return "\n".join(out)


def sigma_check(all_rows):
    e = np.array([(r["dmap"] - r["ref"]) / r["dmap_sigma"] for r in all_rows if r["ref"] is not None and r.get("dmap") is not None and r.get("dmap_sigma")])
    if len(e) == 0:
        return ""
    return (f"\nσ 점검(전 구간): 정규화 오차 |err|/σ 중앙값 {np.median(np.abs(e)):.2f}, |err| ≤ 1σ 비율 {np.mean(np.abs(e) <= 1):.2f}, "
            f"≤ 2σ {np.mean(np.abs(e) <= 2):.2f} (N={len(e)})\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("main", nargs="+", help="<seg>=<json>")
    ap.add_argument("--sens", nargs="*", default=[], help="<size>:<seg>=<json> (sensitivity, incl. the main 518x924 dumps)")
    a = ap.parse_args()
    parts, all_rows, sens = [], [], []
    for item in a.main:
        seg, path = item.split("=", 1)
        summ, rows = load(path)
        parts.append(seg_tables(seg, summ, rows))
        all_rows += rows
    parts.append(pooled_table(all_rows))
    parts.append(sigma_check(all_rows))
    for item in a.sens:
        size, rest = item.split(":", 1)
        seg, path = rest.split("=", 1)
        sens.append((size, seg, load(path)[1]))
    if sens:
        parts.append("\n### 민감도\n" + sens_table(sens))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
