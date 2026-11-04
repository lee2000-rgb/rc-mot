#!/usr/bin/env python3
"""Run / summarise the docs/21 evaluation suite (GT segments of scripts/eval_mot_gt.py) per tag.

  python3 scripts/eval_suite.py run  --tag e5_win06 [--set tracker.static_window_s=0.6 ...] [--segs conti,cf,val] [-j 2]
  python3 scripts/eval_suite.py show --tags v18_base,e5_win06 [--segs conti,cf,val]

Segments: conti = conti_swing_025/065/105 (swing), cf = cf_vib_100/160 (Arizona shake), val = the three Boryeong
regression segments (day_halt_short_085, val_day_haltmid_020, val_day_haltshort_020). `show` prints, per tag and
segment, the worker HOTA / IDF1 / IDsw / FP / FN, the motion metrics over ALL GT classes (st = GT-static objects:
shown speed p50, p90, fraction > 0.3 m/s, static flag; wkFlag = GT-walking objects wrongly frozen; standstill entry
delay), the far-worker recall (< 60 px boxes) and per class the static workers (stW) and static equipment (stE). The last run of each (tag, segment) wins.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {"conti": ["conti_swing_025", "conti_swing_065", "conti_swing_105"],
          "cf": ["cf_vib_100", "cf_vib_160"],
          "val": ["day_halt_short_085", "val_day_haltmid_020", "val_day_haltshort_020"]}
LOG = ROOT / "data/labels/eval_log.jsonl"


def segs_of(groups: str) -> list[str]:
    return [s for g in groups.split(",") for s in GROUPS[g]]


def run(a):
    def one(seg):
        cmd = [sys.executable, str(ROOT / "scripts/eval_mot_gt.py"), str(ROOT / "data/labels" / seg), "--tag", a.tag]
        for kv in a.set:
            cmd += ["--set", kv]
        r = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("worker")), r.stderr.strip()[-300:])
        return f"{seg:24s} {line}"
    with ThreadPoolExecutor(max_workers=max(1, min(a.j, 2))) as ex:       # never more than 2 at once (docs/21 §0)
        for out in ex.map(one, segs_of(a.segs)):
            print(out, flush=True)


def _f(v, fmt="{:.3f}"):
    return "-" if v is None else fmt.format(v)


def _trip(d):
    return "-" if not d else f"{d['p50']:.2f}/{d['p90']:.2f}/{d['frac_gt0.3']:.2f}"


def show(a):
    last = {}
    for ln in open(LOG):
        d = json.loads(ln)
        last[(d.get("tag"), d.get("seg"))] = d
    print(f"{'tag':14s} {'segment':22s} {'HOTA':>6s} {'IDF1':>6s} {'IDsw':>4s} {'FP':>4s} {'FN':>4s} | "
          f"{'st p50':>6s} {'st p90':>6s} {'st>0.3':>6s} {'stFlag':>6s} {'wkFlag':>6s} {'entry':>5s} | {'far rec':>7s} | "
          f"{'stW p50/p90/>.3':>16s} {'stE p50/p90/>.3':>16s}")
    for tag in a.tags.split(","):
        for seg in segs_of(a.segs):
            d = last.get((tag, seg))
            if d is None:
                print(f"{tag:14s} {seg:22s}  (not run)")
                continue
            w = d["results"].get("worker", {})
            mo = d.get("motion") or {}
            st, wk, en = mo.get("static", {}), mo.get("walk", {}), mo.get("static_entry", {})
            far = (d.get("recall_by_height") or {}).get("h0-60", {})
            print(f"{tag:14s} {seg:22s} {_f(w.get('HOTA')):>6s} {_f(w.get('IDF1')):>6s} {w.get('IDsw', '-'):>4} "
                  f"{w.get('FP', '-'):>4} {w.get('FN', '-'):>4} | {_f(st.get('p50'), '{:.2f}'):>6s} "
                  f"{_f(st.get('p90'), '{:.2f}'):>6s} {_f(st.get('frac_gt0.3'), '{:.2f}'):>6s} "
                  f"{_f(st.get('frac_static_flag'), '{:.2f}'):>6s} {_f(wk.get('frac_static_flag'), '{:.2f}'):>6s} "
                  f"{_f(en.get('delay_p50_s'), '{:.1f}'):>5s} | {_f(far.get('recall'), '{:.2f}'):>7s} | "
                  f"{_trip(mo.get('static_worker')):>16s} {_trip(mo.get('static_heavy_equipment')):>16s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--set", action="append", default=[])
    r.add_argument("--segs", default="conti,cf,val")
    r.add_argument("-j", type=int, default=2)
    s = sub.add_parser("show")
    s.add_argument("--tags", required=True)
    s.add_argument("--segs", default="conti,cf,val")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else show(a)


if __name__ == "__main__":
    main()
