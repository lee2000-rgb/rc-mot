"""Doc 22 figures: day timeline (bags by clock time, colour = mount) and per-bag strips (yaw rate / people count)."""
import json, glob, sys, datetime as D
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"); _FN = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
plt.rcParams.update({"font.family": _FN, "font.size": 10, "axes.edgecolor": "#d8dee4", "axes.labelcolor": "#656d76",
                     "xtick.color": "#656d76", "ytick.color": "#656d76", "axes.spines.top": False, "axes.spines.right": False})
O = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
MOUNT = json.loads(sys.argv[3])                       # bag -> mount key
COL = {"tripod": "#2a78d6", "exc_low": "#eb6834", "exc_high": "#1baf7a", "indoor": "#eda100"}
NAME = {"tripod": "삼각대", "exc_low": "굴착기 장착 A (정면 시야)", "exc_high": "굴착기 장착 B (버킷 내려다봄)", "indoor": "실내"}
SURF, INK, MUTED, GRID = "#fbfbfa", "#1f2328", "#656d76", "#e6eaee"
reps = {Path(f).stem: json.load(open(f)) for f in glob.glob(str(O / "hdx_*.json")) if not f.endswith("ground.json")}

# --- timeline (field bags only; the indoor 03:24 bag is listed in the table) ---
field = sorted([r for r in reps.values() if MOUNT[r["bag"]] != "indoor"], key=lambda r: r["start_epoch"])
fig, ax = plt.subplots(figsize=(12, 4.4), dpi=110); fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
kst = D.timezone(D.timedelta(hours=9))
for i, r in enumerate(field):
    t0 = D.datetime.fromtimestamp(r["start_epoch"], kst); h0 = t0.hour + t0.minute / 60 + t0.second / 3600
    w = max(r["duration_s"] / 3600, 0.025)          # very short bags stay visible; the label gives the length
    ax.barh(i, w, left=h0, height=0.62, color=COL[MOUNT[r["bag"]]], linewidth=0)
    ax.text(h0 + w + 0.03, i, f"{r['bag'].replace('hdx_20261026_', '').replace('hdx_cal_20261026_', 'cal ')}  {r['duration_s']/60:.1f}분",
            va="center", fontsize=9, color=INK)
ax.set_yticks([]); ax.invert_yaxis(); ax.set_xlim(10.5, 17.6)
ax.set_xticks(range(11, 18)); ax.set_xticklabels([f"{h}:00" for h in range(11, 18)]); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=COL[k], label=NAME[k]) for k in ("tripod", "exc_low", "exc_high")], loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3, frameon=False, fontsize=9)
ax.set_xlabel("2026-10-26 기록 시각 (KST, bag 첫 메시지 기준)")
fig.tight_layout(); fig.savefig(OUT / "timeline.png", facecolor=SURF); plt.close(fig)

# --- per-bag strips: two small multiples on a shared time axis (no dual axis) ---
for r in reps.values():
    yaw = r.get("motion", {}).get("yaw_series", []); ppl = r.get("people", {}).get("per_s", [])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 2.6), dpi=100, sharex=True); fig.patch.set_facecolor(SURF)
    for a in (a1, a2): a.set_facecolor(SURF); a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True)
    a1.plot(range(len(yaw)), yaw, color="#2a78d6", lw=2); a1.set_ylim(0, max(10, max(yaw or [0]) * 1.1))
    a1.set_title("선회 속도 |ω_z| (°/s, Ouster IMU 0.5 s 평균)", loc="left", fontsize=9.5, color=INK, pad=3)
    a2.step(range(len(ppl)), ppl, where="post", color="#eb6834", lw=2); a2.set_ylim(0, max(3, max(ppl or [0]) + 1))
    a2.set_title("영상 속 사람 수 (yolo11s, 1 Hz, conf 0.35)", loc="left", fontsize=9.5, color=INK, pad=3)
    a2.set_xlabel("bag 시작 후 [s]"); a2.set_xlim(0, max(len(yaw), len(ppl)))
    fig.tight_layout(h_pad=0.6); fig.savefig(OUT / f"{r['bag']}_strip.png", facecolor=SURF); plt.close(fig)
print("ok", len(reps))
