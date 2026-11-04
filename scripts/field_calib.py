#!/usr/bin/env python3
"""On-site extrinsic calibration (camera / 4D radar -> Ouster os_sensor) from one stationary capture.

  # 0) edit configs/field_calib.yaml: topics + rough initial transforms (tape measure)
  # 1) all in one: check topics -> record 40 s -> solve -> open the frame viewer on the capture
  python3 scripts/field_calib.py run --name site_A
  # or step by step
  python3 scripts/field_calib.py record --name site_A            # needs a sourced ROS 2 (ros2 bag record)
  python3 scripts/field_calib.py solve  --bag data/calib_captures/site_A_<time>
  python3 scripts/field_calib.py view   --bag data/calib_captures/site_A_<time>
  # same rig moved to another mount (tripod -> excavator): keep the extrinsics, new ground from the LiDAR of a bag
  python3 scripts/field_calib.py reground --base configs/calib/field_site_A_<time>.json --bag <bag on the new mount>

Scene: machine stationary; 2-4 people 5-15 m in front of the camera (standing or walking slowly); 3+ corner
reflectors on tripods 5-25 m away, spread over the field of view, ~1 m clear around each.
Output: configs/calib/field_<name>_<time>.json (+ report inside), overlays/ next to the capture.
Method details: docs/07_field_calibration.html
"""
from __future__ import annotations

import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEG = ROOT / "models/yolo11s-seg.pt"


def ros_cmd(cmd: str) -> list[str]:
    """Run a ros2 command in a shell with ROS sourced (if the current env is not sourced already)."""
    if os.environ.get("ROS_DISTRO"):
        return ["bash", "-c", cmd]
    setup = sorted(Path("/opt/ros").glob("*/setup.bash"))
    if not setup:
        sys.exit("ROS 2 not found under /opt/ros; source your ROS 2 setup first")
    return ["bash", "-c", f"source {setup[-1]} && {cmd}"]


def check_topics(cfg) -> bool:
    print("checking topics (3 s) ...")
    out = subprocess.run(ros_cmd("ros2 topic list --no-daemon --spin-time 3"), capture_output=True, text=True).stdout.split()
    ok = True
    for k, tp in cfg["topics"].items():
        state = "ok" if tp in out else "MISSING"
        ok &= tp in out or k == "imu"
        print(f"  {k:12s} {tp:55s} {state}")
    return ok


def record(cfg, name: str) -> Path:
    rc = cfg["record"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / rc["out_dir"] / f"{name}_{stamp}"
    out.parent.mkdir(parents=True, exist_ok=True)
    topics = " ".join(list(cfg["topics"].values()) + ["/tf_static"])
    qos = out.parent / "tf_static_qos.yaml"                      # latched topic: subscribe transient_local
    qos.write_text("/tf_static:\n  durability: transient_local\n  reliability: reliable\n  history: keep_all\n")
    print(f"\nrecording {rc['duration_s']} s -> {out}\n  keep the machine still; people in view; reflectors set up")
    p = subprocess.Popen(ros_cmd(f"exec ros2 bag record -s mcap --qos-profile-overrides-path {qos} -o {out} {topics}"), start_new_session=True)
    try:
        for s in range(int(rc["duration_s"]), 0, -1):
            print(f"\r  {s:3d} s left ", end="", flush=True)
            time.sleep(1)
    finally:
        os.killpg(p.pid, signal.SIGINT)
        p.wait(timeout=30)
    print("\r  done            ")
    return out


def parse_windows(text: str | None) -> list[tuple[float, float]] | None:
    """'5-25,35-55,65-85' -> [(5, 25), (35, 55), (65, 85)] (seconds from the start of the bag)."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        a, b = part.split("-")
        if float(b) <= float(a):
            raise SystemExit(f"--windows: bad range {part!r}")
        out.append((float(a), float(b)))
    return out


def solve(cfg, bag: Path, out_json: Path | None, windows=None) -> Path:
    from rcmot.calib.field import solve as field_solve

    out_json = out_json or ROOT / "configs/calib" / f"field_{Path(bag).name}.json"
    field_solve(str(bag), cfg, str(out_json), str(SEG), overlay_dir=str(Path(bag) / "overlays"), windows=windows)
    return out_json


def reground(cfg, bag: Path, base: Path, out_json: Path | None, step: float) -> Path:
    from rcmot.calib.common import Topics
    from rcmot.calib.field import reground as field_reground

    out_json = out_json or ROOT / "configs/calib" / f"field_{Path(bag).name}.json"
    field_reground(str(bag), str(base), str(out_json), step=step, topics=Topics(**cfg["topics"]))
    return out_json


def view(bag: Path, calib: Path, config: str):
    subprocess.run([sys.executable, str(ROOT / "scripts/calib_viewer.py"), "--bag", str(bag), "--calib", str(calib),
                    "--config", config])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "record", "solve", "view", "check", "reground"])
    ap.add_argument("--config", default=str(ROOT / "configs/field_calib.yaml"))
    ap.add_argument("--name", default="field")
    ap.add_argument("--bag", help="capture to solve / view (solve, view); bag on the new mount (reground)")
    ap.add_argument("--base", help="reground: calib json whose camera / radar extrinsics are kept")
    ap.add_argument("--step", type=float, default=2.0, help="reground: one LiDAR scan every STEP s")
    ap.add_argument("--out", help="output calib json (default configs/calib/field_<capture>.json)")
    ap.add_argument("--duration", type=float, help="override record.duration_s")
    ap.add_argument("--no-view", action="store_true", help="run: do not open the viewer at the end")
    ap.add_argument("--windows", help="solve: explicit still windows in seconds from the bag start, e.g. "
                                      "'5-25,35-55,65-85' = one reflector carried to three spots (one window each); "
                                      "default: automatic stationary split")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.duration:
        cfg["record"]["duration_s"] = a.duration

    if a.cmd == "check":
        check_topics(cfg)
    elif a.cmd == "record":
        check_topics(cfg) or print("WARNING: some topics are missing")
        record(cfg, a.name)
    elif a.cmd == "solve":
        solve(cfg, Path(a.bag), Path(a.out) if a.out else None, parse_windows(a.windows))
    elif a.cmd == "reground":
        reground(cfg, Path(a.bag), Path(a.base), Path(a.out) if a.out else None, a.step)
    elif a.cmd == "view":
        out = Path(a.out) if a.out else ROOT / "configs/calib" / f"field_{Path(a.bag).name}.json"
        view(Path(a.bag), out, a.config)
    else:
        if not check_topics(cfg) and input("topics missing - record anyway? [y/N] ").lower() != "y":
            sys.exit(1)
        input("\nset up: machine still, 2-4 people 5-15 m in view, 3+ reflectors 5-25 m. Press Enter to record ...")
        bag = record(cfg, a.name)
        out = solve(cfg, bag, Path(a.out) if a.out else None)
        print(f"\nresult: {out}\noverlays: {bag / 'overlays'}")
        if not a.no_view:
            view(bag, out, a.config)


if __name__ == "__main__":
    main()
