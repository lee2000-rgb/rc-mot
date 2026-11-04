#!/usr/bin/env python3
"""RC-MOT GUI: replay a rosbag2 (mcap / sqlite3, seekable) or run live on ROS 2 topics (docs/11).

  # open a profile (configs/datasets.yaml): a bag replay (datasets:) or a live ROS 2 source (live:); the profile
  # box in the GUI switches between all of them without restarting (bag <-> live included)
  python3 scripts/rcmot_app.py --profile xpanner_construction_field
  python3 scripts/rcmot_app.py --profile hdx_live_mid            # AGX: source env_agx.sh first

  # replay any bag (no ROS install needed)
  python3 scripts/rcmot_app.py --config configs/hdx_boryeong.yaml \
      --bag data/bags/hdx_boryeong_pg/20251125_4R_bosch_halt-short_LC \
      --calib configs/calib/hdx_boryeong_pg_day1125.json [--start 85]

  # live (Orin / field PC):  source /opt/ros/humble/setup.bash
  python3 scripts/rcmot_app.py --config configs/hdx_boryeong.yaml --live

Keys: Space play/pause | A / <- , D / -> one frame | Shift+<- / -> 5 s | Home start | End live
      [ / ] speed | timeline: click or drag to seek (cached part = instant) | wheel on BEV: zoom
"""
from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rcmot.alarm import Alarm  # noqa: E402
from rcmot.gui.playback import BagWorker, FrameCache  # noqa: E402
from rcmot.pipeline import build_pipeline  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def model_items(current: str | None) -> list:
    """Built-in detector sizes found in models/ (n/s/m/l/x seg + plain), plus the current weights."""
    items = []
    for name in ("yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg", "yolo11s", "yolo11x"):
        for ext in (".engine", ".pt"):
            p = ROOT / "models" / f"{name}{ext}"
            if p.exists():
                items.append((f"{name}{ext}", str(p)))
    if current and all(str(current) != p for _, p in items):
        items.insert(0, (Path(current).name, str(current)))
    return items


def attach_alarm(win, cfg: dict) -> Alarm:
    """One Alarm for the window's lifetime: the workers call decide() per frame, the viewer play() per shown frame."""
    al = cfg.get("alarm", {})
    win.alarm = Alarm(zones=al.get("zones", (10.0, 20.0, 30.0)), sounds=al.get("sounds"), hysteresis=al.get("hysteresis", 1.0),
                      root=ROOT, confirmed_only=al.get("confirmed_only", True),     # counting rule of docs/38 §10.4 by default
                      lost_grace_s=al.get("lost_grace_s", 0.5), radar_only_grace_s=al.get("radar_only_grace_s", 1.0))
    win.set_zones(al.get("zones", (10.0, 20.0, 30.0)))          # also the BEV rings / box colours
    if al.get("enable", False):
        win.btn_alarm.setChecked(True)
    return win.alarm


def apply_alarm_cfg(win, alarm: Alarm, cfg: dict) -> None:
    """Profile switch: zones, hysteresis and counting rule of the NEW profile (docs/19 §5: profile overrides switch without
    a restart). The on/off button and the sounds stay as the user left them."""
    al = cfg.get("alarm", {})
    alarm.zones = tuple(sorted(float(z) for z in al.get("zones", (10.0, 20.0, 30.0))))
    alarm.hyst = al.get("hysteresis", 1.0)
    alarm.confirmed_only = al.get("confirmed_only", True)
    alarm.lost_grace_s = al.get("lost_grace_s", 0.5)
    alarm.radar_only_grace_s = al.get("radar_only_grace_s", 1.0)
    win.set_zones(alarm.zones)


def make_publisher(cfg: dict, calib: RigCalib, meta: dict, *, enabled: bool, node=None, stats_fn=None):
    """ResultPublisher from cfg['ros'] (docs/20), or None when rclpy / rcmot_msgs are not importable."""
    r = cfg.get("ros", {})
    try:
        from rcmot.ros.publisher import ResultPublisher

        return ResultPublisher(calib, meta=meta, namespace=r.get("namespace", "/rcmot"), tf_parent=r.get("tf_parent", "radar"),
                               markers=bool(r.get("markers", False)), status_hz=float(r.get("status_hz", 1.0)),
                               enabled=enabled, node=node, stats_fn=stats_fn)
    except Exception as e:                                        # noqa: BLE001  no rclpy / rcmot_msgs on this machine
        print(f"[rcmot] ROS publishing unavailable: {e!r}", file=sys.stderr)
        return None


LIVE_PREFIX = "LIVE: "


def _have_rclpy() -> bool:
    try:
        import rclpy  # noqa: F401
        return True
    except Exception:
        return False


def load_presets(path: Path) -> dict:
    """Profiles usable on this machine: `datasets:` whose bag exists (kind bag) and, when rclpy is importable,
    `live:` entries (kind live). Both carry config / calib / optional `overrides` (cfg a.b: v) and `note`."""
    y = yaml.safe_load(open(path)) if path.exists() else {}
    out = {}
    for k, v in (y.get("datasets") or {}).items():
        if _abs(v["bag"]).exists():
            out[k] = dict(v, kind="bag")
    if _have_rclpy():
        for k, v in (y.get("live") or {}).items():
            out[k] = dict(v, kind="live")
    return out


def apply_overrides(cfg: dict, items) -> dict:
    """items: {"a.b": v} dict or ["a.b=v", ...] list; nested keys are created."""
    pairs = items.items() if isinstance(items, dict) else [kv.split("=", 1) for kv in (items or [])]
    for k, v in pairs:
        d = cfg
        for part in str(k).split(".")[:-1]:
            d = d.setdefault(part, {})
        d[str(k).split(".")[-1]] = yaml.safe_load(v) if isinstance(v, str) else v
    return cfg


class ReplaySession:
    """One opened recording: pipeline + cache + worker + player bound to the shared window."""

    kind = "bag"

    def __init__(self, win, bag: str, cfg: dict, calib: RigCalib, topics: dict, *, start: float, warmup: float,
                 ahead: float, cache_frames: int, weights: str | None, alarm=None, publish: bool | None = None,
                 meta: dict | None = None):
        from rcmot.gui.app import Player
        from rcmot.sensors.bag_source import bag_span, iterate_bag

        self.cfg = cfg
        self.pipe = build_pipeline(cfg, calib, ROOT, weights)
        self.cache = FrameCache(cache_frames)
        t0, t1 = bag_span(bag)
        depth = topics.get("depth") if cfg.get("work", {}).get("enable", False) else None   # stereo for the work monitor
        frames_from = lambda t: iterate_bag(bag, calib, topics["radar"], topics["image"], topics.get("imu"),  # noqa: E731
                                            start=t - t0, depth_topic=depth)
        self.worker = BagWorker(self.pipe, frames_from, self.cache, ahead_s=ahead, warmup_s=warmup, t_begin=t0)
        self.worker.alarm = alarm
        self.publisher = None if publish is None else make_publisher(cfg, calib, meta or {}, enabled=publish)
        if self.publisher is not None:
            self.worker.sinks.append(self.publisher.publish)
        g = cfg.get("gui", {})
        win.set_calib(calib, Path(bag).name, max_range=g.get("bev_range"), danger_m=g.get("danger_m"), zones=g.get("zones"))
        self.player = Player(win, self.cache, self.worker, live=False, t_range=(t0, t1))
        self.player.seek(t0 + start)
        self.worker.start()

    def stop(self):
        self.player.close()
        self.worker.stop()
        if self.publisher is not None:
            self.publisher.close()


class LiveSession:
    """Live ROS 2 source: pipeline + LiveWorker (rclpy node) + player in live mode, bound to the shared window."""

    kind = "live"

    def __init__(self, win, title: str, cfg: dict, calib: RigCalib, topics: dict, *, cache_frames: int,
                 weights: str | None, alarm=None, publish: bool | None = None, meta: dict | None = None):
        from rcmot.gui.app import Player
        from rcmot.gui.playback import LiveWorker

        self.cfg = cfg
        self.pipe = build_pipeline(cfg, calib, ROOT, weights)
        self.worker = LiveWorker(self.pipe, calib, topics, FrameCache(cache_frames))
        self.worker.alarm = alarm
        self.cache = self.worker.cache
        w = self.worker
        self.publisher = None if publish is None else make_publisher(
            cfg, calib, meta or {}, enabled=publish, node=w.node,
            stats_fn=lambda: {"cam_lag_s": w.cam_lag, "frames_dropped": w.n_drop})
        if self.publisher is not None:
            w.sinks.append(lambda res: self.publisher.publish(res, w.radar_frame_id))
        g = cfg.get("gui", {})
        win.set_calib(calib, f"LIVE {title}", max_range=g.get("bev_range"), danger_m=g.get("danger_m"), zones=g.get("zones"))
        self.player = Player(win, self.cache, self.worker, live=True)
        self.worker.start()

    def stop(self):
        self.player.close()
        self.worker.join_worker()                          # the in-flight step (and its sink calls) finishes first
        if self.publisher is not None:
            self.publisher.close()                         # before the worker destroys the node / shuts rclpy down
        self.worker.stop()                                 # destroys the node and shuts rclpy down (next session re-inits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bag", help="rosbag2 directory or .mcap file")
    src.add_argument("--profile", "--dataset", dest="profile", help="profile name from configs/datasets.yaml (bag or live)")
    src.add_argument("--live", action="store_true", help="subscribe the ROS 2 topics of --config (ad-hoc live profile)")
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--calib", help="calib json overriding the config (e.g. another recording session)")
    ap.add_argument("--radar-topic")
    ap.add_argument("--image-topic")
    ap.add_argument("--weights")
    ap.add_argument("--start", type=float, help="replay: initial position [s from bag start]")
    ap.add_argument("--warmup", type=float, default=1.5, help="replay: seconds processed before a seek target")
    ap.add_argument("--ahead", type=float, default=20.0, help="replay: max seconds processed ahead of the cursor")
    ap.add_argument("--cache-frames", type=int, default=3000, help="processed frames kept for scrubbing")
    ap.add_argument("--presets", default=str(ROOT / "configs/datasets.yaml"))
    pub = ap.add_mutually_exclusive_group()
    pub.add_argument("--publish", dest="publish", action="store_true", default=None,
                     help="publish results on ROS 2 (/rcmot/tracks, alarm, detections, status; docs/20); PUB button toggles at runtime")
    pub.add_argument("--no-publish", dest="publish", action="store_false", help="create the publisher but start with it off")
    ap.add_argument("--set", action="append", default=[], metavar="a.b=v",
                    help="config default applied to every opened profile before its own overrides, e.g. "
                         "detector.weights=models/yolo11s-seg.engine (--weights forces one detector for all)")
    a = ap.parse_args()

    presets = load_presets(Path(a.presets))
    if a.profile and a.profile not in presets:
        sys.exit(f"unknown / unavailable profile {a.profile!r}; available: {sorted(presets)}")

    def spec(name: str | None) -> dict:
        """Everything needed to open one source: kind, cfg, calib, topics, bag/start (bag) or title (live)."""
        if name is not None:
            p = presets[name]
            cfg = apply_overrides(yaml.safe_load(open(_abs(p["config"]))), a.set)   # --set: machine defaults (engines ...)
            apply_overrides(cfg, p.get("overrides"))                  # profile-level settings win (model, imgsz, zones ...)
            calib_path = p.get("calib") or str(_abs(p["config"]).parent / cfg["calib"])
            topics = dict(cfg["topics"], **(p.get("topics") or {}))
            d = dict(kind=p["kind"], name=name, cfg=cfg, calib=RigCalib.load(str(_abs(calib_path))), topics=topics,
                     calib_path=str(calib_path))
            if p["kind"] == "bag":
                d.update(bag=str(_abs(p["bag"])), start=float(p.get("start", 0.0)) if a.start is None else a.start)
            return d
        cfg = apply_overrides(yaml.safe_load(open(a.config)), a.set)
        calib_path = a.calib or str((Path(a.config).parent / cfg["calib"]).resolve())
        calib = RigCalib.load(calib_path)
        topics = dict(cfg["topics"])
        topics["radar"] = a.radar_topic or topics["radar"]
        topics["image"] = a.image_topic or topics["image"]
        if a.live:
            return dict(kind="live", name=Path(a.config).stem, cfg=cfg, calib=calib, topics=topics, calib_path=calib_path)
        return dict(kind="bag", name=Path(a.bag).name, cfg=cfg, calib=calib, topics=topics, bag=a.bag, start=a.start or 0.0,
                    calib_path=calib_path)

    from PyQt5 import QtCore, QtWidgets

    from rcmot.gui.viewer import Viewer

    app = QtWidgets.QApplication(sys.argv)
    first = spec(a.profile)
    g = first["cfg"].get("gui", {})
    win = Viewer(first["calib"], max_range=g.get("bev_range", 30.0), danger_m=g.get("danger_m", 5.0),
                 zones=tuple(g.get("zones", (5.0, 10.0, 15.0))), title=first["name"])
    state = {"session": None}
    alarm = attach_alarm(win, first["cfg"])
    publish = a.publish if a.publish is not None else (True if first["cfg"].get("ros", {}).get("publish") else None)

    def meta_of(d: dict) -> dict:
        return {"profile": d["name"], "detector": Path(a.weights or d["cfg"]["detector"]["weights"]).name,
                "calib": Path(d.get("calib_path", "")).name}
    labels = {k: (LIVE_PREFIX + k if v["kind"] == "live" else k) for k, v in presets.items()}   # name -> combo text
    if first["name"] not in presets:                                                             # ad-hoc --bag / --live
        labels[first["name"]] = (LIVE_PREFIX if first["kind"] == "live" else "") + first["name"]
    by_label = {v: k for k, v in labels.items()}
    notes = {labels[k]: v.get("note", "") for k, v in presets.items()}

    def open_session(label: str | None):
        name = first["name"] if label is None else by_label.get(label, label)
        d = first if name == first["name"] and name not in presets else spec(name)
        if state["session"] is not None:
            state["session"].stop()                       # join the old pipeline thread first (GPU, bag reader / rclpy)
            state["session"] = None
        alarm.reset()                                     # hysteresis state of the previous source
        apply_alarm_cfg(win, alarm, d["cfg"])             # zones / hysteresis / counting rule of this profile
        pub_on = None if publish is None else (win.btn_pub.isChecked() if win.btn_pub.isVisible() else publish)
        if d["kind"] == "live":
            state["session"] = LiveSession(win, d["name"], d["cfg"], d["calib"], d["topics"], cache_frames=a.cache_frames,
                                           weights=a.weights, alarm=alarm, publish=pub_on, meta=meta_of(d))
        else:
            state["session"] = ReplaySession(win, d["bag"], d["cfg"], d["calib"], d["topics"], start=d["start"], warmup=a.warmup,
                                             ahead=a.ahead, cache_frames=a.cache_frames, weights=a.weights, alarm=alarm,
                                             publish=pub_on, meta=meta_of(d))
        if win.chk_people.isChecked():                    # keep '사람만' across profile switches
            state["session"].pipe.set_only_classes({"worker"})
        p = state["session"].publisher
        win.set_publishing(None if p is None else p.enabled)
        cur = a.weights or str(_abs(d["cfg"]["detector"]["weights"]))
        win.set_models(model_items(cur), cur)
        if len(labels) > 1:
            win.set_datasets(list(labels.values()), labels[d["name"]], notes)

    # SIGTERM / SIGINT (HDX GUI "Stop" sends SIGTERM to the process group; docs/20): a clean Qt quit -> aboutToQuit
    # stops the worker, the publisher, rclpy and the sound loop instead of leaving a paplay orphan. Installed before the
    # first session (engine load ~15 s); a signal during that window is honoured by the tick once exec_() runs.
    quit_req = {"v": False}

    def on_signal(*_):
        quit_req["v"] = True
        app.quit()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    tick = QtCore.QTimer()                                        # lets Python run its signal handlers while Qt idles
    tick.timeout.connect(lambda: quit_req["v"] and app.quit())
    tick.start(200)

    win.open_dataset.connect(open_session)
    open_session(None)
    win.change_model.connect(lambda path: state["session"] and state["session"].pipe.request_detector(path))
    win.people_only.connect(lambda on: state["session"] and state["session"].pipe.set_only_classes({"worker"} if on else None))

    def set_publish(on: bool):
        s_ = state["session"]
        if s_ is not None and s_.publisher is not None:
            s_.publisher.set_enabled(on)

    win.publish_toggled.connect(set_publish)
    app.aboutToQuit.connect(alarm.stop)                           # kill the sound loop's player first (no respawn)
    app.aboutToQuit.connect(lambda: state["session"] and state["session"].stop())
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
