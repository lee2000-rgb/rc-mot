"""Distance alarm for the PoC demo (docs/18): zone-specific warning sounds over the speaker.

Zones (default, PoC scenario 2026-10-26): danger <= 10 m, warning <= 20 m, caution <= 30 m, each with its own
wav (continuous / fast / slow beep). The level is the nearest displayed track (workers and machines, radial
distance on the ground); a zone is left only when the object is `hysteresis` metres beyond its boundary so
the sound does not flicker at a boundary. The chosen wav is looped by an external player (paplay / aplay /
ffplay, whichever exists) in a worker thread; changing the level kills the current loop and starts the next.
With `confirmed_only` only camera-confirmed tracks count (CONFIRMED, or LOST for at most `lost_grace_s`), and a
radar-only track only for `radar_only_grace_s` after the camera last saw it (docs/38 §10.4: a person the detector
loses for a moment -- crouching, partly hidden -- stays in the alarm while radar holds the track; long radar-only
tracks, which may follow another object, do not count; 0 = radar-only tracks never count).

Two halves (docs/20): `decide(tracks)` is the stateful decision (hysteresis), called by the processing worker for
every frame so the result is part of the FrameResult (published, recorded, reproducible on replay); `play(level)`
drives the sound and is called by the GUI for the frame being shown, so on replay the sound follows the display.
`update()` does both (live use / tests).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from rcmot.types import AlarmState, TrackState

LEVELS = ("none", "caution", "warning", "danger")


def zone_of(r: float | None, zones) -> int:
    """0 none, 1 caution, 2 warning, 3 danger for a horizontal distance and sorted (danger, warning, caution) zones."""
    if r is None:
        return 0
    for k, z in enumerate(zones):
        if r <= z:
            return 3 - k
    return 0


class Alarm:
    def __init__(self, zones=(10.0, 20.0, 30.0), sounds: dict | None = None, hysteresis: float = 1.0,
                 player: str | None = None, root: Path | None = None,
                 confirmed_only: bool = True, lost_grace_s: float = 0.5, radar_only_grace_s: float = 1.0):
        self.zones = tuple(sorted(float(z) for z in zones))          # (danger, warning, caution)
        root = Path(root) if root else Path(__file__).resolve().parents[2]
        sounds = sounds or {"danger": "assets/sounds/beep_800Hz_3_continuous.wav",
                            "warning": "assets/sounds/beep_800Hz_2_fast.wav",
                            "caution": "assets/sounds/beep_800Hz_1_slow.wav"}
        self.sounds = {k: str(root / v if not Path(v).is_absolute() else v) for k, v in sounds.items()}
        self.hyst = hysteresis
        self.confirmed_only = confirmed_only
        self.lost_grace_s = lost_grace_s
        self.radar_only_grace_s = radar_only_grace_s
        self.player = player or next((p for p in ("paplay", "aplay", "ffplay") if shutil.which(p)), None)
        self.enabled = False
        self.level = 0                                               # decision state (hysteresis): 0 none .. 3 danger
        self.nearest: float | None = None
        self.nearest_tid: int | None = None
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._want = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ state
    def set_zones(self, zones) -> None:
        self.zones = tuple(sorted(float(z) for z in zones))

    def set_enabled(self, on: bool) -> None:
        self.enabled = bool(on)
        if not on:
            self._set_level(0)

    def reset(self) -> None:
        """Forget the hysteresis state (pipeline reset / seek)."""
        self.level, self.nearest, self.nearest_tid = 0, None, None

    def counted(self, tracks) -> list:
        if not self.confirmed_only:
            return list(tracks)
        cam_max = max(self.lost_grace_s, self.radar_only_grace_s)
        return [t for t in tracks if t.state == TrackState.CONFIRMED
                or (t.state == TrackState.LOST and t.time_since_update <= self.lost_grace_s
                    # a radar-only track that misses one radar update turns LOST with a small time_since_update: it
                    # still needs a camera within cam_max (else the radar-only limit is bypassed; found 11-04 by the
                    # deck session in val_day_haltmid 46.98-47.11 s)
                    and getattr(t, "time_since_camera", 0.0) <= cam_max)
                or (t.state == TrackState.RADAR_ONLY
                    and getattr(t, "time_since_camera", float("inf")) <= self.radar_only_grace_s)]

    def decide(self, tracks) -> AlarmState:
        """Level of this frame from its tracks (stateful: hysteresis when leaving a zone). No sound."""
        best, tid = None, None
        for t in self.counted(tracks):
            r = float(np.linalg.norm(t.pos[:2]))
            if best is None or r < best:
                best, tid = r, t.tid
        self.nearest, self.nearest_tid = best, tid
        lvl = zone_of(best, self.zones)
        if lvl < self.level and best is not None:                    # hysteresis when leaving a zone
            if best <= self.zones[3 - self.level] + self.hyst:
                lvl = self.level
        self.level = lvl
        return AlarmState(lvl, tid, best, self.zones, self.enabled, self.confirmed_only)

    def play(self, level: int) -> None:
        """Sound for the frame being displayed (silent while disabled)."""
        self._set_level(int(level) if self.enabled else 0)

    def update(self, tracks) -> int:
        """decide + play for live use; returns the level (also when disabled, for the GUI)."""
        st = self.decide(tracks)
        self.play(st.level)
        return st.level

    # ------------------------------------------------------------------ playback
    def _set_level(self, lvl: int) -> None:
        with self._lock:
            self._want = lvl

    def _cmd(self, path: str) -> list[str]:
        if self.player == "ffplay":
            return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]
        return [self.player, path]

    def _loop(self) -> None:
        playing = 0
        while True:
            with self._lock:
                want = self._want
            if want != playing:
                if self._proc is not None and self._proc.poll() is None:
                    self._proc.kill()
                self._proc, playing = None, want
            if playing and self.player and (self._proc is None or self._proc.poll() is not None):
                path = self.sounds.get(LEVELS[playing])
                if path and Path(path).exists():
                    self._proc = subprocess.Popen(self._cmd(path), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    playing = 0
            time.sleep(0.02)

    def stop(self) -> None:
        self._set_level(0)
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
