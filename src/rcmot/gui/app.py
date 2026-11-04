"""GUI controller: plays a FrameCache filled by a BagWorker (replay, seekable) or a LiveWorker (ROS 2 topics).

Replay: a play clock advances at real time x speed; the newest cached frame at or before the clock is shown.
When processing is slower than real time the clock waits at the end of the cache (status: "processing").
Seeking inside the cache is instant; outside it restarts the worker at (target - warm-up).
Live: follows the newest processed frame; pause / step / timeline browse the recent cache (DVR) while
processing continues; LIVE (End) jumps back to the newest frame.
"""
from __future__ import annotations

import time

from PyQt5 import QtCore

from rcmot.gui.playback import FrameCache


class Player(QtCore.QObject):
    def __init__(self, win, cache: FrameCache, worker, *, live: bool, t_range: tuple[float, float] | None = None):
        super().__init__()
        self.win, self.cache, self.worker, self.live = win, cache, worker, live
        self.t_range = t_range
        self.clock = t_range[0] if t_range else None           # timeline time being shown
        self.shown_t = None
        self.target = t_range[0] if t_range else None          # waiting for the worker to reach this time
        self.wall = time.time()
        self.follow = True
        self.proc = {"n": 0, "t": time.time(), "fps": 0.0, "len": 0, "gen": cache.gen}
        win.set_live_mode(live)
        win.timeline.seek.connect(self.seek)
        win.timeline.scrub.connect(self.scrub)
        win.step.connect(self.step)
        win.jump.connect(self.jump)
        win.go_live.connect(self.go_live)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(15)

    def close(self):
        """Detach from the window (another Player takes over, e.g. a different recording)."""
        self.timer.stop()
        for sig, fn in ((self.win.timeline.seek, self.seek), (self.win.timeline.scrub, self.scrub), (self.win.step, self.step),
                        (self.win.jump, self.jump), (self.win.go_live, self.go_live)):
            try:
                sig.disconnect(fn)
            except TypeError:
                pass

    # ------------------------------------------------------------------ navigation
    def seek(self, t: float):
        if self.live:
            span = self.cache.span()
            if span:
                self.follow = t >= span[1] - 0.05
                self.clock = min(max(t, span[0]), span[1])
                if not self.follow:
                    self.win.set_paused(True)
                self._show_at(self.clock)
            return
        self.clock = t
        if self.cache.index_at(t) is not None:
            self.target = None
            self._show_at(t)
        else:
            self.target = t
            self.worker.seek(t)
        self.worker.cursor_t = t

    def scrub(self, t: float):
        if self.cache.index_at(t) is not None:                  # preview cached frames while dragging
            self._show_at(t)

    def jump(self, dt: float):
        if self.clock is None:
            return
        t0 = self.cache.span()[0] if self.live and self.cache.span() else (self.t_range[0] if self.t_range else 0.0)
        self.seek(max(self.clock + dt, t0) if dt > -1e8 else t0)

    def step(self, d: int):
        self.win.set_paused(True)
        self.follow = False
        if self.shown_t is None:
            return
        i = self.cache.index_at(self.shown_t)
        e = None if i is None else self.cache.get(i + d)
        if e is not None:
            self.clock = e.t
            self._show(e)
        elif d > 0 and not self.live:                           # one past the cache end: wait for it
            self.target = self.shown_t + 1e-3
            self.clock = self.target
        self.worker.cursor_t = self.clock if self.clock is not None else 0.0

    def go_live(self):
        if self.live:
            self.follow = True
            self.win.set_paused(False)

    # ------------------------------------------------------------------ display
    def _show(self, e):
        self.shown_t = e.t
        self.win.show_frame(e.image(), e.res, self.proc["fps"], self._note())

    def _show_at(self, t):
        i = self.cache.index_at(t)
        if i is not None:
            self._show(self.cache.get(i))

    def _note(self) -> str:
        if self.live:
            span = self.cache.span()
            lag = (time.time() - span[1]) if span else float("nan")
            lag_s = f"wall-stamp lag {lag:.2f}s" if abs(lag) < 60 else "stamps not wall clock (bag play?)"
            return f"LIVE{'' if self.follow else ' (browsing history)'} | {lag_s} | {self.worker.health()}"
        span = self.cache.span()
        ahead = (span[1] - (self.clock or 0)) if span else 0.0
        st = "end of bag" if self.worker.done else ("buffered" if not self.worker.busy else "processing")
        err = f" | ERROR {self.worker.error}" if self.worker.error else ""
        return f"REPLAY | {st}, {ahead:.1f}s ahead | processing {self.proc['fps']:.1f} FPS{err}"

    def _proc_rate(self):
        p, now, n = self.proc, time.time(), len(self.cache)
        if self.cache.gen != p["gen"]:
            p["gen"], p["len"] = self.cache.gen, 0
        p["n"] += max(n - p["len"], 0)
        p["len"] = n
        if now - p["t"] > 1.0:
            p["fps"] = 0.5 * p["fps"] + 0.5 * p["n"] / (now - p["t"])
            p["n"], p["t"] = 0, now

    def tick(self):
        self._proc_rate()
        now, dt = time.time(), time.time() - self.wall
        self.wall = now
        span = self.cache.span()
        if self.live:
            if span and self.follow and not self.win.paused:
                self.clock = span[1]
            if span and self.clock is not None and (self.shown_t is None or abs(self.clock - self.shown_t) > 1e-6):
                self._show_at(self.clock)
            if span:
                self.win.set_timeline(span[0], span[1], self.shown_t or span[1], span,
                                      "● LIVE" if self.follow else f"-{span[1] - (self.shown_t or span[1]):.1f} s")
            elif int(now * 2) != int((now - dt) * 2):                 # no frame yet: show the topic health at 2 Hz
                self.win.status.setText(f"LIVE | waiting for data | {self.worker.health()}")
            return
        t0, t1 = self.t_range
        if self.target is not None:                            # after a seek: wait for the first frame >= target
            if span and span[1] >= self.target - 1e-6:
                e = self.cache.get(self.cache.find(self.target))
                if e is not None and e.t >= self.target - 0.2:
                    self.target, self.clock = None, e.t
                    self._show(e)
        elif not self.win.paused and span:
            self.clock = min(self.clock + dt * self.win.speed, span[1])   # real-time x speed, capped by processing
            i = self.cache.index_at(self.clock)
            if i is not None:
                e = self.cache.get(i)
                if e is not None and e.t != self.shown_t:
                    self._show(e)
            if self.worker.done and self.clock >= span[1]:
                self.win.set_paused(True)
        if self.clock is not None:
            self.worker.cursor_t = self.clock
            wait = " (seeking...)" if self.target is not None else ""
            self.win.set_timeline(t0, t1, self.shown_t if self.shown_t is not None else self.clock, span,
                                  f"{(self.shown_t or self.clock) - t0:6.1f} / {t1 - t0:.1f} s{wait}")
        if self.win.last is not None and self.target is None:
            self.win.note = self._note()
