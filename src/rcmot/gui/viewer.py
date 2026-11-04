"""RC-MOT GUI v2 (PyQt5 + OpenCV rendering + pyqtgraph 3D).

 +--------------------------------------------------------------------------------------------------+
 | [|<] [<] [play] [>] [speed] [LIVE]  ===== timeline (buffered / cursor, click-drag) =====  time   |
 | status: FPS | ego | sigma_v | counts | timings                                                    |
 +-----------------------------------------------+--------------------------------------------------+
 | CAMERA: corner boxes + label tags, fading     | [BEV] [3D] [작업] [재구성]  BEV: rings, zones, FOV,  |
 | position tails (comet), heading arrows,       |  footprints, gradient tails, velocity arrows,     |
 | radar dots, unclassified movers               |  predicted path, 2-sigma ellipses, radar, movers  |
 +-----------------------------------------------+--------------------------------------------------+
 | GALLERY: card per track (thumbnail, ID, class, speed + compass, range, state badge, source,      |
 |          6 s speed sparkline), nearest first, red frame inside the danger zone, click = focus     |
 +--------------------------------------------------------------------------------------------------+
"""
from __future__ import annotations

import os

import cv2
import numpy as np

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)                # opencv-python points Qt at its own plugins
import pyqtgraph as pg  # noqa: E402
import pyqtgraph.opengl as gl  # noqa: E402
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

from rcmot.types import TrackState  # noqa: E402
from rcmot.utils.calib import object_box  # noqa: E402

STATE_STYLE = {TrackState.CONFIRMED: ("TRACKING", (90, 200, 90)), TrackState.LOST: ("LOST", (40, 170, 255)),
               TrackState.RADAR_ONLY: ("RADAR ONLY", (230, 120, 200)), TrackState.TENTATIVE: ("NEW", (180, 180, 180))}
CLS_SHORT = {"worker": "WORKER", "heavy_equipment": "MACHINE"}
TAIL_S = 4.0

ZONE_COL = {3: (0, 0, 255), 2: (0, 140, 255), 1: (0, 215, 255)}   # BGR: danger red, warning orange, caution yellow
ZONE_NAME = {3: "위험", 2: "경고", 1: "주의"}
ZONE_CSS = {3: "#ff3030", 2: "#ff8c00", 1: "#ffd700"}


def zone_of(r: float, zones) -> int:
    """3 = inside the nearest zone (danger), 2, 1, 0 = beyond all; zones sorted ascending (danger, warning, caution)."""
    for k, z in enumerate(zones):
        if r <= z:
            return 3 - k
    return 0



def id_color(tid: int) -> tuple[int, int, int]:
    """BGR, well separated hues (golden ratio), bright."""
    h = (tid * 0.61803398875) % 1.0
    c = QtGui.QColor.fromHsvF(h, 0.80, 1.0)
    return c.blue(), c.green(), c.red()


def _bgr_to_pixmap(bgr, w=None, h=None):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    img = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
    pm = QtGui.QPixmap.fromImage(img.copy())
    if w and h:
        pm = pm.scaled(w, h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
    return pm


def _blend_layers(img, layers):
    """layers: list of (alpha, draw_fn(canvas)). Draw each alpha bucket on a copy and blend once."""
    for a, fns in layers:
        if not fns:
            continue
        ov = img.copy()
        for fn in fns:
            fn(ov)
        cv2.addWeighted(ov, a, img, 1 - a, 0, img)


def _dashed_rect(img, p1, p2, color, th, dash=10):
    x1, y1 = p1
    x2, y2 = p2
    for (a, b) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        L = int(np.hypot(b[0] - a[0], b[1] - a[1]))
        for s in range(0, L, 2 * dash):
            e = min(s + dash, L)
            pa = (int(a[0] + (b[0] - a[0]) * s / max(L, 1)), int(a[1] + (b[1] - a[1]) * s / max(L, 1)))
            pb = (int(a[0] + (b[0] - a[0]) * e / max(L, 1)), int(a[1] + (b[1] - a[1]) * e / max(L, 1)))
            cv2.line(img, pa, pb, color, th, cv2.LINE_AA)


def _corner_box(img, p1, p2, color, th=2):
    x1, y1 = p1
    x2, y2 = p2
    L = int(0.22 * min(x2 - x1, y2 - y1)) + 4
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * L, cy), color, th + 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * L), color, th + 1, cv2.LINE_AA)
    cv2.rectangle(img, p1, p2, color, 1, cv2.LINE_AA)


def _label(img, x, y, text, color, scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
    y = max(y, th + 8)
    x = min(max(x, 0), img.shape[1] - tw - 10)
    ov = img.copy()
    cv2.rectangle(ov, (x, y - th - 8), (x + tw + 10, y + 2), (25, 25, 25), -1)
    cv2.addWeighted(ov, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (x, y - th - 8), (x + 4, y + 2), color, -1)
    cv2.putText(img, text, (x + 7, y - 3), cv2.FONT_HERSHEY_DUPLEX, scale, (245, 245, 245), 1, cv2.LINE_AA)




def projected_box(calib, cls, pos):
    """Image box of an upright object of the class size standing at the 3D position (for predicted tracks)."""
    return object_box(calib, cls, pos)


# ---------------------------------------------------------------------------------------------- camera view
def render_camera(bgr, res, calib, *, focus=None, show_radar=True, show_tails=True, brighten=False, zones=None,
                  show_movers=True):
    from rcmot.calib.camera_lidar import enhance_dark

    img = enhance_dark(bgr) if brighten else bgr.copy()
    H, W = img.shape[:2]
    now = res.stamp
    if show_radar and res.radar_xyz is not None and len(res.radar_xyz):
        uv, front = calib.project(res.radar_xyz)
        rng = np.linalg.norm(res.radar_xyz[:, :2], axis=1)
        col = cv2.applyColorMap(np.uint8(np.clip(rng / 40 * 255, 0, 255))[:, None], cv2.COLORMAP_TURBO)[:, 0]
        dyn = res.radar_dynamic if res.radar_dynamic is not None else np.zeros(len(rng), bool)
        for (u, v), c, f, dy in zip(uv, col, front, dyn):
            if f and 0 <= u < W and 0 <= v < H:
                cv2.circle(img, (int(u), int(v)), 4 if dy else 2, tuple(int(x) for x in c), -1 if dy else 1, cv2.LINE_AA)

    buckets = {a: [] for a in (0.15, 0.3, 0.5, 0.75)}
    for t in res.tracks:
        col = id_color(t.tid)
        if show_tails and len(t.history) > 1:                      # fading comet tail, oldest faintest
            hp, ht = np.array(t.history), np.array(t.history_t) if t.history_t else None
            ground = hp.copy()
            ground[:, 2] = 0.0
            uv, fr = calib.project(ground)
            age = (ht[-1] - ht) if ht is not None else np.linspace(TAIL_S, 0, len(hp))
            for i in range(len(hp) - 1):
                if not (fr[i] and fr[i + 1]) or age[i] > TAIL_S:
                    continue
                k = 1 - age[i] / TAIL_S
                a = min(buckets, key=lambda b: abs(b - k * 0.75))
                p1, p2 = tuple(np.int32(uv[i])), tuple(np.int32(uv[i + 1]))
                th = int(1 + 5 * k)
                buckets[a].append(lambda o, p1=p1, p2=p2, th=th, col=col: cv2.line(o, p1, p2, col, th, cv2.LINE_AA))
    _blend_layers(img, sorted(buckets.items()))

    for m in res.movers if show_movers else ():                    # unclassified radar movers
        uv, fr = calib.project(m.pos[None])
        if fr[0] and 0 <= uv[0, 0] < W and 0 <= uv[0, 1] < H:
            c = (0, 140, 255)
            cv2.drawMarker(img, tuple(np.int32(uv[0])), c, cv2.MARKER_DIAMOND, 26, 2, cv2.LINE_AA)
            _label(img, int(uv[0, 0]) + 14, int(uv[0, 1]), f"? moving {abs(m.v_radial):.1f}m/s", c, 0.45)

    for t in sorted(res.tracks, key=lambda t: -np.linalg.norm(t.pos[:2])):   # far first, near on top
        box = t.box2d if t.time_since_update == 0 else projected_box(calib, t.cls, t.pos)   # predicted, not stale
        if box is None:
            continue
        col = id_color(t.tid)
        zk = zone_of(float(np.linalg.norm(t.pos[:2])), zones) if zones else 0
        if zk:                                                     # inside an alarm zone: zone colour, thicker
            col = ZONE_COL[zk]
        name, scol = STATE_STYLE[t.state]
        x1, y1, x2, y2 = (int(v) for v in np.clip(box, [-W, -H, -W, -H], [2 * W, 2 * H, 2 * W, 2 * H]))
        th = 3 if (focus == t.tid or zk == 3) else 2
        if t.state == TrackState.CONFIRMED and t.time_since_update == 0:
            _corner_box(img, (x1, y1), (x2, y2), col, th)
        elif t.state == TrackState.RADAR_ONLY:
            _dashed_rect(img, (x1, y1), (x2, y2), (230, 120, 200), th, 5)
        else:
            _dashed_rect(img, (x1, y1), (x2, y2), col, th - 1, 10)
        g0 = np.r_[t.pos[:2], 0.0]
        sp = shown_speed(t)
        if sp > MOVE_MIN:                                          # heading arrow: 1 s ahead on the ground
            uv, fr = calib.project(np.stack([g0, g0 + np.r_[t.vel[:2], 0.0] * 1.0]))
            if fr.all():
                cv2.arrowedLine(img, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), (0, 0, 0), 6, cv2.LINE_AA, 0, 0.3)
                cv2.arrowedLine(img, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), col, 3, cv2.LINE_AA, 0, 0.3)
        rng = float(np.linalg.norm(t.pos[:2]))
        tag = f"ID{t.tid} {CLS_SHORT.get(t.cls, t.cls)}  {sp:.1f}m/s  {rng:.1f}m"
        if t.state != TrackState.CONFIRMED:
            tag += f"  [{name}]"
        _label(img, x1, y1 - 4, tag, col if t.state == TrackState.CONFIRMED else scol)
    return img


# ------------------------------------------------------------------------------------------------- BEV view
def render_bev(res, *, size=820, rng_m=30.0, focus=None, zones=(5.0, 10.0, 15.0), fov_deg=(-36, 36), show_radar=True,
               ego_len=6.0, ego_wid=2.6, world_mode=False, show_movers=True):
    """world_mode (docs/24): draw in the world frame of res.world (fixed at the start of the processed segment, north-up
    = the heading at that moment), centred on the swing axis (or the sensor): the machine turns and moves, the scene
    stays; grid lines are world-fixed, the sensor / axis trajectory is drawn."""
    img = np.full((size, size, 3), (28, 26, 24), np.uint8)
    s = size / (2 * rng_m)
    W = res.world if world_mode else None
    oy = size // 2 if W is not None else int(size * 0.80)          # ego a bit below the centre (look ahead)
    if W is not None:
        RW, ow = W.R(), W.origin
        cen = W.axis if W.axis is not None else W.origin

    def wto(q):                                                    # world xy -> pixel
        q = np.atleast_2d(q)[:, :2] - cen
        return np.c_[size / 2 - q[:, 1] * s, oy - q[:, 0] * s].astype(np.int32)

    def to(p):
        p = np.atleast_2d(p)
        if W is not None:
            return wto(p[:, :2] @ RW.T + ow)
        return np.c_[size / 2 - p[:, 1] * s, oy - p[:, 0] * s].astype(np.int32)

    ego_px = tuple(to(np.zeros(3))[0])
    if W is not None:                                              # world-fixed grid (motion visible when translating)
        g0 = np.floor((cen - rng_m * 1.5) / 5.0) * 5.0
        for k in range(int(rng_m * 3 / 5.0) + 2):
            for axis_i in (0, 1):
                v = g0[axis_i] + 5.0 * k
                e = np.array([[v, cen[1] - rng_m * 2], [v, cen[1] + rng_m * 2]]) if axis_i == 0 else \
                    np.array([[cen[0] - rng_m * 2, v], [cen[0] + rng_m * 2, v]])
                q = wto(e)
                cv2.line(img, tuple(q[0]), tuple(q[1]), (44, 42, 40), 1, cv2.LINE_AA)
    zc = [ZONE_COL[3], ZONE_COL[2], ZONE_COL[1]]
    ov = img.copy()
    for z, c in reversed(list(zip(zones, zc))):
        cv2.circle(ov, ego_px, int(z * s), c, -1, cv2.LINE_AA)
    cv2.addWeighted(ov, 0.12, img, 0.88, 0, img)
    for z, c in zip(zones, zc):                                    # zone boundaries
        cv2.circle(img, ego_px, int(z * s), tuple(int(v * 0.6) for v in c), 1, cv2.LINE_AA)
    for r in range(5, int(rng_m * 1.6) + 1, 5):
        cv2.circle(img, ego_px, int(r * s), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.putText(img, f"{r} m", (ego_px[0] + 4, ego_px[1] - int(r * s) + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130, 130, 130), 1)
    for a in fov_deg:
        e = np.radians(a)
        cv2.line(img, ego_px, tuple(to(np.array([np.cos(e), np.sin(e), 0]) * rng_m * 2)[0]), (90, 90, 40), 1, cv2.LINE_AA)
    if W is not None:
        tr = W.trail()
        if len(tr) > 1:                                            # sensor (level-frame origin) path
            cv2.polylines(img, [wto(tr)], False, (150, 120, 60), 2, cv2.LINE_AA)
        at = W.axis_trail()
        if at is not None and len(at) > 1:                         # swing-axis path (= the undercarriage when driving)
            cv2.polylines(img, [wto(at)], False, (60, 160, 230), 2, cv2.LINE_AA)
    if W is not None and W.lever is not None:                      # excavator: upper structure turning about the axis
        ax_l = -W.lever                                            # axis in the level frame
        L = np.linalg.norm(W.lever)
        body = np.array([[ax_l[0] + max(W.lever[0], 0) + 0.6, ax_l[1] - 1.5], [ax_l[0] + max(W.lever[0], 0) + 0.6, ax_l[1] + 1.5],
                         [ax_l[0] - 2.6, ax_l[1] + 1.5], [ax_l[0] - 2.6, ax_l[1] - 1.5]])
        poly = to(np.c_[body, np.zeros(4)])
        cv2.fillPoly(img, [poly], (70, 70, 70), cv2.LINE_AA)
        cv2.polylines(img, [poly], True, (200, 200, 200), 1, cv2.LINE_AA)
        axp = tuple(wto(W.axis)[0])
        cv2.circle(img, axp, max(int(L * s), 3), (110, 110, 110), 1, cv2.LINE_AA)        # sensor's swing circle
        cv2.line(img, axp, ego_px, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(img, axp, (60, 160, 230), cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)
        cv2.circle(img, ego_px, 5, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        ego = np.array([[ego_len * 0.25, -ego_wid / 2, 0], [ego_len * 0.25, ego_wid / 2, 0], [-ego_len * 0.75, ego_wid / 2, 0],
                        [-ego_len * 0.75, -ego_wid / 2, 0]])
        cv2.fillPoly(img, [to(ego)], (80, 80, 80), cv2.LINE_AA)
        cv2.polylines(img, [to(ego)], True, (200, 200, 200), 1, cv2.LINE_AA)
    ev = res.ego_vel[:2]
    if np.linalg.norm(ev) > 0.1:
        cv2.arrowedLine(img, ego_px, tuple(to(np.r_[ev * 2.0, 0])[0]), (200, 200, 200), 2, cv2.LINE_AA, 0, 0.25)

    if show_radar and res.radar_xyz is not None and len(res.radar_xyz):
        q = to(res.radar_xyz)
        dyn = res.radar_dynamic if res.radar_dynamic is not None else np.zeros(len(q), bool)
        vc = res.radar_vcomp if res.radar_vcomp is not None else np.zeros(len(q))
        for (u, v), d, vv in zip(q, dyn, vc):
            if 0 <= u < size and 0 <= v < size:
                if d:
                    c = (60, 60, 255) if vv > 0 else (255, 140, 60)
                    cv2.circle(img, (int(u), int(v)), 3, c, -1, cv2.LINE_AA)
                else:
                    img[v, u] = (120, 120, 120)

    for m in res.movers if show_movers else ():
        c = tuple(to(m.pos)[0])
        cv2.drawMarker(img, c, (0, 140, 255), cv2.MARKER_DIAMOND, 16, 2, cv2.LINE_AA)

    for t in res.tracks:
        col = id_color(t.tid)
        zk = zone_of(float(np.linalg.norm(t.pos[:2])), zones)
        if zk:
            col = ZONE_COL[zk]
        hp = np.array(t.history) if len(t.history) else t.pos[None]
        ht = np.array(t.history_t) if t.history_t else np.zeros(len(hp))
        q = to(hp)
        for i in range(len(q) - 1):                                # gradient tail: thickness + fade with age
            age = ht[-1] - ht[i]
            if age > TAIL_S:
                continue
            k = 1 - age / TAIL_S
            c = tuple(int(x * (0.25 + 0.75 * k) + 28 * (1 - k)) for x in col)
            cv2.line(img, tuple(q[i]), tuple(q[i + 1]), c, int(1 + 4 * k), cv2.LINE_AA)
        p = t.pos
        sp = shown_speed(t)
        if t.pos_cov is not None:                                  # 2-sigma position ellipse
            C2 = t.pos_cov[:2, :2] if W is None else RW @ t.pos_cov[:2, :2] @ RW.T
            w, V = np.linalg.eigh(C2)
            ang = np.degrees(np.arctan2(V[0, 1], V[1, 1]))
            axes = (max(int(2 * np.sqrt(max(w[1], 0)) * s), 2), max(int(2 * np.sqrt(max(w[0], 0)) * s), 2))
            cv2.ellipse(img, tuple(to(p)[0]), axes, -ang, 0, 360, tuple(int(0.5 * x) for x in col), 1, cv2.LINE_AA)
        if t.cls == "worker":
            cv2.circle(img, tuple(to(p)[0]), max(int(0.4 * s), 5), col, -1 if t.state == TrackState.CONFIRMED else 2, cv2.LINE_AA)
        else:                                                      # oriented machine footprint
            hd = np.arctan2(t.vel[1], t.vel[0]) if sp > MOVE_MIN else 0.0
            c_, s_ = np.cos(hd), np.sin(hd)
            L, Wd = 5.0, 2.6
            corners = np.array([[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]]) @ np.array([[c_, s_], [-s_, c_]])
            poly = to(np.c_[corners + p[:2], np.zeros(4)])
            cv2.polylines(img, [poly], True, col, 2, cv2.LINE_AA)
        if sp > MOVE_MIN:                                          # velocity arrow (2 s) + dashed CV prediction (3 s)
            for k in range(3):
                a, b = p + np.r_[t.vel[:2], 0] * k, p + np.r_[t.vel[:2], 0] * (k + 0.5)
                cv2.line(img, tuple(to(a)[0]), tuple(to(b)[0]), tuple(int(0.6 * x) for x in col), 1, cv2.LINE_AA)
            cv2.arrowedLine(img, tuple(to(p)[0]), tuple(to(p + np.r_[t.vel[:2], 0] * 2.0)[0]), col, 2, cv2.LINE_AA, 0, 0.25)
        u, v = to(p)[0]
        txt = f"{t.tid}"
        if focus == t.tid:
            cv2.circle(img, (u, v), max(int(1.2 * s), 14), (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, txt, (u + 9, v - 9), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (u + 9, v - 9), cv2.FONT_HERSHEY_DUPLEX, 0.6, col, 1, cv2.LINE_AA)
    cv2.putText(img, f"BEV {2 * rng_m:.0f} m  (wheel: zoom)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
    if W is not None:
        mode = "swing-axis model" if W.lever is not None else "Doppler dead reckoning"
        txt = f"WORLD (start of segment = up), heading {np.degrees(W.theta):+.0f} deg, {mode}" + ("  [still]" if W.still else "")
        cv2.putText(img, txt, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 200, 230), 1, cv2.LINE_AA)
        if W.axis is not None:
            d = np.linalg.norm(W.axis - W.axis_trail()[0]) if W.axis_trail() is not None and len(W.axis_trail()) else 0.0
            cv2.putText(img, f"axis moved {d:.1f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 160, 230), 1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------------------------------------------ timeline
MOVE_MIN = 0.3   # [m/s] below this a track is drawn as standing (no arrow, speed 0); docs/16


def shown_speed(t) -> float:
    """Displayed speed: 0 while the tracker holds the object in standstill mode."""
    return 0.0 if getattr(t, "static", False) else float(np.linalg.norm(t.vel[:2]))


class Timeline(QtWidgets.QWidget):
    """Seek bar: full time range, processed (cached) span, cursor; click / drag to scrub, release to seek."""
    seek = QtCore.pyqtSignal(float)
    scrub = QtCore.pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(30)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.t0, self.t1, self.cur, self.buf, self.drag = 0.0, 1.0, 0.0, None, None
        self.live = False

    def set_state(self, t0, t1, cur, buf):
        self.t0, self.t1, self.cur, self.buf = t0, max(t1, t0 + 1e-3), cur, buf
        self.update()

    def _x(self, t):
        return 8 + (self.width() - 16) * (t - self.t0) / (self.t1 - self.t0)

    def _t(self, x):
        return float(np.clip(self.t0 + (x - 8) / max(self.width() - 16, 1) * (self.t1 - self.t0), self.t0, self.t1))

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        h, y = self.height(), self.height() // 2 - 4
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(50, 52, 56))
        qp.drawRoundedRect(QtCore.QRectF(8, y, self.width() - 16, 8), 4, 4)
        if self.buf:
            qp.setBrush(QtGui.QColor(70, 110, 150))
            a, b = self._x(self.buf[0]), self._x(self.buf[1])
            qp.drawRoundedRect(QtCore.QRectF(a, y, max(b - a, 2), 8), 4, 4)
        span = self.t1 - self.t0                                       # tick marks every 1/2/5/10/30/60 s
        step = next(s for s in (1, 2, 5, 10, 30, 60, 120, 300, 1e9) if span / s <= 12)
        qp.setPen(QtGui.QColor(120, 120, 120))
        qp.setFont(QtGui.QFont("Sans", 7))
        k = 0
        while k * step <= span + 1e-6:
            x = self._x(self.t0 + k * step)
            qp.drawLine(QtCore.QPointF(x, y + 9), QtCore.QPointF(x, y + 12))
            lab = f"-{span - k * step:.0f}s" if self.live else f"{k * step:.0f}s"
            qp.drawText(QtCore.QPointF(x + 2, h - 1), lab)
            k += 1
        cur = self.drag if self.drag is not None else self.cur
        x = self._x(cur)
        qp.setPen(QtCore.Qt.NoPen)
        qp.setBrush(QtGui.QColor(255, 80, 80) if self.live else QtGui.QColor(240, 240, 240))
        qp.drawEllipse(QtCore.QPointF(x, y + 4), 7, 7)
        if self.drag is not None:
            qp.setPen(QtGui.QColor(255, 255, 255))
            qp.setFont(QtGui.QFont("Sans", 8, QtGui.QFont.Bold))
            qp.drawText(QtCore.QPointF(min(x + 10, self.width() - 60), y - 2), f"{self.drag - self.t0:.1f} s")

    def mousePressEvent(self, e):
        self.drag = self._t(e.x())
        self.scrub.emit(self.drag)
        self.update()

    def mouseMoveEvent(self, e):
        if self.drag is not None:
            self.drag = self._t(e.x())
            self.scrub.emit(self.drag)
            self.update()

    def mouseReleaseEvent(self, e):
        if self.drag is not None:
            t, self.drag = self._t(e.x()), None
            self.seek.emit(t)
            self.update()


# ------------------------------------------------------------------------------------------------ gallery
class Sparkline(QtWidgets.QWidget):
    def __init__(self, color):
        super().__init__()
        self.setFixedHeight(26)
        self.vals, self.color = [], QtGui.QColor(*color[::-1])

    def set(self, vals):
        self.vals = list(vals)[-60:]
        self.update()

    def paintEvent(self, _):
        if len(self.vals) < 2:
            return
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        vmax = max(max(self.vals), 1.0)
        pts = [QtCore.QPointF(i * (w - 1) / (len(self.vals) - 1), h - 2 - (h - 4) * v / vmax) for i, v in enumerate(self.vals)]
        qp.setPen(QtGui.QPen(self.color, 1.6))
        qp.drawPolyline(QtGui.QPolygonF(pts))
        qp.setPen(QtGui.QColor(150, 150, 150))
        qp.setFont(QtGui.QFont("Sans", 7))
        qp.drawText(2, 9, f"{vmax:.1f}")


class Compass(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(30, 30)
        self.ang, self.on = 0.0, False

    def set(self, vx, vy):
        self.on = np.hypot(vx, vy) > 0.2
        self.ang = np.degrees(np.arctan2(-vy, vx))                 # screen: forward = up
        self.update()

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90), 1))
        qp.drawEllipse(2, 2, 26, 26)
        if not self.on:
            qp.setBrush(QtGui.QColor(150, 150, 150))
            qp.drawEllipse(12, 12, 6, 6)
            return
        qp.translate(15, 15)
        qp.rotate(self.ang - 90 + 180)
        qp.setBrush(QtGui.QColor(240, 240, 240))
        qp.setPen(QtCore.Qt.NoPen)
        qp.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(0, -11), QtCore.QPointF(5, 6), QtCore.QPointF(0, 2), QtCore.QPointF(-5, 6)]))


class Card(QtWidgets.QFrame):
    clicked = QtCore.pyqtSignal(int)

    def __init__(self, tid):
        super().__init__()
        self.tid = tid
        self.setFixedWidth(178)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(5, 5, 5, 5)
        lay.setSpacing(2)
        self.head = QtWidgets.QLabel()
        self.head.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.head)
        mid = QtWidgets.QHBoxLayout()
        self.pic = QtWidgets.QLabel()
        self.pic.setFixedSize(84, 112)
        self.pic.setAlignment(QtCore.Qt.AlignCenter)
        self.pic.setStyleSheet("background:#111;border-radius:4px")
        mid.addWidget(self.pic)
        right = QtWidgets.QVBoxLayout()
        self.speed = QtWidgets.QLabel()
        self.compass = Compass()
        right.addWidget(self.speed)
        right.addWidget(self.compass, 0, QtCore.Qt.AlignHCenter)
        self.info = QtWidgets.QLabel()
        self.info.setStyleSheet("color:#bbb;font-size:10px")
        right.addWidget(self.info)
        right.addStretch(1)
        mid.addLayout(right)
        lay.addLayout(mid)
        self.badge = QtWidgets.QLabel()
        self.badge.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.badge)
        col = id_color(tid)
        self.spark = Sparkline(col)
        lay.addWidget(self.spark)

    def mousePressEvent(self, e):
        self.clicked.emit(self.tid)

    def set(self, t, now, danger, focus, speeds):
        b, g, r = id_color(t.tid)
        name, (sb, sg, sr) = STATE_STYLE[t.state]
        zk = danger if isinstance(danger, int) and not isinstance(danger, bool) else (3 if danger else 0)
        border = ZONE_CSS[zk] if zk else ("#ffffff" if focus else f"rgb({r},{g},{b})")
        style = "dashed" if t.state != TrackState.CONFIRMED else "solid"
        self.setStyleSheet(f"Card{{background:#1e1f22;border:{3 if zk or focus else 2}px {style} {border};border-radius:8px}}"
                           "QLabel{color:#eee;border:none}")
        self.head.setText(f"<span style='background:rgb({r},{g},{b});color:#000;font-weight:700;padding:1px 6px;"
                          f"border-radius:3px'>ID {t.tid}</span> <b>{CLS_SHORT.get(t.cls, t.cls)}</b>")
        if t.thumbnail is not None and t.thumbnail.size:
            from rcmot.calib.camera_lidar import enhance_dark
            self.pic.setPixmap(_bgr_to_pixmap(enhance_dark(t.thumbnail), 84, 112))
        sp = shown_speed(t)
        self.speed.setText(f"<span style='font-size:20px;font-weight:700'>{sp:.1f}</span><span style='color:#aaa'> m/s</span>")
        self.compass.set(t.vel[0], t.vel[1])
        self.info.setText(f"r {np.linalg.norm(t.pos[:2]):.1f} m<br>age {t.age:.0f} s<br>{t.source or '-'}"
                          + (f"<br>radar pts {t.n_radar}" if t.n_radar else ""))
        extra = f" {t.time_since_update:.1f}s" if t.state != TrackState.CONFIRMED else ""
        self.badge.setText(f"<span style='background:rgb({sr},{sg},{sb});color:#000;font-weight:700;"
                           f"padding:1px 8px;border-radius:7px;font-size:10px'>{name}{extra}</span>")
        self.spark.set(speeds)


class Gallery(QtWidgets.QScrollArea):
    focus = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFixedHeight(232)
        self.setStyleSheet("QScrollArea{background:#141516;border:none}")
        self.inner = QtWidgets.QWidget()
        self.inner.setStyleSheet("background:#141516")
        self.lay = QtWidgets.QHBoxLayout(self.inner)
        self.lay.setAlignment(QtCore.Qt.AlignLeft)
        self.setWidget(self.inner)
        self.cards: dict[int, Card] = {}
        self.speeds: dict[int, list] = {}

    def update_tracks(self, tracks, now, danger_m, focus, zones=None):
        live = {t.tid for t in tracks}
        for tid in list(self.cards):
            if tid not in live:
                c = self.cards.pop(tid)
                c.setParent(None)
                c.deleteLater()
        for t in tracks:
            hist = self.speeds.setdefault(t.tid, {})
            hist[round(now, 3)] = shown_speed(t)
            if len(hist) > 400:
                for k in sorted(hist)[:-300]:
                    del hist[k]
            if t.tid not in self.cards:
                c = Card(t.tid)
                c.clicked.connect(self.focus.emit)
                self.cards[t.tid] = c
        order = sorted(tracks, key=lambda t: np.linalg.norm(t.pos[:2]))
        for i, t in enumerate(order):
            c = self.cards[t.tid]
            if self.lay.indexOf(c) != i:
                self.lay.insertWidget(i, c)
            sp = [v for k, v in sorted(self.speeds[t.tid].items()) if now - 6.0 <= k <= now]
            r_ = float(np.linalg.norm(t.pos[:2]))
            c.set(t, now, zone_of(r_, zones) if zones else (r_ < danger_m), focus == t.tid, sp)


# ------------------------------------------------------------------------------------------------ 3D view
class View3D(gl.GLViewWidget):
    def __init__(self, max_range=40.0):
        super().__init__()
        self.setBackgroundColor((24, 24, 26))
        self.setCameraPosition(distance=max_range * 0.9, elevation=32, azimuth=-180)
        self.opts["center"] = pg.Vector(max_range * 0.35, 0, 0)
        grid = gl.GLGridItem()
        grid.setSize(2 * max_range, 2 * max_range)
        grid.setSpacing(5, 5)
        self.addItem(grid)
        for rr in (10, 20, 30):
            a = np.linspace(0, 2 * np.pi, 120)
            self.addItem(gl.GLLinePlotItem(pos=np.c_[rr * np.cos(a), rr * np.sin(a), np.zeros_like(a)],
                                           color=(0.45, 0.45, 0.45, 0.6), width=1))
        self.addItem(gl.GLBoxItem(size=pg.Vector(6, 2.6, 2.5), color=(200, 200, 200, 160)))
        self.radar = gl.GLScatterPlotItem(size=3, pxMode=True)
        self.addItem(self.radar)
        self.dyn = []

    def show(self, res):
        for it in self.dyn:
            self.removeItem(it)
        self.dyn = []
        if res.radar_xyz is not None and len(res.radar_xyz):
            col = np.tile([0.6, 0.6, 0.6, 0.5], (len(res.radar_xyz), 1))
            if res.radar_dynamic is not None:
                col[res.radar_dynamic] = [1.0, 0.35, 0.2, 1.0]
            self.radar.setData(pos=res.radar_xyz, color=col)
        for t in res.tracks:
            b, g, r = (x / 255 for x in id_color(t.tid))
            p = t.pos
            size = (0.7, 0.7, 1.8) if t.cls == "worker" else (5.0, 2.6, 3.0)
            alpha = {TrackState.LOST: 0.4, TrackState.RADAR_ONLY: 0.7}.get(t.state, 1.0)
            box = gl.GLBoxItem(size=pg.Vector(*size), color=(int(r * 255), int(g * 255), int(b * 255), int(alpha * 255)))
            box.translate(p[0] - size[0] / 2, p[1] - size[1] / 2, 0.0)
            items = [box]
            if len(t.history) > 1:
                hp = np.array(t.history).copy()
                hp[:, 2] = 0.05
                k = np.linspace(0.05, 1.0, len(hp))[:, None]
                items.append(gl.GLLinePlotItem(pos=hp, color=np.c_[np.tile([r, g, b], (len(hp), 1)), k], width=3, antialias=True))
            if shown_speed(t) > MOVE_MIN:
                tip = p + np.r_[t.vel[:2], 0] * 2.0
                items.append(gl.GLLinePlotItem(pos=np.array([p, tip]), color=(r, g, b, 1.0), width=4, antialias=True))
            items.append(gl.GLTextItem(pos=p + np.array([0, 0, size[2] + 0.4]), text=f"ID{t.tid} {shown_speed(t):.1f}m/s",
                                       color=QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))))
            for it in items:
                self.addItem(it)
                self.dyn.append(it)


# ------------------------------------------------------------------------------------------------ work monitor (docs/24)
H_LO, H_HI = -0.3, 2.0                                             # fixed colour scale [m] (comparable over time)
_HCOL = np.array([[58, 43, 29], [214, 120, 42], [255, 228, 207]], float)   # BGR: dark navy -> #2a78d6 -> light (one hue)


def _height_lut() -> np.ndarray:
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, [0, 0.55, 1], _HCOL[:, c]) for c in range(3)], 1).astype(np.uint8)


HEIGHT_LUT = _height_lut()
PILE_BGR = (52, 104, 235)                                          # pile outline (#eb6834)


def render_work(res, *, size=640, half=18.0, pile_thr=0.2):
    """Height map of res.work.snapshot around the swing axis (world frame, north-up = segment start), with the machine,
    its trail, the pile outline and the workers; a colour bar on the right."""
    img = np.full((size, size + 70, 3), (28, 26, 24), np.uint8)
    W, ws = res.world, res.work
    if W is None or ws is None or ws.snapshot is None:
        cv2.putText(img, "work monitor off (config work.enable) or no data yet", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (170, 170, 170), 1, cv2.LINE_AA)
        return img
    sn = ws.snapshot
    cen = W.axis if W.axis is not None else W.origin
    E, cell = sn.half_extent, sn.cell
    H = sn.heights()
    i0 = int(np.floor((cen[0] - half + E) / cell)); j0 = int(np.floor((cen[1] - half + E) / cell))
    m = int(round(2 * half / cell))
    crop = np.full((m, m), np.nan, np.float32)
    a0, b0 = max(i0, 0), max(j0, 0)
    a1, b1 = min(i0 + m, H.shape[0]), min(j0 + m, H.shape[1])
    if a1 > a0 and b1 > b0:
        crop[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = H[a0:a1, b0:b1]
    view = crop[::-1, ::-1]                                        # rows: x descending (up), cols: y descending (left)
    fin = np.isfinite(view)
    k = np.clip((np.nan_to_num(view, nan=H_LO) - H_LO) / (H_HI - H_LO), 0, 1)
    col = HEIGHT_LUT[(k * 255).astype(np.uint8)]
    col[~fin] = (40, 38, 36)
    big = cv2.resize(col, (size, size), interpolation=cv2.INTER_NEAREST)
    pile = cv2.resize((fin & (view > pile_thr)).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    cnt, _ = cv2.findContours(pile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(big, cnt, -1, PILE_BGR, 2, cv2.LINE_AA)
    s = size / (2 * half)

    def wto(q):
        q = np.atleast_2d(q)[:, :2] - cen
        return np.c_[size / 2 - q[:, 1] * s, size / 2 - q[:, 0] * s].astype(np.int32)

    for r in (5, 10, 15):
        cv2.circle(big, tuple(wto(cen)[0]), int(r * s), (90, 90, 90), 1, cv2.LINE_AA)
    for x, y, _z, v, *_ in getattr(sn, "piles", ()):                # volume of each separate pile
        q = tuple(wto(np.array([x, y]))[0])
        txt = f"{v:.1f} m3"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        o_ = (q[0] - tw // 2, q[1] + th // 2)
        cv2.putText(big, txt, o_, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(big, txt, o_, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    tr = W.trail()
    if len(tr) > 1:
        cv2.polylines(big, [wto(tr)], False, (150, 120, 60), 1, cv2.LINE_AA)
    o = tuple(wto(W.origin)[0])
    for a in (-36, 36):                                            # camera FOV wedge
        e = np.radians(a) + W.theta
        cv2.line(big, o, tuple(wto(W.origin + 12 * np.array([np.cos(e), np.sin(e)]))[0]), (120, 120, 60), 1, cv2.LINE_AA)
    if W.axis is not None:
        axp = tuple(wto(W.axis)[0])
        cv2.line(big, axp, o, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.drawMarker(big, axp, (60, 160, 230), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    cv2.circle(big, o, 6, (255, 255, 255), -1, cv2.LINE_AA)
    for t in res.tracks:
        if t.cls == "worker" and t.state in (TrackState.CONFIRMED, TrackState.LOST):
            q = tuple(wto(W.to_world(t.pos))[0])
            cv2.circle(big, q, 6, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(big, str(t.tid), (q[0] + 7, q[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(big, f"{2 * half:.0f} m, cell {cell * 100:.0f} cm, up = start heading", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    img[:, :size] = big
    x0, y0, y1 = size + 14, 30, size - 30                          # colour bar
    for y in range(y0, y1):
        kk = (y1 - y) / (y1 - y0)
        img[y, x0:x0 + 16] = HEIGHT_LUT[int(kk * 255)]
    for hv in (0.0, 0.5, 1.0, 1.5, 2.0):
        y = int(y1 - (hv - H_LO) / (H_HI - H_LO) * (y1 - y0))
        cv2.putText(img, f"{hv:.1f}", (x0 + 20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "m", (x0 + 20, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return img


WORK_HALF_M = 10.0                                                 # work views: +-10 m around the swing axis (the pile)


def _hillshade(z: np.ndarray, fin: np.ndarray, cell: float, light=(-0.62, 0.48, 0.62), ambient: float = 0.22) -> np.ndarray:
    """Display only: Lambert shading factor per cell of a height map (slopes facing the light brighter), so the
    shape of the pile reads in the 3D view; cells next to no-data get no slope (no fake walls at map holes)."""
    zz = np.where(fin, z, np.nan)
    gx, gy = np.gradient(np.nan_to_num(zz, nan=0.0), cell)
    ok = fin.copy()
    ok[1:-1, 1:-1] &= fin[:-2, 1:-1] & fin[2:, 1:-1] & fin[1:-1, :-2] & fin[1:-1, 2:]
    gx, gy = np.where(ok, gx, 0.0), np.where(ok, gy, 0.0)
    L = np.asarray(light, float) / np.linalg.norm(light)
    lam = (-gx * L[0] - gy * L[1] + L[2]) / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    return (ambient + (1 - ambient) * np.clip(lam, 0, 1)).astype(np.float32)


def _fill_gaps(H: np.ndarray, base: np.ndarray, max_gap_cells: int = 6):
    """Display only: no-data cells within max_gap_cells of observed cells (the boom's shadow, LiDAR / camera
    occlusions) get a linear interpolation of the surrounding surface (as a triangle mesh over the observed cells
    would) instead of dropping to the ground as a fake crater; farther no-data cells sit on the ground plane.
    Returns (heights, interpolated mask)."""
    from scipy import ndimage
    fin = np.isfinite(H)
    hole = ~fin & (ndimage.distance_transform_edt(~fin) <= max_gap_cells) if fin.any() else ~fin & False
    out = np.where(fin, H, base).astype(np.float32)
    if hole.any() and fin.sum() >= 3:
        from scipy.interpolate import griddata
        rim = fin & ndimage.binary_dilation(hole, iterations=2)
        ii, jj = np.nonzero(rim)
        hi, hj = np.nonzero(hole)
        v = griddata(np.c_[ii, jj], H[ii, jj], (hi, hj), method="linear")
        out[hi, hj] = np.where(np.isfinite(v), v, base[hi, hj])
        hole[hi, hj] = np.isfinite(v)                               # outside the observed hull: ground, not filled
    return out, hole


def _fill_holes(H: np.ndarray, rgb: np.ndarray, min_nb: int = 4):
    """Display only: an empty cell with >= min_nb observed 3x3 neighbours gets their mean height / colour (the 4-pixel
    keyframe sampling leaves isolated empty cells near the machine, drawn as a checkerboard)."""
    fin = np.isfinite(H)
    Hz = np.where(fin, H, 0).astype(np.float32)
    cnt = cv2.boxFilter(fin.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    sm = cv2.boxFilter(Hz, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    fill = ~fin & (cnt >= min_nb)
    H = H.copy()
    H[fill] = sm[fill] / cnt[fill]
    has = rgb.any(-1)
    cc = cv2.boxFilter(has.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cs = cv2.boxFilter(rgb.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cf = ~has & (cc >= min_nb)
    rgb = rgb.copy()
    rgb[cf] = (cs[cf] / cc[cf][:, None]).astype(np.uint8)
    return H, rgb


class Pile3D(gl.GLViewWidget):
    """Rotatable 3D surface of the pile height map (mouse: drag = rotate, wheel = zoom), coloured by the camera
    (VGGT-Omega pixels) or by height; machine, swing axis and workers drawn in the same world frame (docs/24)."""

    def __init__(self):
        super().__init__()
        self.setBackgroundColor((24, 24, 26))
        g = gl.GLGridItem()
        g.setSize(2 * WORK_HALF_M, 2 * WORK_HALF_M)
        g.setSpacing(1, 1)
        g.setColor((90, 90, 90, 90))
        self.addItem(g)
        self.surf = gl.GLSurfacePlotItem(shader=None, smooth=False, computeNormals=False, glOptions="opaque")
        self.addItem(self.surf)
        self.body = gl.GLLinePlotItem(color=(0.85, 0.85, 0.85, 1.0), width=2, antialias=True)
        self.arm = gl.GLLinePlotItem(color=(0.95, 0.95, 0.95, 1.0), width=3, antialias=True)
        self.workers = gl.GLScatterPlotItem(size=14, color=(1.0, 0.2, 0.2, 1.0), pxMode=True)
        for it in (self.body, self.arm, self.workers):
            self.addItem(it)
        self.labels: list = []                                     # QLabel per separate pile (volume) over the GL view;
        self.label_pts: list = []                                  # GLTextItem is not captured by grab() / recordings
        self.color_mode = "camera"
        self.shading = True                                        # hillshade the surface colours (shape readable)
        self._snap_id = None
        self._seg = None

    def reset_view(self, theta: float = 0.0):
        """Look over the machine towards the pile (behind the cab, 35 deg down)."""
        f = np.array([np.cos(theta), np.sin(theta)])
        self.opts["center"] = pg.Vector(4.5 * f[0], 4.5 * f[1], 0.5)
        self.setCameraPosition(distance=17, elevation=32, azimuth=np.degrees(theta) + 180)

    def _set_labels(self, piles, cen):
        """Volume label floating over each pile inside the view (+-WORK_HALF_M)."""
        piles = [q for q in piles if abs(q[0] - cen[0]) < WORK_HALF_M and abs(q[1] - cen[1]) < WORK_HALF_M][:6]
        while len(self.labels) < len(piles):
            lb = QtWidgets.QLabel(self)
            lb.setStyleSheet("QLabel{color:#fff;background:rgba(20,22,26,170);border:1px solid rgba(255,255,255,90);"
                             "border-radius:4px;padding:1px 6px;font-size:16px;font-weight:bold}")
            lb.hide()
            self.labels.append(lb)
        self.label_pts = [(x - cen[0], y - cen[1], z + 0.4) for x, y, z, *_ in piles]
        for i, lb in enumerate(self.labels):
            if i < len(piles):
                lb.setText(f"{piles[i][3]:.1f} m³")
                lb.adjustSize()
        self._place_labels()

    def _place_labels(self):
        """Project the label anchors with the current camera (called after every GL paint)."""
        m = self.projectionMatrix() * self.viewMatrix()
        w, h = self.width(), self.height()
        for i, lb in enumerate(self.labels):
            if i >= len(self.label_pts):
                lb.hide()
                continue
            v = m.map(QtGui.QVector4D(*self.label_pts[i], 1.0))
            if v.w() <= 1e-6:
                lb.hide()
                continue
            sx = (v.x() / v.w() + 1) / 2 * w - lb.width() / 2
            sy = (1 - v.y() / v.w()) / 2 * h - lb.height()
            if not (-lb.width() < sx < w and -lb.height() < sy < h):
                lb.hide()
                continue
            q = QtCore.QPoint(int(sx), int(sy))
            if (lb.pos() - q).manhattanLength() > 1:               # moving a child repaints; no-op when unchanged
                lb.move(q)
            lb.show()

    def paintGL(self, *a, **k):
        super().paintGL(*a, **k)
        if not a and not k.get("region"):                          # not the picking / export passes
            self._place_labels()

    def show_res(self, res):
        W, ws = res.world, res.work
        if W is None or ws is None or ws.snapshot is None:
            return
        sn = ws.snapshot
        cen = W.axis if W.axis is not None else W.origin
        seg = float(sn.series_t[0]) if len(sn.series_t) else None
        if seg != self._seg:                                       # new segment (seek / new bag): camera behind the cab
            self._seg = seg
            self.reset_view(W.theta)
        if id(sn) != self._snap_id:
            self._snap_id = id(sn)
            E, cell = sn.half_extent, sn.cell
            m = int(round(2 * WORK_HALF_M / cell))
            i0 = int(np.floor((cen[0] - WORK_HALF_M + E) / cell)); j0 = int(np.floor((cen[1] - WORK_HALF_M + E) / cell))
            H = sn.heights()
            crop = np.full((m, m), np.nan, np.float32)
            rgb = np.zeros((m, m, 3), np.uint8)
            a0, b0, a1, b1 = max(i0, 0), max(j0, 0), min(i0 + m, H.shape[0]), min(j0 + m, H.shape[1])
            if a1 > a0 and b1 > b0:
                crop[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = H[a0:a1, b0:b1]
                if sn.rgb is not None:
                    rgb[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = sn.rgb[a0:a1, b0:b1]
            xs = (np.arange(m) + i0 + 0.5) * cell - E - cen[0]
            ys = (np.arange(m) + j0 + 0.5) * cell - E - cen[1]
            crop, rgb = _fill_holes(crop, rgb)                    # display only: no checkerboard between samples
            fin = np.isfinite(crop)
            G = sn.ground()
            gcrop = np.zeros((m, m), np.float32)
            if a1 > a0 and b1 > b0:
                gcrop[a0 - i0:a1 - i0, b0 - j0:b1 - j0] = G[a0:a1, b0:b1]
            z, hole = _fill_gaps(crop, gcrop)                      # gaps in the surface: interpolated, grey
            k = np.clip((z - H_LO) / (H_HI - H_LO), 0, 1)
            col = HEIGHT_LUT[(k * 255).astype(np.uint8)][..., ::-1].astype(np.float32) / 255.0      # BGR LUT -> RGB
            if self.color_mode == "camera":
                has = rgb.any(-1)
                col[has] = rgb[has].astype(np.float32) / 255.0
            col[hole] = (0.55, 0.55, 0.57)                         # interpolated (not observed)
            if self.shading:
                col *= _hillshade(z, fin | hole, cell)[..., None]
            col[~fin & ~hole] = (0.16, 0.16, 0.17)
            # per-vertex colours flattened row-major (pyqtgraph 0.13 indexes them by vertex, not (w, h, 4))
            self.surf.setData(x=xs, y=ys, z=z, colors=np.dstack([col, np.ones((m, m, 1), np.float32)]).reshape(-1, 4))
            self._set_labels(getattr(sn, "piles", ()), cen)
        # machine: upper structure footprint turning with the heading, arm axis -> sensor
        R = W.R()
        ax_w = (W.axis if W.axis is not None else W.origin) - cen
        o_w = W.origin - cen
        L = np.linalg.norm(W.lever) if W.lever is not None else 1.5
        body = np.array([[1.0, 1.5], [1.0, -1.5], [-2.6, -1.5], [-2.6, 1.5], [1.0, 1.5]]) @ R.T + ax_w
        self.body.setData(pos=np.c_[body, np.full(len(body), 2.0)])
        self.arm.setData(pos=np.array([[ax_w[0], ax_w[1], 0.0], [ax_w[0], ax_w[1], 3.0], [o_w[0], o_w[1], 3.0]]))
        wk = [W.to_world(t.pos)[0] - cen for t in res.tracks if t.cls == "worker" and t.state in (TrackState.CONFIRMED, TrackState.LOST)]
        self.workers.setData(pos=np.c_[np.array(wk).reshape(-1, 2), np.full(len(wk), 0.9)] if wk else np.zeros((0, 3)))


class WorkView(QtWidgets.QWidget):
    """'작업' tab: pile height map, pile volume over time, cumulative cut / fill and cycle statistics (docs/24)."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QtWidgets.QHBoxLayout()
        self.color_box = QtWidgets.QComboBox()
        self.color_box.addItems(["카메라 색", "높이 색"])
        self.btn_view = QtWidgets.QPushButton("시점 초기화")
        self.chk_log = QtWidgets.QCheckBox("작업 기록")
        self.chk_log.setChecked(True)
        self.chk_plot = QtWidgets.QCheckBox("그래프")
        bar.addWidget(QtWidgets.QLabel("흙더미 (선회축 ±10 m)  3D: 드래그 = 회전, 휠 = 확대 |"))
        bar.addWidget(self.color_box)
        bar.addWidget(self.btn_view)
        bar.addWidget(self.chk_log)
        bar.addWidget(self.chk_plot)
        bar.addStretch(1)
        lay.addLayout(bar)
        hud = QtWidgets.QHBoxLayout()                              # live numbers, always shown
        hud.setSpacing(4)
        self.cards = {}
        for key, title, stretch in (("vol", "흙더미 부피", 3), ("cf", "누적 굴착 · 성토", 2), ("last", "이번 작업", 4),
                                    ("cyc", "작업 사이클", 2), ("state", "상태", 3)):
            c = QtWidgets.QLabel()
            c.setTextFormat(QtCore.Qt.RichText)
            c.setStyleSheet("QLabel{background:#202226;border:1px solid #33363c;border-radius:4px;padding:3px 8px;color:#ddd}")
            c.setMinimumHeight(64)
            c.setProperty("title", title)
            hud.addWidget(c, stretch)
            self.cards[key] = c
        lay.addLayout(hud)
        self.views = QtWidgets.QTabWidget()
        self.views.setStyleSheet("QTabBar::tab{background:#2a2c30;padding:2px 10px} QTabBar::tab:selected{background:#3d5a80}")
        self.pile3d = Pile3D()
        self.map = QtWidgets.QLabel()
        self.map.setAlignment(QtCore.Qt.AlignCenter)
        self.map.setMinimumSize(360, 300)
        self.views.addTab(self.pile3d, "3D")
        self.views.addTab(self.map, "평면도")
        lay.addWidget(self.views, 5)
        self.log = QtWidgets.QTableWidget(0, 7)                    # one row per work cycle (scoop -> put down)
        self.log.setHorizontalHeaderLabels(["#", "시각 [s]", "작업 전 [m³]", "퍼낸 뒤 [m³]", "퍼낸 양 [m³]",
                                            "내려놓은 곳 [m³]", "내려놓은 양 [m³]"])
        self.log.verticalHeader().setVisible(False)
        self.log.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.log.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.log.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.log.setStyleSheet("QTableWidget{background:#1b1c1f;color:#ddd;gridline-color:#333;font-size:12px}"
                               "QHeaderView::section{background:#2a2c30;color:#ccc;border:none;padding:2px}")
        self.log.setMaximumHeight(150)
        lay.addWidget(self.log, 1)
        self._log_key = None
        self.chk_log.toggled.connect(self.log.setVisible)
        self.last_res = None
        self.color_box.currentIndexChanged.connect(self._on_color)
        self.btn_view.clicked.connect(lambda: self.last_res is not None and self.last_res.world is not None
                                      and self.pile3d.reset_view(self.last_res.world.theta))
        self.views.currentChanged.connect(lambda *_: self.last_res is not None and self.show(self.last_res))
        self.stats = QtWidgets.QLabel("")
        self.stats.setStyleSheet("font-family:monospace;font-size:12px;color:#ddd;padding:2px 4px")
        self.stats.setWordWrap(True)
        lay.addWidget(self.stats)
        self.p_vol, self.p_cf = pg.PlotWidget(), pg.PlotWidget()
        for pw, title in ((self.p_vol, "흙더미 부피 [m³] (선회축 9 m 안, 지면 위 0.2 m 초과)"),
                          (self.p_cf, "누적 굴착(cut)·성토(fill) [m³] (기준 표면 대비)")):
            pw.setBackground("#18191b")
            pw.setTitle(title, color="#cccccc", size="9pt")
            pw.showGrid(x=True, y=True, alpha=0.15)
            pw.setMinimumHeight(120)
            lay.addWidget(pw, 1)
        self.p_cf.setLabel("bottom", "구간 시작 후 [s]")
        self.p_cf.setXLink(self.p_vol)
        self.c_vol = self.p_vol.plot(pen=pg.mkPen("#2a78d6", width=2))
        leg = self.p_cf.addLegend(offset=(-10, 2), labelTextColor="#cccccc", colCount=2)
        self.c_cut = self.p_cf.plot(pen=pg.mkPen("#2a78d6", width=2), name="굴착 cut")
        self.c_fill = self.p_cf.plot(pen=pg.mkPen("#eb6834", width=2), name="성토 fill")
        leg.setBrush(pg.mkBrush(24, 25, 27, 200))
        self.now = [pg.InfiniteLine(angle=90, pen=pg.mkPen("#888888", width=1)) for _ in range(2)]
        self.p_vol.addItem(self.now[0])
        self.p_cf.addItem(self.now[1])
        self.cyc: list = []
        self.show_info = False                                      # stats text + plots under the map (hidden for now)
        for w_ in (self.stats, self.p_vol, self.p_cf):
            w_.setVisible(self.show_info)
        self.chk_plot.toggled.connect(self._on_plot)

    def _on_plot(self, on: bool):
        self.show_info = on
        for w_ in (self.stats, self.p_vol, self.p_cf):
            w_.setVisible(on)
        if self.last_res is not None:
            self.show(self.last_res)

    @staticmethod
    def _card(title: str, big: str, sub: str = "", color: str = "#ffffff", big_pt: int = 20) -> str:
        return (f"<div style='font-size:11px;color:#9aa0a8'>{title}</div>"
                f"<div style='font-size:{big_pt}px;font-weight:bold;color:{color}'>{big}</div>"
                f"<div style='font-size:11px;color:#b8bcc4'>{sub}</div>")

    def _show_hud(self, ws):
        C, f = self.cards, (lambda v, fmt="5.1f": format(v, fmt) if v is not None and np.isfinite(v) else "–")  # noqa: E731
        if ws is None or ws.snapshot is None:
            for k, c in C.items():
                c.setText(self._card(c.property("title"), "–", "작업 모니터 꺼짐 또는 데이터 없음" if k == "vol" else ""))
            return
        piles = getattr(ws.snapshot, "piles", ())
        hmax = max((q[5] for q in piles), default=float("nan"))
        C["vol"].setText(self._card("흙더미 부피 (선회축 9 m 안)", f"{ws.volume:.1f} m³",
                                    f"면적 {ws.area:.1f} m² · 최고 {f(hmax, '.2f')} m · 더미 {len(piles)}개"
                                    f" · 보간 {getattr(ws, 'fill_frac', 0.0) * 100:.0f} %", "#7fb8ff", 26))
        C["cf"].setText(self._card("누적 굴착 · 성토 (기준 표면 대비)", f"{max(ws.cut, 0.0):.1f} · {max(ws.fill, 0.0):.1f} m³", ""))
        led = getattr(ws, "ledger", ())
        if led:
            c = led[-1]
            big = (f"−{f(c['cut'], '.2f')} m³ 퍼냄" if np.isfinite(c["cut"]) else "") + \
                  ("  " if np.isfinite(c["cut"]) and np.isfinite(c["fill"]) else "") + \
                  (f"+{f(c['fill'], '.2f')} m³ 내려놓음" if np.isfinite(c["fill"]) else "")
            sub = (f"작업 전 {f(c['dig_before'], '.1f')} → 퍼낸 뒤 {f(c['dig_after'], '.1f')} m³" if np.isfinite(c["t_dig"]) else "") + \
                  (f" · 내려놓은 곳 {f(c['dump_before'], '.1f')} → {f(c['dump_after'], '.1f')} m³" if np.isfinite(c["t_dump"]) else "")
            C["last"].setText(self._card(f"이번 작업 ({len(led)}번째, 버킷 기준)", big or "–", sub, "#ffd08a"))
        else:
            C["last"].setText(self._card("이번 작업 (버킷 기준)", "–", "버킷이 흙을 퍼고 빠지면 표시"))
        cyc_h = 3600.0 / ws.cycle_s if np.isfinite(ws.cycle_s) and ws.cycle_s > 0 else float("nan")
        C["cyc"].setText(self._card("선회 사이클", f"{ws.n_cycles}회", f"평균 {f(ws.cycle_s, '.0f')} s · 시간당 {f(cyc_h, '.0f')}회"))
        if ws.swinging:
            st, col = "선회 중 · 갱신 멈춤", "#e0b050"
        elif ws.bucket_in_pile:
            st, col = "버킷 작업 중 · 갱신 멈춤", "#e0b050"
        elif ws.active:
            st, col = "버킷 동작 · 갱신 멈춤", "#e0b050"
        else:
            st, col = "지도 갱신 중", "#7fd18b"
        age = ws.t - ws.t_map if np.isfinite(getattr(ws, "t_map", float("nan"))) else float("nan")
        src = {"vggt_omega": "VGGT-Ω×레이더", "stereo": "스테레오×레이더"}.get(ws.source, "레이더만")
        C["state"].setText(self._card("상태", st, f"지도 {f(age, '.1f')} s 전 갱신 · {src}", col, 17))

    def _show_log(self, ws):
        led = getattr(ws, "ledger", ()) if ws is not None else ()
        t0 = float(ws.snapshot.series_t[0]) if ws is not None and ws.snapshot is not None and len(ws.snapshot.series_t) else 0.0
        key = (len(led), tuple(round(v, 3) for c in led[-1:] for v in (c["cut"], c["fill"], c["dig_after"], c["dump_after"])
                               if np.isfinite(v)), t0)
        if key == self._log_key:
            return
        self._log_key = key
        f = lambda v, fmt=".2f": format(v, fmt) if np.isfinite(v) else "–"   # noqa: E731
        self.log.setRowCount(len(led))
        for r, c in enumerate(reversed(led)):                      # newest on top
            tt = c["t_dig"] if np.isfinite(c["t_dig"]) else c["t_dump"]
            vals = [str(len(led) - r), f(tt - t0, ".0f"), f(c["dig_before"], ".1f"), f(c["dig_after"], ".1f"),
                    f(c["cut"]), f"{f(c['dump_before'], '.1f')} → {f(c['dump_after'], '.1f')}" if np.isfinite(c["t_dump"]) else "–",
                    f(c["fill"])]
            for k, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                self.log.setItem(r, k, it)

    def _on_color(self, i: int):
        self.pile3d.color_mode = "camera" if i == 0 else "height"
        self.pile3d._snap_id = None
        if self.last_res is not None:
            self.show(self.last_res)

    def show(self, res, pile_thr: float = 0.2):
        self.last_res = res
        if self.views.currentIndex() == 0:
            self.pile3d.show_res(res)
        else:
            side = max(min(self.map.width() - 70, self.map.height()), 260)
            self.map.setPixmap(_bgr_to_pixmap(render_work(res, size=side, half=WORK_HALF_M, pile_thr=pile_thr)))
        ws = res.work
        self._show_hud(ws)
        if self.log.isVisible():
            self._show_log(ws)
        if not self.show_info:
            return
        if ws is None or ws.snapshot is None:
            self.stats.setText("작업 모니터 꺼짐 (프로파일 / config 의 work.enable) 또는 아직 데이터 없음")
            for c in (self.c_vol, self.c_cut, self.c_fill):
                c.setData([], [])
            return
        sn = ws.snapshot
        t0 = float(sn.series_t[0]) if len(sn.series_t) else ws.t
        x = sn.series_t - t0
        self.c_vol.setData(x, sn.series_v)
        self.c_cut.setData(x, sn.series_cut)
        self.c_fill.setData(x, sn.series_fill)
        for ln in self.now:
            ln.setValue(ws.t - t0)
        for pw, it in self.cyc:
            pw.removeItem(it)
        self.cyc = []
        for c in sn.cycles_t:                                       # cycle completions (dashed)
            for pw in (self.p_vol, self.p_cf):
                it = pg.InfiniteLine(pos=float(c - t0), angle=90, pen=pg.mkPen("#777777", width=1, style=QtCore.Qt.DashLine))
                pw.addItem(it)
                self.cyc.append((pw, it))
        cyc_h = 3600.0 / ws.cycle_s if np.isfinite(ws.cycle_s) and ws.cycle_s > 0 else float("nan")
        f = lambda v, fmt: format(v, fmt) if np.isfinite(v) else "–"   # noqa: E731
        self.stats.setText(
            f"흙더미 {ws.volume:6.1f} m³ / {ws.area:5.1f} m² (관측 {ws.observed:4.0f} m²) | 굴착 {ws.cut:5.1f} m³, 성토 {ws.fill:5.1f} m³\n"
            f"사이클 {ws.n_cycles}회 (나갔다 돌아오는 선회 한 쌍), 평균 {f(ws.cycle_s, '.1f')} s, 시간당 {f(cyc_h, '.0f')}회, "
            f"마지막 사이클 굴착 {f(ws.last_dcut, '+.2f')} m³" + ("  [선회 중: 맵 갱신 멈춤]" if ws.swinging else "")
            + ("  [버킷이 흙더미 안: 맵 갱신 멈춤]" if ws.bucket_in_pile else ("  [버킷 동작: 맵 갱신 멈춤]" if ws.active else "")) + "\n"
            f"버킷·붐 가림 {ws.mask_frac * 100:.0f} % (색 분할), 정지 중 갱신 멈춤 비율 {ws.paused_frac * 100:.0f} %\n"
            + (f"깊이: VGGT-Ω 키프레임 {ws.n_engine}개 ({ws.engine_ms:.0f} ms), 레이더 스케일 {ws.scale:.2f}" if ws.source == "vggt_omega" else
               f"깊이: {'OAK 스테레오' if ws.source == 'stereo' else '레이더만'}, 스케일(레이더/스테레오) {ws.scale:.3f}")
            + f" | 이번 프레임 점: 조밀 {ws.n_stereo_pts}, 레이더 {ws.n_radar_pts}")


class ReconView(QtWidgets.QWidget):
    """'재구성' tab: the drive 3D map built while the machine moves (rcmot.work.recon, docs/26 §6) in the world
    frame of the segment, with the keyframe trajectory, the gyro / radar trail and the machine's current pose.
    Mouse: drag = rotate, wheel = zoom, middle drag = pan."""

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        bar = QtWidgets.QHBoxLayout()
        self.chk_follow = QtWidgets.QCheckBox("따라가기")
        self.chk_follow.setChecked(True)
        self.chk_follow.setToolTip("시점을 기계 위치에 맞춰 옮김 (끄면 마우스로 둘러보기)")
        self.btn_top = QtWidgets.QPushButton("위에서")
        self.btn_back = QtWidgets.QPushButton("기계 뒤에서")
        self.size_box = QtWidgets.QComboBox()
        self.size_box.addItems(["점 1 px", "점 2 px", "점 3 px", "점 4 px"])
        self.size_box.setCurrentIndex(1)
        bar.addWidget(QtWidgets.QLabel("주행 3D 재구성 (VGGT-Ω 묶음, 레이더 스케일, 0.1 m) |"))
        for w_ in (self.chk_follow, self.btn_top, self.btn_back, self.size_box):
            bar.addWidget(w_)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.gl = gl.GLViewWidget()
        self.gl.setBackgroundColor((24, 24, 26))
        g = gl.GLGridItem()
        g.setSize(200, 200)
        g.setSpacing(5, 5)
        g.setColor((90, 90, 90, 70))
        self.gl.addItem(g)
        self.pts = gl.GLScatterPlotItem(size=2, pxMode=True, glOptions="opaque")
        self.traj = gl.GLLinePlotItem(color=(0.16, 0.47, 0.84, 1.0), width=2, antialias=True)      # keyframes (VGGT chain)
        self.trail = gl.GLLinePlotItem(color=(0.92, 0.41, 0.2, 1.0), width=1.5, antialias=True)   # gyro + Doppler trail
        self.body = gl.GLLinePlotItem(color=(0.95, 0.95, 0.95, 1.0), width=2, antialias=True)
        for it in (self.pts, self.traj, self.trail, self.body):
            self.gl.addItem(it)
        lay.addWidget(self.gl, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setStyleSheet("font-family:monospace;font-size:12px;color:#ddd;padding:2px 4px")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self._snap_id = None
        self._seg = None
        self.last_res = None
        self.btn_top.clicked.connect(lambda: self._view(top=True))
        self.btn_back.clicked.connect(lambda: self._view(top=False))
        self.size_box.currentIndexChanged.connect(lambda i: (self.pts.setData(size=i + 1), None)[1])

    def _view(self, top: bool):
        th = self.last_res.world.theta if self.last_res is not None and self.last_res.world is not None else 0.0
        if top:
            self.gl.setCameraPosition(distance=90, elevation=90, azimuth=np.degrees(th) - 90)
        else:
            self.gl.setCameraPosition(distance=45, elevation=30, azimuth=np.degrees(th) + 180)

    def show(self, res):
        self.last_res = res
        rs, W = res.recon, res.world
        if rs is None:
            self.status.setText("주행 재구성 꺼짐: 프로파일 overrides 또는 설정의 recon.enable (주행 프로파일 140718·142258은 켜짐); "
                                "models/vggt/vggt_omega_1b_512.pt 필요")
            return
        if W is not None:
            seg = float(W.trail_ref[0][0]) if W.trail_ref else None
            if self._seg != (seg, id(W.trail_ref)):                    # new segment (seek / new bag): reset the view
                self._seg = (seg, id(W.trail_ref))
                self._view(top=False)
            R = W.R()
            body = np.array([[1.0, 1.5], [1.0, -1.5], [-2.6, -1.5], [-2.6, 1.5], [1.0, 1.5], [2.5, 0.0], [1.0, -1.5]]) @ R.T \
                + (W.axis if W.axis is not None else W.origin)
            self.body.setData(pos=np.c_[body, np.full(len(body), 1.5)])
            tr = W.trail()
            self.trail.setData(pos=np.c_[tr, np.full(len(tr), 0.3)] if len(tr) > 1 else np.zeros((0, 3)))
            if self.chk_follow.isChecked():
                self.gl.opts["center"] = pg.Vector(float(W.origin[0]), float(W.origin[1]), 0.0)
                self.gl.update()
        sn = rs.snapshot
        if sn is not None and id(sn) != self._snap_id:
            self._snap_id = id(sn)
            xyz, rgb = sn.xyz, sn.rgb
            if len(xyz) > 2_000_000:                                   # display budget
                sel = np.random.default_rng(0).choice(len(xyz), 2_000_000, replace=False)
                xyz, rgb = xyz[sel], rgb[sel]
            col = np.empty((len(xyz), 4), np.float32)
            col[:, :3] = rgb.astype(np.float32) / 255.0
            col[:, 3] = 1.0
            self.pts.setData(pos=xyz, color=col)
            self.traj.setData(pos=sn.kf_xyz if len(sn.kf_xyz) > 1 else np.zeros((0, 3)))
        elif sn is None and self._snap_id is not None:
            self._snap_id = None
            self.pts.setData(pos=np.zeros((0, 3)), color=np.zeros((0, 4), np.float32))
            self.traj.setData(pos=np.zeros((0, 3)))
        busy = f", 처리 중 {rs.queued}" if rs.queued else ""
        n_pts = f"{rs.points / 1e6:.2f} M" if rs.points >= 1e5 else f"{rs.points}"
        self.status.setText(
            f"키프레임 {rs.n_keyframes} (지도에 {rs.n_mapped}, 대기 {rs.n_keyframes - rs.n_mapped}){busy} | 묶음 {rs.n_chunks} | 점 {n_pts}"
            + (f" | {rs.note}" if rs.note else " | 첫 묶음은 키프레임 16개(약 12 m 주행) 뒤에 나옵니다")
            + (f"\nERROR {rs.error}" if rs.error else "")
            + "\n파랑 = 재구성 키프레임 궤적, 주황 = 자이로+레이더 궤적, 흰 = 기계 (화살표가 앞)")


# ------------------------------------------------------------------------------------------------ main window
class ZoomLabel(QtWidgets.QLabel):
    zoom = QtCore.pyqtSignal(int)

    def wheelEvent(self, e):
        self.zoom.emit(1 if e.angleDelta().y() > 0 else -1)


SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]


class Viewer(QtWidgets.QMainWindow):
    step = QtCore.pyqtSignal(int)          # +1 / -1 frame
    jump = QtCore.pyqtSignal(float)        # relative seconds (-1e9 = to the start)
    go_live = QtCore.pyqtSignal()
    open_dataset = QtCore.pyqtSignal(str)        # replay preset chosen in the dataset box (configs/datasets.yaml)
    change_model = QtCore.pyqtSignal(str)        # detector weights chosen in the model box (path)
    publish_toggled = QtCore.pyqtSignal(bool)    # PUB button: ROS 2 result publishing on / off (docs/20)
    alarm_toggled = QtCore.pyqtSignal(bool)
    people_only = QtCore.pyqtSignal(bool)         # '사람만': heavy-equipment detection off

    def __init__(self, calib, max_range: float = 20.0, danger_m: float = 5.0, zones=(5.0, 10.0, 15.0), title="RC-MOT"):
        super().__init__()
        self.calib, self.bev_range, self.danger_m, self.zones = calib, max_range, danger_m, zones
        self.setWindowTitle(f"{title}  |  4D radar + camera 3D tracking")
        self.resize(1880, 1040)
        self.setStyleSheet("QMainWindow,QWidget{background:#18191b;color:#ddd} QPushButton{background:#2a2c30;border:1px solid #444;"
                           "padding:4px 10px;border-radius:4px} QPushButton:checked{background:#3d5a80} QCheckBox{color:#ccc}")
        self.focus = None
        self.paused, self.speed = False, 1.0
        self.last, self.note = None, ""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        bar = QtWidgets.QHBoxLayout()                  # row 1: source, playback, timeline (gets the free width)
        root.addLayout(bar)
        bar2 = QtWidgets.QHBoxLayout()                 # row 2: alarm / zones / publishing, display and detection options
        root.addLayout(bar2)
        self.ds_box = QtWidgets.QComboBox()           # replay presets; hidden until set_datasets()
        self.ds_box.setVisible(False)
        self.ds_box.setMinimumWidth(220)
        self.model_box = QtWidgets.QComboBox()        # detector weights; hidden until set_models()
        self.model_box.setVisible(False)
        self.model_box.setMinimumWidth(150)
        self.btn_alarm = QtWidgets.QPushButton("경보 OFF")
        self.btn_alarm.setCheckable(True)
        self.btn_alarm.setStyleSheet("QPushButton:checked{background:#a33;color:#fff;font-weight:700}")
        self.alarm_lbl = QtWidgets.QLabel("")
        self.alarm_lbl.setStyleSheet("font-weight:700;padding:0 6px")
        self.btn_pub = QtWidgets.QPushButton("PUB OFF")    # ROS 2 result publishing (docs/20); hidden until set_publishing()
        self.btn_pub.setCheckable(True)
        self.btn_pub.setVisible(False)
        self.btn_pub.setToolTip("publish /rcmot/tracks, /rcmot/alarm, /rcmot/detections, /rcmot/status")
        self.btn_pub.setStyleSheet("QPushButton:checked{background:#2a7a3a;color:#fff;font-weight:700}")
        self.zone_boxes = []                           # danger / warning / caution [m], editable (docs/18)
        for k, (lab, v, css) in enumerate(((" 위험", 10.0, ZONE_CSS[3]), (" 경고", 20.0, ZONE_CSS[2]), (" 주의", 30.0, ZONE_CSS[1]))):
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(1.0, 200.0); sb.setDecimals(0); sb.setSingleStep(1.0); sb.setValue(v); sb.setSuffix(" m"); sb.setPrefix(lab)
            sb.setStyleSheet(f"QDoubleSpinBox{{color:{css};font-weight:700;background:#2a2c30;border:1px solid {css};border-radius:4px;padding:2px}}")
            sb.setMaximumWidth(110)
            self.zone_boxes.append(sb)
        self.alarm = None                              # rcmot.alarm.Alarm, set by the app
        self.btn_home = QtWidgets.QPushButton("|<")
        self.btn_back = QtWidgets.QPushButton("< (A)")
        self.btn_play = QtWidgets.QPushButton("Pause (Space)")
        self.btn_step = QtWidgets.QPushButton("(D) >")
        self.speed_box = QtWidgets.QComboBox()
        self.speed_box.addItems([f"x{v:g}" for v in SPEEDS])
        self.speed_box.setCurrentIndex(SPEEDS.index(1.0))
        self.btn_live = QtWidgets.QPushButton("● LIVE")
        self.btn_live.setVisible(False)
        self.btn_live.setStyleSheet("QPushButton{color:#ff5050;font-weight:700}")
        self.timeline = Timeline()
        self.time_lbl = QtWidgets.QLabel("--")
        self.time_lbl.setStyleSheet("font-family:monospace;font-size:13px;padding:0 6px")
        self.time_lbl.setMinimumWidth(150)
        for w_ in (self.ds_box, self.model_box, self.btn_home, self.btn_back, self.btn_play, self.btn_step, self.speed_box,
                   self.btn_live):
            bar.addWidget(w_)
        bar.addWidget(self.timeline, 1)
        bar.addWidget(self.time_lbl)
        for w_ in (self.btn_alarm, *self.zone_boxes, self.alarm_lbl, self.btn_pub):
            bar2.addWidget(w_)
        bar2.addStretch(1)
        self.chk = {k: QtWidgets.QCheckBox(k) for k in ("radar", "tails", "brighten", "movers")}
        self.chk["movers"].setText("미탐지 ?")
        self.chk["movers"].setToolTip("카메라가 못 본 레이더 이동체(주황 ◇ '? moving'): 끄면 영상·BEV에서 숨김 (처리는 그대로)")
        for k, c in self.chk.items():
            c.setChecked(k != "brighten")
            c.toggled.connect(lambda *_: self.redraw())
            bar2.addWidget(c)
        self.chk_world = QtWidgets.QCheckBox("월드")                   # BEV in the world frame (docs/24)
        self.chk_world.setToolTip("BEV를 구간 시작 시점의 월드 좌표로 그림: 기계가 돌고 움직이며 궤적이 남음 (선회축 모델은 프로파일의 world.lever_arm)")
        self.chk_world.stateChanged.connect(lambda *_: self.redraw())
        bar2.addWidget(self.chk_world)
        self.chk_people = QtWidgets.QCheckBox("사람만")
        self.chk_people.setToolTip("작업자(사람)만 검출: 중장비 검출(트럭·버스·승용차 → 중장비)을 끔. 다음 프레임부터 적용")
        self.chk_people.toggled.connect(self.people_only.emit)
        bar2.addWidget(self.chk_people)
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("font-family:monospace;font-size:12px;color:#ccc;padding:2px 4px")
        root.addWidget(self.status)

        split = QtWidgets.QHBoxLayout()
        root.addLayout(split, 1)
        self.cam = QtWidgets.QLabel()
        self.cam.setMinimumSize(960, 540)
        self.cam.setAlignment(QtCore.Qt.AlignCenter)
        self.cam.setStyleSheet("background:#000")
        split.addWidget(self.cam, 3)
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet("QTabBar::tab{background:#2a2c30;padding:4px 14px} QTabBar::tab:selected{background:#3d5a80}")
        self.bev = ZoomLabel()
        self.bev.setAlignment(QtCore.Qt.AlignCenter)
        self.bev.zoom.connect(self._zoom)
        self.view3d = View3D(40.0)
        self.workview = WorkView()
        self.reconview = ReconView()
        self.tabs.addTab(self.bev, "BEV")
        self.tabs.addTab(self.view3d, "3D")
        self.tabs.addTab(self.workview, "작업")
        self.tabs.addTab(self.reconview, "재구성")
        self.tabs.setCurrentIndex(int(os.environ.get("RCMOT_TAB", 0)))       # start on another tab (tests / screenshots)
        self.tabs.currentChanged.connect(lambda *_: self.redraw())
        split.addWidget(self.tabs, 2)
        self.gallery = Gallery()
        self.gallery.focus.connect(self._focus)
        root.addWidget(self.gallery)

        self.btn_play.clicked.connect(self.toggle_pause)
        self.btn_step.clicked.connect(lambda: self.step.emit(1))
        self.btn_back.clicked.connect(lambda: self.step.emit(-1))
        self.btn_home.clicked.connect(lambda: self.jump.emit(-1e9))
        self.btn_live.clicked.connect(self.go_live.emit)
        self.ds_box.activated.connect(lambda i: self.open_dataset.emit(self.ds_box.itemText(i)))
        self.model_box.activated.connect(self._on_model)
        self.btn_alarm.toggled.connect(self._on_alarm)
        self.btn_pub.toggled.connect(self._on_pub)
        for sb in self.zone_boxes:
            sb.valueChanged.connect(self._on_zones)
        self.speed_box.currentIndexChanged.connect(lambda i: setattr(self, "speed", SPEEDS[i]))
        keys = {QtCore.Qt.Key_Space: self.toggle_pause,
                QtCore.Qt.Key_D: lambda: self.step.emit(1), QtCore.Qt.Key_Right: lambda: self.step.emit(1),
                QtCore.Qt.Key_A: lambda: self.step.emit(-1), QtCore.Qt.Key_Left: lambda: self.step.emit(-1),
                QtCore.Qt.SHIFT + QtCore.Qt.Key_Right: lambda: self.jump.emit(5.0),
                QtCore.Qt.SHIFT + QtCore.Qt.Key_Left: lambda: self.jump.emit(-5.0),
                QtCore.Qt.Key_Home: lambda: self.jump.emit(-1e9), QtCore.Qt.Key_End: self.go_live.emit,
                QtCore.Qt.Key_BracketRight: lambda: self._speed_step(1),
                QtCore.Qt.Key_BracketLeft: lambda: self._speed_step(-1)}
        for k, f in keys.items():
            QtWidgets.QShortcut(QtGui.QKeySequence(k), self, activated=f)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Escape), self, activated=lambda: self._focus(None))
        for c in self.chk.values():
            c.stateChanged.connect(lambda *_: self.redraw())

    # ------------------------------------------------------------------ controls
    def toggle_pause(self):
        self.set_paused(not self.paused)

    def set_paused(self, p: bool):
        self.paused = p
        self.btn_play.setText("Play (Space)" if p else "Pause (Space)")

    def _speed_step(self, d):
        self.speed_box.setCurrentIndex(int(np.clip(self.speed_box.currentIndex() + d, 0, len(SPEEDS) - 1)))

    def set_models(self, items: list, current: str):
        """items: (label, path) of the detector weights offered; a last entry opens a file dialog."""
        self._models = list(items)
        self.model_box.blockSignals(True)
        self.model_box.clear()
        for lab, _ in self._models:
            self.model_box.addItem(lab)
        self.model_box.addItem("파일에서 열기...")
        for k, (_, path) in enumerate(self._models):
            if path == current:
                self.model_box.setCurrentIndex(k)
        self.model_box.blockSignals(False)
        self.model_box.setVisible(True)

    def _on_model(self, i: int):
        if i < len(self._models):
            self.change_model.emit(self._models[i][1])
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "검출기 가중치", "", "YOLO weights (*.pt *.engine *.onnx)")
        if path:
            self._models.append((os.path.basename(path), path))
            self.model_box.insertItem(len(self._models) - 1, os.path.basename(path))
            self.model_box.setCurrentIndex(len(self._models) - 1)
            self.change_model.emit(path)

    def _on_zones(self, *_):
        z = tuple(sorted(float(sb.value()) for sb in self.zone_boxes))
        self.zones, self.danger_m = z, z[0]
        if self.alarm is not None:
            self.alarm.set_zones(z)
        self.redraw()

    def set_zones(self, zones):
        for sb, v in zip(self.zone_boxes, sorted(zones)):
            sb.blockSignals(True); sb.setValue(float(v)); sb.blockSignals(False)
        self._on_zones()

    def _on_alarm(self, on: bool):
        self.btn_alarm.setText("경보 ON" if on else "경보 OFF")
        if self.alarm is not None:
            self.alarm.set_enabled(on)
        self.alarm_toggled.emit(on)

    def _on_pub(self, on: bool):
        self.btn_pub.setText("PUB ON" if on else "PUB OFF")
        self.publish_toggled.emit(on)

    def set_publishing(self, on: bool | None):
        """Show the PUB toggle (a publisher exists) in the given state; None hides it."""
        self.btn_pub.setVisible(on is not None)
        if on is not None:
            self.btn_pub.blockSignals(True); self.btn_pub.setChecked(bool(on)); self.btn_pub.blockSignals(False)
            self.btn_pub.setText("PUB ON" if on else "PUB OFF")

    def set_datasets(self, names, current: str, notes=None):
        self.ds_box.blockSignals(True)
        self.ds_box.clear()
        self.ds_box.addItems(list(names))
        for k, n in enumerate(names):
            if notes and notes.get(n):
                self.ds_box.setItemData(k, notes[n], QtCore.Qt.ToolTipRole)
        if current in names:
            self.ds_box.setCurrentIndex(list(names).index(current))
        self.ds_box.blockSignals(False)
        self.ds_box.setVisible(True)

    def set_calib(self, calib, title: str, max_range=None, danger_m=None, zones=None):
        """Switch the recording: projection calib, BEV settings, title; the frame / gallery state is cleared."""
        self.calib = calib
        if max_range is not None:
            self.bev_range = max_range
        if danger_m is not None:
            self.danger_m = danger_m
        if zones is not None and self.alarm is None:               # with an alarm the zone boxes rule
            self.zones = tuple(zones)
        self.setWindowTitle(f"{title}  |  4D radar + camera 3D tracking")
        self.last, self.note, self.focus = None, "", None
        self.set_paused(False)

    def set_live_mode(self, live: bool):
        self.timeline.live = live
        self.btn_live.setVisible(live)
        self.btn_home.setVisible(not live)
        self.speed_box.setVisible(not live)

    def set_timeline(self, t0, t1, cur, buf, text):
        self.timeline.set_state(t0, t1, cur, buf)
        self.time_lbl.setText(text)

    def _zoom(self, d):
        self.bev_range = float(np.clip(self.bev_range * (0.85 if d > 0 else 1.18), 6, 80))
        self.redraw()

    def _focus(self, tid):
        self.focus = None if tid == self.focus else tid
        self.redraw()

    # ------------------------------------------------------------------ drawing
    def show_frame(self, bgr, res, fps: float = 0.0, note: str = ""):
        self.last = (bgr, res, fps)
        self.note = note
        self.redraw()

    def redraw(self):
        if self.last is None:
            return
        bgr, res, fps = self.last
        cam = render_camera(bgr, res, self.calib, focus=self.focus, show_radar=self.chk["radar"].isChecked(),
                            show_tails=self.chk["tails"].isChecked(), brighten=self.chk["brighten"].isChecked(), zones=self.zones,
                            show_movers=self.chk["movers"].isChecked())
        if self.alarm is not None:
            st = res.alarm                                   # decided on the worker thread (docs/20) ...
            if st is None:                                   # ... or, for a frame processed without it, here
                st = self.alarm.decide(res.tracks)
            lvl = st.level
            if tuple(st.zones) != tuple(self.zones):         # zones edited after this (cached) frame was decided:
                from rcmot.alarm import zone_of              # show the plain zone of its nearest track (no hysteresis)
                lvl = zone_of(st.nearest_range, self.zones)
            self.alarm.play(lvl)                             # the sound follows the frame being shown
            names = {0: "", 1: f"주의 {self.zones[2]:.0f} m", 2: f"경고 {self.zones[1]:.0f} m", 3: f"위험 {self.zones[0]:.0f} m"}
            cols = ZONE_COL
            near = st.nearest_range
            self.alarm_lbl.setText(f"{names[lvl]}{'' if near is None else f'  {near:.1f} m'}" if lvl else
                                   ("" if near is None else f"최근접 {near:.1f} m"))
            self.alarm_lbl.setStyleSheet("font-weight:700;padding:0 6px;color:%s" % {0: "#aaa", 1: "#ffd34d", 2: "#ff9a3c", 3: "#ff5050"}[lvl])
            if lvl and self.alarm.enabled:
                h = 46
                cv2.rectangle(cam, (0, 0), (cam.shape[1], h), cols[lvl], -1)
                en = {1: "CAUTION", 2: "WARNING", 3: "DANGER"}
                cv2.putText(cam, f"{en[lvl]} <= {self.zones[3 - lvl]:.0f} m   ID {st.nearest_tid}  {near:.1f} m", (16, 33), cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (255, 255, 255), 2, cv2.LINE_AA)
        self.cam.setPixmap(_bgr_to_pixmap(cam, self.cam.width(), self.cam.height()))
        tab = self.tabs.currentIndex()
        if tab == 0:
            side = max(min(self.bev.width(), self.bev.height()), 300)
            bev = render_bev(res, size=side, rng_m=self.bev_range, focus=self.focus, zones=self.zones,
                             show_radar=self.chk["radar"].isChecked(), show_movers=self.chk["movers"].isChecked(),
                             world_mode=self.chk_world.isChecked() and res.world is not None)
            self.bev.setPixmap(_bgr_to_pixmap(bev))
        elif tab == 1:
            self.view3d.show(res)
        elif tab == 2:
            self.workview.show(res)
        else:
            self.reconview.show(res)
        self.gallery.update_tracks(res.tracks, res.stamp, self.danger_m, self.focus, zones=self.zones)
        n = {s: sum(t.state == s for t in res.tracks) for s in STATE_STYLE}
        ev = res.ego_vel
        tm = "  ".join(f"{k} {v:.0f}" for k, v in res.timing_ms.items())
        self.status.setText(
            f"{fps:4.1f} FPS | ego {np.linalg.norm(ev[:2]):.2f} m/s{'' if res.ego_valid else ' (hold)'} | "
            f"radar σv {res.sigma_v * 100:.1f} cm/s, dyn pts {int(np.sum(res.radar_dynamic)) if res.radar_dynamic is not None else 0}"
            f" | tracks {n[TrackState.CONFIRMED]} tracking, {n[TrackState.LOST]} lost, {n[TrackState.RADAR_ONLY]} radar-only"
            f" | movers {len(res.movers)} | dets {res.n_dets} | ms: {tm}"
            + (f" | 재구성 {res.recon.n_chunks}묶음 {res.recon.points // 1000}k점" if res.recon is not None else "")
            + (f"\n{self.note}" if self.note else ""))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.redraw()
