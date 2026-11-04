#!/usr/bin/env python3
"""GT label review GUI (docs/13).  python3 scripts/label_review.py data/labels/<segment>

Keys
  A / <-, D / ->   previous / next frame          Space  play / pause (label rate)
  click box        select                         drag on empty area: draw a new box (asks the id)
  selected box     drag inside = move, drag a handle = resize (src -> manual)
  image            wheel = zoom at the cursor, right-drag = pan, 0 = fit, Z = zoom to the selected box
                   LiDAR / Radar checkboxes = project the frame's points on the image (LiDAR dots by range, radar x by Doppler)
  top-down panel   drag a box = move its ref3d exactly where dropped (hold Shift to snap to a LiDAR cluster within 0.4 m)
                   wheel zoom, right-drag pan;  T  hide / show;  X  clear ref3d;  Radar checkbox = radar scan
  N                set the id of the selected box (a box of that id in this frame swaps ids)
  M                merge the selected id into another id (all frames)
  S                split the selected id from this frame on into a new id
  I                interpolate the selected id between its labelled frames
  F                flags of the selected box (occluded / truncated / crouch / ignore) + occ none/partial/full
  Del              delete the selected box        Shift+Del  delete the selected id everywhere
  E                edit the description / class of the selected id
  C                mark this frame checked and go to the next     K  mark the selected id checked
  R                toggle open / resolved of the selected note
  B                brighten (night)               Ctrl+Z  undo
Box label: id, M = equipment, flags, [src]; below the box: ref3d range (green dense / yellow sparse / grey none).
Band under the slider: meta.ego_state (static grey / drive blue / swing orange / drive+swing red).
Top-down panel: LiDAR points of the frame (cache/bev_points.npz from scripts/label_bev_cache.py; grey = ground,
blue -> yellow = height), camera FOV, filled circle = ref3d.xy_level of a box (id colour), hollow square = box
without a 3D reference shows only its edge-ray wedge and "id ?", white = the selected box; small x = radar
points of the frame's scan (raw Doppler: red = approaching, blue = receding, grey = still; while the ego moves, static
points carry the ego speed too).
C also counts review.passes.user.frames_stepped (schema v2).
Every edit is saved to gt.json (timestamped backups in backup/ at most once a minute).
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)               # opencv-python points Qt at its own plugins
from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rcmot.labeling.gt import FLAGS, GT, OCC  # noqa: E402
from rcmot.utils.calib import RigCalib  # noqa: E402

STYLE = ("QMainWindow,QWidget{background:#18191b;color:#ddd} QPushButton{background:#2a2c30;border:1px solid #444;"
         "padding:4px 10px;border-radius:4px} QListWidget{background:#141516;border:1px solid #333}"
         "QListWidget::item:selected{background:#3d5a80} QSlider::groove:horizontal{height:6px;background:#333}"
         "QSlider::handle:horizontal{background:#ddd;width:12px;margin:-4px 0;border-radius:6px}")
REF_COLOR = {"dense": QtGui.QColor(80, 220, 80), "sparse": QtGui.QColor(240, 210, 60), "none": QtGui.QColor(150, 150, 150)}
EGO_COLOR = {"static": QtGui.QColor(90, 90, 95), "drive": QtGui.QColor(60, 120, 220), "swing": QtGui.QColor(235, 140, 40),
             "drive+swing": QtGui.QColor(220, 60, 60)}
HANDLE_PX = 7                                                      # half-size of the resize handles' hit area [px]
HANDLE_CURSOR = [QtCore.Qt.SizeFDiagCursor, QtCore.Qt.SizeVerCursor, QtCore.Qt.SizeBDiagCursor, QtCore.Qt.SizeHorCursor,
                 QtCore.Qt.SizeFDiagCursor, QtCore.Qt.SizeVerCursor, QtCore.Qt.SizeBDiagCursor, QtCore.Qt.SizeHorCursor]
SNAP_M = 0.4                                                       # BEV drop: snap to the LiDAR points within this radius


def id_qcolor(tid: int) -> QtGui.QColor:
    return QtGui.QColor.fromHsv((tid * 94) % 360, 200, 255)


class EgoBand(QtWidgets.QWidget):
    """Thin band under the frame slider: meta.ego_state segments coloured by state, with the current-frame cursor."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(10)
        self.segs, self.n, self.cur = [], 1, 0
        self.setToolTip("ego_state (static grey / drive blue / swing orange / drive+swing red)")

    def set_state(self, segs, n, cur):
        self.segs, self.n, self.cur = list(segs or []), max(int(n), 1), int(cur)
        self.update()

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        qp.fillRect(self.rect(), QtGui.QColor(40, 40, 44))
        for f0, f1, st in self.segs:
            x0, x1 = w * f0 / self.n, w * (f1 + 1) / self.n
            qp.fillRect(QtCore.QRectF(x0, 1, max(x1 - x0, 1), h - 2), EGO_COLOR.get(st, QtGui.QColor(200, 0, 200)))
        x = w * (self.cur + 0.5) / self.n
        qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 2))
        qp.drawLine(QtCore.QPointF(x, 0), QtCore.QPointF(x, h))


def to_pix(bgr: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QtGui.QPixmap.fromImage(QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy())


class FrameView(QtWidgets.QWidget):
    clicked = QtCore.pyqtSignal(float, float)          # image coords
    drawn = QtCore.pyqtSignal(list)                    # [x1, y1, x2, y2] image coords: a new box
    edited = QtCore.pyqtSignal(list)                   # [x1, y1, x2, y2] image coords: the selected box moved / resized

    def __init__(self):
        super().__init__()
        self.setMinimumSize(960, 540)
        self.setMouseTracking(True)
        self.pix, self.boxes, self.sel, self.drag = None, [], None, None
        self.scale, self.off = 1.0, (0, 0)
        self.mode = None                               # (kind 'move'|'resize', handle index, press pos, original xyxy)
        self.live = None                               # xyxy of the selected box while it is being dragged
        self.zoom, self.pan = 1.0, [0.0, 0.0]          # view zoom (1 = fit) and pan [px], kept across frames
        self.panning = None
        self.overlay = {}                              # "lidar": (uv (N,2), range (N,)), "radar": (uv, v_r)
        self.show_lidar, self.show_radar = False, False

    def set_frame(self, pix, boxes, sel, overlay=None):
        self.pix, self.boxes, self.sel, self.live = pix, boxes, sel, None
        self.overlay = overlay or {}
        self.update()

    @staticmethod
    def range_color(r):
        """5 m red -> 20 m yellow/green -> 40 m blue."""
        t = float(np.clip((r - 5.0) / 35.0, 0.0, 1.0))
        return QtGui.QColor.fromHsv(int(240 * t), 230, 255)

    def _draw_overlay(self, qp):
        s, (ox, oy) = self.scale, self.off
        if self.show_lidar and "lidar" in self.overlay:
            uv, rng = self.overlay["lidar"]
            qp.setRenderHint(QtGui.QPainter.Antialiasing, False)
            for k in range(8):                                        # 8 range bins, one pen each
                lo, hi = 5 + 35 * k / 8, 5 + 35 * (k + 1) / 8
                sel = (rng >= lo) & (rng < hi) if k < 7 else (rng >= lo)
                if k == 0:
                    sel |= rng < lo
                if sel.any():
                    qp.setPen(QtGui.QPen(self.range_color((lo + hi) / 2), max(2.0, 1.5 * self.zoom)))
                    qp.drawPoints(QtGui.QPolygonF([QtCore.QPointF(ox + u * s, oy + v * s) for u, v in uv[sel]]))
        if self.show_radar and "radar" in self.overlay:
            uv, vr = self.overlay["radar"]
            for sel, col in ((np.abs(vr) < 0.3, QtGui.QColor(230, 230, 230)), (vr <= -0.3, QtGui.QColor(255, 80, 80)),
                             (vr >= 0.3, QtGui.QColor(90, 160, 255))):
                if sel.any():
                    qp.setPen(QtGui.QPen(col, 2))
                    d = 4
                    for u, v in uv[sel]:
                        x, y = ox + u * s, oy + v * s
                        qp.drawLine(QtCore.QPointF(x - d, y - d), QtCore.QPointF(x + d, y + d))
                        qp.drawLine(QtCore.QPointF(x - d, y + d), QtCore.QPointF(x + d, y - d))

    def _geom(self):
        if self.pix is None:
            return
        s = min(self.width() / self.pix.width(), self.height() / self.pix.height()) * self.zoom
        self.scale = s
        self.off = ((self.width() - self.pix.width() * s) / 2 + self.pan[0], (self.height() - self.pix.height() * s) / 2 + self.pan[1])

    def set_zoom(self, zoom, anchor=None):
        """Zoom so the image point under `anchor` (widget px; default the centre) stays put."""
        if self.pix is None:
            return
        anchor = anchor or QtCore.QPointF(self.width() / 2, self.height() / 2)
        self._geom()
        ix, iy = self.to_img(anchor)
        self.zoom = float(np.clip(zoom, 1.0, 16.0))
        self._geom()
        nx, ny = self.to_img(anchor)
        self.pan[0] += (nx - ix) * self.scale
        self.pan[1] += (ny - iy) * self.scale
        if self.zoom <= 1.0:
            self.pan = [0.0, 0.0]
        self.update()

    def zoom_to(self, xyxy, frac=0.4):
        """Zoom so the box is `frac` of the view height, centred."""
        if self.pix is None:
            return
        x1, y1, x2, y2 = xyxy
        fit = min(self.width() / self.pix.width(), self.height() / self.pix.height())
        self.zoom = float(np.clip(frac * self.height() / max((y2 - y1) * fit, 1e-3), 1.0, 16.0))
        self.pan = [0.0, 0.0]
        self._geom()
        c = QtCore.QPointF(self.off[0] + (x1 + x2) / 2 * self.scale, self.off[1] + (y1 + y2) / 2 * self.scale)
        self.pan = [self.width() / 2 - c.x(), self.height() / 2 - c.y()]
        self.update()

    def wheelEvent(self, e):
        self.set_zoom(self.zoom * (1.25 if e.angleDelta().y() > 0 else 1 / 1.25), QtCore.QPointF(e.pos()))

    def to_img(self, p):
        return ((p.x() - self.off[0]) / self.scale, (p.y() - self.off[1]) / self.scale)

    def _rect(self, xyxy):
        x1, y1, x2, y2 = xyxy
        s, (ox, oy) = self.scale, self.off
        return QtCore.QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)

    @staticmethod
    def _handles(r):
        """TL, T, TR, R, BR, B, BL, L (PowerPoint order, clockwise from the top-left corner)."""
        cx, cy = r.center().x(), r.center().y()
        return [QtCore.QPointF(r.left(), r.top()), QtCore.QPointF(cx, r.top()), QtCore.QPointF(r.right(), r.top()),
                QtCore.QPointF(r.right(), cy), QtCore.QPointF(r.right(), r.bottom()), QtCore.QPointF(cx, r.bottom()),
                QtCore.QPointF(r.left(), r.bottom()), QtCore.QPointF(r.left(), cy)]

    def _hit(self, pos):
        """('resize', k) on a handle of the selected box, ('move', None) inside it, else None."""
        if self.sel is None or self.pix is None:
            return None
        r = self._rect(self.sel["xyxy"])
        for k, h in enumerate(self._handles(r)):
            if abs(h.x() - pos.x()) <= HANDLE_PX and abs(h.y() - pos.y()) <= HANDLE_PX:
                return ("resize", k)
        if r.contains(QtCore.QPointF(pos)):
            return ("move", None)
        return None

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.fillRect(self.rect(), QtGui.QColor(0, 0, 0))
        if self.pix is None:
            return
        self._geom()
        s, (ox, oy) = self.scale, self.off
        qp.drawPixmap(QtCore.QRectF(ox, oy, self.pix.width() * s, self.pix.height() * s), self.pix,
                      QtCore.QRectF(self.pix.rect()))
        self._draw_overlay(qp)
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        for b in self.boxes:
            is_sel = b is self.sel
            r = self._rect(self.live if (is_sel and self.live) else b["xyxy"])
            col = QtGui.QColor(150, 150, 150) if "ignore" in b["flags"] else id_qcolor(b["id"])
            pen = QtGui.QPen(col, 2)
            pen.setStyle({"det": QtCore.Qt.SolidLine, "interp": QtCore.Qt.DashLine}.get(b["src"], QtCore.Qt.DotLine))
            if is_sel:
                qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 5))
                qp.drawRect(r)
            qp.setPen(pen)
            qp.drawRect(r)
            tag = f"{b['id']}" + ("" if b["cls"] == "worker" else " M") + \
                  "".join(f" {f[:3]}" for f in b["flags"]) + ("" if b["src"] == "det" else f" [{b['src']}]")
            qp.setFont(QtGui.QFont("Sans", 10, QtGui.QFont.Bold))
            w = qp.fontMetrics().width(tag) + 6
            qp.fillRect(QtCore.QRectF(r.left(), r.top() - 17, w, 16), QtGui.QColor(0, 0, 0, 170))
            qp.setPen(col)
            qp.drawText(QtCore.QPointF(r.left() + 3, r.top() - 4), tag)
            ref = b.get("ref3d")
            if ref is not None:                                 # v2: LiDAR reference range, coloured by quality
                q = ref.get("quality", "none")
                rtxt = ("--" if ref.get("range_m") is None else f"{ref['range_m']:.1f}m") + ("" if q == "dense" else f" {q[:1]}")
                qp.setFont(QtGui.QFont("Sans", 9))
                w = qp.fontMetrics().width(rtxt) + 6
                qp.fillRect(QtCore.QRectF(r.left(), r.bottom() + 1, w, 14), QtGui.QColor(0, 0, 0, 170))
                qp.setPen(REF_COLOR.get(q, REF_COLOR["none"]))
                qp.drawText(QtCore.QPointF(r.left() + 3, r.bottom() + 12), rtxt)
            if b.get("occ") == "full":
                qp.setPen(QtGui.QPen(col, 1, QtCore.Qt.DotLine))
                qp.drawLine(r.topLeft(), r.bottomRight())
            if is_sel:                                          # resize handles
                qp.setPen(QtGui.QPen(QtGui.QColor(30, 30, 30), 1))
                qp.setBrush(QtGui.QColor(255, 255, 255))
                for h in self._handles(r):
                    qp.drawRect(QtCore.QRectF(h.x() - 4, h.y() - 4, 8, 8))
                qp.setBrush(QtCore.Qt.NoBrush)
        if self.drag:
            qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 2, QtCore.Qt.DashLine))
            qp.drawRect(QtCore.QRectF(self.drag[0], self.drag[1]).normalized())

    def _edited_box(self, pos):
        kind, k, p0, (x1, y1, x2, y2) = self.mode
        dx, dy = (pos.x() - p0.x()) / self.scale, (pos.y() - p0.y()) / self.scale
        if kind == "move":
            x1, y1, x2, y2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
        else:
            if k in (0, 6, 7):
                x1 += dx
            if k in (2, 3, 4):
                x2 += dx
            if k in (0, 1, 2):
                y1 += dy
            if k in (4, 5, 6):
                y2 += dy
        W, H = self.pix.width(), self.pix.height()
        x1, x2 = sorted((min(max(x1, 0), W), min(max(x2, 0), W)))
        y1, y2 = sorted((min(max(y1, 0), H), min(max(y2, 0), H)))
        if x2 - x1 < 2:
            x2 = min(x1 + 2, W)
        if y2 - y1 < 2:
            y2 = min(y1 + 2, H)
        return [x1, y1, x2, y2]

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.RightButton:
            self.panning = (e.pos(), list(self.pan))
            return
        if e.button() != QtCore.Qt.LeftButton:
            return
        hit = self._hit(e.pos())
        if hit:
            self.mode = (hit[0], hit[1], e.pos(), list(self.sel["xyxy"]))
            self.live = list(self.sel["xyxy"])
        else:
            self.drag = [e.pos(), e.pos()]

    def mouseMoveEvent(self, e):
        if self.panning:
            p0, pan0 = self.panning
            self.pan = [pan0[0] + e.pos().x() - p0.x(), pan0[1] + e.pos().y() - p0.y()]
            self.update()
        elif self.mode:
            self.live = self._edited_box(e.pos())
            self.update()
        elif self.drag:
            self.drag[1] = e.pos()
            self.update()
        else:
            hit = self._hit(e.pos())
            self.setCursor(QtGui.QCursor(HANDLE_CURSOR[hit[1]] if hit and hit[0] == "resize" else
                                         QtCore.Qt.SizeAllCursor if hit else QtCore.Qt.CrossCursor))

    def mouseReleaseEvent(self, e):
        if self.panning:
            self.panning = None
            return
        if self.mode:
            kind, k, p0, orig = self.mode
            self.mode = None
            moved = (e.pos() - p0).manhattanLength() >= 3
            box, self.live = self.live, None
            self.update()
            if moved:
                self.edited.emit(box)
            elif kind == "move":                                # a plain click inside the selected box: reselect (smaller box)
                self.clicked.emit(*self.to_img(e.pos()))
            return
        if not self.drag:
            return
        a, b = self.drag
        self.drag = None
        self.update()
        if (a - b).manhattanLength() < 8:
            self.clicked.emit(*self.to_img(b))
        else:
            (x1, y1), (x2, y2) = self.to_img(a), self.to_img(b)
            self.drawn.emit([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])


class BevView(QtWidgets.QWidget):
    """Top-down (level frame, x forward = up, y left = left) view of the frame: LiDAR points, camera FOV and the
    3D references of the boxes.  Dragging a marker moves the box's ref3d.xy_level (snapped to the LiDAR points)."""
    moved = QtCore.pyqtSignal(int, float, float)       # id, x, y [m, level frame]

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(300)
        self.setMouseTracking(True)
        self.above = np.zeros((0, 2))
        self.h = np.zeros(0)
        self.ground = np.zeros((0, 2))
        self.marks: list[dict] = []
        self.radar = (np.zeros((0, 2)), np.zeros(0))
        self.hot = (np.zeros((0, 2)), np.zeros((0, 2)))             # LiDAR / radar points that project inside the selected box
        self.show_radar = True
        self.cam, self.fov, self.msg = (0.0, 0.0), None, ""
        self.cx, self.cy, self.s = 16.0, 0.0, 9.0        # view centre [m] and scale [px/m]
        self.drag, self.pan = None, None

    def set_frame(self, above, h, ground, marks, cam, fov, msg, radar=None, hot=None):
        self.above, self.h, self.ground, self.marks, self.cam, self.fov, self.msg = above, h, ground, marks, cam, fov, msg
        self.radar = radar if radar is not None else (np.zeros((0, 2)), np.zeros(0))
        self.hot = hot if hot is not None else (np.zeros((0, 2)), np.zeros((0, 2)))
        self.update()

    def to_scr(self, x, y):
        return QtCore.QPointF(self.width() / 2 - (y - self.cy) * self.s, self.height() / 2 - (x - self.cx) * self.s)

    def to_lvl(self, p):
        return (self.cx - (p.y() - self.height() / 2) / self.s, self.cy - (p.x() - self.width() / 2) / self.s)

    def _pts(self, xy):
        return [QtCore.QPointF(self.width() / 2 - (y - self.cy) * self.s, self.height() / 2 - (x - self.cx) * self.s) for x, y in xy]

    def paintEvent(self, _):
        qp = QtGui.QPainter(self)
        qp.fillRect(self.rect(), QtGui.QColor(12, 13, 15))
        W, H = self.width(), self.height()
        o = self.to_scr(0.0, 0.0)
        qp.setPen(QtGui.QPen(QtGui.QColor(45, 47, 52), 1))
        qp.setFont(QtGui.QFont("Sans", 8))
        for r in range(5, 101, 5):                                   # range rings about the level origin
            qp.drawEllipse(o, r * self.s, r * self.s)
            if r % 10 == 0:
                qp.drawText(QtCore.QPointF(o.x() + 3, o.y() - r * self.s - 2), f"{r}m")
        if self.fov is not None:                                     # camera field of view
            cx, cy = self.cam
            c = self.to_scr(cx, cy)
            qp.setPen(QtGui.QPen(QtGui.QColor(80, 80, 90), 1, QtCore.Qt.DashLine))
            for dx, dy in self.fov:
                qp.drawLine(c, self.to_scr(cx + 120 * dx, cy + 120 * dy))
            qp.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200), 1))
            qp.drawEllipse(c, 3, 3)
        if len(self.ground):
            qp.setPen(QtGui.QPen(QtGui.QColor(70, 72, 78), 1))
            qp.drawPoints(QtGui.QPolygonF(self._pts(self.ground)))
        if len(self.above):                                          # height -> blue .. yellow in 6 bins
            bins = np.clip((self.h / 3.0 * 6).astype(int), 0, 5)
            for k in range(6):
                sel = bins == k
                if sel.any():
                    t = k / 5
                    col = QtGui.QColor(int(60 + 190 * t), int(90 + 140 * t), int(220 - 170 * t))
                    qp.setPen(QtGui.QPen(col, 2))
                    qp.drawPoints(QtGui.QPolygonF(self._pts(self.above[sel])))
        rxy, rv = self.radar
        if self.show_radar and len(rxy):                             # radar scan: x marks coloured by raw Doppler
            for sel, col in ((np.abs(rv) < 0.3, QtGui.QColor(200, 200, 200)), (rv <= -0.3, QtGui.QColor(255, 90, 90)),
                             (rv >= 0.3, QtGui.QColor(90, 160, 255))):
                if sel.any():
                    qp.setPen(QtGui.QPen(col, 1))
                    for pt in self._pts(rxy[sel]):
                        qp.drawLine(QtCore.QPointF(pt.x() - 3, pt.y() - 3), QtCore.QPointF(pt.x() + 3, pt.y() + 3))
                        qp.drawLine(QtCore.QPointF(pt.x() - 3, pt.y() + 3), QtCore.QPointF(pt.x() + 3, pt.y() - 3))
        hl, hr = self.hot                                            # points seen inside the selected image box: yellow
        if len(hl):
            qp.setPen(QtGui.QPen(QtGui.QColor(255, 230, 0), 5))
            qp.drawPoints(QtGui.QPolygonF(self._pts(hl)))
        if len(hr) and self.show_radar:
            qp.setPen(QtGui.QPen(QtGui.QColor(255, 230, 0), 2))
            for pt in self._pts(hr):
                qp.drawLine(QtCore.QPointF(pt.x() - 5, pt.y() - 5), QtCore.QPointF(pt.x() + 5, pt.y() + 5))
                qp.drawLine(QtCore.QPointF(pt.x() - 5, pt.y() + 5), QtCore.QPointF(pt.x() + 5, pt.y() - 5))
        qp.setRenderHint(QtGui.QPainter.Antialiasing)
        qp.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Bold))
        qp.setBrush(QtCore.Qt.NoBrush)
        cam = self.to_scr(*self.cam)
        for m in self.marks:                                         # outlines only: the LiDAR points stay visible
            p = self.to_scr(*m["xy"])
            x, y = m["xy"]
            dx, dy = x - self.cam[0], y - self.cam[1]
            r = max(float(np.hypot(dx, dy)), 1e-3)
            ux, uy = dx / r, dy / r                                  # unit vector camera -> marker (level xy)
            if m.get("edges") is not None:                           # wedge of the image box's left / right edges
                col = QtGui.QColor(255, 255, 255, 150) if m["sel"] else QtGui.QColor(m["col"].red(), m["col"].green(), m["col"].blue(), 70)
                qp.setPen(QtGui.QPen(col, 1, QtCore.Qt.DotLine))
                for ex, ey in m["edges"]:
                    qp.drawLine(cam, self.to_scr(self.cam[0] + ex * (r + 3.0), self.cam[1] + ey * (r + 3.0)))
            if m["sel"] and m.get("ray") is not None:                # where the box bottom says the feet are
                qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 160), 1))
                q = self.to_scr(*m["ray"])
                qp.drawLine(QtCore.QPointF(q.x() - 5, q.y()), QtCore.QPointF(q.x() + 5, q.y()))
                qp.drawLine(QtCore.QPointF(q.x(), q.y() - 5), QtCore.QPointF(q.x(), q.y() + 5))
            if m["kind"] != "ref":                                   # no reference: only the wedge and the id, no position
                qp.setPen(QtGui.QPen(m["col"], 1))
                qp.drawText(QtCore.QPointF(p.x() + 6, p.y() - 6), f"{m['id']} ?")
                continue
            hw = max(m.get("half_w", 0.35) * self.s, 4.0)            # image box width at this range, in px
            hd = max(0.3 * self.s, 3.0)                              # 0.6 m deep
            poly = QtGui.QPolygonF()
            for a, b in ((-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)):   # rectangle across the viewing ray
                wx = x + (-uy) * a / self.s + ux * b / self.s
                wy = y + ux * a / self.s + uy * b / self.s
                poly.append(self.to_scr(wx, wy))
            if m["sel"]:
                qp.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), 3))
                qp.drawPolygon(poly)
            qp.setPen(QtGui.QPen(m["col"], 2, QtCore.Qt.SolidLine if m["kind"] == "ref" else QtCore.Qt.DashLine))
            qp.drawPolygon(poly)
            qp.drawText(QtCore.QPointF(p.x() + hw + 4, p.y() - 6), str(m["id"]))
        qp.setPen(QtGui.QColor(170, 170, 170))
        qp.setFont(QtGui.QFont("Sans", 8))
        qp.drawText(QtCore.QPointF(6, 14), self.msg)
        qp.drawText(QtCore.QPointF(6, H - 6), "box = ref3d (width = image box width at that range); '?' = no ref; yellow = points inside the selected image box; drop = exact, Shift = snap")

    def _mark_at(self, pos):
        best, bd = None, 12.0
        for m in self.marks:
            d = (self.to_scr(*m["xy"]) - QtCore.QPointF(pos)).manhattanLength()
            if d < bd:
                best, bd = m, d
        return best

    def wheelEvent(self, e):
        f = 1.2 if e.angleDelta().y() > 0 else 1 / 1.2
        x, y = self.to_lvl(e.pos())
        self.s = float(np.clip(self.s * f, 1.5, 80))
        nx, ny = self.to_lvl(e.pos())
        self.cx, self.cy = self.cx + (x - nx), self.cy + (y - ny)
        self.update()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton:
            m = self._mark_at(e.pos())
            if m is not None:
                self.drag = (m, tuple(m["xy"]))
        else:
            self.pan = (e.pos(), self.cx, self.cy)

    def mouseMoveEvent(self, e):
        if self.drag:
            self.drag[0]["xy"] = self.to_lvl(e.pos())
            self.update()
        elif self.pan:
            p0, cx0, cy0 = self.pan
            self.cx, self.cy = cx0 + (e.pos().y() - p0.y()) / self.s, cy0 + (e.pos().x() - p0.x()) / self.s
            self.update()
        else:
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor if self._mark_at(e.pos()) else QtCore.Qt.ArrowCursor))

    def mouseReleaseEvent(self, e):
        if self.pan:
            self.pan = None
            return
        if not self.drag:
            return
        m, xy0 = self.drag
        self.drag = None
        x, y = m["xy"]
        if np.hypot(x - xy0[0], y - xy0[1]) < 0.05:
            m["xy"] = xy0
            self.update()
            return
        snapped = 0
        if (e.modifiers() & QtCore.Qt.ShiftModifier) and len(self.above):   # snap only on request
            d = np.hypot(self.above[:, 0] - x, self.above[:, 1] - y)
            d0 = np.hypot(self.above[:, 0] - xy0[0], self.above[:, 1] - xy0[1])
            near = (d < SNAP_M) & (d0 > SNAP_M)                       # never snap back onto the cluster it was dragged away from
            if near.any():
                x, y = self.above[near].mean(0)
                snapped = int(near.sum())
        m["xy"] = (float(x), float(y))
        self.msg = self.msg.split(" || ")[0] + (f" || snapped to {snapped} LiDAR pts" if snapped else " || placed freely (no snap)")
        self.update()
        self.moved.emit(m["id"], float(x), float(y))


class Review(QtWidgets.QMainWindow):
    def __init__(self, seg: Path):
        super().__init__()
        self.gt = GT(seg / "gt.json")
        self.i, self.sel, self.bright = 0, None, "night" in seg.name
        self.undo: list = []
        self.last_backup = 0.0
        self.thumbs: dict = {}
        self.cal, self.bev_cache = None, None
        try:
            self.cal = RigCalib.load(str(ROOT / self.gt.meta["calib"]))
        except Exception as ex:                                    # noqa: BLE001 - the panel then shows GT points only
            print(f"calib not loaded: {ex}")
        cpath = seg / "cache" / "bev_points.npz"
        if cpath.exists():
            z = np.load(cpath, allow_pickle=True)
            self.bev_cache = {k: z[k] for k in ("xy", "h", "off", "n_above", "zg")}
            if "roff" in z.files:
                self.bev_cache.update({k: z[k] for k in ("rxy", "rv", "roff") + (("rz",) if "rz" in z.files else ())})
        self.ray_gain = 1.0
        self.ray_gain = self._ray_gain()
        self.setWindowTitle(f"GT review | {seg.name}")
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1900, scr.width() - 80), min(1000, scr.height() - 80))
        self.setStyleSheet(STYLE)
        c = QtWidgets.QWidget()
        self.setCentralWidget(c)
        lay = QtWidgets.QHBoxLayout(c)
        left = QtWidgets.QVBoxLayout()
        lay.addLayout(left, 3)
        ihdr = QtWidgets.QHBoxLayout()
        ihdr.addWidget(QtWidgets.QLabel("<b>Image</b> (wheel zoom, right-drag pan, 0 fit, Z zoom to box)"), 1)
        self.lidar_img_cb = QtWidgets.QCheckBox("LiDAR")
        self.lidar_img_cb.setToolTip("project the frame's LiDAR points above ground (dot colour = range: red 5 m .. blue 40 m)")
        self.radar_img_cb = QtWidgets.QCheckBox("Radar")
        self.radar_img_cb.setToolTip("project the frame's radar scan (x: red = approaching, blue = receding, grey = still)")
        ihdr.addWidget(self.lidar_img_cb)
        ihdr.addWidget(self.radar_img_cb)
        left.addLayout(ihdr)
        self.view = FrameView()
        left.addWidget(self.view, 1)
        self.lidar_img_cb.toggled.connect(lambda on: (setattr(self.view, "show_lidar", on), self.view.update()))
        self.radar_img_cb.toggled.connect(lambda on: (setattr(self.view, "show_radar", on), self.view.update()))
        bar = QtWidgets.QHBoxLayout()
        left.addLayout(bar)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, len(self.gt.frames) - 1)
        self.lbl = QtWidgets.QLabel()
        self.lbl.setStyleSheet("font-family:monospace")
        self.lbl.setMinimumWidth(0)
        bar.addWidget(self.slider, 1)
        bar.addWidget(self.lbl)
        self.ego = EgoBand()
        left.addWidget(self.ego)
        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet("font-family:monospace;color:#aaa")
        self.status.setWordWrap(True)                                   # long id descriptions must not widen the window
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        left.addWidget(self.status)
        right = QtWidgets.QVBoxLayout()
        lay.addLayout(right, 1)
        hdr = QtWidgets.QHBoxLayout()
        hdr.addWidget(QtWidgets.QLabel("<b>Top-down</b> (level frame; T = hide)"), 1)
        self.radar_cb = QtWidgets.QCheckBox("Radar")
        self.radar_cb.setChecked(True)
        self.radar_cb.setToolTip("radar scan of the frame: x marks, red = approaching, blue = receding, grey = still (raw Doppler)")
        hdr.addWidget(self.radar_cb)
        right.addLayout(hdr)
        self.bev = BevView()
        self.radar_cb.toggled.connect(lambda on: (setattr(self.bev, "show_radar", on), self.bev.update()))
        right.addWidget(self.bev, 3)
        right.addWidget(QtWidgets.QLabel("<b>IDs</b> (click = first appearance)"))
        self.idlist = QtWidgets.QListWidget()
        self.idlist.setIconSize(QtCore.QSize(150, 50))
        right.addWidget(self.idlist, 3)
        right.addWidget(QtWidgets.QLabel("<b>Notes</b> (review these first; click = jump)"))
        self.notes = QtWidgets.QListWidget()
        self.notes.setWordWrap(True)
        right.addWidget(self.notes, 2)
        helpl = QtWidgets.QLabel(__doc__.split("Keys")[1].split("Box label")[0])
        helpl.setStyleSheet("font-family:monospace;font-size:10px;color:#999")
        right.addWidget(helpl)
        self.slider.valueChanged.connect(self.goto)
        self.view.clicked.connect(self.pick)
        self.view.drawn.connect(self.new_box)
        self.view.edited.connect(self.edit_box)
        self.bev.moved.connect(self.move_ref)
        self.idlist.itemClicked.connect(lambda it: self.goto(self.gt.id_frames(it.data(32))[0], it.data(32)))
        self.notes.itemClicked.connect(lambda it: self.goto(it.data(32)[0], it.data(32)[1]))
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(lambda: self.goto(min(self.i + 1, len(self.gt.frames) - 1)))
        keys = {"A": lambda: self.goto(self.i - 1), "Left": lambda: self.goto(self.i - 1),
                "D": lambda: self.goto(self.i + 1), "Right": lambda: self.goto(self.i + 1),
                "Space": self.play, "N": self.set_id, "M": self.merge, "S": self.split, "I": self.interp,
                "F": self.flags, "Del": self.del_box, "Shift+Del": self.del_id, "E": self.edit_id,
                "C": self.check_frame, "K": self.check_id, "B": self.toggle_bright, "Ctrl+Z": self.do_undo,
                "R": self.toggle_note, "T": lambda: self.bev.setVisible(not self.bev.isVisible()), "X": self.clear_ref,
                "0": lambda: self.view.set_zoom(1.0), "Z": lambda: self.sel and self.view.zoom_to(self.sel["xyxy"])}
        for k, f in keys.items():
            QtWidgets.QShortcut(QtGui.QKeySequence(k), self, activated=f)
        self.refresh_lists()
        self.goto(0)

    # ------------------------------------------------------------------ state
    def frame(self):
        return self.gt.frames[self.i]

    def snapshot(self):
        self.undo.append(copy.deepcopy(self.gt.to_dict()))
        self.undo = self.undo[-50:]

    def commit(self, lists=True):
        now = time.time()
        self.gt.save(backup=now - self.last_backup > 60)
        if now - self.last_backup > 60:
            self.last_backup = now
        if lists:
            self.thumbs.clear()
            self.refresh_lists()
        self.goto(self.i, keep_sel=True)

    def do_undo(self):
        if not self.undo:
            return
        d = self.undo.pop()
        json.dump(d, open(self.gt.path, "w"), indent=1)
        self.gt = GT(self.gt.path)
        self.sel = None
        self.commit()

    # ------------------------------------------------------------------ display
    def image(self, i):
        img = cv2.imread(str(self.gt.root / self.gt.frames[i]["file"]))
        return cv2.convertScaleAbs(img, alpha=2.2, beta=10) if self.bright else img

    def ground_z(self, i):
        """Ground height in the level frame: the cache's per-frame estimate, else 0 (the level frame sits on the ground)."""
        return float(self.bev_cache["zg"][i]) if self.bev_cache is not None else 0.0

    def ray_hit(self, box, zg, gain=None):
        """Where the box's bottom-centre camera ray meets the ground plane (level xy), scaled by ray_gain: the plain
        ground intersection is 10-35 % too far on the HDX rigs (feet cut, pitch) and 3x too near on xpanner (level frame
        not on the ground), so the range is calibrated once per segment against the boxes that have a LiDAR reference."""
        if self.cal is None:
            return None
        x1, _, x2, y2 = box["xyxy"]
        p = self.cal.pixel_ray_to_ground((x1 + x2) / 2, y2, ground_z=zg)
        if p is None or np.hypot(p[0], p[1]) > 150:
            return None
        g = self.ray_gain if gain is None else gain
        return (float(p[0] * g), float(p[1] * g))

    def _ray_gain(self):
        rat = []
        for i, f in enumerate(self.gt.frames):
            zg = self.ground_z(i)
            for b in f["boxes"]:
                ref = b.get("ref3d")
                if not ref or ref.get("range_m") is None or b.get("occ") != "none" or "ignore" in b["flags"]:
                    continue
                p = self.ray_hit(b, zg, gain=1.0)
                if p is not None and np.hypot(*p) > 1.0:
                    rat.append(ref["range_m"] / np.hypot(*p))
        return float(np.clip(np.median(rat), 0.2, 3.0)) if len(rat) >= 10 else 1.0

    def inbox_points(self, box):
        """(LiDAR xy, radar xy) of the cache points whose image projection falls inside the box (10 % x / 5 % y margin)."""
        empty = (np.zeros((0, 2)), np.zeros((0, 2)))
        if box is None or self.cal is None or self.bev_cache is None:
            return empty
        x1, y1, x2, y2 = box["xyxy"]
        w, h = x2 - x1, y2 - y1
        inside = lambda uv: (uv[:, 0] > x1 + 0.1 * w) & (uv[:, 0] < x2 - 0.1 * w) & (uv[:, 1] > y1 + 0.05 * h) & (uv[:, 1] < y2 - 0.05 * h)
        z = self.bev_cache
        a, b, na = int(z["off"][self.i]), int(z["off"][self.i + 1]), int(z["n_above"][self.i])
        xy = z["xy"][a:b][:na].astype(np.float64) / 100.0
        hl = np.zeros((0, 2))
        if len(xy):
            zz = self.ground_z(self.i) + z["h"][a:b][:na].astype(np.float64) * 0.05
            uv, fr = self.cal.project(np.column_stack([xy, zz]))
            ok = fr & np.isfinite(uv).all(1)
            ok[ok] &= inside(uv[ok])
            hl = xy[ok]
        hr = np.zeros((0, 2))
        if "roff" in z:
            ra, rb = int(z["roff"][self.i]), int(z["roff"][self.i + 1])
            if rb > ra:
                rxy = z["rxy"][ra:rb].astype(np.float64) / 100.0
                rzz = z["rz"][ra:rb].astype(np.float64) if "rz" in z else np.full(rb - ra, 1.0)
                uv, fr = self.cal.project(np.column_stack([rxy, rzz]))
                ok = fr & np.isfinite(uv).all(1)
                ok[ok] &= inside(uv[ok])
                hr = rxy[ok]
        return hl, hr

    def image_overlay(self):
        """Frame points projected into the image: {"lidar": (uv, range), "radar": (uv, v_r)} from the BEV cache."""
        out = {}
        if self.cal is None or self.bev_cache is None:
            return out
        z = self.bev_cache
        a, b, na = int(z["off"][self.i]), int(z["off"][self.i + 1]), int(z["n_above"][self.i])
        xy = z["xy"][a:b][:na].astype(np.float64) / 100.0
        if len(xy):
            zz = self.ground_z(self.i) + z["h"][a:b][:na].astype(np.float64) * 0.05
            uv, fr = self.cal.project(np.column_stack([xy, zz]))
            ok = fr & np.isfinite(uv).all(1)
            out["lidar"] = (uv[ok], np.hypot(xy[ok, 0], xy[ok, 1]))
        if "roff" in z:
            ra, rb = int(z["roff"][self.i]), int(z["roff"][self.i + 1])
            if rb > ra:
                rxy = z["rxy"][ra:rb].astype(np.float64) / 100.0
                rzz = z["rz"][ra:rb].astype(np.float64) if "rz" in z else np.full(rb - ra, 1.0)
                uv, fr = self.cal.project(np.column_stack([rxy, rzz]))
                ok = fr & np.isfinite(uv).all(1)
                out["radar"] = (uv[ok], z["rv"][ra:rb].astype(np.float64)[ok])
        return out

    def update_bev(self):
        zg = self.ground_z(self.i)
        above, h, ground, radar = np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2)), None
        msg = "LiDAR cache missing: PYTHONPATH=src python3 scripts/label_bev_cache.py <seg>"
        if self.bev_cache is not None:
            z = self.bev_cache
            a, b, na = int(z["off"][self.i]), int(z["off"][self.i + 1]), int(z["n_above"][self.i])
            xy = z["xy"][a:b].astype(np.float64) / 100.0
            above, h, ground = xy[:na], z["h"][a:b][:na].astype(np.float64) * 0.05, xy[na:]
            msg = f"f{self.i}: LiDAR {na} pts above ground (z_ground {zg:+.2f} m, ray x{self.ray_gain:.2f})"
            if "roff" in z:
                ra, rb = int(z["roff"][self.i]), int(z["roff"][self.i + 1])
                radar = (z["rxy"][ra:rb].astype(np.float64) / 100.0, z["rv"][ra:rb].astype(np.float64))
                msg += f", radar {rb - ra} pts"
        marks = []
        cam = tuple(self.cal.camera_center()[:2]) if self.cal is not None else (0.0, 0.0)
        for b in self.frame()["boxes"]:
            ref = b.get("ref3d")
            ray = self.ray_hit(b, zg)
            col = QtGui.QColor(150, 150, 150) if "ignore" in b["flags"] else id_qcolor(b["id"])
            if ref and ref.get("xy_level") is not None:
                m = {"id": b["id"], "xy": tuple(ref["xy_level"]), "kind": "ref", "col": col, "sel": b is self.sel, "ray": ray}
            elif ray is not None:
                m = {"id": b["id"], "xy": ray, "kind": "ray", "col": col, "sel": b is self.sel, "ray": ray}
            else:
                continue
            if self.cal is not None:                                 # image box left / right edge rays -> width at the marker range
                x1, y1, x2, y2 = b["xyxy"]
                v = (y1 + y2) / 2
                edges = []
                for u in (x1, x2):
                    d = self.cal.pixel_ray(u, v)[:2]
                    edges.append(tuple(d / max(np.linalg.norm(d), 1e-6)))
                r = float(np.hypot(m["xy"][0] - cam[0], m["xy"][1] - cam[1]))
                ang = float(np.arccos(np.clip(np.dot(edges[0], edges[1]), -1.0, 1.0)))
                m["edges"], m["half_w"] = edges, r * np.tan(ang / 2)
            marks.append(m)
        fov = None
        if self.cal is not None:
            W, H = self.cal.image_size
            fov = []
            for u in (0, W):
                d = self.cal.pixel_ray(u, H / 2)[:2]
                fov.append(tuple(d / max(np.linalg.norm(d), 1e-6)))
        hot = self.inbox_points(self.sel)
        if self.sel is not None:
            msg += f" | selected box: {len(hot[0])} LiDAR, {len(hot[1])} radar pts inside"
        self.bev.set_frame(above, h, ground, marks, cam, fov, msg, radar, hot)

    def goto(self, i, tid=None, keep_sel=False):
        self.i = int(np.clip(i, 0, len(self.gt.frames) - 1))
        if tid is not None:
            self.sel = self.gt.box_of(self.i, tid)
        elif not keep_sel:
            sid = self.sel["id"] if self.sel else None
            self.sel = self.gt.box_of(self.i, sid) if sid is not None else None
        self.slider.blockSignals(True)
        self.slider.setValue(self.i)
        self.slider.blockSignals(False)
        f = self.frame()
        self.view.set_frame(to_pix(self.image(self.i)), f["boxes"], self.sel, self.image_overlay())
        self.update_bev()
        rv = self.gt.review
        st = rv["frames"].get(self.gt.review_key(self.i), "draft")
        nchk = sum(v == "checked" for v in rv["frames"].values())
        ichk = sum(v == "checked" for v in rv["ids"].values())
        ego = self.gt.ego_state_at(self.i)
        self.lbl.setText(f"f{self.i:4d}/{len(self.gt.frames) - 1}  t={f['bag_t'] - self.gt.frames[0]['bag_t']:5.1f}s  [{st}]"
                         + (f" ego={ego}" if ego else ""))
        self.ego.set_state(self.gt.meta.get("ego_state"), len(self.gt.frames), self.i)
        s = self.sel
        extra = ""
        if s and "occ" in s:
            ref = s.get("ref3d")
            extra = (f" occ={s['occ']} ign={s.get('ignore_reason')}"
                     + (f" ref3d={ref.get('range_m')}m/{ref.get('quality')}/{ref.get('src')}" if ref else " ref3d=-"))
        self.status.setText(f"frames checked {nchk}/{len(self.gt.frames)} | ids checked {ichk}/{len(self.gt.ids)} | "
                            + (f"selected id {s['id']} {s['cls']} src={s['src']} conf={s['conf']:.2f} flags={s['flags']}"
                               f"{extra} | {self.gt.ids.get(s['id'], {}).get('desc', '')}" if s else "no selection"))

    def thumb(self, tid):
        if tid not in self.thumbs:
            fr = self.gt.id_frames(tid)
            pick = [fr[int(k)] for k in np.linspace(0, len(fr) - 1, min(5, len(fr)))] if fr else []
            crops = []
            for i in pick:
                img = self.image(i)
                x1, y1, x2, y2 = np.array(self.gt.box_of(i, tid)["xyxy"]).astype(int)
                c = img[max(y1, 0):max(y2, y1 + 2), max(x1, 0):max(x2, x1 + 2)]
                crops.append(cv2.resize(c, (30, 50)) if c.size else np.zeros((50, 30, 3), np.uint8))
            crops += [np.zeros((50, 30, 3), np.uint8)] * (5 - len(crops))
            self.thumbs[tid] = QtGui.QIcon(to_pix(np.hstack(crops)))
        return self.thumbs[tid]

    def refresh_lists(self):
        self.idlist.clear()
        for tid in sorted(self.gt.ids):
            fr = self.gt.id_frames(tid)
            if not fr:
                continue
            v = self.gt.ids[tid]
            it = QtWidgets.QListWidgetItem(self.thumb(tid), f"{tid}  {v['cls'][:6]}  n={len(fr)}  f{fr[0]}-{fr[-1]}  "
                                           f"[{self.gt.review['ids'].get(tid, 'draft')}]\n{v.get('desc', '')}")
            it.setForeground(id_qcolor(tid))
            it.setData(32, tid)
            self.idlist.addItem(it)
        self.notes.clear()
        for j, n in enumerate(self.gt.notes):
            k = n.get("frame", self.gt.frame_index(n["stamp"]))
            resolved = n.get("status", "open") == "resolved"
            head = f"[{n.get('kind', '-')}/{'ok' if resolved else 'open'}] "
            it = QtWidgets.QListWidgetItem(f"{head}f{k} id {n.get('id')}: {n['text']}")
            it.setData(32, (k, n.get("id")))
            it.setData(33, j)
            if resolved:
                ft = it.font()
                ft.setStrikeOut(True)
                it.setFont(ft)
                it.setForeground(QtGui.QColor(120, 120, 120))
            self.notes.addItem(it)

    # ------------------------------------------------------------------ actions
    def pick(self, x, y):
        hits = [b for b in self.frame()["boxes"] if b["xyxy"][0] <= x <= b["xyxy"][2] and b["xyxy"][1] <= y <= b["xyxy"][3]]
        self.sel = min(hits, key=lambda b: (b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1])) if hits else None
        self.goto(self.i, keep_sel=True)

    def ask_id(self, title, default):
        v, ok = QtWidgets.QInputDialog.getInt(self, title, f"id (new = {self.gt.new_id()})", default, 1, 9999)
        return v if ok else None

    def new_box(self, xyxy):
        tid = self.ask_id("new box", self.sel["id"] if self.sel else self.gt.new_id())
        if tid is None:
            return
        self.snapshot()
        self.gt.add_box(self.i, tid, xyxy, src="manual")
        self.sel = self.gt.box_of(self.i, tid)
        self.commit()

    def edit_box(self, xyxy):
        """The selected box was moved / resized by hand: new corners, src -> manual (the detector box is gone)."""
        if not self.sel:
            return
        self.snapshot()
        self.sel["xyxy"] = [round(float(v), 1) for v in xyxy]
        self.sel["src"] = "manual"
        self.commit(lists=False)

    def move_ref(self, tid, x, y):
        """A marker was dragged in the top-down panel: the box's 3D reference is set by hand (quality sparse, src manual)."""
        if self.gt.box_of(self.i, tid) is None:
            return
        self.snapshot()
        self.gt.set_ref3d(self.i, tid, {"range_m": float(np.hypot(x, y)), "xy_level": [x, y], "n_lidar": 0,
                                        "quality": "sparse", "src": "manual"})
        self.sel = self.gt.box_of(self.i, tid)
        self.commit(lists=False)

    def clear_ref(self):
        if not self.sel or self.sel.get("ref3d") is None:
            return
        self.snapshot()
        self.gt.set_ref3d(self.i, self.sel["id"], None)
        self.commit(lists=False)

    def set_id(self):
        if not self.sel:
            return
        tid = self.ask_id("set id", self.sel["id"])
        if tid is None:
            return
        self.snapshot()
        self.gt.set_id(self.i, self.sel, tid)
        self.commit()

    def merge(self):
        if not self.sel:
            return
        src = self.sel["id"]
        dst = self.ask_id(f"merge id {src} into", src)
        if dst is None or dst == src:
            return
        self.snapshot()
        self.gt.merge(src, dst)
        self.sel = self.gt.box_of(self.i, dst)
        self.commit()

    def split(self):
        if not self.sel:
            return
        self.snapshot()
        nid = self.gt.split(self.sel["id"], self.i)
        self.sel = self.gt.box_of(self.i, nid)
        self.commit()

    def interp(self):
        if not self.sel:
            return
        self.snapshot()
        n = self.gt.interpolate(self.sel["id"])
        self.statusBar().showMessage(f"interpolated {n} boxes", 3000)
        self.commit()

    def flags(self):
        if not self.sel:
            return
        menu = QtWidgets.QMenu(self)
        for f in FLAGS:
            a = menu.addAction(f)
            a.setCheckable(True)
            a.setChecked(f in self.sel["flags"])
        menu.addSeparator()
        cur = self.sel.get("occ", "partial" if "occluded" in self.sel["flags"] else "none")
        for o in OCC:
            a = menu.addAction(f"occ {o}")
            a.setCheckable(True)
            a.setChecked(o == cur)
        a = menu.exec_(QtGui.QCursor.pos())
        if a:
            self.apply_flag_action(a.text())

    def apply_flag_action(self, text: str):
        """'occluded' | 'truncated' | ... toggles the flag; 'occ none|partial|full' sets the occlusion level."""
        if not self.sel:
            return
        self.snapshot()
        if text.startswith("occ "):
            self.gt.set_occ(self.i, self.sel["id"], text.split()[1])
        else:
            self.gt.toggle_flag(self.sel, text)
        self.commit(lists=False)

    def del_box(self):
        if not self.sel:
            return
        self.snapshot()
        self.frame()["boxes"].remove(self.sel)
        self.sel = None
        self.commit()

    def del_id(self):
        if not self.sel:
            return
        tid = self.sel["id"]
        if QtWidgets.QMessageBox.question(self, "delete id", f"delete id {tid} in all frames?") != QtWidgets.QMessageBox.Yes:
            return
        self.snapshot()
        self.gt.delete_id(tid)
        self.sel = None
        self.commit()

    def edit_id(self):
        if not self.sel:
            return
        tid = self.sel["id"]
        v = self.gt.ids.setdefault(tid, {"cls": self.sel["cls"], "desc": ""})
        desc, ok = QtWidgets.QInputDialog.getText(self, f"id {tid}", "description", text=v.get("desc", ""))
        if not ok:
            return
        cls, ok = QtWidgets.QInputDialog.getItem(self, f"id {tid}", "class", ["worker", "heavy_equipment"],
                                                 0 if v["cls"] == "worker" else 1, False)
        self.snapshot()
        v["desc"] = desc
        if ok and cls != v["cls"]:
            v["cls"] = cls
            for f in self.gt.frames:
                for b in f["boxes"]:
                    if b["id"] == tid:
                        b["cls"] = cls
        self.commit()

    def check_frame(self):
        k = self.gt.review_key(self.i)
        rv = self.gt.review["frames"]
        rv[k] = "draft" if rv.get(k) == "checked" else "checked"
        if self.gt.schema_version >= 2:
            self.gt.bump_pass("user", "user", "frames_stepped")
        self.gt.save(backup=False)
        self.goto(self.i + 1)

    def toggle_note(self):
        it = self.notes.currentItem()
        if it is None:
            return
        j = it.data(33)
        self.gt.resolve_note(j, by="user")
        self.gt.save(backup=False)
        self.refresh_lists()
        self.notes.setCurrentRow(j)

    def check_id(self):
        if not self.sel:
            return
        tid = self.sel["id"]
        rv = self.gt.review["ids"]
        rv[tid] = "draft" if rv.get(tid) == "checked" else "checked"
        self.commit()

    def toggle_bright(self):
        self.bright = not self.bright
        self.thumbs.clear()
        self.refresh_lists()
        self.goto(self.i, keep_sel=True)

    def play(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(int(1000 / max(self.gt.meta.get("rate_hz", 5), 1)))


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    app = QtWidgets.QApplication(sys.argv)
    w = Review(Path(sys.argv[1]))
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
