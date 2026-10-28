# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.player                                          ║
# ║  « watch the bout, not the neighbours »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Clip player: animal zoom, skeleton overlay, context marked in   ║
# ║  the frame, loop, speed and frame stepping.                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Clip player: frame view with overlay and context marking, and its controls."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from boutscope import constants as bc
from boutscope.browser import Strip
from boutscope.gui.timeline import Timeline
from boutscope.pose import PoseSlice
from boutscope.project import Skeleton
from boutscope.video import Clip

BORDER_PX: int = 6
KEYPOINT_RADIUS_PX: float = 3.0


class FrameView(QWidget):
    """One frame, scaled to fit, with skeleton overlay and a context / bout border."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 240)
        self.image: QImage | None = None
        self._buffer: np.ndarray | None = None
        self.points: np.ndarray | None = None  # (keypoints, 2) clip pixels
        self.visible: np.ndarray | None = None
        self.edges: list[tuple[int, int]] = []
        self.border = QColor(bc.CONTEXT_COLOUR)
        self.in_bout = False
        self.message = ""

    def show_frame(
        self, frame: np.ndarray, points, visible, edges, border: str, in_bout: bool
    ) -> None:
        self._buffer = np.ascontiguousarray(frame)
        h, w = self._buffer.shape[:2]
        self.image = QImage(self._buffer.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.points, self.visible, self.edges = points, visible, edges
        self.border, self.in_bout, self.message = QColor(border), in_bout, ""
        self.update()

    def show_message(self, text: str) -> None:
        self.image, self.message = None, text
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#202020"))
        if self.image is None:
            painter.setPen(QColor("white"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        inner = self.rect().adjusted(BORDER_PX, BORDER_PX, -BORDER_PX, -BORDER_PX)
        scale = min(inner.width() / self.image.width(), inner.height() / self.image.height())
        w, h = self.image.width() * scale, self.image.height() * scale
        x0 = inner.x() + (inner.width() - w) / 2
        y0 = inner.y() + (inner.height() - h) / 2
        target = QRectF(x0, y0, w, h)
        pen = QPen(self.border, BORDER_PX if self.in_bout else BORDER_PX / 2)
        if not self.in_bout:
            pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawRect(
            target.adjusted(-BORDER_PX / 2, -BORDER_PX / 2, BORDER_PX / 2, BORDER_PX / 2)
        )
        painter.drawImage(target, self.image)
        if self.points is not None:
            self._paint_overlay(painter, x0, y0, scale)
        painter.end()

    def _paint_overlay(self, painter: QPainter, x0: float, y0: float, scale: float) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        xy = [QPointF(x0 + (x + 0.5) * scale, y0 + (y + 0.5) * scale) for x, y in self.points]
        painter.setPen(QPen(QColor(bc.SKELETON_COLOUR), 2))
        for a, b in self.edges:
            if self.visible[a] and self.visible[b]:
                painter.drawLine(xy[a], xy[b])
        painter.setPen(QPen(QColor(bc.WONG["black"]), 1))
        painter.setBrush(QColor(bc.KEYPOINT_COLOUR))
        for point, shown in zip(xy, self.visible):
            if shown:
                painter.drawEllipse(point, KEYPOINT_RADIUS_PX, KEYPOINT_RADIUS_PX)


class Player(QWidget):
    previous_requested = Signal()
    next_requested = Signal()
    zoom_toggled = Signal(bool)

    def __init__(self, fps: float, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.fps = fps
        self.clip: Clip | None = None
        self.pose: PoseSlice | None = None
        self.skeleton: Skeleton | None = None
        self.bout_colour = bc.WONG["black"]
        self.index = 0
        self.view = FrameView()
        self.timeline = Timeline()
        self.info = QLabel("")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.play_button = QPushButton("Play")
        self.loop = QCheckBox("loop", checked=True)
        self.speed = QComboBox()
        for s in bc.PLAYBACK_SPEEDS:
            self.speed.addItem(f"{s:g}x", s)
        self.speed.setCurrentIndex(bc.PLAYBACK_SPEEDS.index(bc.DEFAULT_SPEED))
        self.zoom = QCheckBox("zoom on animal", checked=True)
        self.overlay = QCheckBox("skeleton", checked=True)
        controls = QHBoxLayout()
        for widget in (
            _button("◀◀ bout", self.previous_requested.emit),
            _button("◀ frame", lambda: self.step(-1)),
            self.play_button,
            _button("frame ▶", lambda: self.step(1)),
            _button("bout ▶▶", self.next_requested.emit),
            self.loop,
            self.speed,
            self.zoom,
            self.overlay,
        ):
            controls.addWidget(widget)
        controls.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addWidget(self.view, stretch=1)
        layout.addWidget(self.timeline)
        layout.addLayout(controls)

        self.play_button.clicked.connect(self.toggle_play)
        self.speed.currentIndexChanged.connect(self._update_interval)
        self.overlay.toggled.connect(lambda _: self._render())
        self.zoom.toggled.connect(self.zoom_toggled.emit)
        self.timeline.seek_requested.connect(self.seek)
        self._update_interval()

    def set_clip(
        self,
        clip: Clip,
        pose: PoseSlice | None,
        skeleton: Skeleton | None,
        bout_colour: str,
        strips: list[Strip],
        info: str,
    ) -> None:
        self.clip, self.pose, self.skeleton, self.bout_colour = clip, pose, skeleton, bout_colour
        request = clip.request
        self.timeline.set_clip(
            len(clip.frames),
            (request.bout_start, min(request.bout_stop, len(clip.frames))),
            bout_colour,
            strips,
        )
        self.info.setText(
            info + (f" — truncated to the first {bc.MAX_CLIP_S:g} s" if request.truncated else "")
        )
        self.index = 0
        self._render()
        if not self.timer.isActive():
            self.toggle_play()

    def show_message(self, text: str) -> None:
        self.timer.stop()
        self.play_button.setText("Play")
        self.clip = None
        self.info.setText("")  # never describe a bout that is not on screen
        self.view.show_message(text)

    def toggle_play(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Play")
        elif self.clip is not None:
            self.timer.start()
            self.play_button.setText("Pause")

    def step(self, delta: int) -> None:
        if self.clip is None:
            return
        self.timer.stop()
        self.play_button.setText("Play")
        self.seek(self.index + delta)

    def seek(self, index: int) -> None:
        if self.clip is None:
            return
        self.index = min(max(index, 0), len(self.clip.frames) - 1)
        self._render()

    def change_speed(self, steps: int) -> None:
        self.speed.setCurrentIndex(
            min(max(self.speed.currentIndex() + steps, 0), self.speed.count() - 1)
        )

    def _tick(self) -> None:
        if self.clip is None:
            return
        if self.index + 1 < len(self.clip.frames):
            self.index += 1
        elif self.loop.isChecked():
            self.index = 0
        else:
            self.toggle_play()
            return
        self._render()

    def _render(self) -> None:
        if self.clip is None:
            return
        request = self.clip.request
        in_bout = request.bout_start <= self.index < request.bout_stop
        points = visible = None
        edges: list[tuple[int, int]] = []
        if self.overlay.isChecked() and self.pose is not None and self.skeleton is not None:
            points = self.clip.to_clip_xy(self.pose.coords[self.index])
            visible = self.pose.visible(self.skeleton.min_confidence)[self.index]
            names = list(self.skeleton.keypoints)
            edges = [(names.index(a), names.index(b)) for a, b in self.skeleton.edges]
        self.view.show_frame(
            self.clip.frames[self.index],
            points,
            visible,
            edges,
            self.bout_colour if in_bout else bc.CONTEXT_COLOUR,
            in_bout,
        )
        self.timeline.set_cursor(self.index)

    def _update_interval(self) -> None:
        speed = self.speed.currentData() or bc.DEFAULT_SPEED
        self.timer.setInterval(max(1, int(round(1000 / (self.fps * speed)))))


def _button(text: str, slot) -> QPushButton:
    button = QPushButton(text)
    button.clicked.connect(slot)
    return button
