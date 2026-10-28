# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.timeline                                        ║
# ║  « one strip per labelling »                                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The clip's timeline: bout and context, a cursor, and one colour ║
# ║  strip per labelling for the same frames.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""The clip's timeline: bout and context, cursor, one strip per labelling."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from boutscope import constants as bc
from boutscope.browser import Strip

NAME_WIDTH: int = 90
ROW_HEIGHT: int = 14
ROW_GAP: int = 3


class Timeline(QWidget):
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.n_frames = 0
        self.bout = (0, 0)
        self.bout_colour = bc.WONG["black"]
        self.strips: list[Strip] = []
        self.cursor = 0
        self._update_height()

    def set_clip(
        self, n_frames: int, bout: tuple[int, int], bout_colour: str, strips: list[Strip]
    ) -> None:
        self.n_frames, self.bout, self.bout_colour, self.strips = (
            n_frames,
            bout,
            bout_colour,
            strips,
        )
        self.cursor = 0
        self._update_height()
        self.update()

    def set_cursor(self, index: int) -> None:
        self.cursor = index
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.n_frames == 0:
            return
        painter = QPainter(self)
        width = self.width() - NAME_WIDTH
        dx = width / self.n_frames
        rows = [("bout", None), *((s.labelling, s) for s in self.strips)]
        for row, (name, strip) in enumerate(rows):
            y = row * (ROW_HEIGHT + ROW_GAP)
            painter.setPen(QColor(bc.WONG["black"]))
            painter.drawText(
                QRectF(0, y, NAME_WIDTH - 4, ROW_HEIGHT),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                name,
            )
            for start, stop, colour in self._segments(strip):
                painter.fillRect(
                    QRectF(NAME_WIDTH + start * dx, y, (stop - start) * dx, ROW_HEIGHT),
                    QColor(colour),
                )
        x = NAME_WIDTH + (self.cursor + 0.5) * dx
        painter.setPen(QPen(QColor(bc.WONG["vermilion"]), 2))
        painter.drawLine(QPointF(x, 0), QPointF(x, self.height()))
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        self._seek(event.position().x())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        index = self._index_at(event.position().x())
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek(event.position().x())
        if index is not None and self.strips:
            lines = [f"frame {index - self.bout[0]:+d}"]
            lines += [
                f"{s.labelling}: {s.lookup.name_of(int(s.codes[index]))}" for s in self.strips
            ]
            QToolTip.showText(event.globalPosition().toPoint(), "\n".join(lines), self)

    def _segments(self, strip: Strip | None):
        """Runs of equal colour, so long clips paint few rectangles."""
        if strip is None:
            yield 0, self.bout[0], bc.CONTEXT_COLOUR
            yield self.bout[0], self.bout[1], self.bout_colour
            yield self.bout[1], self.n_frames, bc.CONTEXT_COLOUR
            return
        codes = strip.codes
        start = 0
        for i in range(1, len(codes) + 1):
            if i == len(codes) or codes[i] != codes[start]:
                yield start, i, strip.lookup.colour_of(int(codes[start]))
                start = i

    def _index_at(self, x: float) -> int | None:
        if self.n_frames == 0 or x < NAME_WIDTH:
            return None
        index = int((x - NAME_WIDTH) / (self.width() - NAME_WIDTH) * self.n_frames)
        return min(max(index, 0), self.n_frames - 1)

    def _seek(self, x: float) -> None:
        index = self._index_at(x)
        if index is not None:
            self.seek_requested.emit(index)

    def _update_height(self) -> None:
        self.setFixedHeight((len(self.strips) + 1) * (ROW_HEIGHT + ROW_GAP))
