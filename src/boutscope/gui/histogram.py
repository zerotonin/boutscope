# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.histogram                                       ║
# ║  « choose the durations you want to see »                        ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Bout-duration histogram on a log axis with a draggable range,   ║
# ║  and the per-stratum count of bouts in range.                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Bout-duration histogram with a draggable duration range."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from boutscope import constants as bc

RANGE_DEBOUNCE_MS: int = 150
TOP_STRATA_SHOWN: int = 6
TICK_LABEL_PX: int = 55  # room one tick label needs


class SecondsAxis(pg.AxisItem):
    """x is log10(seconds); ticks read as ms below 1 s, s above."""

    def tickValues(self, minVal, maxVal, size):  # noqa: N802, N803 (pyqtgraph naming)
        """Ticks at 1, 2, 5 x 10^k seconds, thinned to what the axis width fits."""
        decades = range(int(np.floor(minVal)) - 1, int(np.ceil(maxVal)) + 1)
        per_decade = size / max(maxVal - minVal, 1e-9) / TICK_LABEL_PX
        mantissas = (1, 2, 5) if per_decade >= 3 else (1, 3) if per_decade >= 2 else (1,)
        major = [np.log10(m) + d for d in decades for m in mantissas]
        return [(1.0, [v for v in major if minVal <= v <= maxVal])]

    def tickStrings(self, values, scale, spacing):  # noqa: N802 (Qt naming)
        out = []
        for v in values:
            seconds = 10.0**v
            out.append(f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.3g} s")
        return out


class HistogramPanel(QWidget):
    range_changed = Signal(object, object)  # min_s, max_s (None = open end)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.plot = pg.PlotWidget(axisItems={"bottom": SecondsAxis("bottom")}, background="w")
        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.hideButtons()
        self.plot.setLabel("left", "bouts")
        self.plot.setLabel("bottom", "bout duration")
        self.plot.setMinimumHeight(160)
        self.bars: pg.BarGraphItem | None = None
        self.region = pg.LinearRegionItem(brush=pg.mkBrush(86, 180, 233, 60))
        self.region.setZValue(10)
        self.plot.addItem(self.region)
        self.summary = QLabel("Pick a label.")
        self.summary.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        layout.addWidget(self.summary)
        self._edges = np.zeros(1)
        self._debounce = QTimer(self, singleShot=True, interval=RANGE_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._emit_range)
        self.region.sigRegionChanged.connect(self._debounce.start)

    def set_histogram(self, counts: np.ndarray, edges: np.ndarray, colour: str) -> None:
        if self.bars is not None:
            self.plot.removeItem(self.bars)
        self._edges = edges
        log_edges = np.log10(edges)
        self.bars = pg.BarGraphItem(
            x0=log_edges[:-1],
            x1=log_edges[1:],
            height=counts,
            brush=pg.mkBrush(colour),
            pen=pg.mkPen(bc.WONG["black"], width=0.5),
        )
        self.plot.addItem(self.bars)
        self.region.blockSignals(True)
        self.region.setBounds((log_edges[0], log_edges[-1]))
        self.region.setRegion((log_edges[0], log_edges[-1]))
        self.region.blockSignals(False)
        self.plot.setXRange(log_edges[0], log_edges[-1], padding=0.02)
        self._emit_range()

    def selected_range(self) -> tuple[float | None, float | None]:
        """Duration range in seconds; an end at the histogram's edge is open."""
        if len(self._edges) < 2:
            return None, None
        lo, hi = (10.0**v for v in self.region.getRegion())
        open_lo = lo <= self._edges[0] * 1.0001
        open_hi = hi >= self._edges[-1] * 0.9999
        return (None if open_lo else lo), (None if open_hi else hi)

    def set_available(self, counts: pd.Series, total: int) -> None:
        n = int(counts.sum())
        share = n / total * 100 if total else 0.0
        top = ", ".join(
            f"{_stratum_name(k)}: {v}" for k, v in counts.head(TOP_STRATA_SHOWN).items()
        )
        more = (
            f" (+{len(counts) - TOP_STRATA_SHOWN} more strata)"
            if len(counts) > TOP_STRATA_SHOWN
            else ""
        )
        self.summary.setText(
            f"<b>{n:,}</b> bouts in range of {total:,} ({share:.1f} %) in {len(counts)} strata"
            f"<br>{top}{more}"
        )

    def _emit_range(self) -> None:
        self.range_changed.emit(*self.selected_range())


def _stratum_name(key) -> str:
    return " / ".join(map(str, key)) if isinstance(key, tuple) else str(key)
