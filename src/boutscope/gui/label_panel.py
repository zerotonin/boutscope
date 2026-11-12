# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.label_panel                                     ║
# ║  « pick a labelling, pick a label »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Labelling chooser and the label table: name, colour, review     ║
# ║  state, proposed names, share of frames, bouts, median duration. ║
# ║  Click a header to sort; names sort naturally (PM 2 < PM 10).    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Labelling chooser and label table."""

from __future__ import annotations

import re

import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

COLUMNS: tuple[str, ...] = (
    "",
    "label",
    "scoped",
    "proposed name",
    "frames %",
    "bouts",
    "median ms",
)
NAME_COLUMN: int = 1
SCOPED_COLUMN: int = 2
PROPOSED_COLUMN: int = 3
SORT_ROLE: int = Qt.ItemDataRole.UserRole + 1
SCOPED_MARK: dict[str, str] = {"done": "✔", "started": "◐", "": ""}
SCOPED_RANK: dict[str, int] = {"done": 2, "started": 1, "": 0}


class LabelPanel(QWidget):
    labelling_selected = Signal(str)
    label_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.labelling_box = QComboBox()
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.sortByColumn(NAME_COLUMN, Qt.SortOrder.AscendingOrder)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Labelling"))
        layout.addWidget(self.labelling_box)
        layout.addWidget(self.table)
        self.labelling_box.currentTextChanged.connect(self._on_labelling)
        self.table.itemSelectionChanged.connect(self._on_selection)

    def set_labellings(self, names: list[str]) -> None:
        self.labelling_box.blockSignals(True)
        self.labelling_box.clear()
        self.labelling_box.addItems(names)
        self.labelling_box.blockSignals(False)

    def set_labels(self, table: pd.DataFrame, progress: pd.DataFrame | None = None) -> None:
        """Rows of :meth:`Browser.label_table`, index = label key.

        *progress* (:meth:`Browser.label_progress`) fills the scoped and
        proposed-name columns; the current sort order is kept.
        """
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.clearContents()
        self.table.setRowCount(len(table))
        for row, (label, entry) in enumerate(table.iterrows()):
            swatch = QTableWidgetItem("")
            swatch.setBackground(QColor(entry["colour"]))
            swatch.setData(Qt.ItemDataRole.UserRole, str(label))
            name = _SortItem(str(entry["name"]))
            name.setData(SORT_ROLE, natural_key(str(entry["name"])))
            cells = [
                swatch,
                name,
                _SortItem(""),
                _SortItem(""),
                _number(entry["share_of_frames"] * 100, "{:.2f}"),
                _number(entry["n_bouts"], "{:,.0f}"),
                _number(entry["median_duration_s"] * 1000, "{:.0f}"),
            ]
            for column, item in enumerate(cells):
                self.table.setItem(row, column, item)
        if progress is not None:
            self._fill_progress(progress)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)

    def set_progress(self, progress: pd.DataFrame) -> None:
        """Refresh the scoped and proposed-name columns, keep selection and order."""
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self._fill_progress(progress)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)

    def _fill_progress(self, progress: pd.DataFrame) -> None:
        for row in range(self.table.rowCount()):
            label = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            if label not in progress.index:
                continue
            entry = progress.loc[label]
            scoped = self.table.item(row, SCOPED_COLUMN)
            rated = int(entry["rated"])
            scoped.setText(f"{SCOPED_MARK[entry['scoped']]} {rated}".strip() if rated else "")
            scoped.setData(SORT_ROLE, f"{SCOPED_RANK[entry['scoped']]}{rated:012d}")
            scoped.setToolTip(
                {
                    "done": f"a session of this label is fully rated; {rated} bouts with a verdict",
                    "started": f"{rated} bouts with a verdict, no finished session yet",
                }.get(entry["scoped"], "")
            )
            proposed = self.table.item(row, PROPOSED_COLUMN)
            proposed.setText(entry["proposed"])
            proposed.setData(SORT_ROLE, entry["proposed"].casefold())
            proposed.setToolTip(entry["proposed_detail"])
            self.table.item(row, NAME_COLUMN).setToolTip(entry["proposed_detail"])

    def selected_label(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def select_row(self, row: int) -> None:
        self.table.selectRow(row)

    def select_label(self, label: str) -> None:
        """Select the row of *label* (its place depends on the sort order)."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == label:
                self.table.selectRow(row)
                return

    def _on_labelling(self, name: str) -> None:
        if name:
            self.labelling_selected.emit(name)

    def _on_selection(self) -> None:
        label = self.selected_label()
        if label is not None:
            self.label_selected.emit(label)


def natural_key(text: str) -> str:
    """Sort key that orders embedded numbers by value: ``PM 2`` before ``PM 10``."""
    return re.sub(r"\d+", lambda m: m.group().zfill(12), text.casefold())


class _SortItem(QTableWidgetItem):
    """Table item that sorts by :data:`SORT_ROLE` when set, else by its text."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        mine, theirs = self.data(SORT_ROLE), other.data(SORT_ROLE)
        if mine is None or theirs is None:
            return super().__lt__(other)
        return mine < theirs


def _number(value: float, fmt: str) -> QTableWidgetItem:
    item = _SortItem(fmt.format(value))
    item.setData(SORT_ROLE, float(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item
