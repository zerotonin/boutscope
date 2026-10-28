# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.label_panel                                     ║
# ║  « pick a labelling, pick a label »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Labelling chooser and the label table: name, colour, share of   ║
# ║  frames, bouts, median duration.                                 ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Labelling chooser and label table."""

from __future__ import annotations

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

COLUMNS: tuple[str, ...] = ("", "label", "frames %", "bouts", "median ms")


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

    def set_labels(self, table: pd.DataFrame) -> None:
        """Rows of :meth:`Browser.label_table`, index = label key."""
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(table))
        for row, (label, entry) in enumerate(table.iterrows()):
            swatch = QTableWidgetItem("")
            swatch.setBackground(QColor(entry["colour"]))
            swatch.setData(Qt.ItemDataRole.UserRole, str(label))
            cells = [
                swatch,
                QTableWidgetItem(str(entry["name"])),
                _number(entry["share_of_frames"] * 100, "{:.2f}"),
                _number(entry["n_bouts"], "{:,.0f}"),
                _number(entry["median_duration_s"] * 1000, "{:.0f}"),
            ]
            for column, item in enumerate(cells):
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)

    def selected_label(self) -> str | None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def select_row(self, row: int) -> None:
        self.table.selectRow(row)

    def _on_labelling(self, name: str) -> None:
        if name:
            self.labelling_selected.emit(name)

    def _on_selection(self) -> None:
        label = self.selected_label()
        if label is not None:
            self.label_selected.emit(label)


def _number(value: float, fmt: str) -> QTableWidgetItem:
    item = QTableWidgetItem(fmt.format(value))
    item.setData(Qt.ItemDataRole.UserRole + 1, float(value))
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item
