# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.dialogs                                         ║
# ║  « annotator, blind set, resume »                                ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Small dialogs: who is annotating, which labels a blind session  ║
# ║  mixes, and which stored session to resume.                      ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Annotator name, blind label set, and session resume."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

MAX_MS: float = 3_600_000.0


def ask_annotator(parent: QWidget, current: str = "") -> str | None:
    name, ok = QInputDialog.getText(
        parent, "boutscope", "Who is annotating? (stored with every verdict)", text=current
    )
    return name.strip() if ok and name.strip() else None


@dataclass(frozen=True)
class BlindSettings:
    labels: tuple[str, ...]
    n: int
    min_duration_s: float | None
    max_duration_s: float | None


class BlindDialog(QDialog):
    """Pick the labels to mix (two or more) and the duration range for all of them."""

    def __init__(self, labels: pd.DataFrame, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New blind session")
        self.list = QListWidget()
        for key, entry in labels.iterrows():
            item = QListWidgetItem(f"{entry['name']}  ({entry['share_of_frames'] * 100:.1f} %)")
            item.setData(Qt.ItemDataRole.UserRole, str(key))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.list.addItem(item)
        self.n = QSpinBox(minimum=2, maximum=10_000, value=30)
        self.min_ms = QDoubleSpinBox(maximum=MAX_MS, decimals=0, specialValueText="open")
        self.max_ms = QDoubleSpinBox(maximum=MAX_MS, decimals=0, specialValueText="open")
        form = QFormLayout()
        form.addRow("bouts", self.n)
        form.addRow("shortest (ms)", self.min_ms)
        form.addRow("longest (ms)", self.max_ms)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        self.hint = QLabel(
            "Tick two or more labels. Bouts are mixed in equal shares, "
            "label colours and strips stay hidden until you answer."
        )
        self.hint.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.hint)
        layout.addWidget(self.list)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def checked_labels(self) -> tuple[str, ...]:
        return tuple(
            self.list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list.count())
            if self.list.item(i).checkState() == Qt.CheckState.Checked
        )

    def settings(self) -> BlindSettings:
        return BlindSettings(
            labels=self.checked_labels(),
            n=self.n.value(),
            min_duration_s=self.min_ms.value() / 1000 or None,
            max_duration_s=self.max_ms.value() / 1000 or None,
        )

    def _accept_if_valid(self) -> None:
        if len(self.checked_labels()) >= 2:
            self.accept()
        else:
            self.hint.setText("<b>Tick at least two labels.</b>")


class ResumeDialog(QDialog):
    """Stored sessions, newest first; the chosen one is resumed."""

    COLUMNS: tuple[str, ...] = (
        "id",
        "annotator",
        "labelling",
        "label",
        "blind",
        "drawn",
        "rated",
        "available",
        "updated",
    )

    def __init__(self, sessions: pd.DataFrame, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resume a session")
        self.resize(760, 360)
        self.table = QTableWidget(len(sessions), len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for row, entry in enumerate(sessions.itertuples(index=False)):
            values = entry._asdict()
            values["label"] = "(blind)" if values["blind"] else values["label"]
            values["blind"] = "yes" if values["blind"] else ""
            for column, name in enumerate(self.COLUMNS):
                self.table.setItem(row, column, QTableWidgetItem(str(values[name])))
        if len(sessions):
            self.table.selectRow(0)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.table.doubleClicked.connect(lambda _: self.accept())
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(buttons)

    def selected_id(self) -> int | None:
        rows = self.table.selectionModel().selectedRows()
        return int(self.table.item(rows[0].row(), 0).text()) if rows else None
