# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.sampler_panel                                   ║
# ║  « how many, from whom, which seed »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Strata, mode, number of bouts, seed and per-animal cap;         ║
# ║  Draw starts a session, More continues it.                       ║
# ╚══════════════════════════════════════════════════════════════════╝
"""How many bouts, stratified by what, in which mode, from which seed."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QWidget,
)

from boutscope import constants as bc

DEFAULT_N: int = 20
MAX_SEED: int = 2**31 - 1


@dataclass(frozen=True)
class DrawSettings:
    n: int
    strata: tuple[str, ...]
    mode: str
    seed: int
    max_per_animal: int | None


class SamplerPanel(QWidget):
    settings_changed = Signal()
    draw_requested = Signal(object)  # DrawSettings
    more_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.strata = QListWidget()
        self.strata.setMaximumHeight(90)
        self.mode = QComboBox()
        self.mode.addItems(bc.SAMPLING_MODES)
        self.n = QSpinBox(minimum=1, maximum=10_000, value=DEFAULT_N)
        self.seed = QSpinBox(minimum=0, maximum=MAX_SEED, value=bc.DEFAULT_SEED)
        self.cap = QSpinBox(minimum=0, maximum=10_000, value=0, specialValueText="no cap")
        self.draw_button = QPushButton("Draw")
        self.more_button = QPushButton("More")
        self.more_button.setEnabled(False)
        buttons = QHBoxLayout()
        buttons.addWidget(self.draw_button)
        buttons.addWidget(self.more_button)
        form = QFormLayout(self)
        form.addRow("stratify by", self.strata)
        form.addRow("mode", self.mode)
        form.addRow("bouts", self.n)
        form.addRow("seed", self.seed)
        form.addRow("max per animal", self.cap)
        form.addRow(buttons)
        self.draw_button.clicked.connect(lambda: self.draw_requested.emit(self.settings()))
        self.more_button.clicked.connect(lambda: self.more_requested.emit(self.n.value()))
        self.strata.itemChanged.connect(lambda _: self.settings_changed.emit())
        self.mode.currentTextChanged.connect(lambda _: self.settings_changed.emit())

    def set_strata(self, columns: list[str], checked: tuple[str, ...] = bc.DEFAULT_STRATA) -> None:
        self.strata.blockSignals(True)
        self.strata.clear()
        for column in columns:
            item = QListWidgetItem(column)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked if column in checked else Qt.CheckState.Unchecked
            )
            self.strata.addItem(item)
        self.strata.blockSignals(False)

    def set_typicality_available(self, available: bool) -> None:
        model = self.mode.model()
        for row, mode in enumerate(bc.SAMPLING_MODES):
            if mode != "stratified":
                model.item(row).setEnabled(available)
        if not available:
            self.mode.setCurrentText("stratified")

    def set_can_draw_more(self, can: bool) -> None:
        self.more_button.setEnabled(can)

    def selected_strata(self) -> tuple[str, ...]:
        return tuple(
            self.strata.item(i).text()
            for i in range(self.strata.count())
            if self.strata.item(i).checkState() == Qt.CheckState.Checked
        )

    def settings(self) -> DrawSettings:
        return DrawSettings(
            n=self.n.value(),
            strata=self.selected_strata(),
            mode=self.mode.currentText(),
            seed=self.seed.value(),
            max_per_animal=self.cap.value() or None,
        )
