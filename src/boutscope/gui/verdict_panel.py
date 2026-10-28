# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.verdict_panel                                   ║
# ║  « one key per verdict »                                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Verdict buttons, looks-like, note and a proposed label name;    ║
# ║  in blind mode the answer list replaces the verdict buttons.     ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Verdict entry beside the player: four verdicts, or the blind answer list."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from boutscope import constants as bc
from boutscope.annotations import Verdict

KEY_OF_VERDICT: dict[str, str] = {"fits": "1", "doesnt_fit": "2", "artefact": "3", "unsure": "4"}
BLIND_EXTRA_KEYS: dict[str, str] = {"artefact": "A", "unsure": "U"}
MAX_BLIND_KEYS: int = 9


class VerdictPanel(QWidget):
    verdict_given = Signal(object)  # Verdict
    name_proposed = Signal(str, str)  # name, note

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.blind_options: list[tuple[str, str]] = []  # (label key, display name)
        self.progress = QLabel("")
        self.reveal = QLabel("")
        self.reveal.setWordWrap(True)

        self.verdict_box = QGroupBox("Verdict")
        verdict_layout = QVBoxLayout(self.verdict_box)
        self.buttons: dict[str, QPushButton] = {}
        for verdict in bc.VERDICTS:
            button = QPushButton(f"[{KEY_OF_VERDICT[verdict]}]  {bc.VERDICT_TEXT[verdict]}")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, v=verdict: self.give(v))
            verdict_layout.addWidget(button)
            self.buttons[verdict] = button
        self.looks_like = QComboBox()
        self.note = QLineEdit(placeholderText="note (optional)")
        verdict_layout.addWidget(QLabel("looks like"))
        verdict_layout.addWidget(self.looks_like)
        verdict_layout.addWidget(self.note)

        self.blind_box = QGroupBox("Which label is this?")
        self.blind_layout = QVBoxLayout(self.blind_box)
        self.blind_buttons: list[QPushButton] = []
        blind_extra = QHBoxLayout()
        for verdict, key in BLIND_EXTRA_KEYS.items():
            button = QPushButton(f"[{key}]  {bc.VERDICT_TEXT[verdict]}")
            button.clicked.connect(lambda _=False, v=verdict: self.give(v))
            blind_extra.addWidget(button)
        self.blind_note = QLineEdit(placeholderText="note (optional)")
        self._blind_footer = QVBoxLayout()
        self._blind_footer.addLayout(blind_extra)
        self._blind_footer.addWidget(self.blind_note)

        self.name_box = QGroupBox("Name for this label")
        name_layout = QVBoxLayout(self.name_box)
        self.name = QLineEdit(placeholderText="proposed name")
        self.name_note = QLineEdit(placeholderText="why (optional)")
        self.propose = QPushButton("Propose name")
        self.propose.clicked.connect(self._on_propose)
        for widget in (self.name, self.name_note, self.propose):
            name_layout.addWidget(widget)

        self.advance = QCheckBox("next bout after a verdict", checked=True)
        layout = QVBoxLayout(self)
        for widget in (
            self.progress,
            self.verdict_box,
            self.blind_box,
            self.reveal,
            self.advance,
            self.name_box,
        ):
            layout.addWidget(widget)
        layout.addStretch()
        self.set_blind(None)
        self.setEnabled(False)

    def set_labels(self, labels: list[tuple[str, str]]) -> None:
        """Options for "looks like": (label key, display name)."""
        self.looks_like.clear()
        self.looks_like.addItem("", None)
        for key, name in labels:
            self.looks_like.addItem(name, key)

    def set_blind(self, options: list[tuple[str, str]] | None) -> None:
        """Blind mode with these answer options, or normal mode with None."""
        for button in self.blind_buttons:
            self.blind_layout.removeWidget(button)
            button.deleteLater()
        self.blind_buttons = []
        self.blind_layout.removeItem(self._blind_footer)
        self.blind_options = options or []
        answers = [*self.blind_options, (bc.BLIND_NONE, "none of these")]
        for i, (key, name) in enumerate(answers):
            shortcut = "0" if key == bc.BLIND_NONE else str(i + 1) if i < MAX_BLIND_KEYS else ""
            button = QPushButton(f"[{shortcut}]  {name}" if shortcut else name)
            button.clicked.connect(lambda _=False, k=key: self.answer(k))
            self.blind_layout.addWidget(button)
            self.blind_buttons.append(button)
        self.blind_layout.addLayout(self._blind_footer)
        blind = options is not None
        self.blind_box.setVisible(blind)
        self.verdict_box.setVisible(not blind)
        self.name_box.setVisible(not blind)  # a name field would reveal the label
        self.reveal.setText("")

    def show_bout(self, verdict: Verdict | None, progress: str) -> None:
        """Reflect the stored verdict of the bout now on screen."""
        self.setEnabled(True)
        self.progress.setText(progress)
        for name, button in self.buttons.items():
            button.setChecked(verdict is not None and verdict.verdict == name)
        index = (
            self.looks_like.findData(verdict.looks_like) if verdict and verdict.looks_like else 0
        )
        self.looks_like.setCurrentIndex(max(index, 0))
        note = verdict.note if verdict and verdict.note else ""
        self.note.setText(note)
        self.blind_note.setText(note)

    def show_reveal(self, text: str, colour: str) -> None:
        self.reveal.setText(text)
        self.reveal.setStyleSheet(f"color: {QColor(colour).name()}; font-weight: bold;")

    def press_key(self, key: str) -> None:
        """Keyboard entry: 1-4 verdicts; blind: 1-9 answers, 0 none, A artefact, U unsure."""
        if not self.isEnabled():
            return
        if self.blind_box.isVisible():
            if key == "0":
                self.answer(bc.BLIND_NONE)
            elif key.isdigit() and 1 <= int(key) <= min(len(self.blind_options), MAX_BLIND_KEYS):
                self.answer(self.blind_options[int(key) - 1][0])
            elif key.upper() in BLIND_EXTRA_KEYS.values():
                self.give(next(v for v, k in BLIND_EXTRA_KEYS.items() if k == key.upper()))
            return
        for verdict, shortcut in KEY_OF_VERDICT.items():
            if key == shortcut:
                self.give(verdict)

    def give(self, verdict: str) -> None:
        if self.blind_box.isVisible():
            self.verdict_given.emit(Verdict(verdict, note=self.blind_note.text()))
            return
        self.verdict_given.emit(
            Verdict(verdict, looks_like=self.looks_like.currentData(), note=self.note.text())
        )

    def answer(self, label_key: str) -> None:
        # "unsure" is a placeholder: the browser judges the answer against the true label
        self.verdict_given.emit(
            Verdict("unsure", note=self.blind_note.text(), blind_answer=label_key)
        )

    def _on_propose(self) -> None:
        if self.name.text().strip():
            self.name_proposed.emit(self.name.text(), self.name_note.text())
