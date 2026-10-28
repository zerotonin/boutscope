# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — gui.app                                             ║
# ║  « one window, one project »                                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  QApplication entry point and the main window that wires the     ║
# ║  panels to the Browser.                                          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Main window: label panel, histogram, sampler, bout list and player, wired to a Browser."""

from __future__ import annotations

import sys
import tempfile
from concurrent.futures import Future
from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from boutscope import config
from boutscope import constants as bc
from boutscope.annotations import AnnotationError, AnnotationStore, Verdict
from boutscope.browser import Browser
from boutscope.gui.dialogs import BlindDialog, ResumeDialog, ask_annotator
from boutscope.gui.histogram import HistogramPanel
from boutscope.gui.label_panel import LabelPanel
from boutscope.gui.player import Player
from boutscope.gui.sampler_panel import DrawSettings, SamplerPanel
from boutscope.gui.verdict_panel import VerdictPanel
from boutscope.pose import PoseStore
from boutscope.project import Project, ProjectError, load_project
from boutscope.sampling import SamplingError
from boutscope.video import ClipLoader, ClipRequest, VideoReader

WINDOW_SIZE: tuple[int, int] = (1600, 920)
SETTINGS_ANNOTATOR: str = "annotator"
VERDICT_KEYS: tuple[str, ...] = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "U")
RATED_MARK: str = "●"


class _ClipBridge(QObject):
    """Carries finished clip futures from the reader thread into the GUI thread."""

    done = Signal(object, object)  # ClipRequest, Future


class MainWindow(QMainWindow):
    def __init__(
        self,
        project: Project | None = None,
        video_root: Path | None = None,
        annotator: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("boutscope")
        self.resize(*WINDOW_SIZE)
        self.video_root = video_root
        self.annotator = annotator
        self.store: AnnotationStore | None = None
        self.browser: Browser | None = None
        self.pose: PoseStore | None = None
        self.loader = ClipLoader(VideoReader())
        self.bridge = _ClipBridge()
        self.bridge.done.connect(self._on_clip_done)
        self.current: ClipRequest | None = None

        self.labels = LabelPanel()
        self.histogram = HistogramPanel()
        self.sampler = SamplerPanel()
        self.bout_list = QListWidget()
        self.player = Player(fps=50.0)
        self.verdicts = VerdictPanel()
        self._build_layout()
        self._build_menu()
        self._build_shortcuts()

        self.labels.labelling_selected.connect(self._on_labelling)
        self.labels.label_selected.connect(self._on_label)
        self.histogram.range_changed.connect(lambda *_: self._update_available())
        self.sampler.settings_changed.connect(self._update_available)
        self.sampler.draw_requested.connect(self._on_draw)
        self.sampler.more_requested.connect(self._on_more)
        self.bout_list.currentRowChanged.connect(self._on_bout_row)
        self.player.previous_requested.connect(lambda: self._move_bout(-1))
        self.player.next_requested.connect(lambda: self._move_bout(1))
        self.player.zoom_toggled.connect(lambda _: self._show_current_bout())
        self.verdicts.verdict_given.connect(self._on_verdict)
        self.verdicts.name_proposed.connect(self._on_name)
        if project is not None:
            self.open_project(project)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Project  « open, demo, labelling, label »                  │
    # └────────────────────────────────────────────────────────────┘

    def open_project(self, project: Project) -> None:
        if self.store is not None:
            self.store.close()
        self.store = AnnotationStore(project.root / bc.DB_FILE)
        annotator = self._resolve_annotator()
        try:
            self.browser = Browser(project, self.video_root, self.store, annotator)
        except ValueError as error:
            QMessageBox.warning(self, "boutscope", str(error))
            return
        self.pose = PoseStore(project)
        self.player.fps = project.fps
        self._update_title()
        self.labels.set_labellings(list(project.labellings))
        self.sampler.set_strata(self.browser.stratum_columns())
        self._on_labelling(self.browser.labelling)

    def _on_labelling(self, labelling: str) -> None:
        self.browser.select_labelling(labelling)
        self.labels.set_labels(self.browser.label_table())
        self.sampler.set_typicality_available(
            self.browser.project.labellings[labelling].has_typicality
        )
        table = self.browser.label_table()
        self.verdicts.set_labels([(key, row["name"]) for key, row in table.iterrows()])
        self._clear_bouts()
        self.labels.select_row(0)

    def _on_label(self, label: str) -> None:
        self.browser.select_label(label)
        counts, edges = self.browser.histogram()
        colour = self.browser.label_table().loc[label, "colour"]
        self.histogram.set_histogram(counts, edges, colour)
        self._clear_bouts()

    def _update_available(self) -> None:
        if self.browser is None or self.browser.label is None:
            return
        lo, hi = self.histogram.selected_range()
        try:
            counts = self.browser.available(
                lo, hi, self.sampler.selected_strata(), self.sampler.settings().mode
            )
        except SamplingError as error:
            self.statusBar().showMessage(str(error), 5000)
            return
        self.histogram.set_available(counts, len(self.browser.label_durations()))

    # ┌────────────────────────────────────────────────────────────┐
    # │ Drawing  « draw, more, bout list »                         │
    # └────────────────────────────────────────────────────────────┘

    def _on_draw(self, settings: DrawSettings) -> None:
        if self.browser is None or self.browser.label is None:
            return
        lo, hi = self.histogram.selected_range()
        try:
            batch = self.browser.draw(
                settings.n,
                min_duration_s=lo,
                max_duration_s=hi,
                strata=settings.strata,
                mode=settings.mode,
                seed=settings.seed,
                max_per_animal=settings.max_per_animal,
            )
        except SamplingError as error:
            QMessageBox.warning(self, "boutscope", str(error))
            return
        self._start_list(batch)

    def _start_list(self, batch: list[int], row: int = 0) -> None:
        self._clear_bouts()
        self.verdicts.set_blind(self._blind_options())
        self._append_bouts(batch)
        self._update_title()
        if self.bout_list.count():
            self.bout_list.setCurrentRow(min(row, self.bout_list.count() - 1))
        else:
            self.player.show_message("No bouts in this range.")

    def _blind_options(self) -> list[tuple[str, str]] | None:
        if not self.browser.is_blind:
            return None
        names = self.browser.label_table()["name"]
        return [(key, names[key]) for key in self.browser.label_set]

    def _on_more(self, n: int) -> None:
        if self.browser is None or self.browser.session is None:
            return
        self._append_bouts(self.browser.more(n))

    def _append_bouts(self, bout_ids: list[int]) -> None:
        for bout_id in bout_ids:
            bout = self.browser.bout(bout_id)
            item = QListWidgetItem(
                f"{self.bout_list.count() + 1:>3}  {bout['animal']}  frame {bout['start']}  "
                f"{bout['duration_s'] * 1000:.0f} ms  ·  {bout['video']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, int(bout_id))
            self.bout_list.addItem(item)
            self._mark_item(item)
        session = self.browser.session
        self.sampler.set_can_draw_more(session is not None and session.remaining > 0)
        self.statusBar().showMessage(
            f"{len(self.browser.drawn)} bouts drawn, {session.remaining:,} more available", 5000
        )

    def _clear_bouts(self) -> None:
        self.bout_list.clear()
        self.sampler.set_can_draw_more(False)
        self.verdicts.setEnabled(False)
        self.player.show_message("Pick a label, a duration range, then Draw.")

    def _mark_item(self, item: QListWidgetItem) -> None:
        """Rated bouts get a dot in their verdict's colour."""
        verdict = self.browser.verdict_for(item.data(Qt.ItemDataRole.UserRole))
        text = item.text().lstrip(RATED_MARK + " ")
        if verdict is None:
            item.setText("  " + text)
            item.setForeground(QColor(bc.WONG["black"]))
        else:
            item.setText(f"{RATED_MARK} {text}")
            item.setForeground(QColor(bc.VERDICT_COLOURS[verdict.verdict]))

    def _move_bout(self, delta: int) -> None:
        row = self.bout_list.currentRow() + delta
        if 0 <= row < self.bout_list.count():
            self.bout_list.setCurrentRow(row)

    # ┌────────────────────────────────────────────────────────────┐
    # │ Clips  « load in the background, show when ready »         │
    # └────────────────────────────────────────────────────────────┘

    def _on_bout_row(self, row: int) -> None:
        if row >= 0:
            self._show_current_bout()

    def _show_current_bout(self) -> None:
        item = self.bout_list.currentItem()
        if item is None or self.browser is None:
            return
        zoom = self.player.zoom.isChecked()
        try:
            request = self.browser.clip_request(item.data(Qt.ItemDataRole.UserRole), zoom)
        except (config.ConfigError, KeyError) as error:
            self.player.show_message(str(error))
            return
        self.current = request
        self.player.show_message("loading …")
        future = self.loader.get(request)
        future.add_done_callback(lambda f, r=request: self.bridge.done.emit(r, f))
        self.loader.prefetch(self._following_requests(zoom))

    def _following_requests(self, zoom: bool) -> list[ClipRequest]:
        row = self.bout_list.currentRow()
        ids = [
            self.bout_list.item(r).data(Qt.ItemDataRole.UserRole)
            for r in range(row + 1, min(row + 3, self.bout_list.count()))
        ]
        try:
            return [self.browser.clip_request(i, zoom) for i in ids]
        except (config.ConfigError, KeyError):
            return []

    def _on_clip_done(self, request: ClipRequest, future: Future) -> None:
        if request != self.current:
            return  # a later bout was chosen meanwhile
        try:
            clip = future.result()
        except Exception as error:  # noqa: BLE001 — shown to the user, never swallowed
            self.player.show_message(f"Cannot show this bout:\n{error}")
            return
        bout = self.browser.bout(request.bout_id)
        stop = request.first_frame + request.n_frames
        pose = self.pose.load_slice(request.video, request.animal, request.first_frame, stop)
        blind = self.browser.is_blind
        colour = (
            bc.BLIND_BOUT_COLOUR
            if blind
            else self.browser.label_table().loc[str(bout["label"]), "colour"]
        )
        row = self.bout_list.currentRow() + 1
        info = (
            f"bout {row}/{self.bout_list.count()}  ·  {request.video}  ·  animal "
            f"{request.animal}  ·  frames {bout['start']}–{bout['stop'] - 1}  ·  "
            f"{bout['duration_s'] * 1000:.0f} ms"
        )
        strips = [] if blind else self.browser.strips(request)  # strips would give it away
        self.player.set_clip(clip, pose, self.browser.project.skeleton, colour, strips, info)
        self._refresh_verdict()

    # ┌────────────────────────────────────────────────────────────┐
    # │ Verdicts  « record, mark, reveal, advance »                │
    # └────────────────────────────────────────────────────────────┘

    def _current_bout_id(self) -> int | None:
        item = self.bout_list.currentItem()
        return None if item is None else int(item.data(Qt.ItemDataRole.UserRole))

    def _refresh_verdict(self) -> None:
        bout_id = self._current_bout_id()
        if bout_id is None or self.browser.session_id is None:
            return
        progress = (
            f"rated {self.browser.n_rated()} of {len(self.browser.drawn)}  ·  "
            f"session {self.browser.session_id}  ·  {self.browser.annotator}"
        )
        self.verdicts.show_bout(self.browser.verdict_for(bout_id), progress)

    def _on_verdict(self, verdict: Verdict) -> None:
        bout_id = self._current_bout_id()
        if bout_id is None:
            return
        try:
            self.browser.record(bout_id, verdict)
        except AnnotationError as error:
            QMessageBox.warning(self, "boutscope", str(error))
            return
        self._mark_item(self.bout_list.currentItem())
        if self.browser.is_blind:
            self._reveal(bout_id)
        if (
            self.verdicts.advance.isChecked()
            and self.bout_list.currentRow() + 1 < self.bout_list.count()
        ):
            self._move_bout(1)
        else:
            self._refresh_verdict()

    def _reveal(self, bout_id: int) -> None:
        stored = self.browser.verdict_for(bout_id)
        table = self.browser.label_table()
        truth = str(self.browser.bout(bout_id)["label"])
        answer = stored.blind_answer
        answered = (
            "none of these"
            if answer == bc.BLIND_NONE
            else table.loc[answer, "name"]
            if answer
            else bc.VERDICT_TEXT[stored.verdict]
        )
        mark = {"fits": "✔", "doesnt_fit": "✘"}.get(stored.verdict, "")
        self.verdicts.show_reveal(
            f"previous bout was <b>{table.loc[truth, 'name']}</b>; you said {answered} {mark}",
            table.loc[truth, "colour"],
        )

    def _on_name(self, name: str, note: str) -> None:
        if self.browser is None or self.browser.label is None or self.browser.is_blind:
            return
        try:
            self.browser.propose_name(self.browser.label, name, note)
        except AnnotationError as error:
            QMessageBox.warning(self, "boutscope", str(error))
            return
        self.statusBar().showMessage(f"proposed name for {self.browser.label}: {name}", 5000)
        self.verdicts.name.clear()
        self.verdicts.name_note.clear()

    # ┌────────────────────────────────────────────────────────────┐
    # │ Sessions  « annotator, blind, resume, export »             │
    # └────────────────────────────────────────────────────────────┘

    def _resolve_annotator(self) -> str | None:
        if self.annotator:
            return self.annotator
        saved = QSettings("boutscope", "boutscope").value(SETTINGS_ANNOTATOR, "")
        self.annotator = saved or ask_annotator(self)
        if self.annotator:
            QSettings("boutscope", "boutscope").setValue(SETTINGS_ANNOTATOR, self.annotator)
        return self.annotator

    def change_annotator(self) -> None:
        name = ask_annotator(self, self.annotator or "")
        if not name or self.browser is None:
            return
        self.annotator = self.browser.annotator = name
        QSettings("boutscope", "boutscope").setValue(SETTINGS_ANNOTATOR, name)
        self.browser._reset_draw()  # a session belongs to one annotator
        self._clear_bouts()
        self._update_title()

    def start_blind(
        self, labels: tuple[str, ...], n: int, min_duration_s=None, max_duration_s=None
    ) -> None:
        settings = self.sampler.settings()
        try:
            batch = self.browser.draw_blind(
                labels,
                n,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
                strata=settings.strata,
                seed=settings.seed,
                max_per_animal=settings.max_per_animal,
            )
        except SamplingError as error:
            QMessageBox.warning(self, "boutscope", str(error))
            return
        self._start_list(batch)

    def resume_session(self, session_id: int) -> None:
        try:
            self.browser.resume(session_id)
        except AnnotationError as error:
            QMessageBox.warning(self, "boutscope", str(error))
            return
        browser = self.browser
        self.labels.labelling_box.blockSignals(True)
        self.labels.labelling_box.setCurrentText(browser.labelling)
        self.labels.labelling_box.blockSignals(False)
        self.labels.set_labels(browser.label_table())
        if browser.label is not None:
            counts, edges = browser.histogram()
            self.histogram.set_histogram(
                counts, edges, browser.label_table().loc[browser.label, "colour"]
            )
        first_open = next(
            (i for i, b in enumerate(browser.drawn) if browser.verdict_for(b) is None),
            len(browser.drawn) - 1,
        )
        self._start_list(list(browser.drawn), row=first_open)

    def _blind_dialog(self) -> None:
        if self.browser is None:
            return
        dialog = BlindDialog(self.browser.label_table(), self)
        if dialog.exec():
            s = dialog.settings()
            self.start_blind(s.labels, s.n, s.min_duration_s, s.max_duration_s)

    def _resume_dialog(self) -> None:
        if self.store is None:
            return
        sessions = self.store.list_sessions()
        if sessions.empty:
            QMessageBox.information(self, "boutscope", "No stored sessions in this project yet.")
            return
        dialog = ResumeDialog(sessions, self)
        if dialog.exec() and dialog.selected_id() is not None:
            self.resume_session(dialog.selected_id())

    def _export_dialog(self) -> None:
        if self.store is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Export verdicts as CSV")
        if folder:
            paths = self.store.export_csv(Path(folder))
            self.statusBar().showMessage(f"wrote {', '.join(p.name for p in paths)}", 8000)

    def _update_title(self) -> None:
        if self.browser is None:
            return
        mode = "  ·  BLIND" if self.browser.is_blind else ""
        who = self.browser.annotator or "no annotator"
        self.setWindowTitle(f"boutscope — {self.browser.project.name}  ·  {who}{mode}")

    # ┌────────────────────────────────────────────────────────────┐
    # │ Window  « layout, menu, shortcuts, close »                 │
    # └────────────────────────────────────────────────────────────┘

    def _build_layout(self) -> None:
        middle = QWidget()
        column = QVBoxLayout(middle)
        column.addWidget(self.histogram)
        column.addWidget(self.sampler)
        column.addWidget(self.bout_list, stretch=1)
        splitter = QSplitter()
        splitter.addWidget(self.labels)
        splitter.addWidget(middle)
        splitter.addWidget(self.player)
        splitter.addWidget(self.verdicts)
        splitter.setSizes([330, 360, 700, 210])
        self.setCentralWidget(splitter)

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu("&File")
        for text, slot, keys in (
            ("&Open project…", self._open_dialog, QKeySequence.StandardKey.Open),
            ("Open &demo project", self._open_demo, None),
            ("&Quit", self.close, QKeySequence.StandardKey.Quit),
        ):
            action = QAction(text, self)
            if keys is not None:
                action.setShortcut(keys)
            action.triggered.connect(slot)
            menu.addAction(action)
        session = self.menuBar().addMenu("&Session")
        for text, slot in (
            ("New &blind session…", self._blind_dialog),
            ("&Resume session…", self._resume_dialog),
            ("&Export verdicts as CSV…", self._export_dialog),
            ("Change &annotator…", self.change_annotator),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            session.addAction(action)

    def _build_shortcuts(self) -> None:
        for key, slot in (
            ("Space", self.player.toggle_play),
            ("Left", lambda: self.player.step(-1)),
            ("Right", lambda: self.player.step(1)),
            ("N", lambda: self._move_bout(1)),
            ("B", lambda: self._move_bout(-1)),
            ("L", self.player.loop.toggle),
            ("+", lambda: self.player.change_speed(1)),
            ("-", lambda: self.player.change_speed(-1)),
        ):
            QShortcut(QKeySequence(key), self, activated=slot)
        for key in VERDICT_KEYS:
            QShortcut(QKeySequence(key), self, activated=lambda k=key: self.verdicts.press_key(k))

    def _open_dialog(self) -> None:
        try:
            start = str(config.project_root())
        except config.ConfigError:
            start = ""
        folder = QFileDialog.getExistingDirectory(self, "Open boutscope project", start)
        if folder:
            try:
                self.open_project(load_project(Path(folder)))
            except ProjectError as error:
                QMessageBox.warning(self, "boutscope", str(error))

    def _open_demo(self) -> None:
        from boutscope.synthetic import make_demo_project

        self.statusBar().showMessage("building the demo project …")
        QApplication.processEvents()
        self.open_project(make_demo_project(Path(tempfile.mkdtemp(prefix="boutscope_demo_"))))

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.player.timer.stop()
        self.current = None  # clips still loading are ignored when they arrive
        self.loader.shutdown()
        if self.store is not None:
            self.store.close()
        super().closeEvent(event)


def main(project_path: Path | None = None, video_root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    project = load_project(project_path) if project_path is not None else None
    window = MainWindow(project, video_root)
    window.show()
    return app.exec()
