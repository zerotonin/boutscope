"""The viewer end to end on the demo project, headless."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
pytest.importorskip("cv2")

from boutscope.gui.app import MainWindow  # noqa: E402
from boutscope.synthetic import DemoSpec, SyntheticSpec, make_demo_project  # noqa: E402

TIMEOUT_MS = 10_000


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    spec = DemoSpec(labels=SyntheticSpec(n_videos=2, n_animals=2, n_frames=500, seed=5))
    return make_demo_project(tmp_path_factory.mktemp("demo"), spec)


@pytest.fixture
def window(qtbot, demo):
    window = MainWindow(demo, annotator="tester")
    qtbot.addWidget(window)
    yield window
    window.close()


def _wait_for_clip(qtbot, window):
    qtbot.waitUntil(lambda: window.player.clip is not None, timeout=TIMEOUT_MS)
    return window.player.clip


def test_label_list_and_histogram(window):
    assert window.labels.labelling_box.count() == 2
    assert window.labels.table.rowCount() == 3
    assert window.browser.label is not None  # first row selected on open
    assert window.histogram.selected_range() == (None, None)  # whole range = open ends
    assert "bouts in range" in window.histogram.summary.text()


def test_draw_play_step_zoom_next(qtbot, window):
    window.sampler.n.setValue(4)
    window.sampler.draw_button.click()
    assert window.bout_list.count() == min(4, len(window.browser.session.order))
    clip = _wait_for_clip(qtbot, window)
    assert clip.frames.shape[1:3] == (150, 150)  # zoomed on the animal's box
    assert len(window.player.timeline.strips) == 2
    assert window.player.timer.isActive()
    qtbot.waitUntil(lambda: window.player.index > 0, timeout=TIMEOUT_MS)
    window.player.step(1)
    index = window.player.index
    window.player.step(-1)
    assert window.player.index == index - 1 and not window.player.timer.isActive()
    assert window.player.view.points is not None  # skeleton overlay drawn

    window.player.clip = None
    window.player.zoom.setChecked(False)
    full = _wait_for_clip(qtbot, window)
    assert full.frames.shape[1:3] == (
        window.browser.project.videos["video0"].height,
        window.browser.project.videos["video0"].width,
    )

    first_bout = window.player.clip.request.bout_id
    window.player.clip = None
    window._move_bout(1)
    assert _wait_for_clip(qtbot, window).request.bout_id != first_bout


def test_bout_frames_marked_in_frame_border(qtbot, window):
    window.sampler.draw_button.click()
    clip = _wait_for_clip(qtbot, window)
    request = clip.request
    window.player.seek(request.bout_start)
    assert window.player.view.in_bout
    window.player.seek(max(request.bout_start - 1, 0))
    assert window.player.view.in_bout == (request.bout_start == 0)


def test_more_appends_without_repeats(qtbot, window):
    window.sampler.n.setValue(2)
    window.sampler.draw_button.click()
    window.sampler.more_button.click()
    ids = [window.bout_list.item(i).data(256) for i in range(window.bout_list.count())]
    assert len(ids) == len(set(ids)) == min(4, len(window.browser.session.order))


def test_missing_video_is_reported(qtbot, demo, tmp_path):
    import shutil

    from boutscope import project as bp

    copy = tmp_path / "copy"
    shutil.copytree(demo.root, copy)
    project = bp.load_project(copy)
    moved = {
        k: bp.VideoEntry(**{**vars(v), "path": str(tmp_path / "gone.avi")})
        for k, v in project.videos.items()
    }
    bp.save_videos(moved, copy / "videos.toml")
    window = MainWindow(bp.load_project(copy), annotator="tester")
    qtbot.addWidget(window)
    window.sampler.draw_button.click()
    qtbot.waitUntil(lambda: "Cannot show" in window.player.view.message, timeout=TIMEOUT_MS)
    window.close()


def test_range_selection_limits_draw(qtbot, window):
    durations = window.browser.label_durations()
    cut = float(np.median(durations))
    lo, hi = window.histogram.region.getRegion()
    window.histogram.region.setRegion((np.log10(cut), hi))
    window.histogram._emit_range()
    window.sampler.n.setValue(50)
    window.sampler.draw_button.click()
    drawn = [window.browser.bout(i)["duration_s"] for i in window.browser.drawn]
    assert drawn and min(drawn) >= cut - 1e-9


def test_verdict_keys_record_mark_and_advance(qtbot, window):
    window.sampler.n.setValue(3)
    window.sampler.draw_button.click()
    _wait_for_clip(qtbot, window)
    first = window._current_bout_id()
    window.verdicts.note.setText("clean")
    window.verdicts.press_key("1")
    assert window.browser.verdict_for(first).verdict == "fits"
    assert window.browser.verdict_for(first).note == "clean"
    assert window.bout_list.currentRow() == 1  # advanced
    assert window.bout_list.item(0).text().startswith("●")
    window.verdicts.advance.setChecked(False)
    window.verdicts.press_key("3")
    second = window._current_bout_id()
    assert window.browser.verdict_for(second).verdict == "artefact"
    assert window.bout_list.currentRow() == 1  # stayed
    assert "rated 2 of 3" in window.verdicts.progress.text()


def test_propose_name(qtbot, window):
    window.sampler.draw_button.click()
    _wait_for_clip(qtbot, window)  # the panel is enabled once a bout is on screen
    window.verdicts.name.setText("forward glide")
    window.verdicts.propose.click()
    names = window.store.label_names(window.browser.labelling)
    assert names.name.tolist() == ["forward glide"]
    assert names.annotator.tolist() == ["tester"]


def test_blind_session_hides_and_reveals(qtbot, window):
    labels = tuple(window.browser.label_table().index[:2])
    window.start_blind(labels, 4)
    assert window.browser.is_blind and "BLIND" in window.windowTitle()
    clip = _wait_for_clip(qtbot, window)
    assert clip is not None
    assert window.player.timeline.strips == []  # no strips in blind mode
    assert window.player.bout_colour == "#0072B2"  # neutral, not the label's colour
    assert not window.verdicts.name_box.isVisible()
    bout = window._current_bout_id()
    truth = str(window.browser.bout(bout)["label"])
    window.verdicts.press_key(str(labels.index(truth) + 1))
    assert window.browser.verdict_for(bout).verdict == "fits"
    assert "previous bout was" in window.verdicts.reveal.text()


def test_resume_continues_the_session(qtbot, window, demo):
    window.sampler.n.setValue(3)
    window.sampler.draw_button.click()
    _wait_for_clip(qtbot, window)
    window.verdicts.press_key("2")
    session_id, drawn = window.browser.session_id, list(window.browser.drawn)
    other = MainWindow(demo, annotator="tester")
    qtbot.addWidget(other)
    other.resume_session(session_id)
    assert other.browser.drawn == drawn
    assert other.bout_list.count() == 3 and other.bout_list.currentRow() == 1  # first unrated
    assert other.bout_list.item(0).text().startswith("●")
    other.close()


def test_dialogs_without_showing_them(qtbot, window):
    from PySide6.QtCore import Qt

    from boutscope.gui.dialogs import BlindDialog, ResumeDialog

    dialog = BlindDialog(window.browser.label_table())
    qtbot.addWidget(dialog)
    dialog._accept_if_valid()
    assert "at least two" in dialog.hint.text()
    for row in (0, 1):
        dialog.list.item(row).setCheckState(Qt.CheckState.Checked)
    dialog.min_ms.setValue(40)
    settings = dialog.settings()
    assert len(settings.labels) == 2 and settings.min_duration_s == 0.04
    assert settings.max_duration_s is None

    window.sampler.draw_button.click()
    resume = ResumeDialog(window.store.list_sessions())
    qtbot.addWidget(resume)
    assert resume.selected_id() == window.browser.session_id
