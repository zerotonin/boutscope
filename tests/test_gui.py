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
    window = MainWindow(demo)
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
    window = MainWindow(bp.load_project(copy))
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
