"""Pose store, strip lookups and the headless browser, on the demo project."""

from __future__ import annotations

import numpy as np
import pytest

from boutscope import project as bp
from boutscope.browser import Browser
from boutscope.pose import PoseStore, write_pose
from boutscope.strips import MISSING, LabelLookup
from boutscope.synthetic import DemoSpec, SyntheticSpec, decode_frame_index, make_demo_project
from boutscope.video import VideoReader

pytest.importorskip("cv2")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    spec = DemoSpec(labels=SyntheticSpec(n_videos=2, n_animals=2, n_frames=400, seed=2))
    return make_demo_project(tmp_path_factory.mktemp("demo"), spec)


def test_pose_slice_pads_outside_the_track(demo):
    store = PoseStore(demo)
    animal = next(iter(demo.videos["video0"].boxes))
    pose = store.load_slice("video0", animal, -5, 5)
    assert pose.coords.shape == (10, 3, 2)
    assert np.isnan(pose.coords[:5]).all() and np.isfinite(pose.coords[5:]).all()
    visible = pose.visible(demo.skeleton.min_confidence)
    assert not visible[:5].any()
    assert store.load_slice("video0", "nobody", 0, 5) is None


def test_write_pose_validates(tmp_path):
    project = bp.create_project(
        tmp_path, "p", 50.0, "generic", {"v": bp.VideoEntry(path="v", fps=50.0)}
    )
    with pytest.raises(bp.ProjectError, match="skeleton"):
        write_pose(project, "v", np.zeros((1, 2, 3, 2)), np.zeros((1, 2, 3)), ["a"])


def test_label_lookup_marks_missing_frames(demo):
    lookup = LabelLookup(demo, "behaviour")
    table = demo.load_labels("behaviour")
    track = table[(table.video == "video0") & (table.animal == table.animal.iloc[0])]
    frames = track.frame.to_numpy()
    first, stop = int(frames[0]) - 3, int(frames[0]) + 50
    codes = lookup.labels("video0", str(track.animal.iloc[0]), first, stop)
    present = np.isin(np.arange(first, stop), frames)
    assert (codes[~present] == MISSING).all() and (codes[present] != MISSING).all()
    assert lookup.colour_of(MISSING) == "#FFFFFF"
    assert (lookup.labels("video9", "x", 0, 4) == MISSING).all()


def test_browser_flow(demo):
    browser = Browser(demo)
    table = browser.label_table()
    assert table.share_of_frames.is_monotonic_decreasing
    browser.select_label("swim")
    counts, edges = browser.histogram()
    assert counts.sum() == len(browser.label_durations())
    available = browser.available(0.1, None, strata=("condition",))
    drawn = browser.draw(5, min_duration_s=0.1, strata=("condition",), seed=4)
    assert len(drawn) == min(5, available.sum())
    assert set(browser.more(3)).isdisjoint(drawn)
    request = browser.clip_request(drawn[0])
    clip = VideoReader().read_clip(request)
    bout = browser.bout(drawn[0])
    assert request.roi is not None and clip.frames.shape[1:3] == (150, 150)
    strips = browser.strips(request)
    assert [s.labelling for s in strips] == ["behaviour", "noisy"]
    inside = strips[0].codes[request.bout_start : request.bout_stop]
    assert (inside == strips[0].lookup.names.index("swim")).all()
    assert bout.duration_s >= 0.1


def test_clip_shows_the_bout_frames(demo):
    browser = Browser(demo)
    browser.select_label("turn")
    bout_id = browser.draw(1)[0]
    request = browser.clip_request(bout_id, zoom=False)
    clip = VideoReader().read_clip(request)
    first_bout_frame = clip.frames[request.bout_start][..., ::-1]
    assert decode_frame_index(first_bout_frame) == int(browser.bout(bout_id).video_frame_start)


def test_browser_needs_a_label(demo):
    browser = Browser(demo)
    with pytest.raises(RuntimeError, match="label"):
        browser.histogram()
    browser.select_labelling("noisy")
    with pytest.raises(RuntimeError, match="draw"):
        browser.more(1)


def test_checks_on_the_demo(demo, capsys):
    from boutscope.checks import check_alignment, check_seeking
    from boutscope.cli import main
    from boutscope.video import resolve_video_path

    report = check_seeking(resolve_video_path(demo, "video0"), n_seeks=10, span=300)
    assert report.n_seeks > 5 and report.wrong == []
    alignment = check_alignment(demo, "behaviour", n_bouts=10)
    assert alignment.best_shift == 0 and alignment.n_bouts > 0
    assert main(["check", str(demo.root), "--labelling", "behaviour"]) == 0
    assert "aligned" in capsys.readouterr().out
