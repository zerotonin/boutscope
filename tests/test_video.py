"""Frame-exact reading, cropping, memory budget, clip requests, prefetch."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from boutscope import project as bp
from boutscope.synthetic import decode_frame_index, write_frame_index_video
from boutscope.video import (
    ClipLoader,
    ClipRequest,
    VideoError,
    VideoReader,
    build_clip_request,
    resolve_video_path,
)

cv2 = pytest.importorskip("cv2")
N_FRAMES = 400


@pytest.fixture(scope="module", params=[("MJPG", ".avi"), ("mp4v", ".mp4")])
def video(tmp_path_factory, request):
    fourcc, suffix = request.param
    path = tmp_path_factory.mktemp("video") / f"index{suffix}"
    return write_frame_index_video(path, N_FRAMES, fourcc=fourcc)


def _request(path, first, n, roi=None) -> ClipRequest:
    return ClipRequest(
        bout_id=0,
        video="v",
        animal="a",
        path=path,
        first_frame=first,
        video_first=first,
        n_frames=n,
        bout_start=0,
        bout_stop=n,
        roi=roi,
        truncated=False,
    )


def test_random_seeks_are_frame_exact(video):
    reader = VideoReader()
    rng = np.random.default_rng(0)
    for first in [0, 1, 399 - 5, *rng.integers(0, N_FRAMES - 6, 15), 3]:  # includes backward seeks
        clip = reader.read_clip(_request(video, int(first), 6))
        assert [decode_frame_index(f[..., ::-1]) for f in clip.frames] == list(
            range(first, first + 6)
        )
    reader.close()


def test_crop_and_coordinates(video):
    roi = bp.AnimalBox(x=10, y=4, w=100, h=50)
    clip = VideoReader().read_clip(_request(video, 0, 3, roi))
    assert clip.frames.shape == (3, 50, 100, 3)
    assert clip.origin == (10, 4) and clip.scale == 1.0
    np.testing.assert_allclose(clip.to_clip_xy([[60, 29]]), [[50, 25]])


def test_budget_downscales(video):
    clip = VideoReader(max_clip_bytes=320 * 240 * 3 * 10 // 4).read_clip(_request(video, 0, 10))
    assert clip.scale == pytest.approx(0.5)
    assert clip.frames.shape[1:3] == (120, 160)


def test_read_past_end_returns_what_exists(video):
    clip = VideoReader().read_clip(_request(video, N_FRAMES - 3, 10))
    assert len(clip.frames) == 3


def test_errors(tmp_path, video):
    with pytest.raises(VideoError, match="not found"):
        VideoReader().read_clip(_request(tmp_path / "missing.avi", 0, 1))
    with pytest.raises(VideoError, match="outside"):
        VideoReader().read_clip(_request(video, 0, 1, bp.AnimalBox(5000, 5000, 10, 10)))


def test_clip_request_context_and_truncation(tmp_path):
    videos = {"v": bp.VideoEntry(path="v.avi", fps=50.0, boxes={"a": bp.AnimalBox(0, 0, 9, 9)})}
    project = bp.create_project(tmp_path / "p", "p", 50.0, "generic", videos)
    bout = SimpleNamespace(
        bout_id=3, video="v", animal="a", start=100, n_frames=10, video_frame_start=110
    )
    request = build_clip_request(project, bout, tmp_path / "v.avi")
    assert (request.first_frame, request.video_first, request.n_frames) == (75, 85, 60)
    assert (request.bout_start, request.bout_stop, request.truncated) == (25, 35, False)
    assert request.roi == bp.AnimalBox(0, 0, 9, 9)
    early = SimpleNamespace(**{**vars(bout), "video_frame_start": 5, "start": 0})
    request = build_clip_request(project, early, tmp_path / "v.avi", zoom=False)
    assert (request.bout_start, request.video_first, request.roi) == (5, 0, None)
    long = SimpleNamespace(**{**vars(bout), "n_frames": 10_000})
    request = build_clip_request(project, long, tmp_path / "v.avi")
    assert request.truncated and request.n_frames == 1500


def test_resolve_video_path(tmp_path, monkeypatch):
    videos = {
        "rel": bp.VideoEntry(path="sub/r.mp4", fps=50.0),
        "abs": bp.VideoEntry(path=str(tmp_path / "a.mp4"), fps=50.0),
    }
    project = bp.create_project(tmp_path / "p", "p", 50.0, "generic", videos)
    assert resolve_video_path(project, "rel", tmp_path) == tmp_path / "sub" / "r.mp4"
    assert resolve_video_path(project, "abs") == tmp_path / "a.mp4"
    monkeypatch.setenv("BOUTSCOPE_VIDEO_ROOT", str(tmp_path / "root"))
    assert resolve_video_path(project, "rel") == tmp_path / "root" / "sub" / "r.mp4"


def test_loader_caches_and_prefetches(video):
    loader = ClipLoader(VideoReader(), keep=3)
    requests = [_request(video, f, 4) for f in (0, 10, 20, 30)]
    requests = [ClipRequest(**{**vars(r), "bout_id": i}) for i, r in enumerate(requests)]
    first = loader.get(requests[0])
    assert loader.get(requests[0]) is first
    loader.prefetch(requests[1:])
    clip = loader.get(requests[1]).result(timeout=10)
    assert decode_frame_index(clip.frames[0][..., ::-1]) == 10
    loader.shutdown()


def test_frame_map_clip_repeats_dropped_frames(tmp_path, video):
    from boutscope.video import fill_missing_frames

    videos = {"v": bp.VideoEntry(path=str(video), fps=50.0)}
    project = bp.create_project(tmp_path / "p", "p", 50.0, "generic", videos)
    frame_map = np.r_[np.arange(0, 50), -1, np.arange(50, 399)]  # project frame 50 dropped
    bp.write_frame_map(project, "v", frame_map)
    bout = SimpleNamespace(
        bout_id=1, video="v", animal="a", start=48, n_frames=6, video_frame_start=48
    )
    request = build_clip_request(
        project, bout, video, context_s=0.04, frame_map=project.load_frame_map("v")
    )
    assert request.first_frame == 46 and request.bout_start == 2
    assert request.video_frames == (46, 47, 48, 49, 49, 50, 51, 52, 53, 54)
    clip = VideoReader().read_clip(request)
    assert [decode_frame_index(f[..., ::-1]) for f in clip.frames] == list(request.video_frames)
    np.testing.assert_array_equal(
        fill_missing_frames(np.array([-1, -1, 3, -1, 5])), [3, 3, 3, 3, 5]
    )


def test_write_frame_map_validates(tmp_path):
    project = bp.create_project(
        tmp_path / "p", "p", 50.0, "generic", {"v": bp.VideoEntry(path="v", fps=50.0)}
    )
    with pytest.raises(bp.ProjectError, match="increase"):
        bp.write_frame_map(project, "v", np.array([0, 2, 1]))
    with pytest.raises(bp.ProjectError, match="not in"):
        bp.write_frame_map(project, "w", np.array([0]))
    assert project.load_frame_map("v") is None
