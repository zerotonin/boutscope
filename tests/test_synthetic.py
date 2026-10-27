"""Synthetic videos encode the frame index; decoding reads it back."""

from __future__ import annotations

import pytest

from boutscope.synthetic import (
    decode_frame_index,
    encode_frame_index,
    write_frame_index_video,
)

cv2 = pytest.importorskip("cv2")


def test_encode_decode_without_video():
    for index in (0, 1, 5, 1023, 123_456):
        assert decode_frame_index(encode_frame_index(index, 320, 240)) == index


@pytest.mark.parametrize("fourcc, suffix", [("MJPG", ".avi"), ("mp4v", ".mp4")])
def test_video_round_trip(tmp_path, fourcc, suffix):
    path = write_frame_index_video(tmp_path / f"v{suffix}", 60, fourcc=fourcc)
    capture = cv2.VideoCapture(str(path))
    indices = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        indices.append(decode_frame_index(frame))
    capture.release()
    assert indices == list(range(60))
