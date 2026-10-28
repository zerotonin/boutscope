# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — checks                                              ║
# ║  « trust, but verify the frame mapping »                         ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Exact seeks against sequential decoding, and keypoint-to-video  ║
# ║  alignment by the darkness under the pose at each frame shift.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Checks that a project shows what it claims: exact seeks, and pose on the animal.

``check_seeking`` compares frames reached by direct seeks with the same
frames reached by sequential decoding.  ``check_alignment`` shifts the video
against the keypoints by -k..+k frames and measures the grey value under
the keypoints: on a dark animal over a light background the darkest shift
is the true alignment, so anything but 0 means the frame mapping is off.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boutscope.pose import PoseStore
from boutscope.project import Project
from boutscope.video import resolve_video_path, seek_frame

MAX_SHIFT: int = 15
ALIGN_BOUT_FRAMES: tuple[int, int] = (10, 60)


@dataclass(frozen=True)
class SeekReport:
    video: str
    n_seeks: int
    wrong: list[int]


@dataclass(frozen=True)
class AlignmentReport:
    shifts: np.ndarray
    darkness: np.ndarray  # mean grey under keypoints minus each bout's mean; lower = darker
    n_bouts: int

    @property
    def best_shift(self) -> int:
        return int(self.shifts[int(np.nanargmin(self.darkness))])


def check_seeking(path: Path, n_seeks: int = 30, span: int = 20_000, seed: int = 0) -> SeekReport:
    """Seek to random frames below *span*; compare with sequential decoding."""
    import cv2

    rng = np.random.default_rng(seed)
    targets = sorted({0, 1, *rng.integers(0, span, n_seeks).tolist()})
    capture = cv2.VideoCapture(str(path))
    reference, wanted = {}, set(targets)
    for index in range(max(targets) + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            reference[index] = hashlib.md5(frame.tobytes()).hexdigest()
    targets = [t for t in targets if t in reference]
    wrong = []
    for target in rng.permutation(targets):
        seek_frame(capture, int(target))
        ok, frame = capture.read()
        if not ok or hashlib.md5(frame.tobytes()).hexdigest() != reference[int(target)]:
            wrong.append(int(target))
    capture.release()
    return SeekReport(video=Path(path).name, n_seeks=len(targets), wrong=sorted(wrong))


def check_alignment(
    project: Project,
    labelling: str,
    n_bouts: int = 40,
    video_root: Path | None = None,
    seed: int = 1,
) -> AlignmentReport:
    """Darkness under the keypoints for frame shifts -MAX_SHIFT..+MAX_SHIFT."""
    import cv2

    bouts = project.load_bouts(labelling)
    lo, hi = ALIGN_BOUT_FRAMES
    pool = bouts[(bouts.n_frames >= lo) & (bouts.n_frames <= hi)]
    rng = np.random.default_rng(seed)
    pool = pool.iloc[rng.choice(len(pool), min(n_bouts, len(pool)), replace=False)]
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1)
    store, rows = PoseStore(project), []
    for bout in pool.itertuples():
        video, animal = str(bout.video), str(bout.animal)
        pose = store.load_slice(video, animal, int(bout.start), int(bout.stop))
        if pose is None:
            continue
        video_frames = _video_frames(project, video, int(bout.start), int(bout.stop))
        first = int(video_frames[video_frames >= 0].min()) - MAX_SHIFT
        capture = cv2.VideoCapture(str(resolve_video_path(project, video, video_root)))
        seek_frame(capture, max(first, 0))
        last = int(video_frames.max()) + MAX_SHIFT
        frames = {}
        for index in range(max(first, 0), last + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        capture.release()
        rows.append([_grey_under_pose(frames, video_frames, pose, s) for s in shifts])
    scores = np.array(rows, dtype=float)
    darkness = np.nanmean(scores - np.nanmean(scores, axis=1, keepdims=True), axis=0)
    return AlignmentReport(shifts=shifts, darkness=darkness, n_bouts=len(scores))


def _video_frames(project: Project, video: str, start: int, stop: int) -> np.ndarray:
    frame_map = project.load_frame_map(video)
    if frame_map is not None:
        return np.asarray(frame_map[start:stop])
    return np.arange(start, stop) + project.videos[video].frame_offset


def _grey_under_pose(frames: dict, video_frames: np.ndarray, pose, shift: int) -> float:
    values = []
    for i, frame_index in enumerate(video_frames):
        image = frames.get(int(frame_index) + shift)
        if frame_index < 0 or image is None:
            continue
        seen = np.isfinite(pose.coords[i]).all(axis=1) & (np.nan_to_num(pose.conf[i]) > 0)
        xy = np.round(pose.coords[i][seen]).astype(int)
        inside = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < image.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < image.shape[0])
        )
        values.extend(image[xy[inside, 1], xy[inside, 0]].tolist())
    return float(np.mean(values)) if values else np.nan


def format_alignment(report: AlignmentReport) -> str:
    table = pd.DataFrame({"shift": report.shifts, "darkness": report.darkness.round(2)})
    verdict = (
        "aligned: the darkest shift is 0"
        if report.best_shift == 0
        else f"MISALIGNED: the darkest shift is {report.best_shift:+d} frames"
    )
    return f"{table.to_string(index=False)}\n{report.n_bouts} bouts; {verdict}"
