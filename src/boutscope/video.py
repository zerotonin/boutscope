# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — video                                               ║
# ║  « the frame you asked for, not its neighbour »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Frame-exact clip reading (seek, verify, decode), cropping to    ║
# ║  the animal, a memory budget, and a prefetching worker thread.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Frame-exact clip reading: one bout plus its context, cropped to the animal.

A :class:`ClipRequest` says which video frames to read; :class:`VideoReader`
reads them (seek, verify, sequential decode), crops to the animal's box and
downscales only if the clip would exceed the memory budget.
:class:`ClipLoader` runs the reader in a worker thread and keeps the next
clips prefetched, so the GUI never waits on a decoder.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from boutscope import config
from boutscope import constants as bc
from boutscope.project import AnimalBox, Project


class VideoError(RuntimeError):
    """A video cannot be opened or read where the project says it is."""


@dataclass(frozen=True)
class ClipRequest:
    """Frames to read for one bout: context before, the bout, context after."""

    bout_id: int
    video: str
    animal: str
    path: Path
    first_frame: int  # project frame of clip frame 0
    video_first: int  # video frame of clip frame 0
    n_frames: int
    bout_start: int  # clip index where the bout begins
    bout_stop: int  # clip index after the bout's last shown frame
    roi: AnimalBox | None
    truncated: bool  # bout longer than MAX_CLIP_S, shown from its start

    def key(self) -> tuple[int, bool]:
        return self.bout_id, self.roi is not None


@dataclass(frozen=True)
class Clip:
    request: ClipRequest
    frames: np.ndarray  # (n, h, w, 3) RGB uint8
    origin: tuple[int, int]  # video pixel of the clip's top-left corner
    scale: float  # clip pixels per video pixel

    def to_clip_xy(self, xy: np.ndarray) -> np.ndarray:
        """Video-pixel coordinates -> clip-pixel coordinates."""
        return (np.asarray(xy, dtype=float) - np.asarray(self.origin)) * self.scale


def resolve_video_path(project: Project, video: str, video_root: Path | None = None) -> Path:
    """Absolute path of *video*: absolute entries as given, else under the video root."""
    entry_path = Path(project.videos[video].path).expanduser()
    if entry_path.is_absolute():
        return entry_path
    root = video_root if video_root is not None else config.video_root()
    return Path(root) / entry_path


def build_clip_request(
    project: Project,
    bout,
    path: Path,
    *,
    context_s: float | None = None,
    zoom: bool = True,
) -> ClipRequest:
    """Clip for one bout-table row (any object with the bout columns as attributes)."""
    fps = project.fps
    ctx = int(round((project.context_s if context_s is None else context_s) * fps))
    max_bout = max(1, int(round(bc.MAX_CLIP_S * fps)) - 2 * ctx)
    shown = min(int(bout.n_frames), max_bout)
    before = min(ctx, int(bout.video_frame_start))  # no frames before the file starts
    video = str(bout.video)
    animal = str(bout.animal)
    return ClipRequest(
        bout_id=int(bout.bout_id),
        video=video,
        animal=animal,
        path=Path(path),
        first_frame=int(bout.start) - before,
        video_first=int(bout.video_frame_start) - before,
        n_frames=before + shown + ctx,
        bout_start=before,
        bout_stop=before + shown,
        roi=project.videos[video].boxes.get(animal) if zoom else None,
        truncated=shown < int(bout.n_frames),
    )


class VideoReader:
    """Open captures per file; thread-safe; frame-exact sequential reads."""

    def __init__(self, max_clip_bytes: int = bc.MAX_CLIP_BYTES) -> None:
        self.max_clip_bytes = max_clip_bytes
        self._captures: dict[Path, object] = {}
        self._lock = threading.Lock()

    def read_clip(self, request: ClipRequest) -> Clip:
        with self._lock:
            cropped, (x0, y0, x1, y1) = self._read_frames(request)
        scale = _budget_scale(len(cropped), (y1 - y0) * (x1 - x0) * 3, self.max_clip_bytes)
        if scale < 1.0:
            import cv2

            size = (max(1, int((x1 - x0) * scale)), max(1, int((y1 - y0) * scale)))
            cropped = [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in cropped]
        return Clip(request=request, frames=np.stack(cropped), origin=(x0, y0), scale=scale)

    def close(self) -> None:
        with self._lock:
            for capture in self._captures.values():
                capture.release()
            self._captures.clear()

    def _read_frames(
        self, request: ClipRequest
    ) -> tuple[list[np.ndarray], tuple[int, int, int, int]]:
        """Decode the clip, cropping each frame as it arrives (full frames never pile up)."""
        import cv2

        capture = self._capture(request.path)
        seek_frame(capture, request.video_first)
        frames, box = [], None
        for _ in range(request.n_frames):
            ok, frame = capture.read()
            if not ok:
                break
            if box is None:
                box = _crop_box(request.roi, frame.shape[1], frame.shape[0])
            x0, y0, x1, y1 = box
            frames.append(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        if not frames:
            raise VideoError(f"{request.path.name}: no frames at {request.video_first}")
        return frames, box

    def _capture(self, path: Path):
        import cv2

        if path not in self._captures:
            if not path.is_file():
                raise VideoError(f"video not found: {path}")
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise VideoError(f"OpenCV cannot open {path}")
            self._captures[path] = capture
        return self._captures[path]


def seek_frame(capture, target: int) -> None:
    """Position *capture* so the next read returns frame *target*.

    Direct seek first; if the decoder reports another position, seek back
    by ``SEEK_BACKOFF_FRAMES`` and step forward frame by frame.
    """
    import cv2

    capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == target:
        return
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - bc.SEEK_BACKOFF_FRAMES))
    position = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    while position < target and capture.grab():
        position += 1
    if position != target:
        raise VideoError(f"cannot seek to frame {target} (stopped at {position})")


class ClipLoader:
    """Reads clips in one worker thread; keeps recent and prefetched clips as futures."""

    def __init__(self, reader: VideoReader, keep: int = bc.PREFETCH_CLIPS + 2) -> None:
        self.reader = reader
        self.keep = keep
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="boutscope-clips")
        self._futures: OrderedDict[tuple[int, bool], Future] = OrderedDict()

    def get(self, request: ClipRequest) -> Future:
        key = request.key()
        if key in self._futures:
            self._futures.move_to_end(key)
        else:
            self._futures[key] = self._pool.submit(self.reader.read_clip, request)
            while len(self._futures) > self.keep:
                self._futures.popitem(last=False)
        return self._futures[key]

    def prefetch(self, requests: list[ClipRequest]) -> None:
        for request in requests[: bc.PREFETCH_CLIPS]:
            self.get(request)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        self.reader.close()


def _crop_box(roi: AnimalBox | None, width: int, height: int) -> tuple[int, int, int, int]:
    if roi is None:
        return 0, 0, width, height
    x0, y0 = max(0, roi.x), max(0, roi.y)
    x1, y1 = min(width, roi.x + roi.w), min(height, roi.y + roi.h)
    if x1 <= x0 or y1 <= y0:
        raise VideoError(f"box {roi} lies outside the {width} x {height} frame")
    return x0, y0, x1, y1


def _budget_scale(n_frames: int, bytes_per_frame: int, budget: int) -> float:
    total = n_frames * bytes_per_frame
    return 1.0 if total <= budget else float(np.sqrt(budget / total))
