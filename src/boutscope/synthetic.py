# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — synthetic                                           ║
# ║  « data whose answers we already know »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Label tables with known bouts and videos that carry their own   ║
# ║  frame index, for tests and demos.                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Synthetic label tables and videos with known answers, for tests and demos.

The videos encode each frame's index as a row of black / white blocks, so
a reader can check that the frame it got is the frame it asked for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_BITS: int = 20  # up to ~1 M frames
BLOCK_PX: int = 16


@dataclass(frozen=True)
class SyntheticSpec:
    n_videos: int = 2
    n_animals: int = 3  # per video; animal ids are global ("a0".."a5")
    n_frames: int = 600
    labels: tuple[str, ...] = ("rest", "swim", "turn")
    mean_bout_frames: float = 12.0
    gap_share: float = 0.01  # frames dropped from the table
    conditions: tuple[str, ...] = ("control", "drug")
    seed: int = 0


def make_label_table(spec: SyntheticSpec = SyntheticSpec()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label table and the true bouts it was built from (gaps included as bout breaks).

    Returns:
        table: video, animal, frame, label, typicality, condition.
        truth: video, animal, label, start, stop (frames, stop exclusive).
    """
    rng = np.random.default_rng(spec.seed)
    rows, truth = [], []
    for v in range(spec.n_videos):
        video = f"video{v}"
        condition = spec.conditions[v % len(spec.conditions)]
        for a in range(spec.n_animals):
            animal = f"a{v * spec.n_animals + a}"
            labels = _draw_runs(rng, spec)
            present = rng.random(spec.n_frames) >= spec.gap_share
            frames = np.flatnonzero(present)
            rows.append(
                pd.DataFrame(
                    {
                        "video": video,
                        "animal": animal,
                        "frame": frames,
                        "label": np.asarray(spec.labels)[labels[frames]],
                        "typicality": rng.random(len(frames)),
                        "condition": condition,
                    }
                )
            )
            truth.append(_runs_of(labels, present, video, animal, spec.labels))
    return pd.concat(rows, ignore_index=True), pd.concat(truth, ignore_index=True)


def write_frame_index_video(
    path: Path,
    n_frames: int,
    fps: float = 50.0,
    size: tuple[int, int] = (320, 240),
    fourcc: str = "MJPG",
) -> Path:
    """Write a video whose frame *i* shows *i* in binary blocks along the top row."""
    import cv2

    width, height = size
    if BLOCK_BITS * BLOCK_PX > width:
        raise ValueError(f"width {width} < {BLOCK_BITS * BLOCK_PX} px needed for the frame code")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV cannot write {fourcc} to {path}")
    try:
        for i in range(n_frames):
            writer.write(encode_frame_index(i, width, height))
    finally:
        writer.release()
    return Path(path)


def encode_frame_index(index: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 128, dtype=np.uint8)
    for bit in range(BLOCK_BITS):
        value = 255 if (index >> bit) & 1 else 0
        frame[:BLOCK_PX, bit * BLOCK_PX : (bit + 1) * BLOCK_PX] = value
    return frame


def decode_frame_index(frame: np.ndarray) -> int:
    """Read the frame index back from a decoded (BGR or grey) frame."""
    grey = frame if frame.ndim == 2 else frame.mean(axis=2)
    centre = BLOCK_PX // 2
    bits = [
        grey[centre - 2 : centre + 2, b * BLOCK_PX + centre - 2 : b * BLOCK_PX + centre + 2].mean()
        > 127
        for b in range(BLOCK_BITS)
    ]
    return int(sum(1 << b for b, on in enumerate(bits) if on))


def _draw_runs(rng: np.random.Generator, spec: SyntheticSpec) -> np.ndarray:
    labels = np.empty(spec.n_frames, dtype=np.int64)
    i, current = 0, int(rng.integers(len(spec.labels)))
    while i < spec.n_frames:
        length = 1 + int(rng.geometric(1.0 / spec.mean_bout_frames))
        labels[i : i + length] = current
        i += length
        current = (current + 1 + int(rng.integers(len(spec.labels) - 1))) % len(spec.labels)
    return labels


def _runs_of(
    labels: np.ndarray, present: np.ndarray, video: str, animal: str, names: tuple[str, ...]
) -> pd.DataFrame:
    runs, start = [], None
    for f in range(len(labels) + 1):
        ends = (
            f == len(labels) or not present[f] or (start is not None and labels[f] != labels[start])
        )
        if start is not None and ends:
            runs.append((video, animal, names[labels[start]], start, f))
            start = None
        if f < len(labels) and present[f] and start is None:
            start = f
    return pd.DataFrame(runs, columns=["video", "animal", "label", "start", "stop"])
