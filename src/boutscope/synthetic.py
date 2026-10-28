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


# ┌────────────────────────────────────────────────────────────┐
# │ Demo project  « a viewable project in a few seconds »      │
# └────────────────────────────────────────────────────────────┘


@dataclass(frozen=True)
class DemoSpec:
    labels: SyntheticSpec = SyntheticSpec(n_videos=2, n_animals=4, n_frames=1500, seed=1)
    fps: float = 50.0
    box_px: int = 150
    flip_share: float = 0.15  # second labelling: this share of frames relabelled at random
    low_conf_share: float = 0.05  # keypoints below the overlay threshold


def make_demo_project(root: Path, spec: DemoSpec = DemoSpec()):
    """Project with two labellings, videos of moving three-point animals, and their pose.

    The animals rest, swim or turn according to the ``behaviour`` labelling;
    ``noisy`` is the same labelling with frames relabelled at random.  Video
    paths are absolute, so the demo needs no ``video_root``.
    """
    from boutscope import pose as bpose
    from boutscope import project as bp

    root = Path(root)
    table, _ = make_label_table(spec.labels)
    rng = np.random.default_rng(spec.labels.seed)
    n = spec.labels.n_animals
    cols, rows = 2, (n + 1) // 2
    width = max(cols * spec.box_px, BLOCK_BITS * BLOCK_PX)
    height = 2 * BLOCK_PX + rows * spec.box_px
    skeleton = bp.Skeleton(
        keypoints=("head", "centre", "tail"),
        edges=(("head", "centre"), ("centre", "tail")),
        min_confidence=0.3,
    )
    videos, poses = {}, {}
    (root / "videos").mkdir(parents=True, exist_ok=True)
    for video, group in table.groupby("video", sort=True):
        animals = sorted(group["animal"].unique())
        boxes = {
            a: bp.AnimalBox(
                x=(i % cols) * spec.box_px,
                y=2 * BLOCK_PX + (i // cols) * spec.box_px,
                w=spec.box_px,
                h=spec.box_px,
            )
            for i, a in enumerate(animals)
        }
        coords = np.stack(
            [_animate(group[group["animal"] == a], boxes[a], spec, rng) for a in animals]
        )
        conf = rng.uniform(0.5, 1.0, coords.shape[:3]).astype(np.float32)
        conf[rng.random(conf.shape) < spec.low_conf_share] = 0.1
        path = root / "videos" / f"{video}.avi"
        _write_demo_video(path, coords, conf, skeleton, (width, height), spec.fps)
        videos[video] = bp.VideoEntry(
            path=str(path.resolve()), fps=spec.fps, width=width, height=height, boxes=boxes
        )
        poses[video] = (coords, conf, animals)
    project = bp.create_project(
        root / "project",
        "demo",
        spec.fps,
        "synthetic",
        videos,
        strata=("condition",),
        skeleton=skeleton,
    )
    for video, (coords, conf, animals) in poses.items():
        bpose.write_pose(project, video, coords, conf, animals)
    meta = {
        "rest": bp.LabelMeta("rest", "#000000"),
        "swim": bp.LabelMeta("swim", "#009E73"),
        "turn": bp.LabelMeta("turn", "#CC79A7"),
    }
    project = bp.add_labelling(project, "behaviour", table, meta)
    noisy = table.copy()
    flip = rng.random(len(noisy)) < spec.flip_share
    noisy.loc[flip, "label"] = rng.choice(spec.labels.labels, int(flip.sum()))
    return bp.add_labelling(project, "noisy", noisy, meta)


def _animate(track: pd.DataFrame, box, spec: DemoSpec, rng: np.random.Generator) -> np.ndarray:
    """(frames, 3 keypoints, 2) for one animal; missing frames keep the last behaviour."""
    labels = (
        pd.Series(track["label"].to_numpy(), index=track["frame"].to_numpy())
        .reindex(range(spec.labels.n_frames))
        .ffill()
        .bfill()
        .to_numpy()
    )
    margin = 20.0
    pos = np.array([box.x + box.w / 2, box.y + box.h / 2], dtype=float)
    heading = float(rng.uniform(0, 2 * np.pi))
    out = np.empty((len(labels), 3, 2), dtype=np.float32)
    for f, label in enumerate(labels):
        if label == "swim":
            step = 1.5 * np.array([np.cos(heading), np.sin(heading)])
            nxt = pos + step
            if not (box.x + margin < nxt[0] < box.x + box.w - margin):
                heading = np.pi - heading
            if not (box.y + margin < nxt[1] < box.y + box.h - margin):
                heading = -heading
            pos = pos + 1.5 * np.array([np.cos(heading), np.sin(heading)])
        elif label == "turn":
            heading += 0.15
        axis = np.array([np.cos(heading), np.sin(heading)])
        out[f] = [pos + 8 * axis, pos, pos - 14 * axis]
    return out


def _write_demo_video(path: Path, coords, conf, skeleton, size, fps: float) -> None:
    import cv2

    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for f in range(coords.shape[1]):
            frame = encode_frame_index(f, width, height)
            frame[2 * BLOCK_PX :] = 200
            for animal in coords[:, f]:
                pts = [tuple(int(round(v)) for v in p) for p in animal]
                cv2.line(frame, pts[0], pts[1], (40, 40, 40), 3)
                cv2.line(frame, pts[1], pts[2], (40, 40, 40), 2)
                cv2.circle(frame, pts[0], 4, (20, 20, 20), -1)
            writer.write(frame)
    finally:
        writer.release()
