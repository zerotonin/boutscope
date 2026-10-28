# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — pose                                                ║
# ║  « keypoints, memory-mapped »                                    ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Write and read per-video keypoint arrays for the skeleton       ║
# ║  overlay; .npy so a clip touches only its own frames.            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Keypoints per video, memory-mapped, for the skeleton overlay.

Layout: ``pose/<video>/coords.npy`` (animals, frames, keypoints, 2) in video
pixels, ``conf.npy`` (animals, frames, keypoints), and ``meta.toml`` with
the animal ids in row order and ``frame0``, the project frame of row 0.
Keypoint order is the project skeleton's.  ``.npy`` (not ``.npz``) so the
arrays memory-map: a clip touches only its own frames.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tomli_w

from boutscope import constants as bc
from boutscope.project import Project, ProjectError

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


@dataclass(frozen=True)
class PoseSlice:
    """Keypoints of one animal over a run of project frames; NaN where unknown."""

    coords: np.ndarray  # (frames, keypoints, 2)
    conf: np.ndarray  # (frames, keypoints)

    def visible(self, min_confidence: float) -> np.ndarray:
        """True where a keypoint is drawn: finite and at least *min_confidence*."""
        finite = np.isfinite(self.coords).all(axis=-1)
        return finite & (np.nan_to_num(self.conf, nan=-1.0) >= min_confidence)


def pose_dir(project: Project, video: str) -> Path:
    return project.root / bc.POSE_DIR / video


def write_pose(
    project: Project,
    video: str,
    coords: np.ndarray,
    conf: np.ndarray,
    animals: list[str],
    frame0: int = 0,
) -> Path:
    """Store one video's keypoints; validated against the project skeleton."""
    if project.skeleton is None:
        raise ProjectError("pose needs a skeleton in project.toml")
    if video not in project.videos:
        raise ProjectError(f"video {video!r} not in {bc.VIDEOS_FILE}")
    n_keypoints = len(project.skeleton.keypoints)
    expected = (len(animals), coords.shape[1], n_keypoints, 2)
    if coords.shape != expected or conf.shape != expected[:3]:
        raise ProjectError(
            f"pose arrays {coords.shape} / {conf.shape}, expected {expected} / {expected[:3]}"
        )
    folder = pose_dir(project, video)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / bc.POSE_COORDS_FILE, coords.astype(np.float32))
    np.save(folder / bc.POSE_CONF_FILE, conf.astype(np.float32))
    meta = {"animals": [str(a) for a in animals], "frame0": int(frame0)}
    (folder / bc.POSE_META_FILE).write_text(tomli_w.dumps(meta), encoding="utf-8")
    return folder


class PoseStore:
    """Read access to all pose folders of a project (memory-mapped, cached per video)."""

    def __init__(self, project: Project) -> None:
        self.project = project
        self._videos: dict[str, tuple[np.ndarray, np.ndarray, dict[str, int], int]] = {}

    def has_pose(self, video: str) -> bool:
        return (pose_dir(self.project, video) / bc.POSE_META_FILE).is_file()

    def load_slice(self, video: str, animal: str, first: int, stop: int) -> PoseSlice | None:
        """Keypoints for project frames ``first .. stop-1``; None without pose for the animal."""
        if not self.has_pose(video):
            return None
        coords, conf, rows, frame0 = self._open(video)
        if animal not in rows:
            return None
        n = stop - first
        out_coords = np.full((n, coords.shape[2], 2), np.nan, dtype=np.float32)
        out_conf = np.full((n, coords.shape[2]), np.nan, dtype=np.float32)
        lo, hi = max(first - frame0, 0), min(stop - frame0, coords.shape[1])
        if hi > lo:
            target = slice(lo + frame0 - first, hi + frame0 - first)
            out_coords[target] = coords[rows[animal], lo:hi]
            out_conf[target] = conf[rows[animal], lo:hi]
        return PoseSlice(out_coords, out_conf)

    def _open(self, video: str):
        if video not in self._videos:
            folder = pose_dir(self.project, video)
            meta = tomllib.loads((folder / bc.POSE_META_FILE).read_text(encoding="utf-8"))
            self._videos[video] = (
                np.load(folder / bc.POSE_COORDS_FILE, mmap_mode="r"),
                np.load(folder / bc.POSE_CONF_FILE, mmap_mode="r"),
                {a: i for i, a in enumerate(meta["animals"])},
                int(meta["frame0"]),
            )
        return self._videos[video]
