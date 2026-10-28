# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — constants                                           ║
# ║  « one source of truth for names and defaults »                  ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Project file names, label-table columns, sampling defaults,     ║
# ║  Wong (2011) colours.  Import these instead of repeating them.   ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Shared names, defaults and colours."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

# ┌────────────────────────────────────────────────────────────┐
# │ Project layout  « file names inside a project folder »     │
# └────────────────────────────────────────────────────────────┘
PROJECT_FILE: str = "project.toml"
VIDEOS_FILE: str = "videos.toml"
LABELS_DIR: str = "labels"
BOUTS_DIR: str = "bouts"
POSE_DIR: str = "pose"
DB_FILE: str = "boutscope.sqlite"
LABEL_TABLE_SUFFIX: str = ".parquet"
LABEL_META_SUFFIX: str = ".meta.toml"
FORMAT_VERSION: int = 1

# ┌────────────────────────────────────────────────────────────┐
# │ Label table  « columns every adapter writes »              │
# └────────────────────────────────────────────────────────────┘
REQUIRED_LABEL_COLUMNS: tuple[str, ...] = ("video", "animal", "frame", "label")
OPTIONAL_LABEL_COLUMNS: tuple[str, ...] = ("video_frame", "typicality")
TRACK_KEYS: tuple[str, ...] = ("video", "animal")

BOUT_COLUMNS: tuple[str, ...] = (
    "bout_id",
    "video",
    "animal",
    "label",
    "start",
    "stop",
    "n_frames",
    "duration_s",
    "video_frame_start",
    "video_frame_stop",
)

# ┌────────────────────────────────────────────────────────────┐
# │ Sampling  « defaults for drawing bouts »                   │
# └────────────────────────────────────────────────────────────┘
DEFAULT_STRATA: tuple[str, ...] = ("animal",)
SAMPLING_MODES: tuple[str, ...] = ("stratified", "typical", "atypical")
TYPICALITY_QUANTILE: float = 0.1  # typical = top decile, atypical = bottom decile
DEFAULT_CONTEXT_S: float = 0.5
DEFAULT_SEED: int = 20261028

# ┌────────────────────────────────────────────────────────────┐
# │ Colours  « Wong (2011), cycled over labels »               │
# └────────────────────────────────────────────────────────────┘
WONG: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
LABEL_COLOUR_CYCLE: tuple[str, ...] = tuple(c for k, c in WONG.items() if k != "black")

# ┌────────────────────────────────────────────────────────────┐
# │ Type aliases  « shared array shapes »                      │
# └────────────────────────────────────────────────────────────┘
BoutIds: TypeAlias = np.ndarray  # int64 bout ids in draw order

# ┌────────────────────────────────────────────────────────────┐
# │ Video  « clip reading and playback »                       │
# └────────────────────────────────────────────────────────────┘
MAX_CLIP_BYTES: int = 512 * 2**20  # decoded frames per clip; larger clips are downscaled
MAX_CLIP_S: float = 30.0  # longer bouts are shown from their start, truncated
SEEK_BACKOFF_FRAMES: int = 250  # re-seek this far back when a direct seek lands wrong
PLAYBACK_SPEEDS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_SPEED: float = 0.25
PREFETCH_CLIPS: int = 2

# ┌────────────────────────────────────────────────────────────┐
# │ Pose  « keypoint arrays per video »                        │
# └────────────────────────────────────────────────────────────┘
POSE_COORDS_FILE: str = "coords.npy"  # (animals, frames, keypoints, 2), video pixels
POSE_CONF_FILE: str = "conf.npy"  # (animals, frames, keypoints)
POSE_META_FILE: str = "meta.toml"  # animals (row order), frame0 (project frame of row 0)

# ┌────────────────────────────────────────────────────────────┐
# │ Viewer colours  « context vs bout, missing labels »        │
# └────────────────────────────────────────────────────────────┘
CONTEXT_COLOUR: str = "#999999"
MISSING_LABEL_COLOUR: str = "#FFFFFF"
KEYPOINT_COLOUR: str = WONG["yellow"]
SKELETON_COLOUR: str = WONG["sky_blue"]
