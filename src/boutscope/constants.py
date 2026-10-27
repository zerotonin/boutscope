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
