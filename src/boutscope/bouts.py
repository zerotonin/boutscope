# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — bouts                                               ║
# ║  « a bout is a run, nothing more »                               ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Run-length encoding of label tables into bouts, plus the label  ║
# ║  summary and duration histogram the label picker shows.          ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Label table -> bout table, and the summaries the label picker shows.

A bout is the longest unbroken run of one label for one animal in one
video over consecutive frame numbers.  It ends where the label changes,
where a frame is missing from the table, or where the track ends.
Labels are taken exactly as given: no merging, no filtering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from boutscope import constants as bc


def build_bout_table(
    table: pd.DataFrame,
    fps: float,
    *,
    strata: Sequence[str] = (),
    frame_offsets: Mapping[str, int] | None = None,
    frame_maps: Mapping[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Run-length encode a canonical label table (sorted by video, animal, frame).

    ``start`` / ``stop`` are frames (stop exclusive).  ``video_frame_start`` /
    ``video_frame_stop`` locate the bout in the video file: from the table's
    ``video_frame`` column if present, else the video's frame map (first and
    last frame of the bout the video actually has), else ``frame + frame_offset``.  Stratum
    columns take their value at the bout's first frame; ``typicality`` is the
    bout mean.
    """
    n = len(table)
    if n == 0:
        return pd.DataFrame(columns=list(bc.BOUT_COLUMNS))
    starts = find_bout_starts(
        table["video"].cat.codes.to_numpy(),
        table["animal"].cat.codes.to_numpy(),
        table["label"].cat.codes.to_numpy(),
        table["frame"].to_numpy(),
    )
    stops = np.r_[starts[1:], n]
    frame = table["frame"].to_numpy()
    bouts = pd.DataFrame(
        {
            "bout_id": np.arange(len(starts), dtype=np.int64),
            "video": _take_categorical(table["video"], starts),
            "animal": _take_categorical(table["animal"], starts),
            "label": _take_categorical(table["label"], starts),
            "start": frame[starts],
            "stop": frame[stops - 1] + 1,
        }
    )
    bouts["n_frames"] = (stops - starts).astype(np.int64)
    bouts["duration_s"] = bouts["n_frames"] / float(fps)
    if "video_frame" in table:
        video_frame = table["video_frame"].to_numpy()
        bouts["video_frame_start"] = video_frame[starts]
        bouts["video_frame_stop"] = video_frame[stops - 1] + 1
    else:
        offsets = dict(frame_offsets or {})
        per_video = np.array([offsets.get(v, 0) for v in bouts["video"].cat.categories], np.int64)
        offset = per_video[bouts["video"].cat.codes.to_numpy()]
        bouts["video_frame_start"] = bouts["start"].to_numpy() + offset
        bouts["video_frame_stop"] = bouts["stop"].to_numpy() + offset
        for video, frame_map in (frame_maps or {}).items():
            rows = (bouts["video"].astype(str) == video).to_numpy()
            if rows.any():
                start, stop = map_bout_to_video(
                    frame_map, bouts["start"].to_numpy()[rows], bouts["stop"].to_numpy()[rows]
                )
                bouts.loc[rows, "video_frame_start"] = start
                bouts.loc[rows, "video_frame_stop"] = stop
    for column in strata:
        values = table[column]
        bouts[column] = (
            _take_categorical(values, starts)
            if isinstance(values.dtype, pd.CategoricalDtype)
            else values.to_numpy()[starts]
        )
    if "typicality" in table:
        bouts["typicality"] = compute_bout_means(table["typicality"].to_numpy(), starts)
    return bouts


def map_bout_to_video(
    frame_map: np.ndarray, start: np.ndarray, stop: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """First and (exclusive) last video frame inside each bout; -1 if it has none."""
    frame_map = np.asarray(frame_map)
    index = np.arange(len(frame_map))
    valid = frame_map >= 0
    next_valid = np.minimum.accumulate(np.where(valid, index, len(frame_map))[::-1])[::-1]
    prev_valid = np.maximum.accumulate(np.where(valid, index, -1))
    first = next_valid[np.clip(start, 0, len(frame_map) - 1)]
    last = prev_valid[np.clip(stop - 1, 0, len(frame_map) - 1)]
    inside = (first <= last) & (first < len(frame_map)) & (start < len(frame_map))
    safe_first = np.clip(first, 0, len(frame_map) - 1)
    safe_last = np.clip(last, 0, len(frame_map) - 1)
    return (
        np.where(inside, frame_map[safe_first], -1),
        np.where(inside, frame_map[safe_last] + 1, -1),
    )


def _take_categorical(values: pd.Series, rows: np.ndarray) -> pd.Categorical:
    """Rows of a categorical column, by code (no string round trip)."""
    return pd.Categorical.from_codes(
        values.cat.codes.to_numpy()[rows], categories=values.cat.categories
    )


def find_bout_starts(
    video: np.ndarray, animal: np.ndarray, label: np.ndarray, frame: np.ndarray
) -> np.ndarray:
    """Row indices where a new bout begins (rows sorted by video, animal, frame)."""
    is_new = np.ones(len(frame), dtype=bool)
    is_new[1:] = (
        (video[1:] != video[:-1])
        | (animal[1:] != animal[:-1])
        | (label[1:] != label[:-1])
        | (frame[1:] != frame[:-1] + 1)
    )
    return np.flatnonzero(is_new)


def compute_bout_means(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Mean of *values* over each bout, ignoring NaN (NaN if a bout has none)."""
    finite = np.isfinite(values)
    sums = np.add.reduceat(np.where(finite, values, 0.0), starts)
    counts = np.add.reduceat(finite.astype(np.int64), starts)
    return np.divide(sums, counts, out=np.full(len(starts), np.nan), where=counts > 0)


# ┌────────────────────────────────────────────────────────────┐
# │ Summaries  « what the label picker and histogram show »    │
# └────────────────────────────────────────────────────────────┘


def summarise_labels(bouts: pd.DataFrame) -> pd.DataFrame:
    """Per label: bouts, frames, share of all frames, median and mean bout duration."""
    grouped = bouts.groupby("label", observed=False)
    summary = pd.DataFrame(
        {
            "n_bouts": grouped.size(),
            "n_frames": grouped["n_frames"].sum(),
            "median_duration_s": grouped["duration_s"].median(),
            "mean_duration_s": grouped["duration_s"].mean(),
        }
    )
    summary["share_of_frames"] = summary["n_frames"] / max(int(summary["n_frames"].sum()), 1)
    return summary


def compute_duration_histogram(
    durations_s: np.ndarray, fps: float, n_bins: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    """Counts and log-spaced bin edges from one frame to the longest bout.

    Edges sit half a frame below each whole-frame duration where bins are
    narrower than a frame, so a bin never splits one duration.
    """
    durations_s = np.asarray(durations_s, dtype=float)
    if durations_s.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(1)
    frame_s = 1.0 / fps
    low, high = 0.5 * frame_s, durations_s.max() + 0.5 * frame_s
    edges = np.geomspace(low, high, n_bins + 1)
    edges = np.unique(
        np.where(edges < 20 * frame_s, (np.round(edges / frame_s - 0.5) + 0.5) * frame_s, edges)
    )
    counts, edges = np.histogram(durations_s, bins=edges)
    return counts, edges
