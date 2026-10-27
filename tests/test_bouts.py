"""Bout index: runs, gaps, track boundaries, durations, summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from boutscope.bouts import (
    build_bout_table,
    compute_bout_means,
    compute_duration_histogram,
    summarise_labels,
)
from boutscope.project import load_project

FPS = 50.0


def _canonical(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for column in ("video", "animal", "label"):
        out[column] = pd.Categorical(
            out[column].astype(str), categories=sorted(out[column].astype(str).unique())
        )
    return out.sort_values(["video", "animal", "frame"], kind="stable").reset_index(drop=True)


def test_runs_break_at_label_gap_and_animal():
    table = _canonical(
        pd.DataFrame(
            {
                "video": "v",
                "animal": ["a"] * 7 + ["b"] * 2,
                "frame": [0, 1, 2, 4, 5, 6, 7, 8, 9],
                "label": ["x", "x", "x", "x", "y", "y", "x", "x", "x"],
            }
        )
    )
    bouts = build_bout_table(table, FPS)
    got = list(zip(bouts.animal.astype(str), bouts.label.astype(str), bouts.start, bouts.stop))
    assert got == [
        ("a", "x", 0, 3),
        ("a", "x", 4, 5),
        ("a", "y", 5, 7),
        ("a", "x", 7, 8),
        ("b", "x", 8, 10),
    ]
    assert bouts.n_frames.tolist() == [3, 1, 2, 1, 2]
    np.testing.assert_allclose(bouts.duration_s, bouts.n_frames / FPS)


def test_matches_synthetic_truth(synthetic):
    table, truth = synthetic
    bouts = build_bout_table(_canonical(table), FPS)
    got = bouts[["video", "animal", "label", "start", "stop"]].astype(
        {"video": str, "animal": str, "label": str}
    )
    want = truth.sort_values(["video", "animal", "start"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want, check_dtype=False)


def test_video_frames_from_offset_or_column():
    table = _canonical(
        pd.DataFrame({"video": "v", "animal": "a", "frame": [0, 1, 2], "label": "x"})
    )
    bouts = build_bout_table(table, FPS, frame_offsets={"v": 10})
    assert (bouts.video_frame_start.item(), bouts.video_frame_stop.item()) == (10, 13)
    table["video_frame"] = [100, 101, 103]
    bouts = build_bout_table(table, FPS)
    assert (bouts.video_frame_start.item(), bouts.video_frame_stop.item()) == (100, 104)


def test_bout_mean_ignores_nan():
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, 4.0])
    np.testing.assert_allclose(compute_bout_means(values, np.array([0, 3, 5])), [2.0, np.nan, 4.0])


def test_ids_do_not_depend_on_input_row_order(project, synthetic):
    table, _ = synthetic
    shuffled = table.sample(frac=1.0, random_state=1)
    from boutscope import project as bp

    other = bp.create_project(
        project.root.parent / "other", "o", FPS, "generic", project.videos, strata=project.strata
    )
    other = bp.add_labelling(other, "behaviour", shuffled)
    pd.testing.assert_frame_equal(project.load_bouts("behaviour"), other.load_bouts("behaviour"))
    assert (
        project.labellings["behaviour"].content_hash == other.labellings["behaviour"].content_hash
    )


def test_summary_and_histogram(project):
    bouts = load_project(project.root).load_bouts("behaviour")
    summary = summarise_labels(bouts)
    assert summary.share_of_frames.sum() == np.float64(1.0)
    assert summary.n_frames.sum() == project.labellings["behaviour"].n_frames
    counts, edges = compute_duration_histogram(bouts.duration_s.to_numpy(), FPS)
    assert counts.sum() == len(bouts)
    assert np.all(np.diff(edges) > 0)
    one_frame = (bouts.n_frames == 1).sum()
    assert counts[0] == one_frame  # the first bin holds exactly the one-frame bouts
