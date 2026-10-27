"""Shared fixtures: a synthetic project with known bouts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from boutscope import project as bp
from boutscope.synthetic import SyntheticSpec, make_label_table

FPS = 50.0


@pytest.fixture
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_label_table(SyntheticSpec(n_videos=2, n_animals=3, n_frames=800, seed=3))


@pytest.fixture
def project(tmp_path: Path, synthetic) -> bp.Project:
    table, _ = synthetic
    videos = {
        v: bp.VideoEntry(path=f"{v}.avi", fps=FPS, frame_offset=10) for v in table["video"].unique()
    }
    project = bp.create_project(
        tmp_path / "proj", "synthetic", FPS, "generic", videos, strata=("condition",)
    )
    return bp.add_labelling(project, "behaviour", table)
