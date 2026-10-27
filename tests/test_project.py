"""Project format: round trip and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from boutscope import project as bp

FPS = 50.0


def _videos(*names: str) -> dict[str, bp.VideoEntry]:
    return {n: bp.VideoEntry(path=f"{n}.mp4", fps=FPS) for n in names}


def test_round_trip(tmp_path):
    skeleton = bp.Skeleton(
        keypoints=("head", "tail"), edges=(("head", "tail"),), min_confidence=0.3
    )
    videos = {
        "v": bp.VideoEntry(
            path="sub/v.mp4",
            fps=FPS,
            frame_offset=10,
            width=640,
            height=480,
            boxes={"a": bp.AnimalBox(1, 2, 30, 40)},
        )
    }
    created = bp.create_project(
        tmp_path, "p", FPS, "generic", videos, strata=("condition",), skeleton=skeleton
    )
    table = pd.DataFrame(
        {
            "video": "v",
            "animal": "a",
            "frame": range(5),
            "label": [2, 2, 10, 10, 2],
            "condition": "c",
        }
    )
    meta = {"2": bp.LabelMeta(name="rest", colour="#000000")}
    created = bp.add_labelling(created, "pm", table, meta)
    loaded = bp.load_project(tmp_path)
    assert loaded == created
    assert loaded.load_label_meta("pm")["2"].name == "rest"
    assert loaded.load_label_meta("pm")["10"].name == "10"
    assert list(loaded.load_labels("pm")["label"].cat.categories) == ["2", "10"]  # numeric order


def test_refuses_existing_project_and_labelling(tmp_path):
    project = bp.create_project(tmp_path, "p", FPS, "generic", _videos("v"))
    with pytest.raises(bp.ProjectError, match="already holds"):
        bp.create_project(tmp_path, "p", FPS, "generic", _videos("v"))
    table = pd.DataFrame({"video": "v", "animal": "a", "frame": [0], "label": "x"})
    project = bp.add_labelling(project, "pm", table)
    with pytest.raises(bp.ProjectError, match="exists"):
        bp.add_labelling(project, "pm", table)
    assert bp.add_labelling(project, "pm", table, overwrite=True).labellings["pm"].n_bouts == 1


@pytest.mark.parametrize(
    "table, message",
    [
        (pd.DataFrame({"video": ["v"], "animal": ["a"], "frame": [0]}), "lacks column"),
        (
            pd.DataFrame({"video": ["w"], "animal": ["a"], "frame": [0], "label": ["x"]}),
            "not in videos.toml",
        ),
        (
            pd.DataFrame({"video": ["v"], "animal": ["a"], "frame": [0.5], "label": ["x"]}),
            "whole numbers",
        ),
        (
            pd.DataFrame(
                {"video": ["v", "v"], "animal": ["a", "a"], "frame": [3, 3], "label": ["x", "y"]}
            ),
            "duplicate",
        ),
        (
            pd.DataFrame({"video": ["v"], "animal": [None], "frame": [0], "label": ["x"]}),
            "must not be empty",
        ),
    ],
)
def test_validation_errors(tmp_path, table, message):
    project = bp.create_project(tmp_path, "p", FPS, "generic", _videos("v"))
    with pytest.raises(bp.ProjectError, match=message):
        bp.add_labelling(project, "pm", table)


def test_missing_stratum_and_bad_meta(tmp_path):
    project = bp.create_project(tmp_path, "p", FPS, "generic", _videos("v"), strata=("condition",))
    table = pd.DataFrame({"video": "v", "animal": "a", "frame": [0], "label": "x"})
    with pytest.raises(bp.ProjectError, match="stratum"):
        bp.add_labelling(project, "pm", table)
    table["condition"] = "c"
    with pytest.raises(bp.ProjectError, match="not in the table"):
        bp.add_labelling(project, "pm", table, {"zzz": bp.LabelMeta("z", "#000000")})


def test_values_differing_only_in_type(tmp_path):
    project = bp.create_project(tmp_path, "p", FPS, "generic", _videos("v"))
    table = pd.DataFrame({"video": "v", "animal": "a", "frame": [0, 1], "label": [1, "1"]})
    with pytest.raises(bp.ProjectError, match="differ only in type"):
        bp.add_labelling(project, "pm", table)


def test_load_errors(tmp_path):
    with pytest.raises(bp.ProjectError, match="no project.toml"):
        bp.load_project(tmp_path)
    project = bp.create_project(tmp_path, "p", FPS, "generic", _videos("v"))
    with pytest.raises(bp.ProjectError, match="no labelling"):
        project.load_bouts("nope")
