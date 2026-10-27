"""Generic adapter and the CLI on top of it."""

from __future__ import annotations

import pandas as pd
import pytest

from boutscope import project as bp
from boutscope.adapters.generic import ColumnMap, GenericImport, import_generic
from boutscope.cli import main


@pytest.fixture
def table_csv(tmp_path, synthetic):
    table, _ = synthetic
    path = tmp_path / "labels.csv"
    table.rename(columns={"video": "movie", "label": "cluster"}).to_csv(path, index=False)
    return path


def test_import_new_project_with_column_map(tmp_path, table_csv, synthetic):
    project = import_generic(
        GenericImport(
            table_path=table_csv,
            project_root=tmp_path / "p",
            labelling="km",
            fps=50.0,
            columns=ColumnMap(video="movie", label="cluster"),
            strata=("condition",),
        )
    )
    assert project.labellings["km"].n_frames == len(synthetic[0])
    assert project.videos["video0"].path == "video0"  # placeholder entry
    assert project.strata == ("condition",)


def test_second_labelling_parquet_with_meta_and_videos(tmp_path, synthetic):
    table, _ = synthetic
    videos = {
        v: bp.VideoEntry(path=f"clips/{v}.mp4", fps=50.0, frame_offset=5)
        for v in table.video.unique()
    }
    bp.save_videos(videos, tmp_path / "videos.toml")
    meta = tmp_path / "meta.csv"
    pd.DataFrame({"label": ["rest"], "name": ["resting"], "colour": ["#000000"]}).to_csv(
        meta, index=False
    )
    table.to_parquet(tmp_path / "a.parquet")
    table.assign(label="all").to_parquet(tmp_path / "b.parquet")
    root = tmp_path / "p"
    import_generic(
        GenericImport(
            tmp_path / "a.parquet",
            root,
            "a",
            fps=50.0,
            videos_toml=tmp_path / "videos.toml",
            meta_csv=meta,
        )
    )
    project = import_generic(GenericImport(tmp_path / "b.parquet", root, "b"))
    assert set(project.labellings) == {"a", "b"}
    assert project.videos["video0"].frame_offset == 5
    assert project.load_label_meta("a")["rest"].name == "resting"
    with pytest.raises(bp.ProjectError, match="strata"):
        import_generic(GenericImport(tmp_path / "b.parquet", root, "c", strata=("condition",)))


def test_new_project_needs_fps(tmp_path, table_csv):
    with pytest.raises(bp.ProjectError, match="fps"):
        import_generic(
            GenericImport(
                table_csv, tmp_path / "p", "km", columns=ColumnMap(video="movie", label="cluster")
            )
        )


def test_cli_import_info_sample(tmp_path, table_csv, capsys):
    root = tmp_path / "p"
    assert (
        main(
            [
                "import",
                "generic",
                str(table_csv),
                str(root),
                "--labelling",
                "km",
                "--fps",
                "50",
                "--video-col",
                "movie",
                "--label-col",
                "cluster",
                "--strata",
                "condition",
            ]
        )
        == 0
    )
    assert main(["info", str(root), "--labelling", "km"]) == 0
    assert (
        main(
            [
                "sample",
                str(root),
                "--labelling",
                "km",
                "--label",
                "swim",
                "-n",
                "3",
                "--strata",
                "condition",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "km:" in out and "bouts available" in out
    assert main(["info", str(tmp_path / "missing")]) == 1
