# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — adapters.generic                                    ║
# ║  « any categoriser, one row per frame »                          ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  CSV or Parquet label tables with configurable column names,     ║
# ║  optional videos.toml and label meta CSV.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Generic adapter: one table, one row per labelled frame, from any categoriser.

Input: CSV or Parquet with a video, an animal, a frame and a label column
(names configurable), plus optional ``video_frame``, ``typicality`` and
stratum columns.  Optional extras: a ``videos.toml`` with paths, fps,
offsets and boxes, and a label meta CSV (``label, name, colour[, description]``).
Without a ``videos.toml`` every video gets a placeholder entry (path = the
video name, the project fps, offset 0) to be edited before viewing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from boutscope import constants as bc
from boutscope import project as bp

ADAPTER_NAME: str = "generic"


@dataclass(frozen=True)
class ColumnMap:
    """Names of the input columns that hold the four required fields."""

    video: str = "video"
    animal: str = "animal"
    frame: str = "frame"
    label: str = "label"

    def rename(self) -> dict[str, str]:
        return {
            getattr(self, k): k
            for k in ("video", "animal", "frame", "label")
            if getattr(self, k) != k
        }


@dataclass(frozen=True)
class GenericImport:
    """One import of a label table into a (new or existing) project."""

    table_path: Path
    project_root: Path
    labelling: str
    fps: float | None = None
    columns: ColumnMap = field(default_factory=ColumnMap)
    videos_toml: Path | None = None
    meta_csv: Path | None = None
    strata: tuple[str, ...] = ()
    project_name: str | None = None
    overwrite: bool = False


def import_generic(spec: GenericImport) -> bp.Project:
    """Add the table as a labelling; create the project first if there is none."""
    table = load_table(spec.table_path).rename(columns=spec.columns.rename())
    if (Path(spec.project_root) / bc.PROJECT_FILE).exists():
        project = bp.load_project(spec.project_root)
        if spec.strata and tuple(spec.strata) != project.strata:
            raise bp.ProjectError(
                f"project strata are {project.strata}; a later import cannot change them"
            )
    else:
        if spec.fps is None:
            raise bp.ProjectError("a new project needs --fps")
        videos = (
            bp.load_videos(spec.videos_toml)
            if spec.videos_toml is not None
            else build_placeholder_videos(table, spec.fps)
        )
        project = bp.create_project(
            spec.project_root,
            spec.project_name or Path(spec.project_root).name,
            spec.fps,
            ADAPTER_NAME,
            videos,
            strata=tuple(spec.strata),
        )
    meta = load_meta_csv(spec.meta_csv) if spec.meta_csv is not None else None
    return bp.add_labelling(project, spec.labelling, table, meta, overwrite=spec.overwrite)


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in (".csv", ".tsv", ".txt"):
        return pd.read_csv(
            path, sep="\t" if path.suffix.lower() == ".tsv" else ",", encoding="utf-8"
        )
    raise bp.ProjectError(f"{path.name}: expected .csv, .tsv or .parquet")


def build_placeholder_videos(table: pd.DataFrame, fps: float) -> dict[str, bp.VideoEntry]:
    if "video" not in table:
        raise bp.ProjectError("label table lacks a video column (see --video-col)")
    return {
        str(v): bp.VideoEntry(path=str(v), fps=float(fps))
        for v in sorted(table["video"].astype(str).unique())
    }


def load_meta_csv(path: Path) -> dict[str, bp.LabelMeta]:
    meta = pd.read_csv(path, dtype=str, encoding="utf-8").fillna("")
    missing = [c for c in ("label", "name", "colour") if c not in meta]
    if missing:
        raise bp.ProjectError(f"{Path(path).name} lacks column(s) {missing}")
    return {
        row.label: bp.LabelMeta(
            name=row.name, colour=row.colour, description=getattr(row, "description", "")
        )
        for row in meta.itertuples(index=False)
    }
