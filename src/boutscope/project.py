# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — project                                             ║
# ║  « the contract between adapters and the GUI »                   ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Create, validate, load and save a project folder: label tables, ║
# ║  label meta, videos, skeleton and the bout index.                ║
# ╚══════════════════════════════════════════════════════════════════╝
"""The project folder: the one format every adapter writes and the GUI reads.

Layout (see ``docs/project_format.rst``)::

    project/
    ├── project.toml              name, fps, adapter, labellings, strata, skeleton
    ├── videos.toml               per video: path, fps, frame offset, boxes per animal
    ├── labels/<labelling>.parquet       one row per labelled frame
    ├── labels/<labelling>.meta.toml     names and colours per label
    ├── bouts/<labelling>.parquet        bout index, written at import
    └── pose/<video>.npz          optional keypoints
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tomli_w

from boutscope import constants as bc
from boutscope.bouts import build_bout_table

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


class ProjectError(ValueError):
    """The project folder or a label table breaks the format."""


# ┌────────────────────────────────────────────────────────────┐
# │ Data classes  « what project.toml and videos.toml hold »   │
# └────────────────────────────────────────────────────────────┘


@dataclass(frozen=True)
class Skeleton:
    keypoints: tuple[str, ...]
    edges: tuple[tuple[str, str], ...] = ()
    min_confidence: float = 0.0


@dataclass(frozen=True)
class AnimalBox:
    """Region of the video frame that shows one animal, in video pixels."""

    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class VideoEntry:
    """One video file.

    The video frame of a project frame comes from, in this order: the label
    table's ``video_frame`` column, the video's frame map
    (``framemap/<video>.npy``), or ``frame + frame_offset``.  ``strata`` are
    per-video stratum values that fill a label table lacking those columns.
    """

    path: str  # relative to the video root
    fps: float
    frame_offset: int = 0
    width: int | None = None
    height: int | None = None
    boxes: dict[str, AnimalBox] = field(default_factory=dict)
    strata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LabelMeta:
    name: str
    colour: str
    description: str = ""


@dataclass(frozen=True)
class LabellingInfo:
    n_frames: int
    n_bouts: int
    n_labels: int
    content_hash: str
    has_typicality: bool
    has_video_frame: bool


@dataclass(frozen=True)
class Project:
    root: Path
    name: str
    fps: float
    adapter: str
    created: str
    labellings: dict[str, LabellingInfo] = field(default_factory=dict)
    strata: tuple[str, ...] = ()
    context_s: float = bc.DEFAULT_CONTEXT_S
    skeleton: Skeleton | None = None
    videos: dict[str, VideoEntry] = field(default_factory=dict)

    def label_path(self, labelling: str) -> Path:
        return self.root / bc.LABELS_DIR / f"{labelling}{bc.LABEL_TABLE_SUFFIX}"

    def meta_path(self, labelling: str) -> Path:
        return self.root / bc.LABELS_DIR / f"{labelling}{bc.LABEL_META_SUFFIX}"

    def bouts_path(self, labelling: str) -> Path:
        return self.root / bc.BOUTS_DIR / f"{labelling}{bc.LABEL_TABLE_SUFFIX}"

    def load_labels(self, labelling: str, columns: list[str] | None = None) -> pd.DataFrame:
        self._require(labelling)
        return pd.read_parquet(self.label_path(labelling), columns=columns)

    def load_bouts(self, labelling: str) -> pd.DataFrame:
        self._require(labelling)
        return pd.read_parquet(self.bouts_path(labelling))

    def frame_map_path(self, video: str) -> Path:
        return self.root / bc.FRAMEMAP_DIR / f"{video}.npy"

    def load_frame_map(self, video: str) -> np.ndarray | None:
        """Video frame per project frame (-1 = none), memory-mapped; None without a map."""
        path = self.frame_map_path(video)
        return np.load(path, mmap_mode="r") if path.is_file() else None

    def load_label_meta(self, labelling: str) -> dict[str, LabelMeta]:
        self._require(labelling)
        raw = tomllib.loads(self.meta_path(labelling).read_text(encoding="utf-8"))
        return {label: LabelMeta(**entry) for label, entry in raw.get("labels", {}).items()}

    def _require(self, labelling: str) -> None:
        if labelling not in self.labellings:
            raise ProjectError(
                f"no labelling {labelling!r} in {self.root}; known: {sorted(self.labellings)}"
            )


# ┌────────────────────────────────────────────────────────────┐
# │ Create, add, load  « the public entry points »             │
# └────────────────────────────────────────────────────────────┘


def create_project(
    root: Path,
    name: str,
    fps: float,
    adapter: str,
    videos: dict[str, VideoEntry],
    *,
    strata: tuple[str, ...] = (),
    context_s: float = bc.DEFAULT_CONTEXT_S,
    skeleton: Skeleton | None = None,
) -> Project:
    """Write an empty project (no labellings yet); refuse to overwrite one."""
    root = Path(root)
    if (root / bc.PROJECT_FILE).exists():
        raise ProjectError(f"{root} already holds a project")
    if fps <= 0:
        raise ProjectError(f"fps must be positive, not {fps}")
    project = Project(
        root=root,
        name=name,
        fps=float(fps),
        adapter=adapter,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        strata=tuple(strata),
        context_s=context_s,
        skeleton=skeleton,
        videos=dict(videos),
    )
    for sub in (bc.LABELS_DIR, bc.BOUTS_DIR):
        (root / sub).mkdir(parents=True, exist_ok=True)
    save_project(project)
    return project


def add_labelling(
    project: Project,
    labelling: str,
    table: pd.DataFrame,
    meta: dict[str, LabelMeta] | None = None,
    *,
    overwrite: bool = False,
) -> Project:
    """Validate *table*, store it with its meta, index its bouts; return the updated project."""
    if labelling in project.labellings and not overwrite:
        raise ProjectError(f"labelling {labelling!r} exists; pass overwrite=True to replace it")
    if not labelling.replace("_", "").replace("-", "").isalnum():
        raise ProjectError(f"labelling name {labelling!r}: use letters, digits, '_' and '-'")
    table = normalise_label_table(table, project)
    videos = [str(v) for v in table["video"].cat.categories]
    bouts = build_bout_table(
        table,
        project.fps,
        strata=project.strata,
        frame_offsets={v: e.frame_offset for v, e in project.videos.items()},
        frame_maps={v: m for v in videos if (m := project.load_frame_map(v)) is not None},
    )
    meta = complete_label_meta(meta or {}, table["label"].cat.categories)
    table.to_parquet(project.label_path(labelling), index=False)
    bouts.to_parquet(project.bouts_path(labelling), index=False)
    project.meta_path(labelling).write_text(
        tomli_w.dumps({"labels": {k: _drop_empty(vars(m)) for k, m in meta.items()}}),
        encoding="utf-8",
    )
    info = LabellingInfo(
        n_frames=len(table),
        n_bouts=len(bouts),
        n_labels=len(table["label"].cat.categories),
        content_hash=compute_content_hash(table),
        has_typicality="typicality" in table,
        has_video_frame="video_frame" in table,
    )
    project = replace(project, labellings={**project.labellings, labelling: info})
    save_project(project)
    return project


def load_project(root: Path) -> Project:
    root = Path(root)
    path = root / bc.PROJECT_FILE
    if not path.is_file():
        raise ProjectError(f"no {bc.PROJECT_FILE} in {root}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if raw.get("format_version") != bc.FORMAT_VERSION:
        raise ProjectError(
            f"{path}: format_version {raw.get('format_version')}, "
            f"this boutscope reads {bc.FORMAT_VERSION}"
        )
    skeleton = raw.get("skeleton")
    return Project(
        root=root,
        name=raw["name"],
        fps=float(raw["fps"]),
        adapter=raw["adapter"],
        created=raw["created"],
        labellings={k: LabellingInfo(**v) for k, v in raw.get("labellings", {}).items()},
        strata=tuple(raw.get("strata", ())),
        context_s=float(raw.get("context_s", bc.DEFAULT_CONTEXT_S)),
        skeleton=None
        if skeleton is None
        else Skeleton(
            keypoints=tuple(skeleton["keypoints"]),
            edges=tuple(tuple(e) for e in skeleton.get("edges", ())),
            min_confidence=float(skeleton.get("min_confidence", 0.0)),
        ),
        videos=load_videos(root / bc.VIDEOS_FILE),
    )


def save_project(project: Project) -> None:
    raw = {
        "format_version": bc.FORMAT_VERSION,
        "name": project.name,
        "fps": project.fps,
        "adapter": project.adapter,
        "created": project.created,
        "strata": list(project.strata),
        "context_s": project.context_s,
        "labellings": {k: vars(v) for k, v in project.labellings.items()},
    }
    if project.skeleton is not None:
        raw["skeleton"] = {
            "keypoints": list(project.skeleton.keypoints),
            "edges": [list(e) for e in project.skeleton.edges],
            "min_confidence": project.skeleton.min_confidence,
        }
    project.root.mkdir(parents=True, exist_ok=True)
    (project.root / bc.PROJECT_FILE).write_text(tomli_w.dumps(raw), encoding="utf-8")
    save_videos(project.videos, project.root / bc.VIDEOS_FILE)


def load_videos(path: Path) -> dict[str, VideoEntry]:
    if not path.is_file():
        raise ProjectError(f"missing {path}")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    videos = {}
    for name, entry in raw.get("videos", {}).items():
        boxes = {a: AnimalBox(**b) for a, b in entry.pop("boxes", {}).items()}
        strata = {k: str(v) for k, v in entry.pop("strata", {}).items()}
        videos[name] = VideoEntry(**entry, boxes=boxes, strata=strata)
    return videos


def save_videos(videos: dict[str, VideoEntry], path: Path) -> None:
    raw = {"videos": {}}
    for name, entry in videos.items():
        item = _drop_empty({k: v for k, v in vars(entry).items() if k not in ("boxes", "strata")})
        if entry.strata:
            item["strata"] = dict(entry.strata)
        if entry.boxes:
            item["boxes"] = {a: vars(b) for a, b in entry.boxes.items()}
        raw["videos"][name] = item
    path.write_text(tomli_w.dumps(raw), encoding="utf-8")


# ┌────────────────────────────────────────────────────────────┐
# │ Validation  « one place that knows the rules »             │
# └────────────────────────────────────────────────────────────┘


def normalise_label_table(table: pd.DataFrame, project: Project) -> pd.DataFrame:
    """Check a label table against the format and return it in canonical dtypes.

    Video, animal and label become sorted categoricals (labels as strings),
    frame becomes int64; rows are sorted by video, animal, frame.
    """
    missing = [c for c in bc.REQUIRED_LABEL_COLUMNS if c not in table]
    if missing:
        raise ProjectError(f"label table lacks column(s) {missing}")
    if table[list(bc.REQUIRED_LABEL_COLUMNS)].isna().any().any():
        raise ProjectError("video, animal, frame and label must not be empty")
    frame = table["frame"].to_numpy()
    if not np.issubdtype(frame.dtype, np.integer) and not np.all(np.mod(frame, 1) == 0):
        raise ProjectError("frame must hold whole numbers")
    video = _sorted_categorical(table["video"])
    unknown = sorted(set(video.categories) - set(project.videos))
    if unknown:
        raise ProjectError(
            f"videos not in {bc.VIDEOS_FILE}: {unknown[:5]}"
            + (f" and {len(unknown) - 5} more" if len(unknown) > 5 else "")
        )
    per_video = {
        c: _strata_from_videos(c, video, project) for c in project.strata if c not in table
    }
    animal = _sorted_categorical(table["animal"])
    frame = frame.astype(np.int64)
    order = np.lexsort((frame, animal.codes, video.codes))
    columns = {
        "video": video[order],
        "animal": animal[order],
        "frame": frame[order],
        "label": _sorted_categorical(table["label"])[order],
    }
    if "video_frame" in table:
        columns["video_frame"] = table["video_frame"].to_numpy(dtype=np.int64)[order]
    if "typicality" in table:
        columns["typicality"] = table["typicality"].to_numpy(dtype=np.float64)[order]
    for column in project.strata:
        if column not in columns:
            values = per_video[column] if column in per_video else table[column]
            columns[column] = _sorted_categorical(values)[order]
    out = pd.DataFrame(columns)
    _raise_on_duplicates(out)
    return out


def _raise_on_duplicates(out: pd.DataFrame) -> None:
    """Sorted rows: a duplicate (video, animal, frame) sits right after its twin."""
    v, a, f = (
        out["video"].cat.codes.to_numpy(),
        out["animal"].cat.codes.to_numpy(),
        out["frame"].to_numpy(),
    )
    duplicated = (v[1:] == v[:-1]) & (a[1:] == a[:-1]) & (f[1:] == f[:-1])
    if duplicated.any():
        first = out.iloc[int(np.flatnonzero(duplicated)[0]) + 1]
        raise ProjectError(
            f"{int(duplicated.sum())} duplicate (video, animal, frame) rows, "
            f"e.g. {first.video}, {first.animal}, frame {first.frame}"
        )


def _strata_from_videos(column: str, video: pd.Categorical, project: Project) -> pd.Series:
    """A stratum column filled from the per-video values in videos.toml."""
    lacking = [v for v in video.categories if column not in project.videos[v].strata]
    if lacking:
        raise ProjectError(
            f"label table lacks stratum column {column!r} and videos.toml gives no value "
            f"for {lacking[:3]}"
        )
    values = np.array([project.videos[v].strata[column] for v in video.categories], dtype=object)
    return pd.Series(pd.Categorical(values[video.codes]))


def write_frame_map(project: Project, video: str, video_frames: np.ndarray) -> Path:
    """Store the video frame of every project frame 0..N-1 (-1 where the video has none)."""
    if video not in project.videos:
        raise ProjectError(f"video {video!r} not in {bc.VIDEOS_FILE}")
    frames = np.asarray(video_frames, dtype=np.int64)
    if frames.ndim != 1 or (frames < -1).any():
        raise ProjectError("a frame map is one video frame (>= 0) or -1 per project frame")
    valid = frames[frames >= 0]
    if np.any(np.diff(valid) <= 0):
        raise ProjectError(f"frame map of {video!r} must increase strictly")
    path = project.frame_map_path(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, frames)
    return path


def complete_label_meta(meta: dict[str, LabelMeta], labels: pd.Index) -> dict[str, LabelMeta]:
    """Meta for every label: given entries kept, missing ones get the label as name and a colour."""
    unknown = sorted(set(meta) - set(labels))
    if unknown:
        raise ProjectError(f"meta for labels not in the table: {unknown[:5]}")
    cycle = bc.LABEL_COLOUR_CYCLE
    return {
        str(label): meta.get(str(label), LabelMeta(name=str(label), colour=cycle[i % len(cycle)]))
        for i, label in enumerate(labels)
    }


def compute_content_hash(table: pd.DataFrame) -> str:
    """SHA-256 over the canonical label table, stable across file formats and versions."""
    digest = hashlib.sha256()
    for column in table.columns:
        values = table[column]
        if isinstance(values.dtype, pd.CategoricalDtype):
            digest.update("\x1f".join(map(str, values.cat.categories)).encode("utf-8"))
            values = values.cat.codes
        digest.update(column.encode("utf-8"))
        digest.update(np.ascontiguousarray(values.to_numpy()).tobytes())
    return digest.hexdigest()


def _sorted_categorical(values: pd.Series) -> pd.Categorical:
    """String categorical; categories in numeric order when every value is an integer.

    Factorises first, so only the distinct values become Python strings.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.remove_unused_categories()
        codes, uniques = values.cat.codes.to_numpy(), values.cat.categories
    else:
        codes, uniques = pd.factorize(values)
    names = pd.Index(uniques).astype(str)
    if names.duplicated().any():
        clash = names[names.duplicated()][0]
        raise ProjectError(f"values that differ only in type, e.g. {clash!r}")
    try:
        order = np.argsort([int(n) for n in names], kind="stable")
    except ValueError:
        order = np.argsort(names.to_numpy(), kind="stable")
    new_code = np.empty(len(order), dtype=np.int64)
    new_code[order] = np.arange(len(order))
    return pd.Categorical.from_codes(new_code[codes], categories=names[order])


def _drop_empty(mapping: dict) -> dict:
    return {k: v for k, v in mapping.items() if v is not None and v != ""}
