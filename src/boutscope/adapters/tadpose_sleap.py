# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — adapters.tadpose_sleap                              ║
# ║  « slots, raw frames, crop offsets »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  TadPose SLEAP bundle: labels on the slot grid, a frame map per  ║
# ║  video, keypoints and animal boxes shifted to raw-video pixels.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""TadPose SLEAP adapter: the feature bundle's labels, frame maps, pose and boxes.

Inputs are files TadPose writes, read directly (boutscope does not import
TadPose):

* ``rows.parquet`` (and ``omitted_rows.parquet``): video_stem, well, slot,
  raw_frame.  ``slot`` is the uniform 50 fps time grid and becomes the
  project frame; slot -> raw_frame becomes each video's frame map.
* one label array per labelling, row-aligned to ``rows.parquet``, with an
  optional PM catalogue JSON (raw id -> label, colour, kind).
* the crop box of every video (``video, x, y, w, h``): SLEAP tracked a
  cropped plate, the viewer reads the raw video, so keypoints and boxes are
  shifted by the crop offset.
* optional: ``animals.csv`` (recording -> animal id), a per-video strata
  CSV (``video`` + stratum columns), and the MoSeq export npz per video
  (keypoints on the slot grid, confidence 0 wherever stage 1 filled in).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from boutscope import pose as bpose
from boutscope import project as bp

ADAPTER_NAME: str = "tadpose_sleap"
RECORDING_SEP: str = "__"
KEYPOINTS: tuple[str, ...] = (
    "left_eye",
    "right_eye",
    "centre",
    "tail1",
    "tail2",
    "tail3",
    "tail_tip",
)
EDGES: tuple[tuple[str, str], ...] = (
    ("left_eye", "right_eye"),
    ("left_eye", "centre"),
    ("right_eye", "centre"),
    ("centre", "tail1"),
    ("tail1", "tail2"),
    ("tail2", "tail3"),
    ("tail3", "tail_tip"),
)
BOX_CENTRE_NODE: str = "centre"


@dataclass(frozen=True)
class TadPoseImport:
    bundle_dir: Path
    labellings: dict[str, Path]
    crop_csv: Path
    project_root: Path
    catalogues: dict[str, Path] = field(default_factory=dict)
    animals_csv: Path | None = None
    video_strata_csv: Path | None = None
    pose_dir: Path | None = None
    video_prefix: str = "cropped_"
    video_suffix: str = ".mp4"
    fps: float = 50.0
    box_px: int = 140
    min_confidence: float = 0.2
    name: str | None = None


def import_tadpose_sleap(spec: TadPoseImport) -> bp.Project:
    """Build a project with frame maps, pose and every labelling of *spec*."""
    rows = load_rows(Path(spec.bundle_dir))
    stems = [str(v) for v in rows["video_stem"].cat.categories]
    offsets = load_crop_offsets(Path(spec.crop_csv), stems, spec.video_prefix)
    strata = load_video_strata(spec.video_strata_csv, stems)
    animal_of = load_animal_ids(spec.animals_csv)
    boxes = {}
    for stem in stems if spec.pose_dir is not None else []:  # pass 1: boxes only
        coords, conf, animals = load_export_pose(spec.pose_dir, stem, offsets[stem], animal_of)
        boxes[stem] = build_boxes(coords, conf, animals, spec.box_px)
    videos = {
        s: bp.VideoEntry(
            path=s.removeprefix(spec.video_prefix) + spec.video_suffix,
            fps=spec.fps,
            strata=strata.get(s, {}),
            boxes=boxes.get(s, {}),
        )
        for s in stems
    }
    stratum_names = tuple(sorted({k for v in strata.values() for k in v}))
    project = bp.create_project(
        Path(spec.project_root),
        spec.name or Path(spec.project_root).name,
        spec.fps,
        ADAPTER_NAME,
        videos,
        strata=stratum_names,
        skeleton=bp.Skeleton(KEYPOINTS, EDGES, spec.min_confidence),
    )
    for stem, frame_map in build_frame_maps(Path(spec.bundle_dir), rows).items():
        bp.write_frame_map(project, stem, frame_map)
    for stem in boxes:  # pass 2: one video's keypoints in memory at a time
        bpose.write_pose(
            project, stem, *load_export_pose(spec.pose_dir, stem, offsets[stem], animal_of)
        )
    animal = build_animal_column(rows, animal_of)
    for labelling, path in spec.labellings.items():
        labels = np.load(path, mmap_mode="r")
        if len(labels) != len(rows):
            raise bp.ProjectError(f"{Path(path).name}: {len(labels)} labels for {len(rows)} rows")
        table = pd.DataFrame(
            {
                "video": rows["video_stem"],
                "animal": animal,
                "frame": rows["slot"].to_numpy(),
                "label": np.asarray(labels),
            }
        )
        meta = load_catalogue(spec.catalogues[labelling]) if labelling in spec.catalogues else None
        project = bp.add_labelling(project, labelling, table, meta)
    return project


# ┌────────────────────────────────────────────────────────────┐
# │ Bundle  « rows, frame maps, animals, strata »              │
# └────────────────────────────────────────────────────────────┘


def load_rows(bundle_dir: Path) -> pd.DataFrame:
    """video_stem and well as categoricals (no per-row strings), slot as int64."""
    table = pq.read_table(
        bundle_dir / "rows.parquet",
        columns=["video_stem", "well", "slot"],
        read_dictionary=["video_stem", "well"],
    )
    return table.to_pandas()


def build_frame_maps(bundle_dir: Path, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """slot -> raw video frame per video, from usable and omitted rows (-1 = dropped)."""
    columns = ["video_stem", "slot", "raw_frame"]
    paths = [bundle_dir / "rows.parquet", bundle_dir / "omitted_rows.parquet"]
    parts = [
        pq.read_table(p, columns=columns, read_dictionary=["video_stem"]).to_pandas()
        for p in paths
        if p.is_file()
    ]
    frames = pd.concat(parts, ignore_index=True).dropna(subset=["raw_frame"])
    frames["video_stem"] = frames["video_stem"].astype(str).astype("category")
    frames = frames.drop_duplicates(["video_stem", "slot"])
    maps = {}
    for stem, group in frames.groupby("video_stem", sort=True, observed=True):
        frame_map = np.full(int(group["slot"].max()) + 1, -1, dtype=np.int64)
        frame_map[group["slot"].to_numpy()] = group["raw_frame"].to_numpy(dtype=np.int64)
        maps[str(stem)] = frame_map
    missing = set(map(str, rows["video_stem"].cat.categories)) - set(maps)
    if missing:
        raise bp.ProjectError(f"no raw frames for {sorted(missing)}")
    return maps


def build_animal_column(rows: pd.DataFrame, animal_of: dict[str, str]) -> pd.Categorical:
    """Animal id per row via (video, well) pair codes; ids from animals.csv where known."""
    video, well = rows["video_stem"].cat, rows["well"].cat
    pair = video.codes.to_numpy().astype(np.int64) * len(well.categories) + well.codes.to_numpy()
    unique, inverse = np.unique(pair, return_inverse=True)
    names = []
    for code in unique:
        recording = (
            f"{video.categories[code // len(well.categories)]}{RECORDING_SEP}"
            f"{well.categories[code % len(well.categories)]}"
        )
        names.append(animal_of.get(recording, recording))
    categories = sorted(set(names))
    index = {n: i for i, n in enumerate(categories)}
    codes = np.array([index[n] for n in names], dtype=np.int64)[inverse]
    return pd.Categorical.from_codes(codes, categories=categories)


def load_animal_ids(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str, encoding="utf-8")
    return dict(zip(table["recording"], table["animal"]))


def load_video_strata(path: Path | None, stems: list[str]) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    table = pd.read_csv(path, dtype=str, encoding="utf-8").set_index("video")
    missing = sorted(set(stems) - set(table.index))
    if missing:
        raise bp.ProjectError(f"{Path(path).name} lacks {missing[:3]}")
    return {s: {k: str(v) for k, v in table.loc[s].items()} for s in stems}


def load_crop_offsets(path: Path, stems: list[str], prefix: str) -> dict[str, tuple[int, int]]:
    """(x, y) of each video's SLEAP crop in raw-video pixels, matched by file stem."""
    table = pd.read_csv(path, encoding="utf-8")
    by_stem = {
        Path(str(v)).stem: (int(x), int(y))
        for v, x, y in zip(table["video"], table["x"], table["y"])
    }
    missing = [s for s in stems if s.removeprefix(prefix) not in by_stem]
    if missing:
        raise bp.ProjectError(f"no crop box for {missing[:3]} in {Path(path).name}")
    return {s: by_stem[s.removeprefix(prefix)] for s in stems}


def load_catalogue(path: Path) -> dict[str, bp.LabelMeta]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key: bp.LabelMeta(
            name=entry["label"], colour=entry["colour"], description=str(entry.get("kind", ""))
        )
        for key, entry in raw.items()
        if not key.startswith("_")
    }


# ┌────────────────────────────────────────────────────────────┐
# │ Pose and boxes  « export npz -> raw-video pixels »         │
# └────────────────────────────────────────────────────────────┘


def load_export_pose(
    pose_dir: Path, stem: str, offset: tuple[int, int], animal_of: dict[str, str]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(animals, slots, keypoints, 2) in raw-video pixels, confidences, animal ids."""
    path = Path(pose_dir) / f"{stem}.npz"
    with np.load(path) as npz:
        coords = npz["coords"].astype(np.float32) + np.asarray(offset, dtype=np.float32)
        conf = npz["conf"].astype(np.float32)
        wells = [str(w) for w in npz["wells"]]
    if coords.shape[2] != len(KEYPOINTS):
        raise bp.ProjectError(
            f"{path.name}: {coords.shape[2]} keypoints, expected {len(KEYPOINTS)}"
        )
    animals = [
        animal_of.get(f"{stem}{RECORDING_SEP}{w}", f"{stem}{RECORDING_SEP}{w}") for w in wells
    ]
    return coords, conf, animals


def build_boxes(
    coords: np.ndarray, conf: np.ndarray, animals: list[str], box_px: int
) -> dict[str, bp.AnimalBox]:
    """A square box per animal around the median observed gut position."""
    node = KEYPOINTS.index(BOX_CENTRE_NODE)
    boxes = {}
    for i, animal in enumerate(animals):
        observed = conf[i, :, node] > 0
        if not observed.any():
            continue
        cx, cy = np.median(coords[i, observed, node], axis=0)
        boxes[animal] = bp.AnimalBox(
            x=int(round(cx - box_px / 2)), y=int(round(cy - box_px / 2)), w=box_px, h=box_px
        )
    return boxes
