# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — adapters.keypoint_moseq                             ║
# ║  « syllables onto the project grid »                             ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  kpms results.h5 plus a recording map: one labelling added to    ║
# ║  an existing project.                                            ║
# ╚══════════════════════════════════════════════════════════════════╝
"""keypoint-MoSeq adapter: syllables from ``results.h5`` as a labelling of a project.

kpms names recordings freely and fits windows that need not start at a
video's first frame, so a recording map CSV places each recording:
``recording, video, animal, first_frame`` (the project frame of the
recording's first syllable).  The project must exist (videos, frame maps,
pose come from another adapter or a generic import); kpms adds one
labelling.  Syllables keep their kpms ids (0 = most used after
reindexing); every frame of a recording is labelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from boutscope import constants as bc
from boutscope import project as bp

ADAPTER_NAME: str = "keypoint_moseq"
MAP_COLUMNS: tuple[str, ...] = ("recording", "video", "animal", "first_frame")


@dataclass(frozen=True)
class KpmsImport:
    results_h5: Path
    recording_map: Path
    project_root: Path
    labelling: str = "moseq"
    overwrite: bool = False


def import_keypoint_moseq(spec: KpmsImport) -> bp.Project:
    project = bp.load_project(spec.project_root)
    places = load_recording_map(spec.recording_map)
    syllables = load_syllables(Path(spec.results_h5), list(places.index))
    parts = [
        pd.DataFrame(
            {
                "video": places.at[rec, "video"],
                "animal": places.at[rec, "animal"],
                "frame": int(places.at[rec, "first_frame"]) + np.arange(len(labels)),
                "label": labels,
            }
        )
        for rec, labels in syllables.items()
    ]
    table = pd.concat(parts, ignore_index=True)
    labels = sorted(table["label"].unique())
    cycle = bc.LABEL_COLOUR_CYCLE
    meta = {
        str(s): bp.LabelMeta(name=f"syllable {s}", colour=cycle[i % len(cycle)])
        for i, s in enumerate(labels)
    }
    return bp.add_labelling(project, spec.labelling, table, meta, overwrite=spec.overwrite)


def load_recording_map(path: Path) -> pd.DataFrame:
    table = pd.read_csv(
        path, dtype={"recording": str, "video": str, "animal": str}, encoding="utf-8"
    )
    missing = [c for c in MAP_COLUMNS if c not in table]
    if missing:
        raise bp.ProjectError(f"{Path(path).name} lacks column(s) {missing}")
    if table["recording"].duplicated().any():
        raise bp.ProjectError(f"{Path(path).name}: a recording is listed twice")
    return table.set_index("recording")


def load_syllables(path: Path, recordings: list[str]) -> dict[str, np.ndarray]:
    """``syllable`` per recording from a kpms results file (needs h5py)."""
    try:
        import h5py
    except ImportError as error:
        raise bp.ProjectError("reading results.h5 needs h5py: pip install h5py") from error
    with h5py.File(path, "r") as results:
        missing = [r for r in recordings if r not in results]
        if missing:
            raise bp.ProjectError(f"{path.name} has no recording(s) {missing[:3]}")
        return {r: np.asarray(results[r]["syllable"], dtype=np.int64) for r in recordings}
