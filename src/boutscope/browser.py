# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — browser                                             ║
# ║  « the viewer's state, without Qt »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Labelling, label, duration range, draws and clip requests; the  ║
# ║  GUI is a thin layer on top, so this is tested headless.         ║
# ╚══════════════════════════════════════════════════════════════════╝
"""What the viewer is looking at, without any Qt: labelling, label, range, draw.

The GUI is a thin layer over :class:`Browser`, so the browsing logic is
tested headless.  Bout tables, label summaries and strip lookups are
loaded once per labelling and cached.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from boutscope import constants as bc
from boutscope.bouts import compute_duration_histogram, summarise_labels
from boutscope.project import Project
from boutscope.sampling import (
    SampleRequest,
    Session,
    count_by_stratum,
    select_candidates,
    start_session,
)
from boutscope.strips import LabelLookup
from boutscope.video import ClipRequest, build_clip_request, resolve_video_path


@dataclass(frozen=True)
class Strip:
    labelling: str
    codes: np.ndarray  # one per clip frame, strips.MISSING where unlabelled
    lookup: LabelLookup


class Browser:
    def __init__(self, project: Project, video_root: Path | None = None) -> None:
        if not project.labellings:
            raise ValueError(f"project {project.name!r} has no labellings yet")
        self.project = project
        self.video_root = video_root
        self._bouts: dict[str, pd.DataFrame] = {}
        self._lookups: dict[str, LabelLookup] = {}
        self._label_tables: dict[str, pd.DataFrame] = {}
        self.labelling = next(iter(project.labellings))
        self.label: str | None = None
        self.request: SampleRequest | None = None
        self.session: Session | None = None
        self.drawn: list[int] = []

    # ┌────────────────────────────────────────────────────────────┐
    # │ Choosing  « labelling, label, duration range »             │
    # └────────────────────────────────────────────────────────────┘

    @property
    def bouts(self) -> pd.DataFrame:
        if self.labelling not in self._bouts:
            self._bouts[self.labelling] = self.project.load_bouts(self.labelling).set_index(
                "bout_id", drop=False
            )
        return self._bouts[self.labelling]

    def select_labelling(self, labelling: str) -> None:
        if labelling not in self.project.labellings:
            raise KeyError(labelling)
        self.labelling, self.label = labelling, None
        self._reset_draw()

    def label_table(self) -> pd.DataFrame:
        """Labels of the current labelling with name, colour and summary, most frames first."""
        if self.labelling not in self._label_tables:
            self._label_tables[self.labelling] = self._build_label_table()
        return self._label_tables[self.labelling]

    def _build_label_table(self) -> pd.DataFrame:
        meta = self.project.load_label_meta(self.labelling)
        summary = summarise_labels(self.bouts)
        summary.index = summary.index.astype(str)
        summary.insert(0, "name", [meta[k].name for k in summary.index])
        summary.insert(1, "colour", [meta[k].colour for k in summary.index])
        return summary.sort_values("share_of_frames", ascending=False, kind="stable")

    def select_label(self, label: str) -> None:
        self.label = str(label)
        self._reset_draw()

    def label_durations(self) -> np.ndarray:
        self._require_label()
        return self.bouts.loc[self.bouts["label"] == self.label, "duration_s"].to_numpy()

    def histogram(self, n_bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
        return compute_duration_histogram(self.label_durations(), self.project.fps, n_bins)

    def available(
        self,
        min_duration_s: float | None,
        max_duration_s: float | None,
        strata: tuple[str, ...] = bc.DEFAULT_STRATA,
        mode: str = "stratified",
    ) -> pd.Series:
        """Bouts in range per stratum (the "N available" readout)."""
        request = self._build_request(min_duration_s, max_duration_s, strata, mode)
        return count_by_stratum(select_candidates(self.bouts, request), request.strata)

    def stratum_columns(self) -> list[str]:
        return ["animal", "video", *self.project.strata]

    # ┌────────────────────────────────────────────────────────────┐
    # │ Drawing  « seeded sample, more, clip requests »            │
    # └────────────────────────────────────────────────────────────┘

    def draw(
        self,
        n: int,
        *,
        min_duration_s: float | None = None,
        max_duration_s: float | None = None,
        strata: tuple[str, ...] = bc.DEFAULT_STRATA,
        mode: str = "stratified",
        seed: int = bc.DEFAULT_SEED,
        max_per_animal: int | None = None,
    ) -> list[int]:
        request = self._build_request(min_duration_s, max_duration_s, strata, mode)
        self.request = replace(request, seed=seed, max_per_animal=max_per_animal)
        self.session = start_session(self.bouts, self.request)
        self.drawn = []
        return self.more(n)

    def more(self, n: int) -> list[int]:
        if self.session is None:
            raise RuntimeError("draw first")
        batch = [int(b) for b in self.session.next_batch(n)]
        self.drawn.extend(batch)
        return batch

    def bout(self, bout_id: int) -> pd.Series:
        return self.bouts.loc[bout_id]

    def clip_request(self, bout_id: int, zoom: bool = True) -> ClipRequest:
        bout = self.bout(bout_id)
        path = resolve_video_path(self.project, str(bout["video"]), self.video_root)
        return build_clip_request(self.project, bout, path, zoom=zoom)

    def strips(self, request: ClipRequest) -> list[Strip]:
        """What every labelling says over the clip's frames, current labelling first."""
        order = [self.labelling, *(n for n in self.project.labellings if n != self.labelling)]
        stop = request.first_frame + request.n_frames
        return [
            Strip(
                name,
                self.lookup(name).labels(request.video, request.animal, request.first_frame, stop),
                self.lookup(name),
            )
            for name in order
        ]

    def lookup(self, labelling: str) -> LabelLookup:
        if labelling not in self._lookups:
            self._lookups[labelling] = LabelLookup(self.project, labelling)
        return self._lookups[labelling]

    def _build_request(self, min_duration_s, max_duration_s, strata, mode) -> SampleRequest:
        self._require_label()
        return SampleRequest(
            labelling=self.labelling,
            label=self.label,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            strata=tuple(strata),
            mode=mode,
        )

    def _require_label(self) -> None:
        if self.label is None:
            raise RuntimeError("select a label first")

    def _reset_draw(self) -> None:
        self.request, self.session, self.drawn = None, None, []
