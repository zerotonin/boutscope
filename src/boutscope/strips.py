# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — strips                                              ║
# ║  « what the other labellings said »                              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Per-frame label lookup for any labelling over a stretch of one  ║
# ║  animal's track, for the timeline strips.                        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Per-frame labels of any labelling for a stretch of one animal's track.

The timeline under the video shows, for the frames of the current clip,
what every other labelling said.  Label tables are canonical (sorted by
video, animal, frame), so each track is a contiguous row range and a
lookup is two binary searches.
"""

from __future__ import annotations

import numpy as np

from boutscope import constants as bc
from boutscope.project import Project

MISSING: int = -1


class LabelLookup:
    """Label codes of one labelling by (video, animal, project frame range)."""

    def __init__(self, project: Project, labelling: str) -> None:
        table = project.load_labels(labelling, columns=["video", "animal", "frame", "label"])
        meta = project.load_label_meta(labelling)
        self.labelling = labelling
        self.names = [meta[str(c)].name for c in table["label"].cat.categories]
        self.colours = [meta[str(c)].colour for c in table["label"].cat.categories]
        self._frame = table["frame"].to_numpy()
        self._label = table["label"].cat.codes.to_numpy()
        video = table["video"].cat.codes.to_numpy()
        animal = table["animal"].cat.codes.to_numpy()
        starts = np.flatnonzero(
            np.r_[True, (video[1:] != video[:-1]) | (animal[1:] != animal[:-1])]
        )
        stops = np.r_[starts[1:], len(table)]
        videos, animals = table["video"].cat.categories, table["animal"].cat.categories
        self._tracks = {
            (str(videos[video[s]]), str(animals[animal[s]])): (int(s), int(e))
            for s, e in zip(starts, stops)
        }

    def labels(self, video: str, animal: str, first: int, stop: int) -> np.ndarray:
        """Codes for frames ``first .. stop-1``; :data:`MISSING` where the table has no row."""
        out = np.full(stop - first, MISSING, dtype=np.int32)
        track = self._tracks.get((str(video), str(animal)))
        if track is None:
            return out
        lo, hi = track
        frames = self._frame[lo:hi]
        a = lo + int(np.searchsorted(frames, first, side="left"))
        b = lo + int(np.searchsorted(frames, stop, side="left"))
        out[self._frame[a:b] - first] = self._label[a:b]
        return out

    def colour_of(self, code: int) -> str:
        return bc.MISSING_LABEL_COLOUR if code == MISSING else self.colours[code]

    def name_of(self, code: int) -> str:
        return "no label" if code == MISSING else self.names[code]
