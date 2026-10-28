# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — sampling                                            ║
# ║  « every stratum gets its turn »                                 ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Seeded, stratified, resumable draw orders of bouts, with        ║
# ║  typical and atypical modes from an adapter's typicality score.  ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Seeded, stratified draws of bouts.

The whole draw order is built at once and then read in batches, so "20
more" continues the same order and never repeats a bout.  Strata take
turns (round-robin), each stratum's bouts in a seeded random order, so no
animal or video dominates.  Typical / atypical modes first restrict the
candidates to the top / bottom typicality quantile.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from boutscope import constants as bc


class SamplingError(ValueError):
    """The request cannot be served (unknown mode, no typicality, missing stratum column)."""


@dataclass(frozen=True)
class SampleRequest:
    """Everything that determines a draw order."""

    labelling: str
    label: str
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    strata: tuple[str, ...] = bc.DEFAULT_STRATA
    mode: str = "stratified"
    seed: int = bc.DEFAULT_SEED
    max_per_animal: int | None = None
    typicality_quantile: float = bc.TYPICALITY_QUANTILE

    def key(self) -> str:
        """Stable text form; feeds the RNG so each request has its own stream."""
        return "|".join(
            str(v)
            for v in (
                self.labelling,
                self.label,
                self.min_duration_s,
                self.max_duration_s,
                ",".join(self.strata),
                self.mode,
                self.max_per_animal,
                self.typicality_quantile,
            )
        )


@dataclass
class Session:
    """A draw order and how far it has been read."""

    request: SampleRequest
    order: np.ndarray
    position: int = 0
    shown: list[int] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return len(self.order) - self.position

    def next_batch(self, n: int) -> np.ndarray:
        """The next *n* bout ids (fewer when the order runs out)."""
        batch = self.order[self.position : self.position + n]
        self.position += len(batch)
        self.shown.extend(int(b) for b in batch)
        return batch


def start_session(bouts: pd.DataFrame, request: SampleRequest) -> Session:
    return Session(request=request, order=build_draw_order(bouts, request))


def select_candidates(bouts: pd.DataFrame, request: SampleRequest) -> pd.DataFrame:
    """Bouts of the label inside the duration range, then the typicality cut."""
    if request.mode not in bc.SAMPLING_MODES:
        raise SamplingError(f"mode must be one of {bc.SAMPLING_MODES}, not {request.mode!r}")
    missing = [c for c in request.strata if c not in bouts]
    if missing:
        raise SamplingError(f"stratum column(s) {missing} not in the bout table")
    keep = (bouts["label"] == str(request.label)).to_numpy(dtype=bool, copy=True)
    if request.min_duration_s is not None:
        keep &= bouts["duration_s"].to_numpy() >= request.min_duration_s - 1e-9
    if request.max_duration_s is not None:
        keep &= bouts["duration_s"].to_numpy() <= request.max_duration_s + 1e-9
    candidates = bouts[keep]
    if request.mode == "stratified":
        return candidates
    if "typicality" not in candidates:
        raise SamplingError(
            f"mode {request.mode!r} needs a typicality score, "
            f"and labelling {request.labelling!r} has none"
        )
    scored = candidates[np.isfinite(candidates["typicality"].to_numpy())]
    if scored.empty:
        return scored
    q = request.typicality_quantile
    if request.mode == "typical":
        return scored[scored["typicality"] >= scored["typicality"].quantile(1 - q)]
    return scored[scored["typicality"] <= scored["typicality"].quantile(q)]


def count_by_stratum(candidates: pd.DataFrame, strata: tuple[str, ...]) -> pd.Series:
    """Bouts available per stratum, largest first."""
    if not strata:
        return pd.Series({"all": len(candidates)}, name="n_bouts")
    counts = candidates.groupby(list(strata), observed=True).size().rename("n_bouts")
    return counts.sort_values(ascending=False, kind="stable")


def build_draw_order(bouts: pd.DataFrame, request: SampleRequest) -> np.ndarray:
    """All candidate bout ids in seeded round-robin order across strata, capped per animal."""
    candidates = select_candidates(bouts, request)
    if candidates.empty:
        return np.zeros(0, dtype=np.int64)
    rng = build_rng(request)
    ids = candidates["bout_id"].to_numpy(dtype=np.int64)
    stratum = (
        candidates.groupby(list(request.strata), observed=True).ngroup().to_numpy()
        if request.strata
        else np.zeros(len(ids), dtype=np.int64)
    )
    shuffled = rng.permutation(len(ids))
    stratum_rank = rng.permutation(stratum.max() + 1)[stratum[shuffled]]
    turn = _rank_within_groups(stratum[shuffled])
    order = shuffled[np.lexsort((stratum_rank, turn))]
    drawn = ids[order]
    if request.max_per_animal is not None:
        animal = candidates["animal"].astype(str).to_numpy()[order]
        drawn = drawn[_rank_within_groups(animal) < request.max_per_animal]
    return drawn


def build_blind_order(
    bouts: pd.DataFrame, request: SampleRequest, labels: tuple[str, ...]
) -> np.ndarray:
    """Draw orders of several labels, interleaved so each label gets the same share.

    Each label keeps its own stratified order; the labels take turns in an
    order that is reshuffled every round, so position does not reveal the
    label.  Rounds stop where the scarcest label runs out, so every label is
    equally represented in any prefix.
    """
    if len(labels) < 2:
        raise SamplingError("a blind session needs at least two labels")
    orders = [build_draw_order(bouts, replace(request, label=str(lab))) for lab in labels]
    rounds = min(len(o) for o in orders)
    rng = build_rng(replace(request, label="|".join(map(str, labels))))
    turns = [rng.permutation(len(labels)) for _ in range(rounds)]
    return np.array([orders[i][r] for r, turn in enumerate(turns) for i in turn], dtype=np.int64)


def build_rng(request: SampleRequest) -> np.random.Generator:
    """Generator from the user's seed and a stable hash of the rest of the request."""
    digest = hashlib.sha256(request.key().encode("utf-8")).digest()
    return np.random.default_rng(
        np.random.SeedSequence([request.seed, int.from_bytes(digest[:8], "little")])
    )


def _rank_within_groups(groups: np.ndarray) -> np.ndarray:
    """0, 1, 2, ... for each element's occurrence within its group, in input order."""
    _, inverse = np.unique(groups, return_inverse=True)
    by_group = np.argsort(inverse, kind="stable")
    sorted_groups = inverse[by_group]
    rank = np.empty(len(groups), dtype=np.int64)
    rank[by_group] = np.arange(len(groups)) - np.searchsorted(
        sorted_groups, sorted_groups, side="left"
    )
    return rank
