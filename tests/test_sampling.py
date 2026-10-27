"""Seeded stratified sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boutscope.sampling import (
    SampleRequest,
    SamplingError,
    build_draw_order,
    count_by_stratum,
    select_candidates,
    start_session,
)


@pytest.fixture
def bouts(project) -> pd.DataFrame:
    return project.load_bouts("behaviour")


def test_same_seed_same_order_other_seed_other_order(bouts):
    request = SampleRequest(labelling="behaviour", label="swim")
    first = build_draw_order(bouts, request)
    np.testing.assert_array_equal(first, build_draw_order(bouts, request))
    other = build_draw_order(bouts, SampleRequest(labelling="behaviour", label="swim", seed=1))
    assert sorted(first) == sorted(other) and not np.array_equal(first, other)


def test_golden_order_is_platform_independent(bouts):
    # Generated once; PCG64 + SeedSequence are platform independent, so CI on all OS must agree.
    order = build_draw_order(bouts, SampleRequest(labelling="behaviour", label="rest", seed=7))
    assert order[:5].tolist() == GOLDEN_REST_SEED7


def test_strata_take_turns(bouts):
    request = SampleRequest(labelling="behaviour", label="turn", strata=("animal",))
    order = build_draw_order(bouts, request)
    animal = bouts.set_index("bout_id").loc[order, "animal"].astype(str).to_numpy()
    n_animals = bouts.animal.nunique()
    per_animal = bouts[bouts.label == "turn"].groupby("animal", observed=True).size()
    rounds = per_animal.min()
    head = animal[: rounds * n_animals]
    for r in range(rounds):
        assert len(set(head[r * n_animals : (r + 1) * n_animals])) == n_animals


def test_duration_filter_and_counts(bouts):
    request = SampleRequest(
        labelling="behaviour",
        label="swim",
        min_duration_s=0.1,
        max_duration_s=0.3,
        strata=("condition",),
    )
    candidates = select_candidates(bouts, request)
    assert candidates.duration_s.between(0.1 - 1e-9, 0.3 + 1e-9).all()
    assert (candidates.label == "swim").all()
    counts = count_by_stratum(candidates, request.strata)
    assert counts.sum() == len(candidates) == len(build_draw_order(bouts, request))


def test_cap_per_animal(bouts):
    order = build_draw_order(
        bouts, SampleRequest(labelling="behaviour", label="rest", max_per_animal=2)
    )
    animals = bouts.set_index("bout_id").loc[order, "animal"].astype(str)
    assert animals.value_counts().max() == 2
    assert len(order) == 2 * bouts.animal.nunique()


def test_session_more_never_repeats(bouts):
    session = start_session(bouts, SampleRequest(labelling="behaviour", label="rest"))
    seen = np.concatenate([session.next_batch(7) for _ in range(5)])
    assert len(set(seen.tolist())) == len(seen) == min(35, len(session.order))
    assert session.remaining == len(session.order) - len(seen)


def test_typical_and_atypical(bouts):
    base = dict(labelling="behaviour", label="rest", strata=())
    typical = select_candidates(bouts, SampleRequest(**base, mode="typical"))
    atypical = select_candidates(bouts, SampleRequest(**base, mode="atypical"))
    assert typical.typicality.min() > atypical.typicality.max()
    n = (bouts.label == "rest").sum()
    assert abs(len(typical) - 0.1 * n) <= 2 and abs(len(atypical) - 0.1 * n) <= 2


def test_errors(bouts):
    with pytest.raises(SamplingError, match="mode"):
        select_candidates(bouts, SampleRequest(labelling="behaviour", label="rest", mode="best"))
    with pytest.raises(SamplingError, match="stratum"):
        select_candidates(
            bouts, SampleRequest(labelling="behaviour", label="rest", strata=("genotype",))
        )
    with pytest.raises(SamplingError, match="typicality"):
        select_candidates(
            bouts.drop(columns="typicality"),
            SampleRequest(labelling="behaviour", label="rest", mode="typical"),
        )
    assert len(build_draw_order(bouts, SampleRequest(labelling="behaviour", label="nope"))) == 0


GOLDEN_REST_SEED7: list[int] = [200, 8, 66, 209, 362]
