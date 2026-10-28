"""Verdict store, sessions, resume and blind draws."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from boutscope import constants as bc
from boutscope.annotations import AnnotationError, AnnotationStore, Verdict, hash_order
from boutscope.browser import Browser, judge_blind_answer
from boutscope.sampling import SampleRequest, SamplingError, build_blind_order


@pytest.fixture
def store(tmp_path):
    store = AnnotationStore(tmp_path / bc.DB_FILE)
    yield store
    store.close()


@pytest.fixture
def browser(project, store):
    return Browser(project, store=store, annotator="bart")


def test_session_round_trip(store):
    request = SampleRequest(
        labelling="pm",
        label="3",
        min_duration_s=0.1,
        strata=("condition",),
        seed=7,
        max_per_animal=2,
    )
    order = np.array([5, 2, 9], dtype=np.int64)
    sid = store.create_session("bart", request, order, "hash")
    store.set_position(sid, 2)
    record = store.load_session(sid)
    assert record.request == request and record.annotator == "bart"
    assert record.order_hash == hash_order(order) and record.n_available == 3
    assert (record.position, record.blind, record.label_set) == (2, False, None)
    sessions = store.list_sessions()
    assert sessions.loc[0, "drawn"] == 2 and sessions.loc[0, "available"] == 3


def test_verdict_upsert_and_validation(store):
    sid = store.create_session("bart", SampleRequest("pm", "3"), np.arange(3), "h")
    store.record_verdict(sid, 1, "3", Verdict("fits", note="  clean  "))
    store.record_verdict(sid, 1, "3", Verdict("artefact", looks_like="7", note=" "))
    verdict = store.verdict_for(sid, 1)
    assert verdict == Verdict("artefact", looks_like="7", note=None)
    assert len(store.verdicts()) == 1
    assert store.verdict_for(sid, 2) is None
    with pytest.raises(AnnotationError, match="verdict must be"):
        store.record_verdict(sid, 2, "3", Verdict("maybe"))
    with pytest.raises(AnnotationError, match="no session"):
        store.record_verdict(999, 2, "3", Verdict("fits"))


def test_names_latest_per_annotator(store):
    store.propose_name("bart", "pm", "3", "rest")
    store.propose_name("bart", "pm", "3", "rest, curled")
    store.propose_name("alex", "pm", "3", "floating")
    names = store.label_names("pm")
    assert dict(zip(names.annotator, names.name)) == {"alex": "floating", "bart": "rest, curled"}
    with pytest.raises(AnnotationError):
        store.propose_name("bart", "pm", "3", "  ")
    with pytest.raises(AnnotationError):
        store.annotator_id(" ")


def test_export_and_schema_guard(store, tmp_path):
    sid = store.create_session("bart", SampleRequest("pm", "3"), np.arange(2), "h")
    store.record_verdict(sid, 0, "3", Verdict("fits"))
    paths = store.export_csv(tmp_path / "out")
    assert [p.name for p in paths] == ["verdicts.csv", "sessions.csv", "label_names.csv"]
    assert pd.read_csv(paths[0]).verdict.tolist() == ["fits"]
    db = sqlite3.connect(store.path)
    with db:
        db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    db.close()
    with pytest.raises(AnnotationError, match="schema"):
        AnnotationStore(store.path)


def test_browser_draw_record_resume(browser, store, project):
    browser.select_label("swim")
    drawn = browser.draw(3, strata=("animal",), seed=5)
    browser.record(drawn[0], Verdict("fits"))
    browser.record(drawn[1], Verdict("doesnt_fit", looks_like="turn"))
    more = browser.more(2)
    assert browser.n_rated() == 2

    again = Browser(project, store=store, annotator="bart")
    again.resume(browser.session_id)
    assert again.label == "swim" and again.drawn == drawn + more
    assert again.verdict_for(drawn[1]).looks_like == "turn"
    assert set(again.more(3)).isdisjoint(drawn + more)
    verdicts = store.verdicts("behaviour")
    assert (verdicts.label == "swim").all() and len(verdicts) == 2


def test_resume_refuses_a_changed_labelling(browser, store, project):
    from boutscope import project as bp

    browser.select_label("rest")
    browser.draw(2)
    table = project.load_labels("behaviour")
    changed = bp.add_labelling(project, "behaviour", table.iloc[5:], overwrite=True)
    with pytest.raises(AnnotationError, match="changed"):
        Browser(changed, store=store, annotator="bart").resume(browser.session_id)


def test_blind_order_balances_labels(project):
    bouts = project.load_bouts("behaviour")
    request = SampleRequest(labelling="behaviour", label="", seed=3)
    order = build_blind_order(bouts, request, ("rest", "swim", "turn"))
    labels = bouts.set_index("bout_id").loc[order, "label"].astype(str).to_numpy()
    for start in range(0, len(labels) - 2, 3):
        assert sorted(labels[start : start + 3]) == ["rest", "swim", "turn"]
    np.testing.assert_array_equal(
        order, build_blind_order(bouts, request, ("rest", "swim", "turn"))
    )
    with pytest.raises(SamplingError, match="two labels"):
        build_blind_order(bouts, request, ("rest",))


def test_blind_session_judges_answers(browser, store):
    drawn = browser.draw_blind(("rest", "turn"), 4, seed=2)
    assert browser.is_blind and browser.label_set == ("rest", "turn")
    truth = [str(browser.bout(b)["label"]) for b in drawn]
    browser.record(drawn[0], Verdict("unsure", blind_answer=truth[0]))
    browser.record(drawn[1], Verdict("unsure", blind_answer=bc.BLIND_NONE))
    browser.record(drawn[2], Verdict("artefact"))
    got = store.verdicts().set_index("bout_id")
    assert got.loc[drawn[0], "verdict"] == "fits"
    assert got.loc[drawn[1], "verdict"] == "doesnt_fit"
    assert got.loc[drawn[2], "verdict"] == "artefact"
    assert got.blind.eq(1).all()
    record = store.load_session(browser.session_id)
    assert record.blind and record.label_set == ("rest", "turn")
    assert judge_blind_answer("turn", "turn") == "fits"


def test_record_needs_a_stored_session(project):
    browser = Browser(project)
    browser.select_label("rest")
    bout = browser.draw(1)[0]
    with pytest.raises(AnnotationError, match="no stored session"):
        browser.record(bout, Verdict("fits"))
    assert browser.verdict_for(bout) is None


def test_resume_rebuilds_and_checks_the_order(browser, store, project):
    browser.select_label("turn")
    browser.draw(2, seed=11)
    db = sqlite3.connect(store.path)
    with db:
        db.execute(
            "UPDATE sessions SET order_hash = 'tampered' WHERE id = ?", (browser.session_id,)
        )
    db.close()
    with pytest.raises(AnnotationError, match="differs"):
        Browser(project, store=store, annotator="bart").resume(browser.session_id)


def test_blind_session_resumes(browser, store, project):
    drawn = browser.draw_blind(("rest", "swim"), 3, seed=4)
    again = Browser(project, store=store, annotator="bart")
    again.resume(browser.session_id)
    assert again.is_blind and again.drawn == drawn
