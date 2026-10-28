# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — annotations                                         ║
# ║  « every verdict traceable to its labels »                       ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  Annotators, resumable sessions with their draw order, verdicts  ║
# ║  and proposed label names in one SQLite file per project.        ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Verdicts, sessions and proposed label names in one SQLite file per project.

A session stores everything that made its draw, including the full draw
order, so it can be resumed on any machine and "more" continues exactly
where it stopped.  Every verdict carries the annotator, the session and
the labelling's content hash, so a verdict can always be traced to the
exact labels it judged.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from boutscope import constants as bc
from boutscope.sampling import SampleRequest

SCHEMA: str = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS annotators (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    labelling TEXT NOT NULL,
    label TEXT NOT NULL,
    label_set TEXT,
    blind INTEGER NOT NULL,
    min_duration_s REAL,
    max_duration_s REAL,
    strata TEXT NOT NULL,
    mode TEXT NOT NULL,
    seed INTEGER NOT NULL,
    max_per_animal INTEGER,
    typicality_quantile REAL NOT NULL,
    content_hash TEXT NOT NULL,
    draw_order BLOB NOT NULL,
    position INTEGER NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    labelling TEXT NOT NULL,
    bout_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    verdict TEXT NOT NULL CHECK (verdict IN ('fits', 'doesnt_fit', 'artefact', 'unsure')),
    blind_answer TEXT,
    looks_like TEXT,
    note TEXT,
    content_hash TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    UNIQUE (session_id, bout_id)
);
CREATE TABLE IF NOT EXISTS label_names (
    id INTEGER PRIMARY KEY,
    annotator_id INTEGER NOT NULL REFERENCES annotators(id),
    labelling TEXT NOT NULL,
    label TEXT NOT NULL,
    name TEXT NOT NULL,
    note TEXT,
    timestamp TEXT NOT NULL
);
"""


class AnnotationError(RuntimeError):
    """The store refuses an operation (unknown verdict, schema or hash mismatch)."""


@dataclass(frozen=True)
class SessionRecord:
    id: int
    annotator: str
    request: SampleRequest
    label_set: tuple[str, ...] | None  # blind sessions: the labels mixed
    blind: bool
    order: np.ndarray
    position: int
    content_hash: str
    created: str


@dataclass(frozen=True)
class Verdict:
    """One judgement of one bout.  Blind sessions fill ``blind_answer``."""

    verdict: str
    looks_like: str | None = None
    note: str | None = None
    blind_answer: str | None = None


class AnnotationStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._db = sqlite3.connect(self.path)
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._check_schema()

    def close(self) -> None:
        self._db.close()

    # ┌────────────────────────────────────────────────────────────┐
    # │ Annotators and sessions  « who, and what they drew »       │
    # └────────────────────────────────────────────────────────────┘

    def annotator_id(self, name: str) -> int:
        """Id of *name*; the annotator is created on first use."""
        name = name.strip()
        if not name:
            raise AnnotationError("an annotator needs a name")
        with self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO annotators (name, created) VALUES (?, ?)", (name, _now())
            )
        return self._db.execute("SELECT id FROM annotators WHERE name = ?", (name,)).fetchone()[0]

    def create_session(
        self,
        annotator: str,
        request: SampleRequest,
        order: np.ndarray,
        content_hash: str,
        *,
        label_set: tuple[str, ...] | None = None,
        position: int = 0,
    ) -> int:
        now = _now()
        with self._db:
            cursor = self._db.execute(
                "INSERT INTO sessions (annotator_id, labelling, label, label_set, blind, "
                "min_duration_s, max_duration_s, strata, mode, seed, max_per_animal, "
                "typicality_quantile, content_hash, draw_order, position, created, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.annotator_id(annotator),
                    request.labelling,
                    str(request.label),
                    None if label_set is None else json.dumps(list(label_set)),
                    int(label_set is not None),
                    request.min_duration_s,
                    request.max_duration_s,
                    json.dumps(list(request.strata)),
                    request.mode,
                    int(request.seed),
                    request.max_per_animal,
                    request.typicality_quantile,
                    content_hash,
                    np.asarray(order, dtype="<i8").tobytes(),
                    int(position),
                    now,
                    now,
                ),
            )
        return int(cursor.lastrowid)

    def set_position(self, session_id: int, position: int) -> None:
        with self._db:
            self._db.execute(
                "UPDATE sessions SET position = ?, updated = ? WHERE id = ?",
                (int(position), _now(), session_id),
            )

    def load_session(self, session_id: int) -> SessionRecord:
        row = self._db.execute(
            "SELECT s.id, a.name, s.labelling, s.label, s.label_set, s.blind, s.min_duration_s, "
            "s.max_duration_s, s.strata, s.mode, s.seed, s.max_per_animal, "
            "s.typicality_quantile, s.content_hash, s.draw_order, s.position, s.created "
            "FROM sessions s JOIN annotators a ON a.id = s.annotator_id WHERE s.id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise AnnotationError(f"no session {session_id}")
        (
            sid,
            annotator,
            labelling,
            label,
            label_set,
            blind,
            lo,
            hi,
            strata,
            mode,
            seed,
            cap,
            quantile,
            content_hash,
            order,
            position,
            created,
        ) = row
        request = SampleRequest(
            labelling=labelling,
            label=label,
            min_duration_s=lo,
            max_duration_s=hi,
            strata=tuple(json.loads(strata)),
            mode=mode,
            seed=seed,
            max_per_animal=cap,
            typicality_quantile=quantile,
        )
        return SessionRecord(
            id=sid,
            annotator=annotator,
            request=request,
            label_set=None if label_set is None else tuple(json.loads(label_set)),
            blind=bool(blind),
            order=np.frombuffer(order, dtype="<i8").astype(np.int64),
            position=position,
            content_hash=content_hash,
            created=created,
        )

    def list_sessions(self) -> pd.DataFrame:
        """All sessions with their progress, newest first (for the resume dialog)."""
        return pd.read_sql_query(
            "SELECT s.id, a.name AS annotator, s.labelling, s.label, s.blind, s.mode, s.seed, "
            "s.position AS drawn, length(s.draw_order) / 8 AS available, "
            "(SELECT count(*) FROM verdicts v WHERE v.session_id = s.id) AS rated, "
            "s.created, s.updated FROM sessions s JOIN annotators a ON a.id = s.annotator_id "
            "ORDER BY s.updated DESC, s.id DESC",
            self._db,
        )

    # ┌────────────────────────────────────────────────────────────┐
    # │ Verdicts and names  « the ground truth itself »            │
    # └────────────────────────────────────────────────────────────┘

    def record_verdict(self, session_id: int, bout_id: int, label: str, verdict: Verdict) -> None:
        """Store or replace the verdict on one bout within one session."""
        if verdict.verdict not in bc.VERDICTS:
            raise AnnotationError(f"verdict must be one of {bc.VERDICTS}, not {verdict.verdict!r}")
        session = self._db.execute(
            "SELECT annotator_id, labelling, content_hash FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise AnnotationError(f"no session {session_id}")
        annotator_id, labelling, content_hash = session
        with self._db:
            self._db.execute(
                "INSERT INTO verdicts (session_id, annotator_id, labelling, bout_id, label, "
                "verdict, blind_answer, looks_like, note, content_hash, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT (session_id, bout_id) DO UPDATE SET verdict = excluded.verdict, "
                "blind_answer = excluded.blind_answer, looks_like = excluded.looks_like, "
                "note = excluded.note, timestamp = excluded.timestamp",
                (
                    session_id,
                    annotator_id,
                    labelling,
                    int(bout_id),
                    str(label),
                    verdict.verdict,
                    verdict.blind_answer,
                    _blank_to_none(verdict.looks_like),
                    _blank_to_none(verdict.note),
                    content_hash,
                    _now(),
                ),
            )

    def verdict_for(self, session_id: int, bout_id: int) -> Verdict | None:
        row = self._db.execute(
            "SELECT verdict, looks_like, note, blind_answer FROM verdicts "
            "WHERE session_id = ? AND bout_id = ?",
            (session_id, int(bout_id)),
        ).fetchone()
        return None if row is None else Verdict(*row)

    def verdicts(self, labelling: str | None = None) -> pd.DataFrame:
        """All verdicts with annotator and session settings, one row each."""
        query = (
            "SELECT v.id, a.name AS annotator, v.session_id, s.blind, s.mode, v.labelling, "
            "v.bout_id, v.label, v.verdict, v.blind_answer, v.looks_like, v.note, "
            "v.content_hash, v.timestamp FROM verdicts v "
            "JOIN annotators a ON a.id = v.annotator_id JOIN sessions s ON s.id = v.session_id"
        )
        params: tuple = ()
        if labelling is not None:
            query += " WHERE v.labelling = ?"
            params = (labelling,)
        return pd.read_sql_query(query + " ORDER BY v.id", self._db, params=params)

    def propose_name(
        self, annotator: str, labelling: str, label: str, name: str, note: str | None = None
    ) -> None:
        """Record a proposed name; the history is kept, the latest per annotator counts."""
        if not name.strip():
            raise AnnotationError("a proposed name must not be empty")
        with self._db:
            self._db.execute(
                "INSERT INTO label_names (annotator_id, labelling, label, name, note, timestamp) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.annotator_id(annotator),
                    labelling,
                    str(label),
                    name.strip(),
                    _blank_to_none(note),
                    _now(),
                ),
            )

    def label_names(self, labelling: str) -> pd.DataFrame:
        """Latest proposed name per (annotator, label)."""
        return pd.read_sql_query(
            "SELECT a.name AS annotator, n.label, n.name, n.note, n.timestamp FROM label_names n "
            "JOIN annotators a ON a.id = n.annotator_id WHERE n.labelling = ? AND n.id = ("
            "SELECT max(m.id) FROM label_names m WHERE m.annotator_id = n.annotator_id "
            "AND m.labelling = n.labelling AND m.label = n.label) ORDER BY n.label, a.name",
            self._db,
            params=(labelling,),
        )

    def export_csv(self, folder: Path) -> list[Path]:
        """``verdicts.csv``, ``sessions.csv`` and ``label_names.csv`` in *folder*."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        names = pd.read_sql_query(
            "SELECT a.name AS annotator, n.labelling, n.label, n.name, n.note, n.timestamp "
            "FROM label_names n JOIN annotators a ON a.id = n.annotator_id ORDER BY n.id",
            self._db,
        )
        out = []
        for stem, table in (
            ("verdicts", self.verdicts()),
            ("sessions", self.list_sessions()),
            ("label_names", names),
        ):
            path = folder / f"{stem}.csv"
            table.to_csv(path, index=False, encoding="utf-8")
            out.append(path)
        return out

    def _check_schema(self) -> None:
        row = self._db.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            with self._db:
                self._db.execute(
                    "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(bc.DB_SCHEMA_VERSION),),
                )
        elif int(row[0]) != bc.DB_SCHEMA_VERSION:
            raise AnnotationError(
                f"{self.path.name}: schema {row[0]}, this boutscope reads {bc.DB_SCHEMA_VERSION}"
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _blank_to_none(text: str | None) -> str | None:
    return None if text is None or not text.strip() else text.strip()
