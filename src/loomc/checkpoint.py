"""SQLite checkpoint store for loop runs."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from loomc._types import StepRecord


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    STALLED = "stalled"


class RunInfo(BaseModel):
    id: str
    spec_hash: str
    status: RunStatus
    created_at: str
    updated_at: str


class CheckpointStore(Protocol):
    def create_run(self, spec_hash: str) -> str: ...
    def update_run_status(self, run_id: str, status: RunStatus) -> None: ...
    def get_run(self, run_id: str) -> RunInfo | None: ...
    def list_runs(self) -> list[RunInfo]: ...
    def save_step(self, run_id: str, record: StepRecord) -> None: ...
    def load_steps(self, run_id: str) -> list[StepRecord]: ...


class SQLiteCheckpointStore:
    """SQLite-backed checkpoint store."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_dir = Path.cwd() / ".loomc"
            db_dir.mkdir(exist_ok=True)
            db_path = db_dir / "checkpoints.db"
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    spec_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS steps (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    step_n INTEGER NOT NULL,
                    state_blob TEXT NOT NULL,
                    cost_usd REAL NOT NULL DEFAULT 0.0,
                    score REAL,
                    timestamp TEXT NOT NULL,
                    error TEXT,
                    PRIMARY KEY (run_id, step_n)
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def create_run(self, spec_hash: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, spec_hash, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, spec_hash, RunStatus.RUNNING.value, now, now),
            )
        return run_id

    def update_run_status(self, run_id: str, status: RunStatus) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, run_id),
            )

    def get_run(self, run_id: str) -> RunInfo | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, spec_hash, status, created_at, updated_at FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return RunInfo(
            id=row[0],
            spec_hash=row[1],
            status=RunStatus(row[2]),
            created_at=row[3],
            updated_at=row[4],
        )

    def list_runs(self) -> list[RunInfo]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, spec_hash, status, created_at, updated_at FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [
            RunInfo(id=r[0], spec_hash=r[1], status=RunStatus(r[2]), created_at=r[3], updated_at=r[4])
            for r in rows
        ]

    def save_step(self, run_id: str, record: StepRecord) -> None:
        state_blob = _serialize_state(record.state)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO steps (run_id, step_n, state_blob, cost_usd, score, timestamp, error) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    record.step_n,
                    state_blob,
                    record.cost_usd,
                    record.score,
                    record.timestamp.isoformat(),
                    record.error,
                ),
            )

    def load_steps(self, run_id: str) -> list[StepRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT step_n, state_blob, cost_usd, score, timestamp, error FROM steps WHERE run_id = ? ORDER BY step_n",
                (run_id,),
            ).fetchall()
        return [
            StepRecord(
                step_n=r[0],
                state=json.loads(r[1]),
                cost_usd=r[2],
                score=r[3],
                timestamp=datetime.fromisoformat(r[4]),
                error=r[5],
            )
            for r in rows
        ]


def _serialize_state(state: Any) -> str:
    if isinstance(state, BaseModel):
        return state.model_dump_json()
    return json.dumps(state)
