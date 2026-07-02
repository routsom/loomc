"""Tests for SQLite checkpoint store."""

import pytest

from loomc._types import StepRecord
from loomc.checkpoint import RunStatus, SQLiteCheckpointStore


@pytest.fixture
def store(tmp_path):
    return SQLiteCheckpointStore(db_path=tmp_path / "test.db")


class TestSQLiteCheckpointStore:
    def test_create_and_get_run(self, store):
        run_id = store.create_run("abc123")
        run = store.get_run(run_id)
        assert run is not None
        assert run.spec_hash == "abc123"
        assert run.status == RunStatus.RUNNING

    def test_update_status(self, store):
        run_id = store.create_run("abc123")
        store.update_run_status(run_id, RunStatus.COMPLETED)
        run = store.get_run(run_id)
        assert run.status == RunStatus.COMPLETED

    def test_save_and_load_dict_state(self, store):
        run_id = store.create_run("abc123")
        record = StepRecord(
            step_n=0,
            state={"code": "print('hello')", "score": 0.5},
            cost_usd=0.01,
            score=0.5,
        )
        store.save_step(run_id, record)
        steps = store.load_steps(run_id)
        assert len(steps) == 1
        assert steps[0].state == {"code": "print('hello')", "score": 0.5}
        assert steps[0].cost_usd == 0.01

    def test_save_and_load_multiple_steps(self, store):
        run_id = store.create_run("abc123")
        for i in range(5):
            store.save_step(
                run_id,
                StepRecord(step_n=i, state={"i": i}, cost_usd=0.01, score=i * 0.1),
            )
        steps = store.load_steps(run_id)
        assert len(steps) == 5
        assert [s.step_n for s in steps] == [0, 1, 2, 3, 4]

    def test_list_runs(self, store):
        store.create_run("hash1")
        store.create_run("hash2")
        runs = store.list_runs()
        assert len(runs) == 2

    def test_get_nonexistent_run(self, store):
        assert store.get_run("nonexistent") is None
