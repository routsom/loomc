"""Integration tests for the @loop decorator."""

import pytest

from loomc import Budget, BudgetExhausted, BestScore, Escalate, StallError, loop, score_plateau, predicate
from loomc.checkpoint import SQLiteCheckpointStore, RunStatus


@pytest.fixture
def store(tmp_path):
    return SQLiteCheckpointStore(db_path=tmp_path / "test.db")


class TestLoopBasic:
    def test_budget_iteration_limit(self, store):
        @loop(budget=Budget(iterations=5), checkpoint=store)
        def increment(state):
            return {**state, "n": state["n"] + 1}

        with pytest.raises(BudgetExhausted, match="iterations"):
            increment({"n": 0})

    def test_convergence_stops_loop(self, store):
        @loop(
            converge_on=score_plateau(patience=2, min_delta=0.01),
            budget=Budget(iterations=20),
            checkpoint=store,
            scorer=lambda s: s["score"],
        )
        def plateau_fn(state):
            # Score plateaus at 0.9 after step 3
            n = state["n"] + 1
            score = min(0.9, n * 0.3)
            return {"n": n, "score": score}

        result = plateau_fn({"n": 0, "score": 0.0})
        assert result["score"] == 0.9
        # Should have stopped well before 20 iterations
        assert result["n"] < 15


class TestBacktracking:
    def test_best_score_restores(self, store):
        call_count = 0

        @loop(
            budget=Budget(iterations=6),
            checkpoint=store,
            scorer=lambda s: s["score"],
            backtrack=BestScore(),
        )
        def regressing_fn(state):
            nonlocal call_count
            call_count += 1
            n = state["n"] + 1
            # Score goes up then down
            scores = [0.3, 0.6, 0.9, 0.4, 0.5, 0.6]
            score = scores[min(n - 1, len(scores) - 1)]
            return {"n": n, "score": score}

        with pytest.raises(BudgetExhausted):
            regressing_fn({"n": 0, "score": 0.0})


class TestResume:
    def test_resume_continues_from_checkpoint(self, store):
        counter = {"n": 0}

        @loop(
            budget=Budget(iterations=10),
            checkpoint=store,
            scorer=lambda s: s["val"],
            converge_on=predicate(lambda h: len(h) > 0 and (h[-1].score or 0) >= 5),
        )
        def failing_fn(state):
            counter["n"] += 1
            val = state["val"] + 1
            if counter["n"] == 3:
                raise RuntimeError("simulated crash")
            return {"val": val}

        # First run — fails at step 3
        with pytest.raises(RuntimeError, match="simulated crash"):
            failing_fn({"val": 0})

        # Find the run
        runs = store.list_runs()
        assert len(runs) == 1
        run_id = runs[0].id

        # Resume — counter resets so no more crash
        counter["n"] = 10  # skip the crash trigger
        result = failing_fn.resume(run_id)
        assert result["val"] >= 5


class TestStallDetection:
    def test_stall_raises(self, store):
        @loop(
            budget=Budget(iterations=20),
            checkpoint=store,
            scorer=lambda s: s["score"],
            on_stall=Escalate(after=3),
        )
        def stuck_fn(state):
            return {"score": 0.5}  # never improves

        with pytest.raises(StallError):
            stuck_fn({"score": 0.0})
