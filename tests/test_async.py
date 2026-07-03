"""Tests for async loop support and nested budget propagation."""

import pytest

from loomc import Budget, BudgetExhausted, loop, predicate, score_plateau
from loomc.checkpoint import SQLiteCheckpointStore
from loomc.loop import _parent_ledger


@pytest.fixture
def store(tmp_path):
    return SQLiteCheckpointStore(db_path=tmp_path / "test.db")


class TestAsyncLoop:
    @pytest.mark.asyncio
    async def test_basic_async_budget(self, store):
        @loop(budget=Budget(iterations=5), checkpoint=store)
        async def increment(state):
            return {**state, "n": state["n"] + 1}

        with pytest.raises(BudgetExhausted):
            await increment({"n": 0})

    @pytest.mark.asyncio
    async def test_async_convergence(self, store):
        @loop(
            converge_on=predicate(lambda h: len(h) > 0 and (h[-1].score or 0) >= 1.0),
            budget=Budget(iterations=10),
            checkpoint=store,
            scorer=lambda s: s["score"],
        )
        async def improve(state):
            return {**state, "score": min(1.0, state["score"] + 0.25)}

        result = await improve({"score": 0.0})
        assert result["score"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_async_score_plateau(self, store):
        @loop(
            converge_on=score_plateau(patience=2, min_delta=0.01),
            budget=Budget(iterations=20),
            checkpoint=store,
            scorer=lambda s: s["score"],
        )
        async def plateau(state):
            n = state["n"] + 1
            return {"n": n, "score": min(0.9, n * 0.3)}

        result = await plateau({"n": 0, "score": 0.0})
        assert result["score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_async_resume(self, store):
        counter = {"n": 0}

        @loop(
            budget=Budget(iterations=10),
            checkpoint=store,
            converge_on=predicate(lambda h: h[-1].state.get("val", 0) >= 5 if h else False),
        )
        async def failing(state):
            counter["n"] += 1
            if counter["n"] == 2:
                raise RuntimeError("simulated crash")
            return {**state, "val": state["val"] + 1}

        with pytest.raises(RuntimeError):
            await failing({"val": 0})

        run_id = store.list_runs()[0].id
        counter["n"] = 10  # bypass crash trigger
        result = await failing.resume(run_id)
        assert result["val"] >= 5

    @pytest.mark.asyncio
    async def test_async_checkpoints_persisted(self, store):
        @loop(
            converge_on=predicate(lambda h: len(h) >= 3),
            budget=Budget(iterations=10),
            checkpoint=store,
        )
        async def counted(state):
            return {**state, "n": state["n"] + 1}

        await counted({"n": 0})
        runs = store.list_runs()
        assert len(runs) == 1
        steps = store.load_steps(runs[0].id)
        assert len(steps) == 3


class TestNestedBudget:
    def test_parent_ledger_cleared_after_loop(self, store):
        @loop(budget=Budget(iterations=2), checkpoint=store)
        def simple(state):
            return {**state, "n": state.get("n", 0) + 1}

        with pytest.raises(BudgetExhausted):
            simple({"n": 0})

        assert _parent_ledger.get() is None

    def test_nested_costs_propagate(self, store):
        inner_store = SQLiteCheckpointStore(db_path=store.db_path)

        @loop(
            budget=Budget(iterations=2),
            checkpoint=inner_store,
            cost_fn=lambda s: 0.10,
        )
        def inner(state):
            return {**state, "inner": state.get("inner", 0) + 1}

        parent_costs = []

        @loop(budget=Budget(iterations=2), checkpoint=store, cost_fn=lambda s: 0.01)
        def outer(state):
            with pytest.raises(BudgetExhausted):
                inner({"inner": 0})
            ledger = _parent_ledger.get()
            if ledger:
                parent_costs.append(ledger.total_cost_usd)
            return {**state, "outer": state.get("outer", 0) + 1}

        with pytest.raises(BudgetExhausted):
            outer({"outer": 0})

        # Inner loop costs ($0.20 total) should have been seen by outer's ledger
        assert len(parent_costs) > 0
        assert parent_costs[0] > 0.0
