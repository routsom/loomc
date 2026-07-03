"""Core loop runtime — the @loop decorator."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import inspect
from datetime import datetime, timezone
from typing import Any, Callable

from loomc._types import BacktrackStrategy, ConvergenceDetector, CostFn, ScorerFn, StepRecord
from loomc.budget import Budget, BudgetExhausted, BudgetLedger
from loomc.checkpoint import RunStatus, SQLiteCheckpointStore

# Context variable for nested budget propagation (M3)
_parent_ledger: contextvars.ContextVar[BudgetLedger | None] = contextvars.ContextVar(
    "loomc_parent_ledger", default=None
)


class StallError(Exception):
    """Raised when the loop stalls (no progress for N iterations)."""

    def __init__(self, iterations: int):
        self.iterations = iterations
        super().__init__(f"Loop stalled: no score improvement for {iterations} iterations")


class Escalate:
    """Stall handler: raise StallError after `after` iterations of no improvement."""

    def __init__(self, after: int = 3):
        self.after = after


def loop(
    converge_on: ConvergenceDetector | None = None,
    budget: Budget | None = None,
    checkpoint: str | Any = "sqlite",
    backtrack: BacktrackStrategy | None = None,
    on_stall: Escalate | None = None,
    cost_fn: CostFn | None = None,
    scorer: ScorerFn | None = None,
) -> Callable:
    """Decorator that compiles a single-iteration function into a resilient loop.

    The decorated function signature must be fn(state) -> state.
    """

    def decorator(fn: Callable) -> LoopRunner | AsyncLoopRunner:
        spec_hash = hashlib.md5(
            f"{fn.__module__}.{fn.__qualname__}".encode()
        ).hexdigest()[:8]

        store = _make_store(checkpoint)
        kwargs = dict(
            fn=fn,
            spec_hash=spec_hash,
            store=store,
            converge_on=converge_on,
            budget=budget,
            backtrack=backtrack,
            on_stall=on_stall,
            cost_fn=cost_fn,
            scorer=scorer,
        )

        if inspect.iscoroutinefunction(fn):
            runner: LoopRunner | AsyncLoopRunner = AsyncLoopRunner(**kwargs)
        else:
            runner = LoopRunner(**kwargs)

        functools.update_wrapper(runner, fn)
        return runner

    return decorator


class LoopRunner:
    """Callable wrapper that executes the loop runtime."""

    def __init__(
        self,
        fn: Callable,
        spec_hash: str,
        store: SQLiteCheckpointStore,
        converge_on: ConvergenceDetector | None,
        budget: Budget | None,
        backtrack: BacktrackStrategy | None,
        on_stall: Escalate | None,
        cost_fn: CostFn | None,
        scorer: ScorerFn | None,
    ):
        self._fn = fn
        self._spec_hash = spec_hash
        self._store = store
        self._converge_on = converge_on
        self._budget = budget
        self._backtrack = backtrack
        self._on_stall = on_stall
        self._cost_fn = cost_fn
        self._scorer = scorer

    def __call__(self, state: Any, *, run_id: str | None = None) -> Any:
        return self._execute(state, run_id=run_id)

    def resume(self, run_id: str) -> Any:
        """Resume a previously checkpointed run."""
        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id!r} not found")
        steps = self._store.load_steps(run_id)
        if not steps:
            raise ValueError(f"Run {run_id!r} has no checkpointed steps")
        last_step = steps[-1]
        state = last_step.state
        return self._execute(state, run_id=run_id, resume_from=steps)

    def _execute(
        self,
        state: Any,
        run_id: str | None = None,
        resume_from: list[StepRecord] | None = None,
    ) -> Any:
        history: list[StepRecord] = []
        ledger = BudgetLedger()

        if run_id is None:
            run_id = self._store.create_run(self._spec_hash)
        else:
            self._store.update_run_status(run_id, RunStatus.RUNNING)

        effective_budget = self._budget

        if resume_from:
            history = list(resume_from)
            for step in history:
                ledger.record(step.cost_usd)
            # Extend budget to account for already-spent resources
            if effective_budget:
                effective_budget = Budget(
                    usd=(effective_budget.usd + ledger.total_cost_usd) if effective_budget.usd is not None else None,
                    iterations=(effective_budget.iterations + ledger.total_iterations) if effective_budget.iterations is not None else None,
                    wall_clock=effective_budget.wall_clock,
                )

        step_n = len(history)
        final_status = RunStatus.COMPLETED

        # Capture outer parent before overwriting context var (M3)
        _outer_parent = _parent_ledger.get()
        _token = _parent_ledger.set(ledger)

        try:
            while True:
                # Pre-flight budget check
                if effective_budget:
                    try:
                        ledger.check_pre(effective_budget)
                    except BudgetExhausted:
                        final_status = RunStatus.BUDGET_EXHAUSTED
                        raise

                # Execute one iteration
                error = None
                cost = 0.0
                try:
                    state = self._fn(state)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    if not self._backtrack:
                        final_status = RunStatus.FAILED
                        raise

                # Extract cost
                if error is None and self._cost_fn:
                    cost = self._cost_fn(state)

                # Score
                score = None
                if error is None and self._scorer:
                    score = self._scorer(state)

                # Record step
                record = StepRecord(
                    step_n=step_n,
                    state=state,
                    cost_usd=cost,
                    score=score,
                    timestamp=datetime.now(timezone.utc),
                    error=error,
                )
                history.append(record)
                self._store.save_step(run_id, record)
                ledger.record(cost)
                step_n += 1

                # Propagate cost to outer parent ledger if nested (M3)
                if _outer_parent is not None:
                    _outer_parent.record(cost)

                # Backtrack check
                if self._backtrack:
                    should_bt, restore_to = self._backtrack.should_backtrack(history)
                    if should_bt and restore_to is not None:
                        for step in history:
                            if step.step_n == restore_to:
                                state = step.state
                                break

                # Convergence check
                if self._converge_on and self._converge_on.should_stop(history):
                    break

                # Stall detection
                if self._on_stall and self._scorer:
                    self._check_stall(history)

        except (BudgetExhausted, StallError):
            self._store.update_run_status(run_id, final_status)
            raise
        except Exception:
            self._store.update_run_status(run_id, RunStatus.FAILED)
            raise
        else:
            self._store.update_run_status(run_id, RunStatus.COMPLETED)
        finally:
            _parent_ledger.reset(_token)

        return state

    def _check_stall(self, history: list[StepRecord]) -> None:
        if not self._on_stall:
            return
        scored = [s for s in history if s.score is not None]
        if len(scored) < self._on_stall.after + 1:
            return
        best = max(s.score for s in scored[: -self._on_stall.after])  # type: ignore[arg-type]
        recent = scored[-self._on_stall.after :]
        if all((s.score or 0) <= best for s in recent):
            raise StallError(self._on_stall.after)


class AsyncLoopRunner(LoopRunner):
    """Async variant of LoopRunner for coroutine iteration functions."""

    async def __call__(self, state: Any, *, run_id: str | None = None) -> Any:  # type: ignore[override]
        return await self._execute_async(state, run_id=run_id)

    async def resume(self, run_id: str) -> Any:  # type: ignore[override]
        run = self._store.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id!r} not found")
        steps = self._store.load_steps(run_id)
        if not steps:
            raise ValueError(f"Run {run_id!r} has no checkpointed steps")
        return await self._execute_async(steps[-1].state, run_id=run_id, resume_from=steps)

    async def _execute_async(
        self,
        state: Any,
        run_id: str | None = None,
        resume_from: list[StepRecord] | None = None,
    ) -> Any:
        history: list[StepRecord] = []
        ledger = BudgetLedger()

        if run_id is None:
            run_id = self._store.create_run(self._spec_hash)
        else:
            self._store.update_run_status(run_id, RunStatus.RUNNING)

        effective_budget = self._budget

        if resume_from:
            history = list(resume_from)
            for step in history:
                ledger.record(step.cost_usd)
            if effective_budget:
                effective_budget = Budget(
                    usd=(effective_budget.usd + ledger.total_cost_usd) if effective_budget.usd is not None else None,
                    iterations=(effective_budget.iterations + ledger.total_iterations) if effective_budget.iterations is not None else None,
                    wall_clock=effective_budget.wall_clock,
                )

        step_n = len(history)
        final_status = RunStatus.COMPLETED
        _outer_parent = _parent_ledger.get()
        _token = _parent_ledger.set(ledger)

        try:
            while True:
                if effective_budget:
                    try:
                        ledger.check_pre(effective_budget)
                    except BudgetExhausted:
                        final_status = RunStatus.BUDGET_EXHAUSTED
                        raise

                error = None
                cost = 0.0
                try:
                    state = await self._fn(state)
                except Exception as e:
                    error = f"{type(e).__name__}: {e}"
                    if not self._backtrack:
                        final_status = RunStatus.FAILED
                        raise

                if error is None and self._cost_fn:
                    cost = self._cost_fn(state)

                score = None
                if error is None and self._scorer:
                    score = self._scorer(state)

                record = StepRecord(
                    step_n=step_n,
                    state=state,
                    cost_usd=cost,
                    score=score,
                    timestamp=datetime.now(timezone.utc),
                    error=error,
                )
                history.append(record)
                self._store.save_step(run_id, record)
                ledger.record(cost)
                step_n += 1

                if _outer_parent is not None:
                    _outer_parent.record(cost)

                if self._backtrack:
                    should_bt, restore_to = self._backtrack.should_backtrack(history)
                    if should_bt and restore_to is not None:
                        for step in history:
                            if step.step_n == restore_to:
                                state = step.state
                                break

                if self._converge_on and self._converge_on.should_stop(history):
                    break

                if self._on_stall and self._scorer:
                    self._check_stall(history)

        except (BudgetExhausted, StallError):
            self._store.update_run_status(run_id, final_status)
            raise
        except Exception:
            self._store.update_run_status(run_id, RunStatus.FAILED)
            raise
        else:
            self._store.update_run_status(run_id, RunStatus.COMPLETED)
        finally:
            _parent_ledger.reset(_token)

        return state


def _make_store(checkpoint: str | Any) -> SQLiteCheckpointStore:
    if checkpoint == "sqlite" or checkpoint is None:
        return SQLiteCheckpointStore()
    if isinstance(checkpoint, str):
        return SQLiteCheckpointStore(db_path=checkpoint)
    return checkpoint
