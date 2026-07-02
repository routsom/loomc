"""Backtracking strategies for agent loops."""

from __future__ import annotations

from loomc._types import ScorerFn, StepRecord


class BestScore:
    """Backtrack to the highest-scoring checkpoint when score regresses."""

    def __init__(self, metric: ScorerFn | None = None):
        self.metric = metric

    def should_backtrack(
        self, history: list[StepRecord]
    ) -> tuple[bool, int | None]:
        scored = [s for s in history if s.score is not None]
        if len(scored) < 2:
            return False, None
        best_idx = max(range(len(scored)), key=lambda i: scored[i].score or 0)
        if best_idx < len(scored) - 1:
            # Latest is not the best — backtrack
            return True, scored[best_idx].step_n
        return False, None


class LastGood:
    """On error, restore the last non-error checkpoint."""

    def should_backtrack(
        self, history: list[StepRecord]
    ) -> tuple[bool, int | None]:
        if not history or history[-1].error is None:
            return False, None
        for step in reversed(history[:-1]):
            if step.error is None:
                return True, step.step_n
        return False, None
