"""Convergence detectors for agent loops."""

from __future__ import annotations

from typing import Callable

from loomc._types import StepRecord


class ScorePlateau:
    """Stop when score hasn't improved for `patience` consecutive iterations."""

    def __init__(self, patience: int = 3, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta

    def should_stop(self, history: list[StepRecord]) -> bool:
        scored = [s for s in history if s.score is not None]
        if len(scored) < self.patience + 1:
            return False
        best = max(s.score for s in scored[:-self.patience])  # type: ignore[arg-type]
        recent = scored[-self.patience :]
        return all((s.score or 0) - best < self.min_delta for s in recent)  # type: ignore[operator]


class SemanticDelta:
    """Stop when output stabilizes based on embedding distance.

    M2 stub — currently never triggers convergence.
    Will use sentence-transformers or an API-based embedding provider
    to compute cosine distance between consecutive iteration outputs.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def should_stop(self, history: list[StepRecord]) -> bool:
        return False


class Predicate:
    """Wrap an arbitrary function as a convergence detector."""

    def __init__(self, fn: Callable[[list[StepRecord]], bool]):
        self.fn = fn

    def should_stop(self, history: list[StepRecord]) -> bool:
        return self.fn(history)


def score_plateau(patience: int = 3, min_delta: float = 0.01) -> ScorePlateau:
    return ScorePlateau(patience=patience, min_delta=min_delta)


def semantic_delta(threshold: float = 0.05) -> SemanticDelta:
    return SemanticDelta(threshold=threshold)


def predicate(fn: Callable[[list[StepRecord]], bool]) -> Predicate:
    return Predicate(fn=fn)
