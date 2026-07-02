"""Core types for loomc."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel


class StepRecord(BaseModel):
    """Record of a single loop iteration."""

    step_n: int
    state: Any
    cost_usd: float = 0.0
    score: float | None = None
    timestamp: datetime = datetime.now()
    error: str | None = None


CostFn = Callable[[Any], float]
ScorerFn = Callable[[Any], float]


@runtime_checkable
class ConvergenceDetector(Protocol):
    def should_stop(self, history: list[StepRecord]) -> bool: ...


@runtime_checkable
class BacktrackStrategy(Protocol):
    def should_backtrack(
        self, history: list[StepRecord]
    ) -> tuple[bool, int | None]: ...
