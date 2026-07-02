"""Budget enforcement for agent loops."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    """Declarative budget limits for a loop."""

    usd: float | None = None
    iterations: int | None = None
    wall_clock: str | None = None

    @property
    def wall_clock_seconds(self) -> float | None:
        if self.wall_clock is None:
            return None
        return _parse_duration(self.wall_clock)


class BudgetExhausted(Exception):
    """Raised when a budget limit is breached."""

    def __init__(self, limit: str, used: float, max_val: float):
        self.limit = limit
        self.used = used
        self.max_val = max_val
        super().__init__(f"Budget exhausted: {limit} ({used:.4f} / {max_val:.4f})")


@dataclass
class BudgetLedger:
    """Tracks cumulative cost and iterations against a budget."""

    total_cost_usd: float = 0.0
    total_iterations: int = 0
    start_time: float = field(default_factory=time.monotonic)

    def check_pre(self, budget: Budget) -> None:
        """Raise BudgetExhausted if the next iteration would exceed limits."""
        if budget.iterations is not None and self.total_iterations >= budget.iterations:
            raise BudgetExhausted(
                "iterations", self.total_iterations, budget.iterations
            )
        if budget.usd is not None and self.total_cost_usd >= budget.usd:
            raise BudgetExhausted("usd", self.total_cost_usd, budget.usd)
        wc = budget.wall_clock_seconds
        if wc is not None:
            elapsed = time.monotonic() - self.start_time
            if elapsed >= wc:
                raise BudgetExhausted("wall_clock", elapsed, wc)

    def record(self, cost_usd: float = 0.0) -> None:
        """Record one iteration's cost."""
        self.total_cost_usd += cost_usd
        self.total_iterations += 1

    def remaining(self, budget: Budget) -> Budget:
        """Return a new Budget with remaining amounts."""
        usd = None if budget.usd is None else max(0, budget.usd - self.total_cost_usd)
        iters = (
            None
            if budget.iterations is None
            else max(0, budget.iterations - self.total_iterations)
        )
        wc = None
        if budget.wall_clock_seconds is not None:
            remaining_s = max(0, budget.wall_clock_seconds - (time.monotonic() - self.start_time))
            wc = f"{remaining_s:.0f}s"
        return Budget(usd=usd, iterations=iters, wall_clock=wc)


def _parse_duration(s: str) -> float:
    """Parse duration strings like '10m', '1h', '30s' into seconds."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(s|m|h)", s.strip())
    if not m:
        raise ValueError(f"Invalid duration: {s!r}. Use format like '10m', '1h', '30s'")
    val = float(m.group(1))
    unit = m.group(2)
    return val * {"s": 1, "m": 60, "h": 3600}[unit]
