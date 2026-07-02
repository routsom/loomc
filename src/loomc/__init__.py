"""loomc — Declarative loop specs compiled into resilient agent execution."""

from loomc.backtrack import BestScore, LastGood
from loomc.budget import Budget, BudgetExhausted
from loomc.checkpoint import RunStatus
from loomc.convergence import predicate, score_plateau, semantic_delta
from loomc.loop import Escalate, LoopRunner, StallError, loop

__all__ = [
    "loop",
    "Budget",
    "BudgetExhausted",
    "BestScore",
    "LastGood",
    "Escalate",
    "StallError",
    "LoopRunner",
    "RunStatus",
    "score_plateau",
    "semantic_delta",
    "predicate",
]
