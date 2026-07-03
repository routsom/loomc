"""loomc — Declarative loop specs compiled into resilient agent execution."""

from loomc.adapters import AnthropicAdapter, OpenAIAdapter
from loomc.backtrack import BestScore, LastGood
from loomc.budget import Budget, BudgetExhausted
from loomc.checkpoint import RunStatus
from loomc.convergence import JudgeDetector, judge, predicate, score_plateau, semantic_delta
from loomc.loop import AsyncLoopRunner, Escalate, LoopRunner, StallError, loop

__all__ = [
    "loop",
    "Budget",
    "BudgetExhausted",
    "BestScore",
    "LastGood",
    "Escalate",
    "StallError",
    "LoopRunner",
    "AsyncLoopRunner",
    "RunStatus",
    "score_plateau",
    "semantic_delta",
    "judge",
    "predicate",
    "JudgeDetector",
    "AnthropicAdapter",
    "OpenAIAdapter",
]
