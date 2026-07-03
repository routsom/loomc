"""Convergence detectors for agent loops."""

from __future__ import annotations

import math
from typing import Any, Callable

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
    """Stop when consecutive iteration outputs stabilize based on embedding distance.

    Requires ``pip install loomc[embeddings]`` (sentence-transformers).

    Args:
        threshold: Cosine distance below which outputs are considered stable.
            Default 0.05 means 95%+ similar.
        text_fn: Callable to extract text from state. Defaults to ``str(state)``.
        model_name: sentence-transformers model. Default ``all-MiniLM-L6-v2``
            (fast, 80MB, good general-purpose embeddings).
    """

    def __init__(
        self,
        threshold: float = 0.05,
        text_fn: Callable[[Any], str] | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.threshold = threshold
        self.text_fn = text_fn or str
        self.model_name = model_name
        self._model: Any = None
        self._cache: dict[int, list[float]] = {}

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "semantic_delta requires sentence-transformers. "
                    "Run: pip install loomc[embeddings]"
                ) from None
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, text: str) -> list[float]:
        return self._get_model().encode(text).tolist()

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
        return 1.0 - (dot / mag if mag > 0 else 0.0)

    def should_stop(self, history: list[StepRecord]) -> bool:
        if len(history) < 2:
            return False
        last, prev = history[-1], history[-2]
        if last.step_n not in self._cache:
            self._cache[last.step_n] = self._embed(self.text_fn(last.state))
        if prev.step_n not in self._cache:
            self._cache[prev.step_n] = self._embed(self.text_fn(prev.state))
        dist = self._cosine_distance(self._cache[prev.step_n], self._cache[last.step_n])
        return dist < self.threshold


class JudgeDetector:
    """Stop when an LLM judge decides the output is done.

    Use via ``AnthropicAdapter.judge()`` or ``OpenAIAdapter.judge()``
    rather than instantiating directly.
    """

    def __init__(self, judge_fn: Callable[[Any], bool]):
        self.judge_fn = judge_fn

    def should_stop(self, history: list[StepRecord]) -> bool:
        if not history or history[-1].error is not None:
            return False
        return self.judge_fn(history[-1].state)


class Predicate:
    """Wrap an arbitrary function as a convergence detector."""

    def __init__(self, fn: Callable[[list[StepRecord]], bool]):
        self.fn = fn

    def should_stop(self, history: list[StepRecord]) -> bool:
        return self.fn(history)


def score_plateau(patience: int = 3, min_delta: float = 0.01) -> ScorePlateau:
    return ScorePlateau(patience=patience, min_delta=min_delta)


def semantic_delta(
    threshold: float = 0.05,
    text_fn: Callable[[Any], str] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
) -> SemanticDelta:
    return SemanticDelta(threshold=threshold, text_fn=text_fn, model_name=model_name)


def judge(judge_fn: Callable[[Any], bool]) -> JudgeDetector:
    """Wrap a callable ``judge_fn(state) -> bool`` as a convergence detector."""
    return JudgeDetector(judge_fn)


def predicate(fn: Callable[[list[StepRecord]], bool]) -> Predicate:
    return Predicate(fn=fn)
