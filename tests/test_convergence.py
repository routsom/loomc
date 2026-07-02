"""Tests for convergence detectors."""

from loomc._types import StepRecord
from loomc.convergence import predicate, score_plateau


def _make_history(scores: list[float]) -> list[StepRecord]:
    return [
        StepRecord(step_n=i, state={}, score=s)
        for i, s in enumerate(scores)
    ]


class TestScorePlateau:
    def test_converges_on_plateau(self):
        detector = score_plateau(patience=3, min_delta=0.01)
        history = _make_history([0.5, 0.7, 0.8, 0.8, 0.8, 0.8])
        assert detector.should_stop(history) is True

    def test_no_convergence_while_improving(self):
        detector = score_plateau(patience=3, min_delta=0.01)
        history = _make_history([0.1, 0.3, 0.5, 0.7])
        assert detector.should_stop(history) is False

    def test_not_enough_history(self):
        detector = score_plateau(patience=3, min_delta=0.01)
        history = _make_history([0.5, 0.5])
        assert detector.should_stop(history) is False

    def test_custom_patience(self):
        detector = score_plateau(patience=2, min_delta=0.01)
        history = _make_history([0.5, 0.8, 0.8, 0.8])
        assert detector.should_stop(history) is True


class TestPredicate:
    def test_custom_predicate(self):
        detector = predicate(lambda h: len(h) >= 3)
        assert detector.should_stop(_make_history([0.1, 0.2])) is False
        assert detector.should_stop(_make_history([0.1, 0.2, 0.3])) is True
