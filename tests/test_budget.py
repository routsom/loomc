"""Tests for budget enforcement."""

import pytest

from loomc.budget import Budget, BudgetExhausted, BudgetLedger, _parse_duration


class TestBudgetParsing:
    def test_parse_seconds(self):
        assert _parse_duration("30s") == 30.0

    def test_parse_minutes(self):
        assert _parse_duration("10m") == 600.0

    def test_parse_hours(self):
        assert _parse_duration("1h") == 3600.0

    def test_parse_invalid(self):
        with pytest.raises(ValueError):
            _parse_duration("10x")

    def test_wall_clock_property(self):
        b = Budget(wall_clock="5m")
        assert b.wall_clock_seconds == 300.0

    def test_wall_clock_none(self):
        b = Budget()
        assert b.wall_clock_seconds is None


class TestBudgetLedger:
    def test_iteration_limit(self):
        ledger = BudgetLedger()
        budget = Budget(iterations=3)
        for _ in range(3):
            ledger.check_pre(budget)
            ledger.record(0.0)
        with pytest.raises(BudgetExhausted, match="iterations"):
            ledger.check_pre(budget)

    def test_usd_limit(self):
        ledger = BudgetLedger()
        budget = Budget(usd=1.0)
        ledger.record(0.6)
        ledger.check_pre(budget)  # still under
        ledger.record(0.5)
        with pytest.raises(BudgetExhausted, match="usd"):
            ledger.check_pre(budget)

    def test_remaining(self):
        ledger = BudgetLedger()
        budget = Budget(usd=2.0, iterations=10)
        ledger.record(0.5)
        ledger.record(0.3)
        rem = ledger.remaining(budget)
        assert rem.usd == pytest.approx(1.2)
        assert rem.iterations == 8
