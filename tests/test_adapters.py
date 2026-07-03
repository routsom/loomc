"""Tests for cost adapters and judge convergence."""

import pytest

from loomc.adapters import AnthropicAdapter, OpenAIAdapter
from loomc.convergence import JudgeDetector
from loomc._types import StepRecord


class FakeUsage:
    def __init__(self, input_tokens=0, output_tokens=0, prompt_tokens=0, completion_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class FakeResponse:
    def __init__(self, **kwargs):
        self.usage = FakeUsage(**kwargs)


class TestAnthropicAdapter:
    def test_cost_sonnet(self):
        adapter = AnthropicAdapter("claude-sonnet-4-6")
        response = FakeResponse(input_tokens=1_000_000, output_tokens=1_000_000)
        assert adapter.cost_from_response(response) == pytest.approx(18.00)

    def test_cost_haiku(self):
        adapter = AnthropicAdapter("claude-haiku-4-5-20251001")
        response = FakeResponse(input_tokens=1_000_000, output_tokens=1_000_000)
        assert adapter.cost_from_response(response) == pytest.approx(4.80)

    def test_cost_fn_dict_state(self):
        adapter = AnthropicAdapter("claude-sonnet-4-6")
        response = FakeResponse(input_tokens=500_000, output_tokens=250_000)
        cost = adapter.cost_fn({"text": "result", "_response": response})
        assert cost == pytest.approx(1.50 + 3.75)

    def test_cost_fn_no_response(self):
        assert AnthropicAdapter().cost_fn({"text": "no response"}) == 0.0

    def test_cost_fn_no_usage(self):
        assert AnthropicAdapter().cost_from_response(object()) == 0.0

    def test_unknown_model_uses_default(self):
        adapter = AnthropicAdapter("claude-unknown")
        response = FakeResponse(input_tokens=1_000_000, output_tokens=0)
        assert adapter.cost_from_response(response) == pytest.approx(3.00)

    def test_judge_returns_detector(self):
        class FakeContent:
            text = "YES"

        class FakeClient:
            class messages:
                @staticmethod
                def create(**kwargs):
                    class R:
                        content = [FakeContent()]
                    return R()

        detector = AnthropicAdapter("claude-haiku-4-5-20251001").judge(
            client=FakeClient(), prompt="Done?"
        )
        assert isinstance(detector, JudgeDetector)
        history = [StepRecord(step_n=0, state={"text": "result"})]
        assert detector.should_stop(history) is True

    def test_judge_no_on_empty_history(self):
        assert AnthropicAdapter().judge(client=object()).should_stop([]) is False


class TestOpenAIAdapter:
    def test_cost_gpt4o(self):
        adapter = OpenAIAdapter("gpt-4o")
        response = FakeResponse(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert adapter.cost_from_response(response) == pytest.approx(12.50)

    def test_cost_mini(self):
        adapter = OpenAIAdapter("gpt-4o-mini")
        response = FakeResponse(prompt_tokens=1_000_000, completion_tokens=1_000_000)
        assert adapter.cost_from_response(response) == pytest.approx(0.75)

    def test_cost_fn_dict_state(self):
        adapter = OpenAIAdapter("gpt-4o")
        response = FakeResponse(prompt_tokens=100_000, completion_tokens=50_000)
        cost = adapter.cost_fn({"text": "result", "_response": response})
        assert cost == pytest.approx(0.25 + 0.50)

    def test_cost_fn_no_response(self):
        assert OpenAIAdapter().cost_fn({"text": "no response"}) == 0.0


class TestJudgeDetector:
    def test_stops_when_true(self):
        detector = JudgeDetector(lambda s: s.get("done", False))
        assert detector.should_stop([StepRecord(step_n=0, state={"done": True})]) is True

    def test_continues_when_false(self):
        detector = JudgeDetector(lambda s: s.get("done", False))
        assert detector.should_stop([StepRecord(step_n=0, state={"done": False})]) is False

    def test_skips_error_steps(self):
        detector = JudgeDetector(lambda s: True)
        assert detector.should_stop([StepRecord(step_n=0, state={}, error="RuntimeError: oops")]) is False
