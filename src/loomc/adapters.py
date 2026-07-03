"""Cost adapters and LLM-as-judge helpers for Anthropic and OpenAI."""

from __future__ import annotations

from typing import Any, Callable

# --- Pricing tables (USD per 1M tokens, as of 2026-07) ---

_ANTHROPIC_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "_default": {"input": 3.00, "output": 15.00},
}

_OPENAI_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "_default": {"input": 2.50, "output": 10.00},
}


class AnthropicAdapter:
    """Cost extraction and LLM-as-judge for Anthropic response objects.

    Usage::

        from anthropic import Anthropic
        from loomc import loop, Budget, AnthropicAdapter

        adapter = AnthropicAdapter("claude-sonnet-4-6")
        client = Anthropic()

        @loop(
            budget=Budget(usd=2.00),
            cost_fn=adapter.cost_fn,
            converge_on=adapter.judge(
                client=client,
                prompt="Reply YES if the report is publication-ready, NO otherwise.",
            ),
        )
        def refine(state):
            response = client.messages.create(...)
            return {**state, "text": response.content[0].text, "_response": response}
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        pricing = _ANTHROPIC_PRICING.get(model, _ANTHROPIC_PRICING["_default"])
        self._input_rate = pricing["input"]
        self._output_rate = pricing["output"]

    def cost_fn(self, state: Any) -> float:
        """Extract cost from state containing an Anthropic response at ``state['_response']``."""
        response = (
            state.get("_response") if isinstance(state, dict)
            else getattr(state, "_response", None)
        )
        return _anthropic_cost(response, self._input_rate, self._output_rate)

    def cost_from_response(self, response: Any) -> float:
        """Extract cost directly from an Anthropic response object."""
        return _anthropic_cost(response, self._input_rate, self._output_rate)

    def judge(
        self,
        client: Any,
        prompt: str = "Reply YES if the task is complete, NO otherwise. One word only.",
        text_fn: Callable[[Any], str] | None = None,
    ) -> "JudgeDetector":
        """Return a convergence detector that uses this Anthropic client as a judge.

        The judge calls the LLM with the current state's text representation
        and stops the loop when the response starts with 'Y'.
        """
        from loomc.convergence import JudgeDetector

        def _judge_fn(state: Any) -> bool:
            text = text_fn(state) if text_fn else str(state)
            response = client.messages.create(
                model=self.model,
                max_tokens=8,
                messages=[{"role": "user", "content": f"{prompt}\n\n---\n{text}"}],
            )
            return response.content[0].text.strip().upper().startswith("Y")

        return JudgeDetector(_judge_fn)


class OpenAIAdapter:
    """Cost extraction and LLM-as-judge for OpenAI response objects.

    Usage::

        from openai import OpenAI
        from loomc import loop, Budget, OpenAIAdapter

        adapter = OpenAIAdapter("gpt-4o")
        client = OpenAI()

        @loop(
            budget=Budget(usd=2.00),
            cost_fn=adapter.cost_fn,
            converge_on=adapter.judge(client=client, prompt="Is this complete? YES or NO."),
        )
        def refine(state):
            response = client.chat.completions.create(...)
            return {**state, "text": response.choices[0].message.content, "_response": response}
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        pricing = _OPENAI_PRICING.get(model, _OPENAI_PRICING["_default"])
        self._input_rate = pricing["input"]
        self._output_rate = pricing["output"]

    def cost_fn(self, state: Any) -> float:
        """Extract cost from state containing an OpenAI response at ``state['_response']``."""
        response = (
            state.get("_response") if isinstance(state, dict)
            else getattr(state, "_response", None)
        )
        return _openai_cost(response, self._input_rate, self._output_rate)

    def cost_from_response(self, response: Any) -> float:
        """Extract cost directly from an OpenAI response object."""
        return _openai_cost(response, self._input_rate, self._output_rate)

    def judge(
        self,
        client: Any,
        prompt: str = "Reply YES if the task is complete, NO otherwise. One word only.",
        text_fn: Callable[[Any], str] | None = None,
    ) -> "JudgeDetector":
        """Return a convergence detector that uses this OpenAI client as a judge."""
        from loomc.convergence import JudgeDetector

        def _judge_fn(state: Any) -> bool:
            text = text_fn(state) if text_fn else str(state)
            response = client.chat.completions.create(
                model=self.model,
                max_tokens=8,
                messages=[{"role": "user", "content": f"{prompt}\n\n---\n{text}"}],
            )
            return response.choices[0].message.content.strip().upper().startswith("Y")

        return JudgeDetector(_judge_fn)


# --- Internal helpers ---

def _anthropic_cost(response: Any, input_rate: float, output_rate: float) -> float:
    if response is None:
        return 0.0
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate


def _openai_cost(response: Any, input_rate: float, output_rate: float) -> float:
    if response is None:
        return 0.0
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
