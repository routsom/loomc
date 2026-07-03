
<div align="center"><pre>
    ██╗      ██████╗  ██████╗ ███╗   ███╗ ██████╗
    ██║     ██╔═══██╗██╔═══██╗████╗ ████║██╔════╝
    ██║     ██║   ██║██║   ██║██╔████╔██║██║
    ██║     ██║   ██║██║   ██║██║╚██╔╝██║██║
    ███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║╚██████╗
    ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝ ╚═════╝
The Circuit Breaker Your Agent Loop Never Had
</pre></div>

**Declarative loop specs compiled into resilient agent execution**

[![CI](https://github.com/routsom/loomc/actions/workflows/ci.yml/badge.svg)](https://github.com/routsom/loomc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/loomc)](https://pypi.org/project/loomc/)
[![Python](https://img.shields.io/pypi/pyversions/loomc)](https://pypi.org/project/loomc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

> Your $4 refinement loop crashed at step 9 of 10. You have no idea what happened. You start again from scratch.

**loomc** is a Python decorator that turns a single-iteration function into a resilient agent loop — with hard budget caps, SQLite checkpointing after every step, and full resume from wherever it stopped.

---

## The problem

Iterative agent loops — refine-until-done, self-critique, agentic code fixers — are the hardest part of agent engineering to get right. Not the LLM call. The loop around it.

Three things go wrong constantly:

**1. No crash recovery.** Your loop runs 9 of 10 iterations, spends $3.80, then crashes. You lose everything and restart from zero. There is no checkpoint. There is no resume.

**2. No visibility.** The loop finishes (or doesn't). You get a final state. You have no idea which iteration produced the best result, what each step cost, or why it stopped.

**3. No hard budget.** You set `max_iterations=15` and hope. There's no wall-clock limit, no USD cap, no way to know you're halfway through your budget mid-run.

## The solution

Write one iteration. loomc runs the loop.

```python
from loomc import loop, Budget, BestScore, score_plateau

@loop(
    budget=Budget(usd=2.00, iterations=15, wall_clock="10m"),
    checkpoint="sqlite",       # persisted after every step
    converge_on=score_plateau(patience=3),
    backtrack=BestScore(),
)
def refine_report(state):
    ...  # your LLM call here — one iteration
```

If it crashes: `refine_report.resume("run_id")` — picks up from the last checkpoint, budget intact.

If you want to inspect it mid-run: `loomc show <run_id>` — every step, every cost, every score.

If the budget runs out: it stops cleanly, status recorded, state preserved.

---

## Install

```bash
pip install loomc
```

---

## What it looks like

Your loop crashed. Check what happened, resume from where it stopped:

```text
$ loomc runs

ID             STATUS    STEPS      COST  CREATED
--------------------------------------------------------------------
a1b2c3d4e5f6   failed        3   $0.1500  2026-07-03T12:00:19
```

```text
$ loomc show a1b2c3d4e5f6

Run: a1b2c3d4e5f6  Status: failed
STEP      COST     CUMUL    SCORE  ERROR
--------------------------------------------------------
   0   $0.0500   $0.0500   0.2500
   1   $0.0500   $0.1000   0.5000
   2   $0.0500   $0.1500   0.7500
   3   $0.0000   $0.1500      -    RuntimeError: rate limit
```

```python
result = refine_report.resume("a1b2c3d4e5f6")  # picks up from step 3, $1.85 still in budget
```

A completed run with backtracking:

```text
$ loomc show b2c3d4e5f6a1

Run: b2c3d4e5f6a1  Status: completed
STEP      COST     CUMUL    SCORE  ERROR
--------------------------------------------------------
   0   $0.0500   $0.0500   0.2500
   1   $0.0500   $0.1000   0.5000
   2   $0.0500   $0.1500   0.7500
   3   $0.0500   $0.2000   0.7400  <- backtrack to step 2
   4   $0.0500   $0.2500   0.8100
   5   $0.0500   $0.3000   0.9200
   6   $0.0500   $0.3500   1.0000  <- converged
```

---

## Quick start

### Iterative code fixer

```python
from pydantic import BaseModel
from loomc import loop, Budget, BestScore, predicate

class State(BaseModel):
    code: str
    iteration: int = 0

def scorer(state) -> float:
    # fraction of tests passing
    ...

@loop(
    converge_on=predicate(lambda h: (h[-1].score or 0) >= 1.0),
    budget=Budget(usd=1.00, iterations=10),
    checkpoint="sqlite",
    backtrack=BestScore(),
    scorer=scorer,
    cost_fn=lambda s: 0.08,
)
def fix_code(state: State) -> State:
    # call your LLM here — one iteration
    ...

result = fix_code(State(code=BUGGY_CODE))
```

---

## Features

- **Checkpoint every step** — SQLite, automatic, zero config. `.loomc/checkpoints.db` in your working directory.
- **Resume anywhere** — `my_fn.resume("run_id")` continues from the last successful step. Budget accounting carries over.
- **Hard budget caps** — USD, iteration count, wall clock. Enforced *before* each call — the loop never starts a step it can't afford.
- **Full step history** — `loomc show <id>` gives you cost, score, and error for every iteration. Know exactly why a loop stopped.
- **Convergence detection** — `score_plateau`, `predicate`, `semantic_delta`, `judge` (LLM-as-judge)
- **Cost adapters** — `AnthropicAdapter` and `OpenAIAdapter` extract token costs from responses automatically
- **Async support** — decorate `async def` iteration functions natively; `AsyncLoopRunner` handles the event loop
- **Nested loops** — child loops propagate costs to the parent ledger via `contextvars`; parent budget sees total spend
- **Backtracking** — `BestScore` restores the highest-scoring checkpoint on regression. `LastGood` restores on error.
- **Stall detection** — `Escalate(after=3)` raises `StallError` when no progress for N steps. Catch it to swap models, escalate to human, or abort cleanly.
- **LLM agnostic** — loomc never calls an LLM. It wraps your code. Bring any model, any SDK.

---

## Budget

```python
Budget(
    usd=2.00,           # hard stop on cumulative cost
    iterations=15,      # hard stop on iteration count
    wall_clock="10m",   # hard stop on elapsed time (10m / 1h / 30s)
)
```

Budgets are enforced **before** each iteration — the loop never starts a call it can't afford to make. Provide `cost_fn` to extract cost from your iteration's return value, or wire in Anthropic/OpenAI usage metadata directly.

---

## Convergence detectors

| Detector | Behaviour |
| --- | --- |
| `score_plateau(patience=3, min_delta=0.01)` | Stops when score hasn't improved by `min_delta` for `patience` consecutive steps |
| `semantic_delta(threshold=0.05)` | Stops when embedding distance between consecutive outputs drops below threshold |
| `judge(judge_fn)` | Calls `judge_fn(state) -> bool` — wire in an LLM-as-judge via `AnthropicAdapter.judge()` or `OpenAIAdapter.judge()` |
| `predicate(fn)` | Wraps any `fn(history) -> bool` |

---

## Backtracking strategies

| Strategy | Behaviour |
| --- | --- |
| `BestScore()` | Restores the highest-scoring checkpoint whenever score regresses |
| `LastGood()` | Restores the last non-error checkpoint on exception |

---

## Stall detection

```python
@loop(
    on_stall=Escalate(after=3),   # raises StallError after 3 steps of no improvement
    scorer=my_scorer,
    ...
)
def my_fn(state): ...
```

Catch `StallError` to implement your own escalation path (human-in-the-loop, model switch, etc.).

---

## CLI reference

```text
loomc runs                         List all runs (id, status, steps, cost, created)
loomc show <run_id>                Step-by-step cost/score table for a run
loomc resume <run_id>              Print the code snippet to resume a run
loomc export <run_id> --json       Export full run data as JSON
```

---

## Checkpoint storage

Everything lives in `.loomc/checkpoints.db` — a plain SQLite file. No cloud, no server, no dashboard required.

| Table | Contents |
| --- | --- |
| `runs` | `id, spec_hash, status, created_at, updated_at` |
| `steps` | `run_id, step_n, state_blob (JSON), cost_usd, score, timestamp, error` |

State must be JSON-serialisable (dict or Pydantic model). Bring your own store by implementing the `CheckpointStore` protocol.

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## What's shipped

All three milestones are complete as of v0.2.0.

| Milestone | What shipped |
| --- | --- |
| **M1** | `@loop` decorator, USD/iteration/wall-clock budgets, SQLite checkpointing, resume, `score_plateau`, `BestScore`, `LastGood`, `Escalate`, CLI |
| **M2** | `semantic_delta` (sentence-transformers), `judge` (LLM-as-judge), `AnthropicAdapter`, `OpenAIAdapter` |
| **M3** | Async support (`AsyncLoopRunner`), nested loop budget propagation via `contextvars` |

**What's next** — custom checkpoint backends (Postgres, S3), streaming step callbacks, a web UI for `loomc runs`, and first-class multi-agent orchestration where each agent is a `@loop`.

---

## License

MIT
