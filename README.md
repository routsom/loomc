# loomc

> Your $4 refinement loop crashed at step 9 of 10. You have no idea what happened. You start again from scratch.

**loomc** is a Python decorator that turns a single-iteration function into a resilient agent loop — with hard budget caps, SQLite checkpointing after every step, and full resume from wherever it stopped.

[![CI](https://github.com/routsom/loomc/actions/workflows/ci.yml/badge.svg)](https://github.com/routsom/loomc/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/loomc)](https://pypi.org/project/loomc/)
[![Python](https://img.shields.io/pypi/pyversions/loomc)](https://pypi.org/project/loomc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

Resume is a first-class workflow — not just crash recovery. Start a loop, inspect it, decide to continue:

```text
$ loomc runs

ID             STATUS              STEPS      COST  CREATED
------------------------------------------------------------------------
a1b2c3d4e5f6   failed                  3   $0.1500  2026-07-02T12:00:19
```

```text
$ loomc show a1b2c3d4e5f6

Run: a1b2c3d4e5f6  Status: failed — crashed at step 3, $1.85 of $2.00 remaining

STEP      COST     CUMUL    SCORE  ERROR
--------------------------------------------------------
   0   $0.0500   $0.0500   0.2500
   1   $0.0500   $0.1000   0.5000
   2   $0.0500   $0.1500   0.7500
   3   $0.0000   $0.1500      -    RuntimeError: rate limit
```

```python
result = refine_report.resume("a1b2c3d4e5f6")  # continues from step 3
```

---

**`loomc runs`** — all runs in the local checkpoint DB:

```text
ID             STATUS              STEPS      COST  CREATED
------------------------------------------------------------------------
a1b2c3d4e5f6   completed               7   $0.3500  2026-07-02T12:00:19
9f8e7d6c5b4a   budget_exhausted       10   $1.0012  2026-07-02T11:47:03
3c2b1a0f9e8d   stalled                 4   $0.1600  2026-07-02T11:31:44
```

**`loomc show a1b2c3d4e5f6`** — step-by-step cost and score:

```text
Run: a1b2c3d4e5f6  Status: completed
Created: 2026-07-02T12:00:00+00:00

STEP      COST     CUMUL    SCORE  ERROR
--------------------------------------------------------
   0   $0.0500   $0.0500   0.2500
   1   $0.0500   $0.1000   0.5000
   2   $0.0500   $0.1500   0.7500
   3   $0.0500   $0.2000   0.7400  <- backtrack triggered
   4   $0.0500   $0.2500   0.8100
   5   $0.0500   $0.3000   0.9200
   6   $0.0500   $0.3500   1.0000  <- converged
```

**`loomc export a1b2c3d4e5f6 --json`** — full run as JSON for analysis.

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

### Resume a crashed run

```bash
$ loomc resume a1b2c3d4e5f6
Run a1b2c3d4e5f6 — failed, 3 steps

To resume programmatically:

    result = fix_code.resume("a1b2c3d4e5f6")
```

```python
result = fix_code.resume("a1b2c3d4e5f6")
```

The loop picks up from the last checkpoint. Budget accounting continues from where it left off.

---

## Features

- **Checkpoint every step** — SQLite, automatic, zero config. `.loomc/checkpoints.db` in your working directory.
- **Resume anywhere** — `my_fn.resume("run_id")` continues from the last successful step. Budget accounting carries over.
- **Hard budget caps** — USD, iteration count, wall clock. Enforced *before* each call — the loop never starts a step it can't afford.
- **Full step history** — `loomc show <id>` gives you cost, score, and error for every iteration. Know exactly why a loop stopped.
- **Convergence detection** — `score_plateau`, `predicate`, `semantic_delta` *(M2)*
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
| `semantic_delta(threshold=0.05)` | *(M2)* Stops when embedding distance between consecutive outputs drops below threshold |
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

## Roadmap

- **M1 (shipped)** — decorator, USD/iteration/wall-clock budgets, SQLite checkpointing, resume, `score_plateau`, `BestScore`, `LastGood`, `Escalate`, CLI
- **M2** — `semantic_delta` convergence, `judge` (LLM-as-judge), Anthropic/OpenAI cost adapters
- **M3** — nested loops with budget ledger, async support

---

## License

MIT
