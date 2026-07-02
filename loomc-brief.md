# loomc — A Compiler for Agent Loops

## One-liner
Declarative loop specs compiled into resilient agent execution: budgets, convergence detection, checkpointing, backtracking, resumability — so nobody hand-rolls `while True: llm()` again.

## Problem
Every agent team reimplements the same inner loop (plan → act → reflect → retry) with ad-hoc budget caps, no checkpoints, no convergence criteria, and no way to resume a crashed run. Control flow is the least-differentiated, most-rewritten code in agent engineering.

## Core Concept
A Python library where the loop contract is declared, not coded:

```python
from loomc import loop, semantic_delta

@loop(
    converge_on=semantic_delta(threshold=0.05),   # stop when output stabilizes
    budget=Budget(usd=2.00, iterations=15, wall_clock="10m"),
    checkpoint="sqlite",                          # every iteration persisted
    backtrack=BestScore(metric=my_scorer),        # regress? restore best checkpoint
    on_stall=Escalate(after=3),                   # no progress → raise to human
)
def refine_report(state: State) -> State:
    ...  # user writes ONE iteration; loomc owns the loop
```

## Key Components
1. **Loop runtime** — executes the decorated iteration fn; enforces budgets pre-call (estimate) and post-call (actual from usage metadata); hard-stops on breach.
2. **Convergence detectors** (pluggable): `semantic_delta` (embedding distance between iteration outputs), `score_plateau` (user metric stops improving), `judge` (LLM-as-judge says done), `predicate` (arbitrary fn).
3. **Checkpoint store** — SQLite default. Schema: `runs(id, spec_hash, status)`, `steps(run_id, n, state_blob, cost, score, ts)`. State must be serializable (pydantic model or dict). `loomc resume <run_id>` continues from last checkpoint.
4. **Backtracking** — strategies: `BestScore` (restore highest-scoring checkpoint on regression), `LastGood` (restore last non-error). Track lineage so a run's step history can branch.
5. **LLM client agnostic** — loomc never calls an LLM itself; it wraps user code. Provide optional cost-extraction adapters for Anthropic/OpenAI SDK response objects; else user supplies `cost_fn`.
6. **CLI** — `loomc runs`, `loomc show <id>` (step-by-step cost/score table), `loomc resume <id>`, `loomc export <id> --json`.

## Architecture Notes
- Pure Python ≥3.10, deps: pydantic, sqlite (stdlib), optional sentence-transformers or API-based embeddings for semantic_delta (make embedding provider pluggable, default to a cheap API call).
- State is the only thing passed between iterations — no hidden globals. Enforce via type signature `fn(State) -> State`.
- Budgets are enforced by the runtime, not trusted to the iteration fn. Wrap iteration in timeout; kill on wall-clock breach.
- Design for nesting: a loop step may itself invoke another loomc loop (sub-agent pattern). Parent budget must cap children (pass a `BudgetLedger` down).

## Milestones
- **M1 (MVP, ~1 week):** decorator, iteration budget + USD budget with manual cost_fn, SQLite checkpointing, resume, `score_plateau` convergence. Ship with one killer example: iterative code-fixer that refines until tests pass under a $1 budget.
- **M2:** semantic_delta + judge convergence, BestScore backtracking, CLI, Anthropic/OpenAI cost adapters.
- **M3:** nested loops with budget ledger, stall detection/escalation hooks, async support.

## Virality Hooks
- README hero: side-by-side of 80 lines of hand-rolled loop vs. 8 lines of loomc.
- `loomc show` output (cost/score per step, backtrack events) is inherently screenshotable.

## Non-Goals (v1)
No prompt management, no multi-agent orchestration graphs (that's LangGraph's turf — loomc is the inner loop), no hosted service.

## Definition of Done (M1)
`pip install loomc`, run the code-fixer example, kill it mid-run, `loomc resume` completes it, budget breach demonstrably halts execution. Tests for budget enforcement and checkpoint round-tripping.
