"""Example: Iterative code fixer that refines until tests pass.

This simulates an LLM fixing buggy code iteration by iteration,
under a $1.00 budget. Each iteration fixes one bug.

Run:
    pip install -e .
    python examples/code_fixer.py

Output looks like:
    Step 0: score=0.25, cost=$0.08  (1/4 tests passing)
    Step 1: score=0.50, cost=$0.08  (2/4 tests passing)
    Step 2: score=0.75, cost=$0.08  (3/4 tests passing)
    Step 3: score=1.00, cost=$0.08  (4/4 tests passing)
    Converged! All tests pass.

    Total cost: $0.32 / $1.00 budget
    Run `loomc show <run_id>` to inspect step-by-step.
"""

from __future__ import annotations

from pydantic import BaseModel

from loomc import Budget, BestScore, loop, predicate, score_plateau

# --- State ---

BUGGY_CODE = """\
def add(a, b):
    return a - b  # bug: subtraction instead of addition

def multiply(a, b):
    return a // b  # bug: floor division instead of multiplication

def greet(name):
    return f"hello {name}"  # bug: lowercase 'h'

def is_even(n):
    return n % 2 == 1  # bug: checks odd instead of even
"""

FIXES = [
    ("return a - b", "return a + b"),
    ("return a // b", "return a * b"),
    ('return f"hello {name}"', 'return f"Hello {name}"'),
    ("return n % 2 == 1", "return n % 2 == 0"),
]

TESTS = [
    ("add(2, 3) == 5", "add"),
    ("multiply(3, 4) == 12", "multiply"),
    ('greet("world") == "Hello world"', "greet"),
    ("is_even(4) == True", "is_even"),
]


class FixerState(BaseModel):
    code: str
    test_output: str = ""
    iteration: int = 0
    passing: bool = False


# --- Scorer ---

def _get_code(state) -> str:
    if isinstance(state, FixerState):
        return state.code
    if isinstance(state, dict):
        return state.get("code", "")
    return ""


def score_code(state) -> float:
    """Run tests against the code, return fraction passing."""
    ns: dict = {}
    try:
        exec(_get_code(state), ns)
    except Exception:
        return 0.0
    passed = 0
    for test_expr, _ in TESTS:
        try:
            if eval(test_expr, ns):
                passed += 1
        except Exception:
            pass
    return passed / len(TESTS)


# --- Simulated LLM iteration ---

@loop(
    converge_on=predicate(lambda h: len(h) > 0 and (h[-1].score or 0) >= 1.0),
    budget=Budget(usd=1.00, iterations=10),
    checkpoint="sqlite",
    backtrack=BestScore(),
    scorer=score_code,
    cost_fn=lambda s: 0.08,  # simulated $0.08 per LLM call
)
def fix_code(state: FixerState) -> FixerState:
    """One iteration: apply the next fix (simulating an LLM edit)."""
    if isinstance(state, dict):
        state = FixerState(**state)
    code = state.code
    iteration = state.iteration

    if iteration < len(FIXES):
        old, new = FIXES[iteration]
        code = code.replace(old, new, 1)

    # Run tests to build output
    ns: dict = {}
    results = []
    try:
        exec(code, ns)
        for test_expr, name in TESTS:
            try:
                ok = eval(test_expr, ns)
                results.append(f"  {'PASS' if ok else 'FAIL'}: {name}")
            except Exception as e:
                results.append(f"  ERROR: {name} — {e}")
    except Exception as e:
        results.append(f"  COMPILE ERROR: {e}")

    test_output = "\n".join(results)
    passing = all("PASS" in r for r in results)

    return FixerState(
        code=code,
        test_output=test_output,
        iteration=iteration + 1,
        passing=passing,
    )


if __name__ == "__main__":
    print("Starting iterative code fixer...")
    print(f"Budget: $1.00 / 10 iterations\n")

    initial = FixerState(code=BUGGY_CODE)

    try:
        result = fix_code(initial)
    except Exception as e:
        print(f"Loop ended: {e}")
        result = None

    if result:
        score = score_code(result)
        print(f"\nFinal score: {score:.0%}")
        print(f"Iterations: {result.iteration}")
        print(f"Tests passing: {'YES' if result.passing else 'NO'}")
        print(f"\nFinal test output:\n{result.test_output}")
        print(f"\nRun `loomc runs` to see all runs.")
