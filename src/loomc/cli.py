"""CLI for loomc — inspect and manage loop runs."""

from __future__ import annotations

import argparse
import json
import sys

from loomc.checkpoint import SQLiteCheckpointStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="loomc", description="Manage agent loop runs")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("runs", help="List all runs")

    show_p = sub.add_parser("show", help="Show step-by-step details for a run")
    show_p.add_argument("run_id")

    resume_p = sub.add_parser("resume", help="Show how to resume a run")
    resume_p.add_argument("run_id")

    export_p = sub.add_parser("export", help="Export run data as JSON")
    export_p.add_argument("run_id")
    export_p.add_argument("--json", action="store_true", default=True)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    try:
        store = SQLiteCheckpointStore()
    except Exception:
        print("No checkpoint database found in .loomc/checkpoints.db", file=sys.stderr)
        sys.exit(1)

    if args.command == "runs":
        _cmd_runs(store)
    elif args.command == "show":
        _cmd_show(store, args.run_id)
    elif args.command == "resume":
        _cmd_resume(store, args.run_id)
    elif args.command == "export":
        _cmd_export(store, args.run_id)


def _cmd_runs(store: SQLiteCheckpointStore) -> None:
    runs = store.list_runs()
    if not runs:
        print("No runs found.")
        return

    print(f"{'ID':<14} {'STATUS':<18} {'STEPS':>5}  {'COST':>8}  {'CREATED'}")
    print("-" * 72)
    for run in runs:
        steps = store.load_steps(run.id)
        total_cost = sum(s.cost_usd for s in steps)
        print(
            f"{run.id:<14} {run.status.value:<18} {len(steps):>5}  ${total_cost:>7.4f}  {run.created_at[:19]}"
        )


def _cmd_show(store: SQLiteCheckpointStore, run_id: str) -> None:
    run = store.get_run(run_id)
    if run is None:
        print(f"Run {run_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    steps = store.load_steps(run_id)
    print(f"Run: {run.id}  Status: {run.status.value}")
    print(f"Created: {run.created_at}  Updated: {run.updated_at}")
    print()

    if not steps:
        print("No steps recorded.")
        return

    cumulative = 0.0
    print(f"{'STEP':>4}  {'COST':>8}  {'CUMUL':>8}  {'SCORE':>7}  {'ERROR'}")
    print("-" * 56)
    for s in steps:
        cumulative += s.cost_usd
        score_str = f"{s.score:.4f}" if s.score is not None else "   -"
        error_str = s.error[:30] if s.error else ""
        print(
            f"{s.step_n:>4}  ${s.cost_usd:>7.4f}  ${cumulative:>7.4f}  {score_str:>7}  {error_str}"
        )


def _cmd_resume(store: SQLiteCheckpointStore, run_id: str) -> None:
    run = store.get_run(run_id)
    if run is None:
        print(f"Run {run_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    steps = store.load_steps(run_id)
    print(f"Run {run.id} — {run.status.value}, {len(steps)} steps")
    print()
    print("To resume programmatically:")
    print()
    print(f'    result = my_loop_fn.resume("{run_id}")')
    print()
    print("The .resume() method loads the last checkpoint and continues the loop.")


def _cmd_export(store: SQLiteCheckpointStore, run_id: str) -> None:
    run = store.get_run(run_id)
    if run is None:
        print(f"Run {run_id!r} not found.", file=sys.stderr)
        sys.exit(1)

    steps = store.load_steps(run_id)
    data = {
        "run": run.model_dump(),
        "steps": [s.model_dump(mode="json") for s in steps],
    }
    print(json.dumps(data, indent=2, default=str))


if __name__ == "__main__":
    main()
