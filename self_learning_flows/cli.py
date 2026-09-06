"""Command-line interface for the package and reproducible benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .storage import SQLiteStore


def _benchmark(args: argparse.Namespace) -> int:
    from mini_office.benchmark import BenchmarkConfig, run_benchmark
    from mini_office.dataset import DEFAULT_DATASET

    summary = run_benchmark(
        BenchmarkConfig(
            dataset_path=args.dataset or DEFAULT_DATASET,
            output_path=args.output,
            seed=args.seed,
        )
    )
    output = summary.to_dict()
    if not args.verbose:
        output.pop("routes", None)
    print(json.dumps(output, indent=2))
    return 0 if summary.successful_tasks == summary.tasks else 1


def _list_workflows(args: argparse.Namespace) -> int:
    workflows = SQLiteStore(args.database).list_workflows(scope=args.scope)
    print(
        json.dumps(
            [
                {
                    "id": workflow.workflow_id,
                    "name": workflow.name,
                    "status": str(workflow.status),
                    "version": workflow.version,
                    "observations": workflow.stats.pattern_observations,
                    "executions": workflow.stats.executions,
                    "reliability": round(workflow.stats.reliability, 4),
                }
                for workflow in workflows
            ],
            indent=2,
        )
    )
    return 0


def _demo(_: argparse.Namespace) -> int:
    from mini_office.agent import ReferenceAgent
    from mini_office.dataset import load_tasks
    from mini_office.world import MiniOfficeWorld

    from .engine import InMemoryStore, SelfLearningFlowEngine
    from .models import TaskRequest

    tasks = [task for task in load_tasks() if task.pattern == "reschedule_and_notify"]
    engine = SelfLearningFlowEngine(InMemoryStore())
    for task in tasks[:3]:
        world = MiniOfficeWorld()
        world.seed(task.variables)
        engine.observe(ReferenceAgent(world).run(task))
    target = tasks[3]
    world = MiniOfficeWorld()
    world.seed(target.variables)
    proposal = engine.propose(
        TaskRequest(
            target.instruction,
            "mini-office",
            target.workflow_variables,
            environment="mini-office-v1",
            available_tools=world.tool_registry().names(),
        )
    )
    if proposal is None:
        print("No workflow proposed")
        return 1
    print(json.dumps(proposal.confirmation_payload(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="self-learning-flows")
    commands = parser.add_subparsers(dest="command", required=True)

    benchmark = commands.add_parser("benchmark", help="run the Mini Office stream")
    benchmark.add_argument("--dataset", type=Path, help="JSONL dataset; defaults to packaged v1")
    benchmark.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/mini_office_v1.json"),
    )
    benchmark.add_argument("--verbose", action="store_true", help="print per-task routes")
    benchmark.add_argument("--seed", type=int, help="shuffle the chronological stream reproducibly")
    benchmark.set_defaults(function=_benchmark)

    listing = commands.add_parser("list", help="list workflows in a SQLite store")
    listing.add_argument("--database", default=".self_learning_flows/workflows.db")
    listing.add_argument("--scope")
    listing.set_defaults(function=_list_workflows)

    demo = commands.add_parser("demo", help="print a real workflow confirmation payload")
    demo.set_defaults(function=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.function(args)
