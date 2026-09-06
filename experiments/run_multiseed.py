"""Run multiple chronological task orders and aggregate core metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from mini_office.benchmark import BenchmarkConfig, run_benchmark


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/mini_office_multiseed.json"),
    )
    args = parser.parse_args()
    runs = [run_benchmark(BenchmarkConfig(seed=seed)).to_dict() for seed in args.seeds]
    metric_names = [
        "task_success_rate",
        "full_agent_calls_avoided_rate",
        "token_reduction_rate",
        "workflow_tasks",
        "model_free_workflow_tasks",
        "workflow_fallbacks",
        "offer_false_positive_rate",
    ]
    aggregate = {
        name: {
            "mean": statistics.mean(run[name] for run in runs),
            "stdev": statistics.stdev(run[name] for run in runs) if len(runs) > 1 else 0.0,
        }
        for name in metric_names
    }
    for run in runs:
        run.pop("routes", None)
    result = {"seeds": args.seeds, "aggregate": aggregate, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"seeds": args.seeds, "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
