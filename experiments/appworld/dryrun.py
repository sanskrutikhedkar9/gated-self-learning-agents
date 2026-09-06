"""No-model AppWorld integration check used before paid experiments."""

from __future__ import annotations

import argparse

from self_learning_flows.adapters.appworld import (
    appworld_tool_schemas_from_docs,
    build_appworld_tool_registry,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--experiment-name", default="slf_registry_dryrun")
    args = parser.parse_args()

    from appworld import AppWorld

    with AppWorld(
        task_id=args.task_id,
        experiment_name=args.experiment_name,
        load_ground_truth=True,
    ) as world:
        schemas = appworld_tool_schemas_from_docs(world.task.api_docs.function_calling())
        registry = build_appworld_tool_registry(world.apis, schemas)
        print(f"schemas={len(schemas)} registered={len(registry.names())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
