"""No-model AppWorld integration check used before paid experiments."""

from __future__ import annotations

import argparse

from self_learning_flows.adapters.appworld import (
    appworld_tool_schemas_from_docs,
    build_appworld_tool_registry,
)

from .treatment_runner import OpenAICompatibleCodeAgent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--experiment-name", default="slf_registry_dryrun")
    args = parser.parse_args()

    from appworld import AppWorld

    with AppWorld(
        task_id=args.task_id,
        experiment_name=args.experiment_name,
        load_ground_truth=False,
    ) as world:
        docs = world.task.api_docs.function_calling()
        schemas = appworld_tool_schemas_from_docs(docs)
        registry = build_appworld_tool_registry(world.apis, schemas)
        catalog = OpenAICompatibleCodeAgent._api_catalog(world)
        candidates = OpenAICompatibleCodeAgent._relevant_api_contracts(world, catalog)
        candidate_names = [str(contract["call"]) for contract in candidates]
        unknown = sorted(set(candidate_names) - set(catalog))
        if unknown:
            raise RuntimeError(f"candidate selector returned unknown APIs: {unknown}")
        if "apis.supervisor.complete_task" not in candidate_names:
            raise RuntimeError("candidate selector omitted supervisor completion API")
        print(
            f"schemas={len(schemas)} registered={len(registry.names())} "
            f"catalog={len(catalog)} grounded_candidates={len(candidates)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
