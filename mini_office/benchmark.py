"""Chronological continual-learning benchmark for the reference package."""

from __future__ import annotations

import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.execution import WorkflowExecutor
from self_learning_flows.models import (
    ComputationKind,
    ConfirmationDecision,
    TaskRequest,
)

from .agent import ReferenceAgent
from .compute import ReferenceComputeBackend
from .dataset import DEFAULT_DATASET, load_tasks
from .verifier import OfficeVerifier, verify_task
from .world import MiniOfficeWorld


@dataclass(slots=True)
class BenchmarkConfig:
    dataset_path: str | Path = DEFAULT_DATASET
    output_path: str | Path | None = None
    auto_confirm: bool = True
    seed: int | None = None


@dataclass(slots=True)
class BenchmarkSummary:
    dataset: str
    seed: int | None
    tasks: int
    successful_tasks: int
    task_success_rate: float
    baseline_agent_tasks: int
    workflow_tasks: int
    workflow_successes: int
    workflow_fallbacks: int
    model_free_workflow_tasks: int
    slm_workflow_tasks: int
    full_agent_reasoning_calls: int
    workflow_generation_calls: int
    all_agent_baseline_tokens: int
    self_learning_total_tokens: int
    baseline_tokens: int
    workflow_tokens: int
    token_reduction_rate: float
    full_agent_calls_avoided_rate: float
    mean_workflow_match_score: float
    negative_probes: int
    false_workflow_offers: int
    offer_false_positive_rate: float
    learned_workflows: dict[str, int]
    routes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _request(task, tools: list[str]) -> TaskRequest:
    return TaskRequest(
        instruction=task.instruction,
        scope="mini-office",
        variables=task.workflow_variables,
        environment="mini-office-v1",
        available_tools=tools,
        metadata={"task_id": task.task_id, "pattern": task.pattern},
    )


def _portable_dataset_path(path: str | Path) -> str:
    """Prefer a repository-relative dataset label in persisted benchmark results."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(resolved)


def run_benchmark(config: BenchmarkConfig | None = None) -> BenchmarkSummary:
    config = config or BenchmarkConfig()
    tasks = load_tasks(config.dataset_path)
    if config.seed is not None:
        random.Random(config.seed).shuffle(tasks)
    store = InMemoryStore()
    engine = SelfLearningFlowEngine(store)

    successful = baseline_tasks = workflow_tasks = workflow_successes = fallbacks = 0
    model_free = slm_tasks = full_reasoning_calls = workflow_generation_calls = 0
    baseline_tokens = workflow_tokens = 0
    all_agent_baseline_tokens = 0
    match_scores: list[float] = []
    routes: list[dict[str, Any]] = []

    for task in tasks:
        comparison_world = MiniOfficeWorld()
        comparison_world.seed(task.variables)
        comparison_episode = ReferenceAgent(comparison_world).run(task)
        if not verify_task(task, comparison_world):
            raise RuntimeError(f"Reference baseline failed verifier for {task.task_id}")
        all_agent_baseline_tokens += (
            comparison_episode.input_tokens + comparison_episode.output_tokens
        )

        world = MiniOfficeWorld()
        world.seed(task.variables)
        registry = world.tool_registry()
        request = _request(task, registry.names())
        proposal = engine.propose(request)

        if proposal is not None and not proposal.missing_variables and config.auto_confirm:
            engine.record_feedback(
                proposal,
                ConfirmationDecision.APPROVE,
                task_instruction=task.instruction,
            )
            executor = WorkflowExecutor(
                registry,
                compute_backend=ReferenceComputeBackend(),
                verifier=OfficeVerifier(task, world),
            )
            result = engine.execute(request, proposal, executor)
            workflow_tasks += 1
            match_scores.append(proposal.match.score)
            workflow_tokens += result.input_tokens + result.output_tokens
            generative_kinds = {
                kind
                for kind in result.executor_kinds
                if kind in {ComputationKind.SLM, ComputationKind.LLM, ComputationKind.FULL_AGENT}
            }
            if not generative_kinds:
                model_free += 1
            if ComputationKind.SLM in generative_kinds:
                slm_tasks += 1
                workflow_generation_calls += result.reasoning_calls
            if result.success and result.verified:
                workflow_successes += 1
                successful += 1
                routes.append(
                    {
                        "task_id": task.task_id,
                        "pattern": task.pattern,
                        "route": "workflow",
                        "workflow": proposal.workflow.name,
                        "match_score": round(proposal.match.score, 4),
                        "verified": True,
                    }
                )
                continue

            # A failed workflow never becomes the final task outcome. Re-run in a
            # clean world through the full agent and learn from that correction.
            fallbacks += 1
            world = MiniOfficeWorld()
            world.seed(task.variables)
            registry = world.tool_registry()

        agent = ReferenceAgent(world)
        episode = agent.run(task)
        episode.verified = verify_task(task, world)
        episode.success = episode.success and episode.verified
        baseline_tasks += 1
        full_reasoning_calls += episode.reasoning_calls
        baseline_tokens += episode.input_tokens + episode.output_tokens
        successful += int(episode.success)
        workflow = engine.observe(episode)
        if workflow is not None:
            hashes = registry.schema_hashes()
            workflow.tool_schema_hashes = {
                name: hashes[name] for name in workflow.required_tools if name in hashes
            }
            store.upsert_workflow(workflow)
        routes.append(
            {
                "task_id": task.task_id,
                "pattern": task.pattern,
                "route": "full_agent" if not proposal else "workflow_fallback",
                "verified": episode.verified,
            }
        )

    status_counts: dict[str, int] = {}
    for workflow in store.list_workflows():
        key = str(workflow.status)
        status_counts[key] = status_counts.get(key, 0) + 1
    probe_variables = {task.pattern: task.workflow_variables for task in tasks}
    negative_requests = [
        TaskRequest(
            "Reschedule the Delta retrospective but do not notify or email anyone.",
            "mini-office",
            probe_variables["reschedule_and_notify"],
        ),
        TaskRequest(
            "Create the issue ticket but do not assign it or email the requester.",
            "mini-office",
            probe_variables["create_assign_ticket"],
        ),
        TaskRequest(
            "Add Wendy as a contact only; do not create an event or send email.",
            "mini-office",
            probe_variables["onboard_employee"],
        ),
        TaskRequest(
            "Prepare the priority digest but do not send it by email.",
            "mini-office",
            probe_variables["priority_digest"],
        ),
        TaskRequest("Delete every calendar and mailbox permanently.", "mini-office"),
        TaskRequest("Purchase twenty laptops for the company.", "mini-office"),
    ]
    false_offers = sum(engine.propose(request) is not None for request in negative_requests)
    self_learning_total_tokens = baseline_tokens + workflow_tokens
    summary = BenchmarkSummary(
        dataset=_portable_dataset_path(config.dataset_path),
        seed=config.seed,
        tasks=len(tasks),
        successful_tasks=successful,
        task_success_rate=successful / len(tasks) if tasks else 0.0,
        baseline_agent_tasks=baseline_tasks,
        workflow_tasks=workflow_tasks,
        workflow_successes=workflow_successes,
        workflow_fallbacks=fallbacks,
        model_free_workflow_tasks=model_free,
        slm_workflow_tasks=slm_tasks,
        full_agent_reasoning_calls=full_reasoning_calls,
        workflow_generation_calls=workflow_generation_calls,
        all_agent_baseline_tokens=all_agent_baseline_tokens,
        self_learning_total_tokens=self_learning_total_tokens,
        baseline_tokens=baseline_tokens,
        workflow_tokens=workflow_tokens,
        token_reduction_rate=(
            1 - self_learning_total_tokens / all_agent_baseline_tokens
            if all_agent_baseline_tokens
            else 0.0
        ),
        full_agent_calls_avoided_rate=workflow_successes / len(tasks) if tasks else 0.0,
        mean_workflow_match_score=statistics.mean(match_scores) if match_scores else 0.0,
        negative_probes=len(negative_requests),
        false_workflow_offers=false_offers,
        offer_false_positive_rate=false_offers / len(negative_requests),
        learned_workflows=status_counts,
        routes=routes,
    )
    if config.output_path:
        output = Path(config.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    return summary
