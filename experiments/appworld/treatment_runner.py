"""End-to-end AppWorld baseline/treatment runner for SelfLearningFlows.

This module imports AppWorld lazily so the core package remains usable without
the benchmark's Pydantic-1 environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from self_learning_flows.adapters.appworld import (
    AppWorldCompletionVerifier,
    AppWorldTraceAdapter,
    AppWorldTraceRecorder,
    appworld_runtime_variables,
    appworld_tool_schemas_from_docs,
    build_appworld_tool_registry,
)
from self_learning_flows.compiler import (
    DeterministicWorkflowCompiler,
    EvidenceGatedWorkflowCompiler,
    ModelAssistedWorkflowCompiler,
)
from self_learning_flows.discovery import HybridEpisodeFamilyDiscoverer
from self_learning_flows.engine import SelfLearningFlowEngine
from self_learning_flows.execution import WorkflowExecutor
from self_learning_flows.extraction import StructuredVariableExtractor
from self_learning_flows.matching import ModelSemanticReranker, WorkflowMatcher
from self_learning_flows.models import TaskRequest
from self_learning_flows.promotion import PromotionConfig, PromotionPolicy
from self_learning_flows.providers import OpenAICompatibleStructuredModel
from self_learning_flows.research_protocol import (
    ExperimentPhase,
    ProtocolConfig,
    ProtocolGuard,
)
from self_learning_flows.storage import SQLiteStore
from self_learning_flows.synthesis import ValidatedWorkflowCompiler


@dataclass(slots=True)
class AgentOutcome:
    completed: bool
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0
    interactions: int = 0
    error: str | None = None

    @property
    def token_usage(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_calls": self.reasoning_calls,
        }


class BaselineAgent(Protocol):
    def run(self, world: Any) -> AgentOutcome: ...


@dataclass(slots=True)
class TaskRun:
    task_id: str
    condition: str
    route: str
    success: bool
    workflow_id: str | None = None
    workflow_version: int | None = None
    fallback_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0
    interactions: int = 0
    elapsed_ms: float = 0.0
    fresh_worlds: int = 1
    agent_input_tokens: int = 0
    agent_output_tokens: int = 0
    agent_reasoning_calls: int = 0
    workflow_input_tokens: int = 0
    workflow_output_tokens: int = 0
    workflow_reasoning_calls: int = 0
    overhead_input_tokens: int = 0
    overhead_output_tokens: int = 0
    overhead_reasoning_calls: int = 0


class OpenAICompatibleCodeAgent:
    """Small ReAct-style AppWorld code agent used identically in both conditions."""

    _CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        max_interactions: int = 25,
        max_tokens: int = 2000,
        timeout_seconds: int = 180,
        retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        self.base_url = endpoint
        self.max_interactions = max_interactions
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def run(self, world: Any) -> AgentOutcome:
        supervisor = self._jsonable_mapping(world.task.supervisor)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Task: {world.task.instruction}\n\n"
                    f"Supervisor account data: {json.dumps(supervisor, default=str)}\n\n"
                    "Write the first Python action."
                ),
            },
        ]
        outcome = AgentOutcome(completed=False)
        for interaction in range(1, self.max_interactions + 1):
            try:
                content, usage = self._completion(messages)
            except Exception as exc:  # provider failures become benchmark data
                outcome.error = f"{type(exc).__name__}: {exc}"
                break
            outcome.input_tokens += int(usage.get("prompt_tokens", 0))
            outcome.output_tokens += int(usage.get("completion_tokens", 0))
            outcome.reasoning_calls += 1
            outcome.interactions = interaction
            match = self._CODE_BLOCK.search(content)
            code = match.group(1).strip() if match else content.strip()
            if not code:
                outcome.error = "Model returned no executable Python"
                break
            environment_output = world.execute(code)
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Environment output:\n"
                            f"{environment_output[-12000:]}\n\n"
                            "Continue with one Python code block. Complete the task through "
                            "the supervisor API when ready."
                        ),
                    },
                ]
            )
            if world.task_completed():
                outcome.completed = True
                break
        if not outcome.completed and outcome.error is None:
            outcome.error = "Maximum agent interactions reached"
        return outcome

    def _completion(self, messages: list[dict[str, str]]) -> tuple[str, dict[str, int]]:
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}",
            },
        )
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"]), dict(body.get("usage", {}))
            except urllib.error.HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    raise RuntimeError(f"Model API returned {exc.code}: {detail}") from exc
                time.sleep(min(2 ** (attempt - 1), 8))
            except urllib.error.URLError as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"Could not reach model API: {exc}") from exc
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Model request failed")

    @staticmethod
    def _jsonable_mapping(value: Any) -> dict[str, Any]:
        try:
            return dict(value)
        except (TypeError, ValueError):
            return dict(vars(value))

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You solve tasks inside AppWorld by writing Python. The namespace already contains "
            "`apis` and `requester`. Return exactly one Python code block per turn. Discover apps "
            "with `apis.api_docs.show_app_descriptions()`, API names with "
            "`apis.api_docs.show_api_descriptions(app_name=...)`, and exact parameters with "
            "`apis.api_docs.show_api_doc(app_name=..., api_name=...)` or "
            "`apis.api_docs.search_api_docs(query=...)`. Inspect results, use the supervisor's "
            "synthetic account credentials when login is required, make the requested changes, "
            "then call the documented supervisor completion API. Do not guess API parameters."
        )


class AppWorldTreatmentRunner:
    """Runs one condition while preserving a hard fresh-world fallback boundary."""

    def __init__(
        self,
        *,
        engine: SelfLearningFlowEngine,
        guard: ProtocolGuard,
        baseline_agent: BaselineAgent,
        world_factory: Callable[[str, str], Any],
        experiment_name: str,
        condition: str,
        records_path: str | Path | None = None,
        run_id: str | None = None,
        overhead_model: Any | None = None,
    ):
        if condition not in {"baseline", "treatment"}:
            raise ValueError("condition must be baseline or treatment")
        self.engine = engine
        self.guard = guard
        self.baseline_agent = baseline_agent
        self.world_factory = world_factory
        self.experiment_name = experiment_name
        self.condition = condition
        self.records_path = Path(records_path) if records_path else None
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:12]}"
        self.overhead_model = overhead_model

    def run_task(self, task_id: str) -> TaskRun:
        started = time.perf_counter()
        overhead_before = self._overhead_usage()
        fallback_reason: str | None = None
        fresh_worlds = 1
        if self.condition == "treatment":
            world = self.world_factory(task_id, self.experiment_name)
            try:
                workflow_result, fallback_reason = self._try_workflow(world)
                if workflow_result is not None:
                    self._add_overhead(workflow_result, overhead_before)
                    workflow_result.elapsed_ms = (time.perf_counter() - started) * 1000
                    self._record_metric(workflow_result)
                    return workflow_result
            finally:
                self._close_world(world)
            # This is deliberately a new AppWorld initialization. It resets the
            # task DB, requester, Python namespace, request counters, and clock.
            fresh_worlds += 1

        world = self.world_factory(task_id, self.experiment_name)
        try:
            task_run = self._run_baseline(world, fallback_reason, fresh_worlds)
        finally:
            self._close_world(world)
        task_run.elapsed_ms = (time.perf_counter() - started) * 1000
        self._add_overhead(task_run, overhead_before)
        self._record_metric(task_run)
        return task_run

    def run(self, task_ids: list[str]) -> list[TaskRun]:
        return [self.run_task(task_id) for task_id in task_ids]

    def _try_workflow(self, world: Any) -> tuple[TaskRun | None, str | None]:
        registry = self._registry(world)
        request = TaskRequest(
            instruction=str(world.task.instruction),
            scope="appworld",
            variables=appworld_runtime_variables(world.task),
            environment=self._environment_name(),
            available_tools=registry.names(),
            metadata={"task_id": str(world.task.id)},
        )
        try:
            proposal = self.engine.propose(request)
        except Exception as exc:
            return None, f"proposal_error:{type(exc).__name__}"
        if proposal is None:
            return None, "no_eligible_workflow"
        unsupported = {
            str(step.kind)
            for step in WorkflowExecutor._walk_steps(proposal.workflow.steps)
            if str(step.kind) not in self.guard.config.supported_step_kinds
        }
        if unsupported:
            return None, "unsupported_step_kinds:" + ",".join(sorted(unsupported))
        if proposal.missing_variables:
            return None, "missing_variables:" + ",".join(proposal.missing_variables)
        ProtocolGuard.approve_in_memory(proposal)
        verifier = AppWorldCompletionVerifier(world)
        executor = WorkflowExecutor(registry, verifier=verifier)
        result = executor.execute(request, proposal.workflow, proposal.confirmed_variables or {})
        if not (result.success and result.verified):
            self.guard.record_execution(self.engine, proposal.workflow.workflow_id, result)
            return None, result.error or "workflow_verification_failed"
        evaluation = world.evaluate()
        if hasattr(evaluation, "to_dict"):
            evaluation_payload = dict(evaluation.to_dict())
        elif isinstance(evaluation, dict):
            evaluation_payload = dict(evaluation)
        else:
            evaluation_payload = {"success": getattr(evaluation, "success", None)}
        official_success = evaluation_payload.get("success")
        if type(official_success) is not bool:
            official_success = False
            result.error = "Official AppWorld evaluator returned no boolean success"
        # Official test feedback scores this one attempt. It is never allowed
        # to select a fallback or mutate frozen test-time memory.
        result.success = official_success
        result.verified = official_success
        if not official_success and result.error is None:
            result.error = "Official AppWorld evaluator rejected completed workflow"
        self.guard.record_execution(self.engine, proposal.workflow.workflow_id, result)
        return (
            TaskRun(
                task_id=str(world.task.id),
                condition=self.condition,
                route="workflow",
                success=official_success,
                workflow_id=proposal.workflow.workflow_id,
                workflow_version=proposal.workflow.version,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                reasoning_calls=result.reasoning_calls,
                workflow_input_tokens=result.input_tokens,
                workflow_output_tokens=result.output_tokens,
                workflow_reasoning_calls=result.reasoning_calls,
                fresh_worlds=1,
            ),
            None,
        )

    def _run_baseline(
        self,
        world: Any,
        fallback_reason: str | None,
        fresh_worlds: int,
    ) -> TaskRun:
        registry = self._registry(world)
        variables = appworld_runtime_variables(world.task)
        with AppWorldTraceRecorder(world.requester) as recorder:
            outcome = self.baseline_agent.run(world)
        evaluation = world.evaluate()
        record = recorder.normalized_record(
            task=world.task,
            evaluation=evaluation,
            tool_schema_hashes=registry.schema_hashes(),
            tool_schemas=registry.schemas(),
            variables=variables,
            token_usage=outcome.token_usage,
        )
        self._append_record(record)
        episode = AppWorldTraceAdapter().normalize(**record)
        if self.guard.allows_learning:
            self.guard.observe(self.engine, episode)
        return TaskRun(
            task_id=str(world.task.id),
            condition=self.condition,
            route="full_agent" if fallback_reason is None else "fresh_world_fallback",
            success=bool(record["passed"]),
            fallback_reason=fallback_reason or outcome.error,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            reasoning_calls=outcome.reasoning_calls,
            agent_input_tokens=outcome.input_tokens,
            agent_output_tokens=outcome.output_tokens,
            agent_reasoning_calls=outcome.reasoning_calls,
            interactions=outcome.interactions,
            fresh_worlds=fresh_worlds,
        )

    @staticmethod
    def _registry(world: Any):
        docs = world.task.api_docs.function_calling()
        schemas = appworld_tool_schemas_from_docs(docs)
        return build_appworld_tool_registry(world.apis, schemas)

    @staticmethod
    def _environment_name() -> str:
        # Tool schema hashes, the freeze manifest, and dependency lock carry
        # concrete version identity; the routing environment remains stable.
        return "appworld"

    def _append_record(self, record: dict[str, Any]) -> None:
        if self.records_path is None:
            return
        self.records_path.parent.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _record_metric(self, task_run: TaskRun) -> None:
        add_metric = getattr(self.engine.store, "add_metric", None)
        if add_metric:
            add_metric(self.run_id, self.condition, asdict(task_run))

    def _overhead_usage(self) -> tuple[int, int, int]:
        model = self.overhead_model
        return (
            int(getattr(model, "input_tokens", 0)),
            int(getattr(model, "output_tokens", 0)),
            int(getattr(model, "reasoning_calls", 0)),
        )

    def _add_overhead(self, task_run: TaskRun, before: tuple[int, int, int]) -> None:
        after = self._overhead_usage()
        task_run.overhead_input_tokens = after[0] - before[0]
        task_run.overhead_output_tokens = after[1] - before[1]
        task_run.overhead_reasoning_calls = after[2] - before[2]
        task_run.input_tokens += task_run.overhead_input_tokens
        task_run.output_tokens += task_run.overhead_output_tokens
        task_run.reasoning_calls += task_run.overhead_reasoning_calls

    @staticmethod
    def _close_world(world: Any) -> None:
        close = getattr(world, "close", None)
        if callable(close):
            close()


def _git_commit(repository: Path) -> str:
    base = [
        "git",
        "-c",
        f"safe.directory={repository.resolve().as_posix()}",
    ]
    status = subprocess.check_output(
        [*base, "status", "--porcelain", "--untracked-files=all"],
        cwd=repository,
        text=True,
    ).strip()
    if status:
        raise RuntimeError("Refusing to seal a run from a dirty repository")
    return subprocess.check_output([*base, "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def _chat_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    return endpoint if endpoint.endswith("/chat/completions") else endpoint + "/chat/completions"


def _load_task_ids(dataset_name: str) -> list[str]:
    from appworld.task import load_task_ids

    return list(load_task_ids(dataset_name=dataset_name))


def _world_factory(random_seed: int) -> Callable[[str, str], Any]:
    from appworld import AppWorld

    def create(task_id: str, experiment_name: str):
        return AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            load_ground_truth=True,
            random_seed=random_seed,
        )

    return create


def _learning_curve(results: list[TaskRun]) -> list[dict[str, Any]]:
    if not results:
        return []
    checkpoints = sorted(
        {max(1, round(len(results) * fraction)) for fraction in (0.25, 0.5, 0.75, 1)}
    )
    curve = []
    for count in checkpoints:
        prefix = results[:count]
        workflow_routes = sum(item.route == "workflow" for item in prefix)
        curve.append(
            {
                "tasks_seen": count,
                "task_success_rate": sum(item.success for item in prefix) / count,
                "workflow_route_rate": workflow_routes / count,
                "full_agent_avoidance_rate": workflow_routes / count,
                "cumulative_reasoning_calls": sum(item.reasoning_calls for item in prefix),
                "cumulative_tokens": sum(item.input_tokens + item.output_tokens for item in prefix),
            }
        )
    return curve


def _workflow_inventory(store: SQLiteStore) -> dict[str, Any]:
    workflows = store.list_workflows(scope="appworld")
    statuses: dict[str, int] = {}
    compilers: dict[str, int] = {}
    for workflow in workflows:
        status = str(workflow.status)
        compiler = str(workflow.metadata.get("compiler", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
        compilers[compiler] = compilers.get(compiler, 0) + 1
    return {
        "workflows": len(workflows),
        "families": len(
            {
                str(workflow.metadata.get("family_id") or workflow.workflow_id)
                for workflow in workflows
            }
        ),
        "statuses": statuses,
        "compilers": compilers,
        "validated_llm_workflows": sum(
            workflow.metadata.get("compiler") == "validated_llm" for workflow in workflows
        ),
        "compilation_failures": sum(
            int(workflow.metadata.get("compilation_failures", 0)) for workflow in workflows
        ),
        "source_episodes": sum(workflow.stats.pattern_observations for workflow in workflows),
        "held_out_executions": sum(workflow.stats.executions for workflow in workflows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=[item.value for item in ExperimentPhase], required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--condition", choices=["baseline", "treatment"], required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--protocol", type=Path, default=Path("experiments/appworld/protocol.json"))
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--seal-after-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--random-seed", type=int, default=100)
    parser.add_argument("--max-interactions", type=int, default=25)
    parser.add_argument(
        "--compiler-mode",
        choices=["deterministic", "annotate", "structural"],
        default="structural",
    )
    parser.add_argument("--family-mode", choices=["exact", "hybrid"], default="hybrid")
    parser.add_argument("--matcher-mode", choices=["lexical", "semantic"], default="semantic")
    parser.add_argument("--min-synthesis-observations", type=int, default=2)
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    config = ProtocolConfig.load(args.protocol)
    phase = ExperimentPhase(args.phase)
    guard = ProtocolGuard(config, phase, args.dataset_name)
    if args.min_synthesis_observations < 1:
        parser.error("--min-synthesis-observations must be positive")
    if not 0 <= args.max_repair_attempts <= 5:
        parser.error("--max-repair-attempts must be between 0 and 5")
    declared_treatment = config.metadata.get("treatment", {})
    actual_treatment = {
        "compiler_mode": args.compiler_mode,
        "family_mode": args.family_mode,
        "matcher_mode": args.matcher_mode,
        "min_synthesis_observations": args.min_synthesis_observations,
        "max_repair_attempts": args.max_repair_attempts,
    }
    mismatches = {
        key: (declared_treatment[key], value)
        for key, value in actual_treatment.items()
        if key in declared_treatment and declared_treatment[key] != value
    }
    if args.condition == "treatment" and mismatches:
        parser.error(
            "runner settings differ from the predeclared protocol: "
            + ", ".join(
                f"{key}={actual!r} (declared {declared!r})"
                for key, (declared, actual) in mismatches.items()
            )
        )
    repository = Path(__file__).resolve().parents[2]
    source_commit = _git_commit(repository)
    if not (repository / "experiments" / "appworld" / "requirements.lock").exists():
        parser.error("experiments/appworld/requirements.lock is missing")
    endpoint = _chat_endpoint(args.base_url)
    structured_model = OpenAICompatibleStructuredModel(
        args.model,
        base_url=endpoint,
        api_key=api_key,
    )
    store = SQLiteStore(args.database)
    deterministic_compiler = DeterministicWorkflowCompiler()
    if args.compiler_mode == "deterministic":
        compiler = deterministic_compiler
    elif args.compiler_mode == "annotate":
        compiler = EvidenceGatedWorkflowCompiler(
            ModelAssistedWorkflowCompiler(structured_model, deterministic_compiler),
            min_observations=args.min_synthesis_observations,
            fallback=deterministic_compiler,
        )
    else:
        compiler = EvidenceGatedWorkflowCompiler(
            ValidatedWorkflowCompiler(
                structured_model,
                fallback=deterministic_compiler,
                max_repair_attempts=args.max_repair_attempts,
            ),
            min_observations=args.min_synthesis_observations,
            fallback=deterministic_compiler,
        )
    matcher = WorkflowMatcher(allow_shadow=phase != ExperimentPhase.TEST)
    promotion_values = config.metadata.get("promotion_config", {})
    promotion_policy = PromotionPolicy(PromotionConfig(**promotion_values))
    engine = SelfLearningFlowEngine(
        store,
        compiler=compiler,
        matcher=matcher,
        promotion_policy=promotion_policy,
        variable_extractor=StructuredVariableExtractor(structured_model),
        family_discoverer=(
            HybridEpisodeFamilyDiscoverer(structured_model)
            if args.family_mode == "hybrid"
            else None
        ),
        semantic_reranker=(
            ModelSemanticReranker(structured_model) if args.matcher_mode == "semantic" else None
        ),
    )
    if phase == ExperimentPhase.TEST and args.condition == "treatment":
        if args.freeze_manifest is None:
            parser.error("--freeze-manifest is required for test")
        guard.validate_frozen(
            store,
            args.freeze_manifest,
            expected_source_commit=source_commit,
        )

    task_ids = args.task_id or _load_task_ids(args.dataset_name)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]
    agent = OpenAICompatibleCodeAgent(
        args.model,
        api_key=api_key,
        base_url=args.base_url,
        max_interactions=args.max_interactions,
    )
    runner = AppWorldTreatmentRunner(
        engine=engine,
        guard=guard,
        baseline_agent=agent,
        world_factory=_world_factory(args.random_seed),
        experiment_name=args.experiment_name,
        condition=args.condition,
        records_path=args.records,
        overhead_model=structured_model,
    )
    results = runner.run(task_ids)
    workflow_routes = sum(item.route == "workflow" for item in results)
    full_agent_routes = sum(item.route != "workflow" for item in results)
    summary = {
        "run_id": runner.run_id,
        "phase": phase.value,
        "split": args.dataset_name,
        "condition": args.condition,
        "source_commit": source_commit,
        "model": args.model,
        "compiler_mode": args.compiler_mode,
        "family_mode": args.family_mode,
        "matcher_mode": args.matcher_mode,
        "tasks": len(results),
        "successes": sum(item.success for item in results),
        "workflow_routes": workflow_routes,
        "successful_workflow_routes": sum(
            item.route == "workflow" and item.success for item in results
        ),
        "workflow_route_success_rate": (
            sum(item.route == "workflow" and item.success for item in results) / workflow_routes
            if workflow_routes
            else None
        ),
        "workflow_execution_model_free_routes": sum(
            item.route == "workflow" and item.workflow_reasoning_calls == 0 for item in results
        ),
        "fully_model_free_routes": sum(
            item.route == "workflow"
            and item.workflow_reasoning_calls == 0
            and item.overhead_reasoning_calls == 0
            for item in results
        ),
        "full_agent_routes": full_agent_routes,
        "full_agent_avoidance_rate": workflow_routes / len(results) if results else 0.0,
        "fresh_world_fallbacks": sum(item.route == "fresh_world_fallback" for item in results),
        "input_tokens": sum(item.input_tokens for item in results),
        "output_tokens": sum(item.output_tokens for item in results),
        "reasoning_calls": sum(item.reasoning_calls for item in results),
        "agent_reasoning_calls": sum(item.agent_reasoning_calls for item in results),
        "workflow_reasoning_calls": sum(item.workflow_reasoning_calls for item in results),
        "overhead_reasoning_calls": sum(item.overhead_reasoning_calls for item in results),
        "structured_model_usage": structured_model.usage_by_category,
        "workflow_inventory": _workflow_inventory(store),
        "learning_curve": _learning_curve(results),
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(summary, indent=2))
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    if phase == ExperimentPhase.TEST and args.condition == "treatment":
        guard.validate_frozen(
            store,
            args.freeze_manifest,
            expected_source_commit=source_commit,
        )

    if args.seal_after_run:
        if args.freeze_manifest is None:
            parser.error("--freeze-manifest is required with --seal-after-run")
        guard.seal(
            store,
            args.freeze_manifest,
            source_commit=source_commit,
            run_config={
                "model": args.model,
                "condition": args.condition,
                "dataset_name": args.dataset_name,
                "random_seed": args.random_seed,
                "max_interactions": args.max_interactions,
                "compiler_mode": args.compiler_mode,
                "family_mode": args.family_mode,
                "matcher_mode": args.matcher_mode,
                "min_synthesis_observations": args.min_synthesis_observations,
                "max_repair_attempts": args.max_repair_attempts,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
