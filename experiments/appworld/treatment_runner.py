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
    EvidenceGatedWorkflowCompiler,
    ModelAssistedWorkflowCompiler,
)
from self_learning_flows.engine import SelfLearningFlowEngine
from self_learning_flows.execution import WorkflowExecutor
from self_learning_flows.extraction import StructuredVariableExtractor
from self_learning_flows.matching import WorkflowMatcher
from self_learning_flows.models import TaskRequest
from self_learning_flows.providers import OpenAICompatibleStructuredModel
from self_learning_flows.research_protocol import (
    ExperimentPhase,
    ProtocolConfig,
    ProtocolGuard,
)
from self_learning_flows.storage import SQLiteStore


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
        task_run.input_tokens += after[0] - before[0]
        task_run.output_tokens += after[1] - before[1]
        task_run.reasoning_calls += after[2] - before[2]

    @staticmethod
    def _close_world(world: Any) -> None:
        close = getattr(world, "close", None)
        if callable(close):
            close()


def _git_commit(repository: Path) -> str:
    base = [
        "git",
        "-c",
        f"safe.directory={repository.resolve()}",
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
    parser.add_argument("--protocol", type=Path, default=Path("experiments/appworld/protocol.json"))
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--seal-after-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--random-seed", type=int, default=100)
    parser.add_argument("--max-interactions", type=int, default=25)
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    if not api_key:
        parser.error(f"environment variable {args.api_key_env} is not set")
    config = ProtocolConfig.load(args.protocol)
    phase = ExperimentPhase(args.phase)
    guard = ProtocolGuard(config, phase, args.dataset_name)
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
    engine = SelfLearningFlowEngine(
        store,
        compiler=EvidenceGatedWorkflowCompiler(ModelAssistedWorkflowCompiler(structured_model)),
        matcher=WorkflowMatcher(allow_shadow=phase != ExperimentPhase.TEST),
        variable_extractor=StructuredVariableExtractor(structured_model),
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
    summary = {
        "run_id": runner.run_id,
        "phase": phase.value,
        "split": args.dataset_name,
        "condition": args.condition,
        "source_commit": source_commit,
        "model": args.model,
        "tasks": len(results),
        "successes": sum(item.success for item in results),
        "workflow_routes": sum(item.route == "workflow" for item in results),
        "fresh_world_fallbacks": sum(item.route == "fresh_world_fallback" for item in results),
        "input_tokens": sum(item.input_tokens for item in results),
        "output_tokens": sum(item.output_tokens for item in results),
        "reasoning_calls": sum(item.reasoning_calls for item in results),
        "results": [asdict(item) for item in results],
    }
    print(json.dumps(summary, indent=2))

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
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
