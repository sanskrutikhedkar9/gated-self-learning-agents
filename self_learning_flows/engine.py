"""Main orchestration API for continuous workflow learning and reuse."""

from __future__ import annotations

import copy
import time
from typing import Any

from .compiler import DeterministicWorkflowCompiler
from .discovery import structural_signature
from .execution import WorkflowExecutor
from .matching import WorkflowMatcher
from .models import (
    ConfirmationDecision,
    ExecutionResult,
    TaskEpisode,
    TaskRequest,
    WorkflowDefinition,
    WorkflowFeedback,
    WorkflowMatch,
    WorkflowProposal,
    WorkflowStatus,
)
from .promotion import PromotionPolicy
from .protocols import VariableExtractor, WorkflowCompiler, WorkflowStore
from .scoring import running_average


class SelfLearningFlowEngine:
    """Continuously learns candidates and routes eligible requests to workflows."""

    def __init__(
        self,
        store: WorkflowStore,
        *,
        compiler: WorkflowCompiler | None = None,
        matcher: WorkflowMatcher | None = None,
        promotion_policy: PromotionPolicy | None = None,
        variable_extractor: VariableExtractor | None = None,
    ):
        self.store = store
        self.compiler = compiler or DeterministicWorkflowCompiler()
        self.matcher = matcher or WorkflowMatcher()
        self.promotion_policy = promotion_policy or PromotionPolicy()
        self.variable_extractor = variable_extractor

    def observe(self, episode: TaskEpisode) -> WorkflowDefinition | None:
        """Record every episode; learn only from verified successes."""
        episode.metadata["structural_signature"] = structural_signature(episode)
        existing_episode = self.store.get_episode(episode.task_id)
        if existing_episode is not None:
            if existing_episode.scope != episode.scope:
                raise ValueError(
                    f"Task ID {episode.task_id!r} already exists in scope "
                    f"{existing_episode.scope!r}"
                )
            if existing_episode.success and existing_episode.verified:
                signature = existing_episode.metadata.get("structural_signature", "")
                return self._workflow_for_signature(signature, existing_episode.scope)
            return None
        self.store.add_episode(episode)
        if not (episode.success and episode.verified and episode.steps):
            return None

        signature = episode.metadata["structural_signature"]
        workflow = self._workflow_for_signature(signature, episode.scope)
        if workflow is not None and workflow.tool_schema_hashes:
            observed_hashes = episode.metadata.get("tool_schema_hashes", {})
            missing_contracts = set(workflow.required_tools) - set(observed_hashes)
            changed_contracts = {
                name
                for name, expected in workflow.tool_schema_hashes.items()
                if name in observed_hashes and observed_hashes[name] != expected
            }
            if changed_contracts:
                workflow.status = WorkflowStatus.QUARANTINED
                workflow.metadata["quarantine_reason"] = "observed_tool_schema_drift:" + ",".join(
                    sorted(changed_contracts)
                )
                self.store.upsert_workflow(workflow)
                return workflow
            if missing_contracts:
                # Store the episode for audit, but do not let evidence without
                # contract provenance alter a replayable workflow.
                return workflow
        episodes = self._episodes_for_workflow(workflow, episode)

        if workflow is None:
            workflow = self.compiler.compile([episode])
        elif workflow.status == WorkflowStatus.CANDIDATE:
            workflow = self.compiler.refine(workflow, episodes)
        # Once a definition is offered for held-out execution, new observations
        # may add support but must not silently rewrite the executable program.

        if episode.task_id not in workflow.stats.source_task_ids:
            workflow.stats.source_task_ids.append(episode.task_id)
        workflow.stats.pattern_observations = len(set(workflow.stats.source_task_ids))
        workflow.status = self.promotion_policy.evaluate(workflow)
        self.store.upsert_workflow(workflow)
        return workflow

    def match(self, request: TaskRequest, *, limit: int = 3) -> list[WorkflowMatch]:
        workflows = self.store.list_workflows(
            scope=request.scope,
            statuses={WorkflowStatus.SHADOW.value, WorkflowStatus.ACTIVE.value},
        )
        return self.matcher.rank(request, workflows)[:limit]

    def propose(self, request: TaskRequest) -> WorkflowProposal | None:
        matches = self.match(request, limit=2)
        if not matches or not matches[0].eligible:
            return None
        if (
            len(matches) > 1
            and matches[1].eligible
            and matches[0].score - matches[1].score < self.matcher.ambiguity_margin
        ):
            return None
        workflow = self.store.get_workflow(matches[0].workflow_id)
        if workflow is None:
            return None
        return self.matcher.propose(request, workflow, matches[0], self.variable_extractor)

    def record_feedback(
        self,
        proposal: WorkflowProposal,
        decision: ConfirmationDecision,
        *,
        final_variables: dict[str, Any] | None = None,
        user_id: str = "anonymous",
        task_instruction: str = "",
    ) -> WorkflowDefinition:
        workflow = self._require_workflow(proposal.workflow.workflow_id)
        if decision == ConfirmationDecision.APPROVE:
            workflow.stats.approvals += 1
        elif decision == ConfirmationDecision.EDIT:
            workflow.stats.edits += 1
        elif decision == ConfirmationDecision.REJECT:
            workflow.stats.rejections += 1
            if task_instruction and task_instruction not in workflow.negative_examples:
                workflow.negative_examples.append(task_instruction)
        elif decision == ConfirmationDecision.FULL_AGENT:
            workflow.stats.full_agent_choices += 1
        confirmed_variables = proposal.variables if final_variables is None else final_variables
        feedback = WorkflowFeedback(
            workflow_id=workflow.workflow_id,
            task_instruction=task_instruction,
            decision=decision,
            proposed_variables=proposal.variables,
            final_variables=confirmed_variables,
            user_id=user_id,
        )
        add_feedback = getattr(self.store, "add_feedback", None)
        if add_feedback:
            add_feedback(feedback)
        proposal.confirmation_decision = decision
        proposal.confirmed_variables = copy.deepcopy(confirmed_variables)
        self.store.upsert_workflow(workflow)
        return workflow

    def execute(
        self,
        request: TaskRequest,
        proposal: WorkflowProposal,
        executor: WorkflowExecutor,
        *,
        variables: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        workflow = self._require_workflow(proposal.workflow.workflow_id)
        if proposal.confirmation_decision not in {
            ConfirmationDecision.APPROVE,
            ConfirmationDecision.EDIT,
        }:
            raise PermissionError("Workflow execution requires an approve or edit decision")
        if proposal.workflow.version != workflow.version:
            raise RuntimeError("Workflow changed after confirmation; request a new proposal")
        if workflow.scope != "*" and request.scope != workflow.scope:
            raise ValueError("Workflow scope does not match the request")
        if workflow.environment != "*" and request.environment != workflow.environment:
            raise ValueError("Workflow environment does not match the request")
        confirmed = proposal.confirmed_variables or {}
        selected_variables = confirmed if variables is None else variables
        if variables is not None and variables != confirmed:
            raise PermissionError("Execution variables differ from the confirmed values")
        result = executor.execute(request, workflow, selected_variables)
        self.record_execution(workflow.workflow_id, result)
        return result

    def record_execution(self, workflow_id: str, result: ExecutionResult) -> WorkflowDefinition:
        workflow = self._require_workflow(workflow_id)
        stats = workflow.stats
        count_before = stats.executions
        stats.executions += 1
        if result.success and result.verified:
            stats.successes += 1
            stats.consecutive_failures = 0
            stats.last_success_at = time.time()
        else:
            stats.failures += 1
            stats.consecutive_failures += 1
            stats.last_failure_at = time.time()
        stats.average_latency_ms = running_average(
            stats.average_latency_ms, count_before, result.latency_ms
        )
        stats.average_reasoning_calls = running_average(
            stats.average_reasoning_calls, count_before, result.reasoning_calls
        )
        for trace in result.steps:
            key = str(trace.executor)
            executor_stats = stats.executor_stats.setdefault(key, self._new_executor_stats())
            executor_stats.executions += 1
            executor_stats.latency_ms += trace.latency_ms
            executor_stats.input_tokens += trace.input_tokens
            executor_stats.output_tokens += trace.output_tokens
            if trace.success:
                executor_stats.successes += 1
            else:
                executor_stats.failures += 1
        workflow.status = self.promotion_policy.evaluate(workflow)
        self.store.upsert_workflow(workflow)
        return workflow

    def _workflow_for_signature(self, signature: str, scope: str) -> WorkflowDefinition | None:
        lookup = getattr(self.store, "workflow_by_signature", None)
        if lookup:
            return lookup(signature, scope=scope)
        for workflow in self.store.list_workflows(scope=scope):
            if workflow.structural_signature == signature:
                return workflow
        return None

    def _episodes_for_workflow(
        self,
        workflow: WorkflowDefinition | None,
        current: TaskEpisode,
    ) -> list[TaskEpisode]:
        if workflow is None:
            return [current]
        episodes: list[TaskEpisode] = []
        for task_id in workflow.stats.source_task_ids:
            episode = self.store.get_episode(task_id)
            if episode is not None:
                episodes.append(episode)
        if all(item.task_id != current.task_id for item in episodes):
            episodes.append(current)
        return episodes

    def _require_workflow(self, workflow_id: str) -> WorkflowDefinition:
        workflow = self.store.get_workflow(workflow_id)
        if workflow is None:
            raise KeyError(f"Unknown workflow: {workflow_id}")
        return workflow

    @staticmethod
    def _new_executor_stats():
        from .models import ExecutorStats

        return ExecutorStats()


class InMemoryStore:
    """Tiny store for embedding the engine in unit tests and notebooks."""

    def __init__(self):
        self.episodes: dict[str, TaskEpisode] = {}
        self.workflows: dict[str, WorkflowDefinition] = {}
        self.workflow_versions: dict[tuple[str, int], WorkflowDefinition] = {}
        self.feedback: list[WorkflowFeedback] = []

    def add_episode(self, episode: TaskEpisode) -> None:
        self.episodes.setdefault(episode.task_id, copy.deepcopy(episode))

    def get_episode(self, task_id: str) -> TaskEpisode | None:
        value = self.episodes.get(task_id)
        return copy.deepcopy(value) if value else None

    def list_episodes(self, *, scope: str | None = None) -> list[TaskEpisode]:
        return [
            copy.deepcopy(item)
            for item in self.episodes.values()
            if scope is None or item.scope in {scope, "*"}
        ]

    def upsert_workflow(self, workflow: WorkflowDefinition) -> None:
        self.workflows[workflow.workflow_id] = copy.deepcopy(workflow)
        self.workflow_versions.setdefault(
            (workflow.workflow_id, workflow.version), copy.deepcopy(workflow)
        )

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition | None:
        value = self.workflows.get(workflow_id)
        return copy.deepcopy(value) if value else None

    def list_workflow_versions(self, workflow_id: str) -> list[WorkflowDefinition]:
        return [
            copy.deepcopy(item)
            for (stored_id, _), item in sorted(
                self.workflow_versions.items(), key=lambda pair: pair[0][1]
            )
            if stored_id == workflow_id
        ]

    def workflow_by_signature(self, signature: str, *, scope: str) -> WorkflowDefinition | None:
        candidates = [
            item
            for item in self.workflows.values()
            if item.structural_signature == signature and item.scope in {scope, "*"}
        ]
        return copy.deepcopy(candidates[-1]) if candidates else None

    def list_workflows(
        self,
        *,
        scope: str | None = None,
        statuses: set[str] | None = None,
    ) -> list[WorkflowDefinition]:
        return [
            copy.deepcopy(item)
            for item in self.workflows.values()
            if (scope is None or item.scope in {scope, "*"})
            and (not statuses or str(item.status) in statuses)
        ]

    def add_feedback(self, feedback: WorkflowFeedback) -> None:
        self.feedback.append(copy.deepcopy(feedback))
