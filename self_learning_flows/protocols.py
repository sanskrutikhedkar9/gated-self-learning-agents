"""Extension ports used to keep the learning core framework and model agnostic."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .models import (
    ComputationKind,
    ExecutionResult,
    TaskEpisode,
    TaskRequest,
    ToolResult,
    WorkflowDefinition,
)


@runtime_checkable
class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text."""


@runtime_checkable
class StructuredModel(Protocol):
    @property
    def model_name(self) -> str: ...

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Produce a JSON object conforming to schema."""


@runtime_checkable
class VariableExtractor(Protocol):
    @property
    def computation_kind(self) -> ComputationKind: ...

    def extract(self, request: TaskRequest, workflow: WorkflowDefinition) -> dict[str, Any]: ...


@runtime_checkable
class WorkflowCompiler(Protocol):
    def compile(self, episodes: list[TaskEpisode]) -> WorkflowDefinition: ...

    def refine(
        self,
        workflow: WorkflowDefinition,
        episodes: list[TaskEpisode],
    ) -> WorkflowDefinition: ...


@runtime_checkable
class ComputeBackend(Protocol):
    def run(
        self,
        *,
        kind: ComputationKind,
        operation: str,
        inputs: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
    ) -> ToolResult: ...


@runtime_checkable
class OutcomeVerifier(Protocol):
    def verify(
        self,
        *,
        request: TaskRequest,
        workflow: WorkflowDefinition,
        result: ExecutionResult,
    ) -> bool: ...


@runtime_checkable
class TraceAdapter(Protocol):
    def normalize(self, framework_run: Any) -> TaskEpisode: ...


@runtime_checkable
class WorkflowStore(Protocol):
    def add_episode(self, episode: TaskEpisode) -> None: ...

    def get_episode(self, task_id: str) -> TaskEpisode | None: ...

    def list_episodes(self, *, scope: str | None = None) -> list[TaskEpisode]: ...

    def upsert_workflow(self, workflow: WorkflowDefinition) -> None: ...

    def get_workflow(self, workflow_id: str) -> WorkflowDefinition | None: ...

    def list_workflows(
        self,
        *,
        scope: str | None = None,
        statuses: set[str] | None = None,
    ) -> list[WorkflowDefinition]: ...
