"""Framework-neutral data contracts for episodes and learned workflows.

The package deliberately stores plain serializable values. Framework-specific
message, graph, and tool objects are normalized by adapters before entering the
learning core.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any


class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ComputationKind(StrEnum):
    """Cheapest permitted executor for a workflow step."""

    DETERMINISTIC = "deterministic"
    SLM = "slm"
    LLM = "llm"
    FULL_AGENT = "full_agent"


class StepKind(StrEnum):
    TOOL = "tool"
    COMPUTE = "compute"
    PAGINATE = "paginate"
    FOREACH = "foreach"
    FILTER = "filter"
    REDUCE = "reduce"
    BRANCH = "branch"


class WorkflowStatus(StrEnum):
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    RETIRED = "retired"


class ConfirmationDecision(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"
    FULL_AGENT = "full_agent"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(slots=True)
class ToolCallTrace:
    """One normalized tool call produced during an agent run."""

    step_id: str
    agent: str
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = True
    error: str | None = None
    depends_on: list[str] = field(default_factory=list)
    executor: ComputationKind = ComputationKind.FULL_AGENT
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ToolCallTrace:
        value = dict(value)
        value["executor"] = ComputationKind(value.get("executor", "full_agent"))
        return cls(**value)


@dataclass(slots=True)
class TaskEpisode:
    instruction: str
    scope: str
    success: bool
    verified: bool
    steps: list[ToolCallTrace]
    task_id: str = field(default_factory=lambda: new_id("task"))
    outcome: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    framework: str = "agnostic"
    user_id: str = "anonymous"
    started_at: float = field(default_factory=time.time)
    completed_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskEpisode:
        value = dict(value)
        value["steps"] = [ToolCallTrace.from_dict(item) for item in value.get("steps", [])]
        return cls(**value)


@dataclass(slots=True)
class TaskRequest:
    instruction: str
    scope: str = "*"
    variables: dict[str, Any] = field(default_factory=dict)
    user_id: str = "anonymous"
    environment: str = "default"
    available_tools: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VariableSpec:
    name: str
    type: str = "string"
    required: bool = True
    description: str = ""
    default: Any = None
    examples: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> VariableSpec:
        return cls(**value)


@dataclass(slots=True)
class WorkflowStep:
    id: str
    kind: StepKind
    operation: str
    agent: str = "*"
    arguments: dict[str, Any] = field(default_factory=dict)
    executor: ComputationKind = ComputationKind.DETERMINISTIC
    fallback_executors: list[ComputationKind] = field(default_factory=list)
    output_key: str | None = None
    condition: dict[str, Any] | None = None
    depends_on: list[str] = field(default_factory=list)
    description: str = ""
    on_error: str = "escalate"
    collection: Any = None
    control: dict[str, Any] = field(default_factory=dict)
    body: list[WorkflowStep] = field(default_factory=list)
    else_body: list[WorkflowStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkflowStep:
        value = dict(value)
        value["kind"] = StepKind(value.get("kind", "tool"))
        value["executor"] = ComputationKind(value.get("executor", "deterministic"))
        value["fallback_executors"] = [
            ComputationKind(item) for item in value.get("fallback_executors", [])
        ]
        value["body"] = [cls.from_dict(item) for item in value.get("body", [])]
        value["else_body"] = [cls.from_dict(item) for item in value.get("else_body", [])]
        return cls(**value)


@dataclass(slots=True)
class ExecutorStats:
    executions: int = 0
    successes: int = 0
    failures: int = 0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutorStats:
        return cls(**value)


@dataclass(slots=True)
class WorkflowStats:
    pattern_observations: int = 0
    executions: int = 0
    successes: int = 0
    failures: int = 0
    approvals: int = 0
    edits: int = 0
    rejections: int = 0
    full_agent_choices: int = 0
    consecutive_failures: int = 0
    source_task_ids: list[str] = field(default_factory=list)
    executor_stats: dict[str, ExecutorStats] = field(default_factory=dict)
    average_latency_ms: float = 0.0
    average_reasoning_calls: float = 0.0
    last_success_at: float | None = None
    last_failure_at: float | None = None

    @property
    def reliability(self) -> float:
        # Beta(1, 1) posterior mean avoids overconfidence at low sample counts.
        return (self.successes + 1) / (self.executions + 2)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkflowStats:
        value = dict(value)
        value["executor_stats"] = {
            key: ExecutorStats.from_dict(item)
            for key, item in value.get("executor_stats", {}).items()
        }
        return cls(**value)


@dataclass(slots=True)
class WorkflowDefinition:
    name: str
    description: str
    scope: str
    variables: list[VariableSpec]
    steps: list[WorkflowStep]
    workflow_id: str = field(default_factory=lambda: new_id("workflow"))
    version: int = 1
    status: WorkflowStatus = WorkflowStatus.CANDIDATE
    intent_examples: list[str] = field(default_factory=list)
    structural_signature: str = ""
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    tool_schema_hashes: dict[str, str] = field(default_factory=dict)
    required_tools: list[str] = field(default_factory=list)
    environment: str = "default"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    stats: WorkflowStats = field(default_factory=WorkflowStats)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkflowDefinition:
        value = dict(value)
        value["status"] = WorkflowStatus(value.get("status", "candidate"))
        value["variables"] = [VariableSpec.from_dict(item) for item in value.get("variables", [])]
        value["steps"] = [WorkflowStep.from_dict(item) for item in value.get("steps", [])]
        value["stats"] = WorkflowStats.from_dict(value.get("stats", {}))
        return cls(**value)


@dataclass(slots=True)
class WorkflowMatch:
    workflow_id: str
    score: float
    intent_score: float
    tool_compatibility: float
    variable_coverage: float
    reliability: float
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    matcher: str = "lexical"


@dataclass(slots=True)
class WorkflowProposal:
    workflow: WorkflowDefinition
    match: WorkflowMatch
    variables: dict[str, Any]
    missing_variables: list[str] = field(default_factory=list)
    ambiguous_variables: list[str] = field(default_factory=list)
    confirmation_decision: ConfirmationDecision | None = None
    confirmed_variables: dict[str, Any] | None = None

    def confirmation_payload(self) -> dict[str, Any]:
        return {
            "type": "workflow_confirmation",
            "workflow_id": self.workflow.workflow_id,
            "workflow_name": self.workflow.name,
            "description": self.workflow.description,
            "match_score": round(self.match.score, 4),
            "variables": self.variables,
            "missing_variables": self.missing_variables,
            "steps": [
                {
                    "id": step.id,
                    "agent": step.agent,
                    "operation": step.operation,
                    "executor": str(step.executor),
                }
                for step in self.workflow.steps
            ],
            "choices": ["approve", "edit", "reject", "full_agent"],
        }


@dataclass(slots=True)
class ToolResult:
    success: bool
    value: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0


@dataclass(slots=True)
class ExecutionResult:
    workflow_id: str
    success: bool
    verified: bool
    outputs: dict[str, Any]
    steps: list[ToolCallTrace]
    executor_kinds: list[ComputationKind]
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0
    escalated: bool = False
    error: str | None = None
    side_effects_may_have_occurred: bool = False


@dataclass(slots=True)
class WorkflowFeedback:
    workflow_id: str
    task_instruction: str
    decision: ConfirmationDecision
    proposed_variables: dict[str, Any]
    final_variables: dict[str, Any]
    user_id: str = "anonymous"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)
