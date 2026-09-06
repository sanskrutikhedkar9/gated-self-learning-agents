"""Evidence-constrained LLM synthesis and offline workflow validation.

The model sees symbolic traces, proposes only the package's bounded workflow IR,
and has no authority to publish or execute a proposal. Local code validates the
program and replays it against every source episode before it can be returned.
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .compiler import DeterministicWorkflowCompiler, _find_value_path, _safe_name, _value_type
from .discovery import action_path, structural_signature
from .execution import ToolRegistry, WorkflowExecutor
from .models import (
    ComputationKind,
    StepKind,
    TaskEpisode,
    ToolResult,
    VariableSpec,
    WorkflowDefinition,
    WorkflowStep,
)
from .protocols import StructuredModel


class WorkflowSynthesisError(ValueError):
    """Raised when neither model synthesis nor the conservative fallback validates."""


@dataclass(slots=True)
class WorkflowValidationReport:
    valid: bool
    static_errors: list[str] = field(default_factory=list)
    replay_errors: list[str] = field(default_factory=list)
    replayed_episode_ids: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [*self.static_errors, *self.replay_errors]

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "static_errors": self.static_errors,
            "replay_errors": self.replay_errors,
            "replayed_episode_ids": self.replayed_episode_ids,
        }


def _json_value_schema() -> dict[str, Any]:
    # Arguments/control are JSON-encoded strings in the model response. This
    # keeps provider-side structured output strict; local parsing then applies
    # the considerably stronger workflow-specific validator below.
    return {"type": "string"}


_VARIABLE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "type", "required", "description"],
    "properties": {
        "name": {"type": "string"},
        "type": {"type": "string"},
        "required": {"type": "boolean"},
        "description": {"type": "string"},
    },
}

_NODE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "parent_id",
        "branch",
        "order",
        "kind",
        "operation",
        "agent",
        "arguments_json",
        "executor",
        "output_key",
        "condition_json",
        "depends_on",
        "description",
        "on_error",
        "collection_json",
        "control_json",
    ],
    "properties": {
        "id": {"type": "string"},
        "parent_id": {"type": "string"},
        "branch": {"type": "string", "enum": ["main", "then", "else", "body"]},
        "order": {"type": "integer", "minimum": 0},
        "kind": {
            "type": "string",
            "enum": ["tool", "compute", "paginate", "foreach", "filter", "reduce", "branch"],
        },
        "operation": {"type": "string"},
        "agent": {"type": "string"},
        "arguments_json": _json_value_schema(),
        "executor": {"type": "string", "enum": ["deterministic", "slm", "llm"]},
        "output_key": {"type": "string"},
        "condition_json": _json_value_schema(),
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "description": {"type": "string"},
        "on_error": {"type": "string", "enum": ["escalate"]},
        "collection_json": _json_value_schema(),
        "control_json": _json_value_schema(),
    },
}

PROGRAM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "description",
        "preconditions",
        "postconditions",
        "variables",
        "nodes",
    ],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "preconditions": {"type": "array", "items": {"type": "string"}},
        "postconditions": {"type": "array", "items": {"type": "string"}},
        "variables": {"type": "array", "items": _VARIABLE_SCHEMA},
        "nodes": {"type": "array", "items": _NODE_SCHEMA},
    },
}


class SymbolicTraceBuilder:
    """Remove concrete argument/result values while retaining data-flow evidence."""

    _REDACTED = re.compile(r"^<redacted(?::[^>]*)?>$")

    def __init__(self) -> None:
        self.literal_values: dict[str, Any] = {}
        self._literal_ids: dict[str, str] = {}

    def build(self, episodes: list[TaskEpisode]) -> dict[str, Any]:
        symbolic = []
        for episode_index, episode in enumerate(episodes):
            traces = [trace for trace in episode.steps if trace.success]
            symbolic_steps = []
            prior: list[tuple[str, Any]] = []
            for step_index, trace in enumerate(traces):
                evidence_id = f"e{episode_index}_s{step_index}"
                symbolic_steps.append(
                    {
                        "evidence_id": evidence_id,
                        "operation": trace.tool,
                        "agent": trace.agent,
                        "executor": str(trace.executor),
                        "arguments": self._symbolize(
                            trace.arguments,
                            episode,
                            prior,
                            hint=_safe_name(trace.tool.split(".")[-1]),
                        ),
                        "result_shape": self._shape(trace.result),
                    }
                )
                prior.append((evidence_id, trace.result))
            symbolic.append(
                {
                    "episode_id": episode.task_id,
                    "instruction": self._instruction_template(episode),
                    "declared_inputs": [
                        {"name": name, "type": _value_type(value)}
                        for name, value in sorted(episode.variables.items())
                    ],
                    "steps": symbolic_steps,
                }
            )
        contracts: dict[str, dict[str, Any]] = {}
        for episode in episodes:
            for name, schema in episode.metadata.get("tool_schemas", {}).items():
                if isinstance(schema, dict):
                    contracts.setdefault(str(name), schema)
        return {
            "episodes": symbolic,
            "tool_contracts": contracts,
            "literal_catalog": [
                {"id": key, "type": _value_type(value)}
                for key, value in self.literal_values.items()
            ],
            "allowed_reference_forms": [
                "$input.<variable>[.<path>]",
                "$steps.<proposed_step_id>[.<result_path>]",
                "$loop.item[.<path>]",
                "$loop.index",
                "$literal.<catalog_id>",
            ],
        }

    @staticmethod
    def _instruction_template(episode: TaskEpisode) -> str:
        instruction = episode.instruction

        def values(value: Any):
            if isinstance(value, dict):
                for child in value.values():
                    yield from values(child)
            elif isinstance(value, list):
                for child in value:
                    yield from values(child)
            elif isinstance(value, str) and len(value) >= 2:
                yield value

        replacements = [
            (candidate, name)
            for name, value in episode.variables.items()
            for candidate in values(value)
        ]
        for trace in episode.steps:
            for argument_name, value in trace.arguments.items():
                replacements.extend(
                    (
                        candidate,
                        f"{_safe_name(trace.tool.split('.')[-1])}_{argument_name}",
                    )
                    for candidate in values(value)
                )
        for candidate, name in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
            instruction = instruction.replace(candidate, "{{" + name + "}}")
        return instruction

    def _symbolize(
        self,
        value: Any,
        episode: TaskEpisode,
        prior: list[tuple[str, Any]],
        *,
        hint: str,
    ) -> Any:
        for name, candidate in episode.variables.items():
            path = _find_value_path(candidate, value)
            if path is not None:
                suffix = f".{path}" if path else ""
                return f"$input.{name}{suffix}"
        for evidence_id, result in reversed(prior):
            path = _find_value_path(result, value)
            if path is not None:
                suffix = f".{path}" if path else ""
                return f"$evidence.{evidence_id}{suffix}"
        if isinstance(value, dict):
            return {
                str(key): self._symbolize(
                    item, episode, prior, hint=f"{hint}_{_safe_name(str(key))}"
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._symbolize(item, episode, prior, hint=f"{hint}_{index}")
                for index, item in enumerate(value)
            ]
        if isinstance(value, str) and self._REDACTED.match(value):
            return f"$sensitive.{hint}"
        encoded = json.dumps([type(value).__name__, value], sort_keys=True, default=str)
        literal_id = self._literal_ids.get(encoded)
        if literal_id is None:
            literal_id = f"literal_{len(self._literal_ids) + 1}"
            self._literal_ids[encoded] = literal_id
            self.literal_values[literal_id] = copy.deepcopy(value)
        return f"$literal.{literal_id}"

    def _shape(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._shape(item) for key, item in value.items()}
        if isinstance(value, list):
            return {"type": "array", "items": self._shape(value[0]) if value else "any"}
        return {"type": _value_type(value)}


class WorkflowProgramValidator:
    """Static capability, reference, contract, and bounded-control validator."""

    _TOOL_KINDS = {StepKind.TOOL, StepKind.PAGINATE, StepKind.FOREACH}
    _REDUCERS = {
        "flatten",
        "unique",
        "first",
        "last",
        "count",
        "sum",
        "min",
        "max",
        "sort",
        "top_k",
    }
    _PREDICATES = {
        "eq",
        "ne",
        "in",
        "not_in",
        "contains",
        "gt",
        "gte",
        "lt",
        "lte",
        "exists",
        "not_exists",
    }
    _CONTROL_KEYS = {
        StepKind.PAGINATE: {
            "page_argument",
            "start",
            "increment",
            "max_iterations",
            "items_path",
            "has_more_path",
            "page_size_argument",
            "allow_truncation",
        },
        StepKind.FOREACH: {"max_iterations", "collect"},
        StepKind.FILTER: {"predicate"},
        StepKind.REDUCE: {"path", "descending", "k"},
        StepKind.BRANCH: set(),
        StepKind.TOOL: set(),
        StepKind.COMPUTE: set(),
    }

    def __init__(self, *, max_nodes: int = 64, max_loop_iterations: int = 100):
        self.max_nodes = max_nodes
        self.max_loop_iterations = max_loop_iterations

    def validate(self, workflow: WorkflowDefinition, episodes: list[TaskEpisode]) -> list[str]:
        errors: list[str] = []
        steps = list(WorkflowExecutor._walk_steps(workflow.steps))
        if not steps:
            errors.append("program has no executable nodes")
        if len(steps) > self.max_nodes:
            errors.append(f"program has {len(steps)} nodes; maximum is {self.max_nodes}")
        ids = [step.id for step in steps]
        if any(not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", item) for item in ids):
            errors.append("step IDs must be short alphanumeric identifiers")
        if len(ids) != len(set(ids)):
            errors.append("step IDs are not unique")

        observed_operations = {
            trace.tool for episode in episodes for trace in episode.steps if trace.success
        }
        schemas = self._schemas(episodes)
        proposed_tool_steps = {
            step.operation
            for step in steps
            if step.kind in {StepKind.TOOL, StepKind.PAGINATE}
            or (step.kind == StepKind.FOREACH and not step.body)
        }
        hashes, hash_errors = self._schema_hashes(episodes, proposed_tool_steps)
        errors.extend(hash_errors)
        declared_inputs = {variable.name for variable in workflow.variables}
        if len(declared_inputs) != len(workflow.variables):
            errors.append("workflow variable names are not unique")
        required_tools: set[str] = set()
        seen: set[str] = set()
        for step in steps:
            if step.on_error != "escalate":
                errors.append(f"{step.id}: on_error must be 'escalate'")
            missing_dependencies = set(step.depends_on) - seen
            if missing_dependencies:
                errors.append(
                    f"{step.id}: unresolved/forward dependencies "
                    + ", ".join(sorted(missing_dependencies))
                )
            invokes_tool = step.kind in {StepKind.TOOL, StepKind.PAGINATE} or (
                step.kind == StepKind.FOREACH and not step.body
            )
            if invokes_tool:
                required_tools.add(step.operation)
                if step.operation not in observed_operations:
                    errors.append(f"{step.id}: unobserved tool {step.operation!r}")
                if step.executor != ComputationKind.DETERMINISTIC:
                    errors.append(f"{step.id}: tool calls must use deterministic execution")
                schema = schemas.get(step.operation)
                if schema:
                    errors.extend(self._validate_argument_contract(step, schema))
            elif step.kind == StepKind.COMPUTE:
                if step.operation not in observed_operations or not step.operation.startswith(
                    "compute."
                ):
                    errors.append(f"{step.id}: compute operation was not observed")
            elif step.kind == StepKind.REDUCE and step.operation not in self._REDUCERS:
                errors.append(f"{step.id}: unsupported reducer {step.operation!r}")
            elif step.kind == StepKind.FILTER:
                errors.extend(self._validate_predicate(step.control.get("predicate"), step.id))
            elif step.kind == StepKind.BRANCH:
                errors.extend(self._validate_condition(step.condition, step.id))
                if (
                    isinstance(step.condition, dict)
                    and step.condition.get("source") == "input"
                    and str(step.condition.get("key", "")).split(".", 1)[0] not in declared_inputs
                ):
                    errors.append(f"{step.id}: branch uses an undeclared input")

            extras = set(step.control) - self._CONTROL_KEYS[step.kind]
            if extras:
                errors.append(f"{step.id}: unsupported control keys {', '.join(sorted(extras))}")
            if step.kind in {StepKind.PAGINATE, StepKind.FOREACH}:
                maximum = step.control.get("max_iterations")
                if not isinstance(maximum, int) or not 1 <= maximum <= self.max_loop_iterations:
                    errors.append(
                        f"{step.id}: max_iterations must be between 1 and "
                        f"{self.max_loop_iterations}"
                    )
            errors.extend(self._validate_references(step, declared_inputs, seen))
            seen.add(step.id)

        if required_tools != set(workflow.required_tools):
            errors.append("required_tools does not exactly match executable tool nodes")
        if set(workflow.tool_schema_hashes) != required_tools:
            errors.append("every required tool must have exactly one schema hash")
        for name in required_tools:
            if name in hashes and workflow.tool_schema_hashes.get(name) != hashes[name]:
                errors.append(f"schema hash does not match evidence for {name}")
        return list(dict.fromkeys(errors))

    @staticmethod
    def _schemas(episodes: list[TaskEpisode]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for episode in episodes:
            for name, schema in episode.metadata.get("tool_schemas", {}).items():
                if isinstance(schema, dict):
                    result.setdefault(str(name), schema)
        return result

    @staticmethod
    def _schema_hashes(
        episodes: list[TaskEpisode], tools: set[str]
    ) -> tuple[dict[str, str], list[str]]:
        values: dict[str, set[str]] = defaultdict(set)
        for episode in episodes:
            for name, value in episode.metadata.get("tool_schema_hashes", {}).items():
                if name in tools and value:
                    values[name].add(str(value))
        errors: list[str] = []
        hashes: dict[str, str] = {}
        for name in tools:
            observed = values.get(name, set())
            if not observed:
                errors.append(f"no schema hash evidence for {name}")
            elif len(observed) > 1:
                errors.append(f"conflicting schema hash evidence for {name}")
            else:
                hashes[name] = next(iter(observed))
        return hashes, errors

    def _validate_argument_contract(self, step: WorkflowStep, schema: dict[str, Any]) -> list[str]:
        if not isinstance(step.arguments, dict):
            return [f"{step.id}: arguments must be an object"]
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        page_argument = (
            str(step.control.get("page_argument")) if step.kind == StepKind.PAGINATE else None
        )
        supplied = set(step.arguments)
        if page_argument:
            supplied.add(page_argument)
        missing = required - supplied
        extras = (
            supplied - set(properties) if schema.get("additionalProperties") is False else set()
        )
        errors = []
        if missing:
            errors.append(f"{step.id}: missing required arguments {', '.join(sorted(missing))}")
        if extras:
            errors.append(f"{step.id}: unexpected arguments {', '.join(sorted(extras))}")
        return errors

    def _validate_references(
        self, step: WorkflowStep, inputs: set[str], seen_steps: set[str]
    ) -> list[str]:
        errors: list[str] = []

        def visit(value: Any) -> None:
            if isinstance(value, str):
                if value.startswith("$input."):
                    name = value.removeprefix("$input.").split(".", 1)[0]
                    if name not in inputs:
                        errors.append(f"{step.id}: undeclared input {name!r}")
                elif value.startswith("$steps."):
                    name = value.removeprefix("$steps.").split(".", 1)[0]
                    if name not in seen_steps:
                        errors.append(f"{step.id}: forward/unknown step reference {name!r}")
                elif value.startswith("$") and not value.startswith("$loop."):
                    errors.append(f"{step.id}: unsupported reference {value!r}")
                if value.startswith("<redacted"):
                    errors.append(f"{step.id}: redacted evidence cannot become a literal")
            elif isinstance(value, dict):
                if set(value) == {"$template"}:
                    for name in re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", str(value["$template"])):
                        if name not in inputs:
                            errors.append(f"{step.id}: template uses undeclared input {name!r}")
                else:
                    for child in value.values():
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for value in (step.arguments, step.collection, step.control, step.condition):
            visit(value)
        return errors

    def _validate_predicate(self, predicate: Any, step_id: str) -> list[str]:
        if not isinstance(predicate, dict):
            return [f"{step_id}: filter predicate must be an object"]
        if "all" in predicate or "any" in predicate:
            key = "all" if "all" in predicate else "any"
            children = predicate.get(key)
            if not isinstance(children, list) or not children:
                return [f"{step_id}: {key} predicate requires children"]
            return [
                error for child in children for error in self._validate_predicate(child, step_id)
            ]
        if predicate.get("op", "eq") not in self._PREDICATES:
            return [f"{step_id}: unsupported predicate operator"]
        if not isinstance(predicate.get("path", ""), str):
            return [f"{step_id}: predicate path must be a string"]
        return []

    @staticmethod
    def _validate_condition(condition: Any, step_id: str) -> list[str]:
        if not isinstance(condition, dict):
            return [f"{step_id}: branch requires a condition"]
        if set(condition) - {"source", "key", "equals"}:
            return [f"{step_id}: branch condition uses unsupported fields"]
        if condition.get("source") not in {"input", "steps", "loop"}:
            return [f"{step_id}: branch condition source is invalid"]
        if not isinstance(condition.get("key"), str) or "equals" not in condition:
            return [f"{step_id}: branch condition requires key and equals"]
        return []


class RecordedTraceReplayValidator:
    """Replay a workflow with recorded results and require exact call consumption."""

    class _AlwaysValid:
        def verify(self, *, request: Any, workflow: Any, result: Any) -> bool:
            del request, workflow, result
            return True

    class _RecordedCompute:
        def __init__(self, traces: list[Any], cursor: dict[str, int]):
            self.traces = traces
            self.cursor = cursor

        def run(
            self,
            *,
            kind: ComputationKind,
            operation: str,
            inputs: dict[str, Any],
            output_schema: dict[str, Any] | None = None,
        ) -> ToolResult:
            del kind, output_schema
            index = self.cursor["value"]
            if index >= len(self.traces):
                return ToolResult(False, error=f"unexpected extra compute call to {operation}")
            expected = self.traces[index]
            if expected.tool != operation or expected.arguments != inputs:
                return ToolResult(False, error=f"compute call {operation} does not match trace")
            self.cursor["value"] += 1
            return ToolResult(True, value=copy.deepcopy(expected.result))

    def validate(
        self, workflow: WorkflowDefinition, episodes: list[TaskEpisode]
    ) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        replayed: list[str] = []
        schemas = WorkflowProgramValidator._schemas(episodes)
        for episode in episodes:
            error = self._replay(workflow, episode, schemas)
            if error:
                errors.append(f"{episode.task_id}: {error}")
            else:
                replayed.append(episode.task_id)
        return errors, replayed

    def _replay(
        self,
        workflow: WorkflowDefinition,
        episode: TaskEpisode,
        schemas: dict[str, dict[str, Any]],
    ) -> str | None:
        traces = [trace for trace in episode.steps if trace.success]
        variables = self._source_variables(workflow, episode, traces)
        missing = [
            variable.name
            for variable in workflow.variables
            if variable.required and variables.get(variable.name) is None
        ]
        if missing:
            return "could not bind inputs: " + ", ".join(missing)

        registry = ToolRegistry()
        cursor = {"value": 0}
        operations = sorted(workflow.required_tools)
        for operation in operations:
            schema = schemas.get(operation) or self._inferred_schema(
                [trace.arguments for trace in traces if trace.tool == operation]
            )

            def invoke(_operation: str = operation, **arguments: Any) -> Any:
                index = cursor["value"]
                if index >= len(traces):
                    raise ValueError(f"unexpected extra call to {_operation}")
                expected = traces[index]
                if expected.tool != _operation:
                    raise ValueError(
                        f"expected {expected.tool} at call {index + 1}, got {_operation}"
                    )
                if arguments != expected.arguments:
                    raise ValueError(f"arguments for {_operation} do not reproduce source trace")
                cursor["value"] += 1
                return copy.deepcopy(expected.result)

            registry.register(operation, invoke, input_schema=schema)
        replay_workflow = copy.deepcopy(workflow)
        replay_workflow.tool_schema_hashes = {
            name: registry.schema_hashes()[name] for name in replay_workflow.required_tools
        }
        from .models import TaskRequest

        result = WorkflowExecutor(
            registry,
            compute_backend=self._RecordedCompute(traces, cursor),
            verifier=self._AlwaysValid(),
        ).execute(
            TaskRequest(
                episode.instruction,
                episode.scope,
                variables,
                environment=replay_workflow.environment,
                available_tools=registry.names(),
            ),
            replay_workflow,
            variables,
        )
        if not result.success:
            return result.error or "replay execution failed"
        if cursor["value"] != len(traces):
            remaining = ", ".join(trace.tool for trace in traces[cursor["value"] :])
            return f"program left source calls unconsumed: {remaining}"
        return None

    def _source_variables(
        self,
        workflow: WorkflowDefinition,
        episode: TaskEpisode,
        traces: list[Any],
    ) -> dict[str, Any]:
        variables = copy.deepcopy(episode.variables)
        observed_operations = {trace.tool for trace in traces}
        for step in WorkflowExecutor._walk_steps(workflow.steps):
            if step.kind != StepKind.BRANCH or not isinstance(step.condition, dict):
                continue
            if step.condition.get("source") != "input":
                continue
            name = str(step.condition.get("key", "")).split(".", 1)[0]
            if not name or name in variables:
                continue
            expected = step.condition.get("equals")
            then_tools = {
                child.operation
                for child in WorkflowExecutor._walk_steps(step.body)
                if child.kind in {StepKind.TOOL, StepKind.PAGINATE, StepKind.COMPUTE}
                or (child.kind == StepKind.FOREACH and not child.body)
            }
            else_tools = {
                child.operation
                for child in WorkflowExecutor._walk_steps(step.else_body)
                if child.kind in {StepKind.TOOL, StepKind.PAGINATE, StepKind.COMPUTE}
                or (child.kind == StepKind.FOREACH and not child.body)
            }
            if then_tools & observed_operations:
                variables[name] = copy.deepcopy(expected)
            elif else_tools & observed_operations and isinstance(expected, bool):
                variables[name] = not expected
            elif (
                then_tools and not (then_tools & observed_operations) and isinstance(expected, bool)
            ):
                variables[name] = not expected
        cursor = 0
        for step in WorkflowExecutor._walk_steps(workflow.steps):
            if step.kind not in {
                StepKind.TOOL,
                StepKind.PAGINATE,
                StepKind.FOREACH,
                StepKind.COMPUTE,
            }:
                continue
            match = next(
                (
                    index
                    for index in range(cursor, len(traces))
                    if traces[index].tool == step.operation
                ),
                None,
            )
            if match is None:
                continue
            self._bind(step.arguments, traces[match].arguments, variables)
            cursor = match + 1
        return variables

    def _bind(self, expression: Any, actual: Any, variables: dict[str, Any]) -> None:
        if isinstance(expression, str) and expression.startswith("$input."):
            path = expression.removeprefix("$input.").split(".")
            if len(path) == 1:
                existing = variables.get(path[0], actual)
                if existing == actual:
                    variables[path[0]] = copy.deepcopy(actual)
            return
        if isinstance(expression, dict) and isinstance(actual, dict):
            for key, child in expression.items():
                if key in actual:
                    self._bind(child, actual[key], variables)
        elif isinstance(expression, list) and isinstance(actual, list):
            for child, item in zip(expression, actual, strict=False):
                self._bind(child, item, variables)

    @staticmethod
    def _inferred_schema(arguments: list[dict[str, Any]]) -> dict[str, Any]:
        keys = set().union(*(item.keys() for item in arguments)) if arguments else set()
        required = set(keys)
        for item in arguments:
            required &= set(item)
        return {
            "type": "object",
            "properties": {name: {} for name in sorted(keys)},
            "required": sorted(required),
            "additionalProperties": False,
        }


class ValidatedWorkflowCompiler:
    """Synthesize, repair, statically validate, and replay a bounded program."""

    def __init__(
        self,
        model: StructuredModel,
        *,
        fallback: DeterministicWorkflowCompiler | None = None,
        max_repair_attempts: int = 2,
        max_nodes: int = 64,
        max_loop_iterations: int = 100,
    ):
        self.model = model
        self.fallback = fallback or DeterministicWorkflowCompiler()
        self.max_repair_attempts = max_repair_attempts
        self.program_validator = WorkflowProgramValidator(
            max_nodes=max_nodes, max_loop_iterations=max_loop_iterations
        )
        self.replay_validator = RecordedTraceReplayValidator()

    def compile(self, episodes: list[TaskEpisode]) -> WorkflowDefinition:
        usable = [episode for episode in episodes if episode.success and episode.verified]
        if not usable:
            raise WorkflowSynthesisError("At least one verified successful episode is required")
        builder = SymbolicTraceBuilder()
        evidence = builder.build(usable)
        previous: dict[str, Any] | None = None
        validation_errors: list[str] = []
        attempts = 0
        for attempts in range(1, self.max_repair_attempts + 2):
            response = self.model.generate_json(
                system=self._system_prompt(),
                prompt=json.dumps(
                    {
                        "task": "Synthesize a reusable bounded workflow program",
                        "evidence": evidence,
                        "limits": {
                            "max_nodes": self.program_validator.max_nodes,
                            "max_loop_iterations": self.program_validator.max_loop_iterations,
                            "repair_attempt": attempts - 1,
                        },
                        "previous_proposal": previous,
                        "validation_errors": validation_errors,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
                schema=PROGRAM_SCHEMA,
            )
            previous = response
            try:
                workflow = self._workflow_from_response(response, usable, builder.literal_values)
                report = self.validate(workflow, usable)
                if report.valid:
                    workflow.metadata.update(
                        {
                            "compiler": "validated_llm",
                            "compiler_model": self.model.model_name,
                            "synthesis_attempts": attempts,
                            "validation_report": report.to_dict(),
                            "source_action_paths": [action_path(item) for item in usable],
                        }
                    )
                    return workflow
                validation_errors = report.errors[:20]
            except (KeyError, TypeError, ValueError) as exc:
                validation_errors = [f"proposal_parse_error: {type(exc).__name__}: {exc}"]

        try:
            workflow = self.fallback.compile(usable)
            report = self.validate(workflow, usable)
            if not report.valid:
                raise WorkflowSynthesisError("; ".join(report.errors))
            workflow.metadata.update(
                {
                    "compiler": "deterministic_after_llm_rejection",
                    "compiler_model": self.model.model_name,
                    "synthesis_attempts": attempts,
                    "llm_validation_errors": validation_errors,
                    "validation_report": report.to_dict(),
                }
            )
            return workflow
        except (TypeError, ValueError) as exc:
            raise WorkflowSynthesisError(
                "LLM proposal failed validation and deterministic fallback could not compile: "
                + "; ".join(validation_errors + [str(exc)])
            ) from exc

    def refine(
        self, workflow: WorkflowDefinition, episodes: list[TaskEpisode]
    ) -> WorkflowDefinition:
        revised = self.compile(episodes)
        revised.workflow_id = workflow.workflow_id
        revised.version = workflow.version + 1
        revised.status = workflow.status
        revised.created_at = workflow.created_at
        revised.stats = copy.deepcopy(workflow.stats)
        revised.stats.pattern_observations = len({item.task_id for item in episodes})
        revised.stats.source_task_ids = list(dict.fromkeys(item.task_id for item in episodes))
        revised.negative_examples = list(workflow.negative_examples)
        revised.metadata = {**workflow.metadata, **revised.metadata}
        return revised

    def validate(
        self, workflow: WorkflowDefinition, episodes: list[TaskEpisode]
    ) -> WorkflowValidationReport:
        static_errors = self.program_validator.validate(workflow, episodes)
        if static_errors:
            return WorkflowValidationReport(False, static_errors=static_errors)
        replay_errors, replayed = self.replay_validator.validate(workflow, episodes)
        return WorkflowValidationReport(
            not replay_errors,
            replay_errors=replay_errors,
            replayed_episode_ids=replayed,
        )

    def _workflow_from_response(
        self,
        response: dict[str, Any],
        episodes: list[TaskEpisode],
        literals: dict[str, Any],
    ) -> WorkflowDefinition:
        if not isinstance(response, dict):
            raise TypeError("model response must be an object")
        raw_variables = response.get("variables")
        raw_nodes = response.get("nodes")
        if not isinstance(raw_variables, list) or not isinstance(raw_nodes, list):
            raise TypeError("variables and nodes must be arrays")
        variables = []
        for item in raw_variables:
            if not isinstance(item, dict):
                raise TypeError("variable entries must be objects")
            name = str(item["name"])
            if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$", name):
                raise ValueError(f"invalid variable name {name!r}")
            variables.append(
                VariableSpec(
                    name=name,
                    type=str(item["type"]),
                    required=bool(item["required"]),
                    description=str(item["description"]),
                )
            )
        steps = self._build_steps(raw_nodes, literals)
        tools = list(
            dict.fromkeys(
                step.operation
                for step in WorkflowExecutor._walk_steps(steps)
                if step.kind in {StepKind.TOOL, StepKind.PAGINATE}
                or (step.kind == StepKind.FOREACH and not step.body)
            )
        )
        observed_tools = {
            trace.tool for episode in episodes for trace in episode.steps if trace.success
        }
        invented = set(tools) - observed_tools
        if invented:
            raise ValueError("unobserved tool(s): " + ", ".join(sorted(invented)))
        hashes, errors = WorkflowProgramValidator._schema_hashes(episodes, set(tools))
        if errors:
            raise ValueError("; ".join(errors))
        exemplar = episodes[-1]
        return WorkflowDefinition(
            name=str(response.get("name", "")).strip() or "Learned Workflow",
            description=str(response.get("description", "")).strip(),
            scope=exemplar.scope,
            variables=variables,
            steps=steps,
            intent_examples=list(dict.fromkeys(item.instruction for item in episodes))[-12:],
            structural_signature=structural_signature(exemplar),
            preconditions=[str(item) for item in response.get("preconditions", [])],
            postconditions=[str(item) for item in response.get("postconditions", [])],
            tool_schema_hashes=hashes,
            required_tools=tools,
            environment=exemplar.metadata.get("environment", "default"),
            metadata={
                "source_frameworks": sorted({item.framework for item in episodes}),
                "pattern_name": exemplar.metadata.get("pattern_name", ""),
            },
        )

    def _build_steps(self, raw_nodes: list[Any], literals: dict[str, Any]) -> list[WorkflowStep]:
        by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        ids: set[str] = set()
        for node in raw_nodes:
            if not isinstance(node, dict):
                raise TypeError("node entries must be objects")
            node_id = str(node["id"])
            if node_id in ids:
                raise ValueError(f"duplicate node ID {node_id!r}")
            ids.add(node_id)
            by_parent[str(node["parent_id"])].append(node)
        if any(parent and parent not in ids for parent in by_parent):
            raise ValueError("node references an unknown parent")

        def children(parent: str, branch: str) -> list[WorkflowStep]:
            selected = [node for node in by_parent.get(parent, []) if node["branch"] == branch]
            selected.sort(key=lambda node: int(node["order"]))
            if len({int(node["order"]) for node in selected}) != len(selected):
                raise ValueError(f"duplicate order within {parent or 'root'}:{branch}")
            return [convert(node) for node in selected]

        def convert(node: dict[str, Any]) -> WorkflowStep:
            node_id = str(node["id"])
            kind = StepKind(str(node["kind"]))
            arguments = self._decode(node["arguments_json"], "arguments_json", literals)
            condition = self._decode(node["condition_json"], "condition_json", literals)
            collection = self._decode(node["collection_json"], "collection_json", literals)
            control = self._decode(node["control_json"], "control_json", literals)
            if not isinstance(arguments, dict) or not isinstance(control, dict):
                raise TypeError(
                    f"{node_id}: arguments_json and control_json must decode to objects"
                )
            if condition is not None and not isinstance(condition, dict):
                raise TypeError(f"{node_id}: condition_json must decode to an object or null")
            if kind == StepKind.BRANCH:
                body = children(node_id, "then")
                else_body = children(node_id, "else")
            elif kind == StepKind.FOREACH:
                body = children(node_id, "body")
                else_body = []
            else:
                nested = by_parent.get(node_id, [])
                if nested:
                    raise ValueError(f"{node_id}: only branch/foreach nodes may have children")
                body = []
                else_body = []
            executor = ComputationKind(str(node["executor"]))
            if kind != StepKind.COMPUTE:
                executor = ComputationKind.DETERMINISTIC
            return WorkflowStep(
                id=node_id,
                kind=kind,
                operation=str(node["operation"]),
                agent=str(node["agent"]) or str(node["operation"]).split(".", 1)[0],
                arguments=arguments,
                executor=executor,
                fallback_executors=(
                    DeterministicWorkflowCompiler._fallbacks(executor)
                    if kind == StepKind.COMPUTE
                    else []
                ),
                output_key=str(node["output_key"]) or None,
                condition=condition,
                depends_on=[str(item) for item in node["depends_on"]],
                description=str(node["description"]),
                on_error=str(node["on_error"]),
                collection=collection,
                control=control,
                body=body,
                else_body=else_body,
            )

        roots = children("", "main")
        non_root_main = [
            node["id"]
            for parent, nodes in by_parent.items()
            if parent
            for node in nodes
            if node["branch"] == "main"
        ]
        if non_root_main:
            raise ValueError("nested nodes cannot use the main branch")
        return roots

    def _decode(self, value: Any, field_name: str, literals: dict[str, Any]) -> Any:
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a JSON string")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field_name} is not valid JSON") from exc

        def materialize(item: Any) -> Any:
            if isinstance(item, str) and item.startswith("$literal."):
                literal_id = item.removeprefix("$literal.")
                if literal_id not in literals:
                    raise ValueError(f"unknown literal evidence {literal_id!r}")
                return copy.deepcopy(literals[literal_id])
            if isinstance(item, str) and item.startswith(("$evidence.", "$sensitive.")):
                raise ValueError(f"unresolved evidence reference {item!r}")
            if isinstance(item, dict):
                return {str(key): materialize(child) for key, child in item.items()}
            if isinstance(item, list):
                return [materialize(child) for child in item]
            return item

        return materialize(decoded)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a constrained workflow compiler. Synthesize one reusable program only from "
            "the verified symbolic traces. You may normalize equivalent paths, remove optional "
            "read-only discovery calls when their output is unused, and introduce bounded "
            "pagination, foreach, filter, reduce, or branches when the evidence supports them. "
            "Never invent a tool or side effect. Use only literal IDs in the supplied catalog; "
            "never guess a hidden value. Turn values that vary by request into declared $input "
            "variables. Convert $evidence references to $steps references using your proposed "
            "step IDs. Sensitive markers must become declared inputs. All loops require an "
            "explicit max_iterations no greater than the given bound. Branch conditions may use "
            "only source/key/equals. Filter predicates use the documented bounded operators. "
            "Represent every arguments, condition, collection, and control value as a valid JSON "
            "string. Use 'null' for absent condition/collection and '{}' for empty objects. "
            "Flatten nesting with parent_id: root nodes use parent_id='' and branch='main'; "
            "branch children use then/else; foreach children use body. Return strict JSON only."
        )
