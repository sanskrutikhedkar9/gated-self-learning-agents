"""Compile successful trajectories into parameterized executable workflows."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .discovery import pagination_argument, structural_signature
from .models import (
    ComputationKind,
    StepKind,
    TaskEpisode,
    ToolCallTrace,
    VariableSpec,
    WorkflowDefinition,
    WorkflowStep,
)
from .protocols import StructuredModel, WorkflowCompiler


def _value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        inner = _value_type(value[0]) if value else "any"
        return f"list[{inner}]"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, str) and re.match(r"^\d{4}-\d{2}-\d{2}T", value):
        return "datetime"
    return "string"


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    return value or "step"


def _find_value_path(container: Any, target: Any, path: str = "") -> str | None:
    if container == target:
        return path.lstrip(".")
    if isinstance(container, dict):
        for key, value in container.items():
            found = _find_value_path(value, target, f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(container, list):
        for index, value in enumerate(container):
            found = _find_value_path(value, target, f"{path}.{index}")
            if found is not None:
                return found
    return None


_MISSING = object()


def _value_at(container: Any, path: tuple[str | int, ...]) -> Any:
    current = container
    try:
        for part in path:
            if isinstance(current, list) and isinstance(part, str) and part.isdigit():
                part = int(part)
            current = current[part]
    except (KeyError, IndexError, TypeError):
        return _MISSING
    return current


def _successful_steps(episode: TaskEpisode) -> list[ToolCallTrace]:
    return [step for step in episode.steps if step.success]


@dataclass(slots=True)
class _TraceGroup:
    representative: ToolCallTrace
    calls: list[ToolCallTrace]
    page_argument: str | None = None


def _group_successful_steps(episode: TaskEpisode) -> list[_TraceGroup]:
    steps = _successful_steps(episode)
    groups: list[_TraceGroup] = []
    index = 0
    while index < len(steps):
        operation = steps[index].tool
        cursor = index + 1
        while cursor < len(steps) and steps[cursor].tool == operation:
            cursor += 1
        run = steps[index:cursor]
        page_argument = pagination_argument(run)
        if page_argument:
            groups.append(_TraceGroup(run[0], run, page_argument))
            index = cursor
        else:
            groups.append(_TraceGroup(steps[index], [steps[index]]))
            index += 1
    return groups


def _list_result_path(value: Any, path: str = "") -> str | None:
    if isinstance(value, list):
        return path
    if isinstance(value, dict):
        for key, child in value.items():
            found = _list_result_path(child, f"{path}.{key}".lstrip("."))
            if found is not None:
                return found
    return None


def _render_template(template: str, variables: dict[str, Any]) -> str:
    return re.sub(
        r"\{\{([a-zA-Z0-9_]+)\}\}",
        lambda match: str(variables.get(match.group(1), match.group(0))),
        template,
    )


def _value_shape(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("$input.", "$steps.")):
        return value
    if isinstance(value, dict):
        if set(value) == {"$template"}:
            return {"$template": "redacted_template"}
        return {str(key): _value_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return {"type": "array", "items": _value_shape(value[0]) if value else "any"}
    return {"type": _value_type(value)}


class DeterministicWorkflowCompiler:
    """Anti-unifies typed inputs and data flow without a model call.

    This compiler is intentionally conservative: it preserves the observed tool
    path, parameterizes known episode variables, and never invents a tool.
    """

    def compile(self, episodes: list[TaskEpisode]) -> WorkflowDefinition:
        usable = [episode for episode in episodes if episode.success and episode.verified]
        if not usable:
            raise ValueError("At least one verified successful episode is required")
        exemplar = usable[-1]
        variables = self._variables(usable)
        steps, generated_types = self._steps(usable, variables)
        known_names = {variable.name for variable in variables}
        for name in sorted(self._referenced_inputs(steps) - known_names):
            variables.append(
                VariableSpec(
                    name=name,
                    type=generated_types.get(name, "string"),
                    required=True,
                    description=f"Generated input for {name.replace('_', ' ')}",
                )
            )
        tools = list(
            dict.fromkeys(
                step.operation
                for step in steps
                if step.kind in {StepKind.TOOL, StepKind.PAGINATE, StepKind.FOREACH}
            )
        )
        observed_hashes = exemplar.metadata.get("tool_schema_hashes", {})
        label = exemplar.metadata.get("pattern_name") or self._label(tools)
        workflow = WorkflowDefinition(
            name=label.replace("_", " ").title(),
            description=f"Learned reusable workflow for {label.replace('_', ' ')}.",
            scope=exemplar.scope,
            variables=variables,
            steps=steps,
            intent_examples=list(dict.fromkeys(item.instruction for item in usable))[-12:],
            structural_signature=structural_signature(exemplar),
            required_tools=tools,
            tool_schema_hashes={
                name: str(observed_hashes[name]) for name in tools if name in observed_hashes
            },
            environment=exemplar.metadata.get("environment", "default"),
            preconditions=list(exemplar.metadata.get("preconditions", [])),
            postconditions=list(exemplar.metadata.get("postconditions", [])),
            metadata={
                "compiler": "deterministic",
                "source_frameworks": sorted({item.framework for item in usable}),
                "pattern_name": exemplar.metadata.get("pattern_name", ""),
            },
        )
        workflow.stats.pattern_observations = len(usable)
        workflow.stats.source_task_ids = [item.task_id for item in usable]
        return workflow

    def refine(
        self,
        workflow: WorkflowDefinition,
        episodes: list[TaskEpisode],
    ) -> WorkflowDefinition:
        revised = self.compile(episodes)
        revised.workflow_id = workflow.workflow_id
        revised.version = workflow.version + 1
        revised.status = workflow.status
        revised.created_at = workflow.created_at
        revised.stats = copy.deepcopy(workflow.stats)
        revised.tool_schema_hashes = copy.deepcopy(workflow.tool_schema_hashes)
        revised.stats.pattern_observations = len({item.task_id for item in episodes})
        revised.stats.source_task_ids = list(dict.fromkeys(item.task_id for item in episodes))
        revised.negative_examples = list(workflow.negative_examples)
        revised.metadata.update(workflow.metadata)
        return revised

    @staticmethod
    def _label(tools: list[str]) -> str:
        meaningful = [_safe_name(tool.split(".")[-1]) for tool in tools]
        return "_then_".join(meaningful[:4]) or "learned_workflow"

    @staticmethod
    def _variables(episodes: list[TaskEpisode]) -> list[VariableSpec]:
        names = sorted({name for episode in episodes for name in episode.variables})
        variables: list[VariableSpec] = []
        for name in names:
            examples = [
                episode.variables[name] for episode in episodes if name in episode.variables
            ]
            variables.append(
                VariableSpec(
                    name=name,
                    type=_value_type(examples[0]) if examples else "string",
                    required=all(name in episode.variables for episode in episodes),
                    description=f"Learned input for {name.replace('_', ' ')}",
                    examples=list(
                        dict.fromkeys(
                            json.dumps(item, sort_keys=True, default=str) for item in examples
                        )
                    )[:5],
                )
            )
        # Restore JSON-encoded examples to their original types.
        for variable in variables:
            variable.examples = [json.loads(item) for item in variable.examples]
        return variables

    def _steps(
        self,
        episodes: list[TaskEpisode],
        variables: list[VariableSpec],
    ) -> tuple[list[WorkflowStep], dict[str, str]]:
        exemplar = episodes[-1]
        group_sets = [_group_successful_steps(episode) for episode in episodes]
        exemplar_groups = group_sets[-1]
        if any(
            len(groups) != len(exemplar_groups)
            or any(
                left.representative.tool != right.representative.tool
                or left.page_argument != right.page_argument
                for left, right in zip(groups, exemplar_groups, strict=True)
            )
            for groups in group_sets
        ):
            raise ValueError("Verified episodes do not share one compilable control-flow path")
        trace_sets = [[group.representative for group in groups] for groups in group_sets]
        step_ids: list[str] = []
        workflow_steps: list[WorkflowStep] = []
        seen_counts: Counter[str] = Counter()
        known_variables = {variable.name for variable in variables}
        generated_types: dict[str, str] = {}

        for index, group in enumerate(exemplar_groups):
            trace = group.representative
            base = _safe_name(trace.tool.split(".")[-1])
            seen_counts[base] += 1
            step_id = base if seen_counts[base] == 1 else f"{base}_{seen_counts[base]}"
            arguments = {
                key: self._generalize_argument(
                    value,
                    key,
                    episodes,
                    trace_sets,
                    index,
                    (key,),
                    known_variables,
                    generated_types,
                    workflow_steps,
                    exemplar,
                )
                for key, value in trace.arguments.items()
                if key != group.page_argument
            }
            control: dict[str, Any] = {}
            step_kind = StepKind.COMPUTE if trace.tool.startswith("compute.") else StepKind.TOOL
            if group.page_argument:
                values = [call.arguments[group.page_argument] for call in group.calls]
                control = {
                    "page_argument": group.page_argument,
                    "start": values[0],
                    "increment": values[1] - values[0],
                    "max_iterations": 100,
                    "items_path": _list_result_path(group.calls[0].result) or "",
                    "allow_truncation": False,
                }
                for candidate in ("page_size", "limit"):
                    if candidate in trace.arguments:
                        control["page_size_argument"] = candidate
                        break
                step_kind = StepKind.PAGINATE
            workflow_steps.append(
                WorkflowStep(
                    id=step_id,
                    kind=step_kind,
                    operation=trace.tool,
                    agent=trace.agent,
                    arguments=arguments,
                    executor=(
                        trace.executor
                        if trace.tool.startswith("compute.")
                        else ComputationKind.DETERMINISTIC
                    ),
                    fallback_executors=(
                        self._fallbacks(trace.executor) if trace.tool.startswith("compute.") else []
                    ),
                    depends_on=list(step_ids[-1:]),
                    description=f"Execute {trace.tool}",
                    control=control,
                )
            )
            step_ids.append(step_id)
        return workflow_steps, generated_types

    @staticmethod
    def _fallbacks(kind: ComputationKind) -> list[ComputationKind]:
        ladder = [
            ComputationKind.DETERMINISTIC,
            ComputationKind.SLM,
            ComputationKind.LLM,
            ComputationKind.FULL_AGENT,
        ]
        return ladder[ladder.index(kind) + 1 :]

    @staticmethod
    def _referenced_inputs(steps: list[WorkflowStep]) -> set[str]:
        names: set[str] = set()

        def visit(value: Any) -> None:
            if isinstance(value, str) and value.startswith("$input."):
                names.add(value.removeprefix("$input.").split(".", 1)[0])
            elif isinstance(value, dict):
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        def visit_step(step: WorkflowStep) -> None:
            visit(step.arguments)
            visit(step.collection)
            visit(step.control)
            for child in [*step.body, *step.else_body]:
                visit_step(child)

        for step in steps:
            visit_step(step)
        return names

    def _generalize_argument(
        self,
        exemplar_value: Any,
        argument_name: str,
        episodes: list[TaskEpisode],
        trace_sets: list[list[Any]],
        step_index: int,
        argument_path: tuple[str | int, ...],
        known_variables: set[str],
        generated_types: dict[str, str],
        prior_steps: list[WorkflowStep],
        exemplar: TaskEpisode,
    ) -> Any:
        for name in sorted(known_variables):
            if all(
                name in episode.variables
                and step_index < len(traces)
                and _value_at(traces[step_index].arguments, argument_path)
                == episode.variables[name]
                for episode, traces in zip(episodes, trace_sets, strict=True)
            ):
                return f"$input.{name}"

        for prior_index in range(step_index - 1, -1, -1):
            trace = trace_sets[-1][prior_index]
            path = _find_value_path(trace.result, exemplar_value)
            result_path = tuple(path.split(".")) if path else ()
            if (
                path is not None
                and prior_index < len(prior_steps)
                and all(
                    step_index < len(traces)
                    and prior_index < len(traces)
                    and _value_at(traces[step_index].arguments, argument_path)
                    == _value_at(traces[prior_index].result, result_path)
                    for traces in trace_sets
                )
            ):
                suffix = f".{path}" if path else ""
                return f"$steps.{prior_steps[prior_index].id}{suffix}"

        if isinstance(exemplar_value, list):
            return [
                self._generalize_argument(
                    item,
                    f"{argument_name}_{index}",
                    episodes,
                    trace_sets,
                    step_index,
                    (*argument_path, index),
                    known_variables,
                    generated_types,
                    prior_steps,
                    exemplar,
                )
                for index, item in enumerate(exemplar_value)
            ]
        if isinstance(exemplar_value, dict):
            return {
                key: self._generalize_argument(
                    item,
                    f"{argument_name}_{key}",
                    episodes,
                    trace_sets,
                    step_index,
                    (*argument_path, key),
                    known_variables,
                    generated_types,
                    prior_steps,
                    exemplar,
                )
                for key, item in exemplar_value.items()
            }

        if isinstance(exemplar_value, str):
            templated = exemplar_value
            replaced = False
            for name in sorted(known_variables):
                variable_value = exemplar.variables.get(name)
                if (
                    isinstance(variable_value, str)
                    and variable_value
                    and variable_value in templated
                ):
                    templated = templated.replace(variable_value, "{{" + name + "}}")
                    replaced = True
            if replaced and all(
                step_index < len(traces)
                and _value_at(traces[step_index].arguments, argument_path)
                == _render_template(templated, episode.variables)
                for episode, traces in zip(episodes, trace_sets, strict=True)
            ):
                return {"$template": templated}

        # If this argument differs across aligned episodes, create a stable input.
        aligned_values: list[Any] = []
        for traces in trace_sets:
            if step_index < len(traces):
                aligned_values.append(_value_at(traces[step_index].arguments, argument_path))
        serialized = {json.dumps(item, sort_keys=True, default=str) for item in aligned_values}
        if len(serialized) > 1 or (
            isinstance(exemplar_value, str) and exemplar_value.startswith("<redacted:")
        ):
            generated_name = (
                f"{_safe_name(trace_sets[-1][step_index].tool.split('.')[-1])}_{argument_name}"
            )
            value_types = {_value_type(item) for item in aligned_values if item is not _MISSING}
            generated_types[generated_name] = (
                next(iter(value_types)) if len(value_types) == 1 else "any"
            )
            return f"$input.{generated_name}"
        return exemplar_value


class ModelAssistedWorkflowCompiler:
    """Let an LLM annotate a conservative draft while preventing tool invention."""

    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "description", "preconditions", "postconditions", "step_executors"],
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "preconditions": {"type": "array", "items": {"type": "string"}},
            "postconditions": {"type": "array", "items": {"type": "string"}},
            "step_executors": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["step_id", "executor"],
                    "properties": {
                        "step_id": {"type": "string"},
                        "executor": {
                            "type": "string",
                            "enum": ["deterministic", "slm", "llm"],
                        },
                    },
                },
            },
        },
    }

    def __init__(
        self, model: StructuredModel, fallback: DeterministicWorkflowCompiler | None = None
    ):
        self.model = model
        self.fallback = fallback or DeterministicWorkflowCompiler()

    def compile(self, episodes: list[TaskEpisode]) -> WorkflowDefinition:
        draft = self.fallback.compile(episodes)
        return self._annotate(draft, episodes)

    def refine(
        self,
        workflow: WorkflowDefinition,
        episodes: list[TaskEpisode],
    ) -> WorkflowDefinition:
        draft = self.fallback.refine(workflow, episodes)
        return self._annotate(draft, episodes)

    def _annotate(
        self,
        draft: WorkflowDefinition,
        episodes: list[TaskEpisode],
    ) -> WorkflowDefinition:
        response = self.model.generate_json(
            system=(
                "You compile repeated successful agent traces into a reusable workflow. "
                "Do not add, remove, or reorder operations. Decide only whether each existing "
                "step can be deterministic, needs a small language model for bounded semantic "
                "work, or needs a large language model for genuine reasoning. Prefer the cheapest "
                "safe executor. Return strict JSON."
            ),
            prompt=json.dumps(
                {
                    "draft": {
                        "name": draft.name,
                        "description": draft.description,
                        "variables": [
                            {
                                "name": variable.name,
                                "type": variable.type,
                                "required": variable.required,
                                "description": variable.description,
                            }
                            for variable in draft.variables
                        ],
                        "steps": [
                            {
                                "id": step.id,
                                "kind": str(step.kind),
                                "operation": step.operation,
                                "argument_shape": _value_shape(step.arguments),
                                "executor": str(step.executor),
                            }
                            for step in draft.steps
                        ],
                    },
                    "episodes": [
                        {
                            "instruction": episode.instruction,
                            "steps": [
                                {
                                    "operation": trace.tool,
                                    "argument_shape": _value_shape(trace.arguments),
                                    "result_type": _value_type(trace.result),
                                    "success": trace.success,
                                }
                                for trace in episode.steps
                            ],
                        }
                        for episode in episodes[-5:]
                    ],
                },
                ensure_ascii=False,
                default=str,
            ),
            schema=self._SCHEMA,
        )
        draft.name = str(response["name"]).strip() or draft.name
        draft.description = str(response["description"]).strip() or draft.description
        draft.preconditions = [str(item) for item in response["preconditions"]]
        draft.postconditions = [str(item) for item in response["postconditions"]]
        raw_executors = response.get("step_executors", [])
        executor_map = {
            str(item.get("step_id")): item.get("executor")
            for item in raw_executors
            if isinstance(item, dict)
        }
        for step in draft.steps:
            proposed = executor_map.get(step.id)
            if proposed in {"deterministic", "slm", "llm"}:
                # Tool invocation is deterministic once arguments are bound. A model may
                # only be assigned to explicit compute steps.
                if step.kind == StepKind.COMPUTE:
                    step.executor = ComputationKind(proposed)
                    step.fallback_executors = self.fallback._fallbacks(step.executor)
                else:
                    step.executor = ComputationKind.DETERMINISTIC
        draft.metadata["compiler"] = "model_assisted"
        draft.metadata["compiler_model"] = self.model.model_name
        return draft


class EvidenceGatedWorkflowCompiler:
    """Delay paid model synthesis until a pattern has enough support."""

    def __init__(
        self,
        model_compiler: WorkflowCompiler,
        *,
        min_observations: int = 3,
        fallback: DeterministicWorkflowCompiler | None = None,
    ):
        self.model_compiler = model_compiler
        self.min_observations = min_observations
        self.fallback = fallback or DeterministicWorkflowCompiler()

    def compile(self, episodes: list[TaskEpisode]) -> WorkflowDefinition:
        if len(episodes) < self.min_observations:
            return self.fallback.compile(episodes)
        return self.model_compiler.compile(episodes)

    def refine(
        self,
        workflow: WorkflowDefinition,
        episodes: list[TaskEpisode],
    ) -> WorkflowDefinition:
        if len(episodes) < self.min_observations:
            try:
                return self.fallback.refine(workflow, episodes)
            except ValueError:
                # A structurally different but semantically related trace is
                # retained as pending evidence. It must not promote the old
                # executable draft before enough evidence exists for synthesis.
                deferred = copy.deepcopy(workflow)
                deferred.metadata["compilation_deferred"] = True
                deferred.metadata["deferred_observations"] = len(episodes)
                return deferred
        return self.model_compiler.refine(workflow, episodes)
