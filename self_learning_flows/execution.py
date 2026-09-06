"""Validated execution of compiled workflows."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import (
    ComputationKind,
    ExecutionResult,
    StepKind,
    TaskRequest,
    ToolCallTrace,
    ToolResult,
    WorkflowDefinition,
)
from .protocols import ComputeBackend, OutcomeVerifier

ToolFunction = Callable[..., Any]


@dataclass(slots=True)
class RegisteredTool:
    name: str
    function: ToolFunction
    input_schema: dict[str, Any]
    description: str = ""
    agent: str = "*"

    @property
    def schema_hash(self) -> str:
        encoded = json.dumps(self.input_schema, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(slots=True)
class _ExecutionState:
    outputs: dict[str, Any] = field(default_factory=dict)
    traces: list[ToolCallTrace] = field(default_factory=list)
    executors: list[ComputationKind] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        name: str,
        function: ToolFunction,
        *,
        input_schema: dict[str, Any] | None = None,
        description: str = "",
        agent: str = "*",
    ) -> None:
        schema = input_schema or self._schema_from_signature(function)
        self._tools[name] = RegisteredTool(name, function, schema, description, agent)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> RegisteredTool | None:
        return self._tools.get(name)

    def schema_hashes(self) -> dict[str, str]:
        return {name: tool.schema_hash for name, tool in self._tools.items()}

    def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Unknown tool: {name}")
        error = self._validate(arguments, tool.input_schema)
        if error:
            return ToolResult(success=False, error=error)
        started = time.perf_counter()
        try:
            value = tool.function(**arguments)
            return ToolResult(
                success=True,
                value=value,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:  # tool boundaries must become data, not crash the runtime
            return ToolResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - started) * 1000,
            )

    @staticmethod
    def _schema_from_signature(function: ToolFunction) -> dict[str, Any]:
        signature = inspect.signature(function)
        properties: dict[str, Any] = {}
        required: list[str] = []
        for name, parameter in signature.parameters.items():
            if name in {"self", "cls"}:
                continue
            properties[name] = {}
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
        return {"type": "object", "properties": properties, "required": required}

    @staticmethod
    def _validate(arguments: dict[str, Any], schema: dict[str, Any]) -> str | None:
        return ToolRegistry._validate_schema(arguments, schema, "arguments")

    @staticmethod
    def _validate_schema(value: Any, schema: dict[str, Any], path: str) -> str | None:
        declared = schema.get("type")
        allowed = declared if isinstance(declared, list) else [declared] if declared else []
        if value is None and "null" in allowed:
            return None
        checks = {
            "object": lambda item: isinstance(item, dict),
            "array": lambda item: isinstance(item, list),
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int) and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float)) and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "null": lambda item: item is None,
        }
        if allowed and not any(checks[kind](value) for kind in allowed if kind in checks):
            return f"{path} has the wrong type; expected {' or '.join(allowed)}"
        if "enum" in schema and value not in schema["enum"]:
            return f"{path} is not one of the allowed values"
        if isinstance(value, dict):
            missing = [name for name in schema.get("required", []) if name not in value]
            if missing:
                return f"Missing required arguments at {path}: {', '.join(missing)}"
            properties = schema.get("properties", {})
            if schema.get("additionalProperties") is False:
                extras = set(value) - set(properties)
                if extras:
                    return f"Unexpected arguments at {path}: {', '.join(sorted(extras))}"
            for name, item in value.items():
                child_schema = properties.get(name)
                if child_schema:
                    error = ToolRegistry._validate_schema(item, child_schema, f"{path}.{name}")
                    if error:
                        return error
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                error = ToolRegistry._validate_schema(item, schema["items"], f"{path}[{index}]")
                if error:
                    return error
        return None


class WorkflowExecutor:
    def __init__(
        self,
        tools: ToolRegistry,
        *,
        compute_backend: ComputeBackend | None = None,
        verifier: OutcomeVerifier | None = None,
        require_verifier: bool = True,
        require_schema_hashes: bool = True,
    ):
        self.tools = tools
        self.compute_backend = compute_backend
        self.verifier = verifier
        self.require_verifier = require_verifier
        self.require_schema_hashes = require_schema_hashes

    def check_contracts(self, workflow: WorkflowDefinition) -> list[str]:
        errors: list[str] = []
        current_hashes = self.tools.schema_hashes()
        step_tools = {
            step.operation
            for step in self._walk_steps(workflow.steps)
            if step.kind in {StepKind.TOOL, StepKind.PAGINATE, StepKind.FOREACH}
        }
        required_tools = set(workflow.required_tools) | step_tools
        if not workflow.steps:
            errors.append("Workflow has no executable steps")
        all_steps = list(self._walk_steps(workflow.steps))
        step_ids = [step.id for step in all_steps]
        if len(step_ids) != len(set(step_ids)):
            errors.append("Workflow step IDs are not unique")
        seen_steps: set[str] = set()
        for step in all_steps:
            unknown = set(step.depends_on) - seen_steps
            if unknown:
                errors.append(
                    f"Step {step.id} has unresolved dependencies: {', '.join(sorted(unknown))}"
                )
            if step.kind in {StepKind.PAGINATE, StepKind.FOREACH}:
                maximum = step.control.get("max_iterations", 100)
                if not isinstance(maximum, int) or not 1 <= maximum <= 1000:
                    errors.append(
                        f"Step {step.id} max_iterations must be an integer from 1 to 1000"
                    )
            seen_steps.add(step.id)
        for tool_name in sorted(required_tools):
            if tool_name not in current_hashes:
                errors.append(f"Required tool unavailable: {tool_name}")
            expected = workflow.tool_schema_hashes.get(tool_name)
            if not expected and self.require_schema_hashes:
                errors.append(f"No learned schema contract: {tool_name}")
            elif expected and current_hashes.get(tool_name) != expected:
                errors.append(f"Tool schema changed: {tool_name}")
        return errors

    @classmethod
    def _walk_steps(cls, steps: list[Any]):
        for step in steps:
            yield step
            yield from cls._walk_steps(step.body)
            yield from cls._walk_steps(step.else_body)

    def execute(
        self,
        request: TaskRequest,
        workflow: WorkflowDefinition,
        variables: dict[str, Any],
    ) -> ExecutionResult:
        started = time.perf_counter()
        contract_errors = self.check_contracts(workflow)
        if self.verifier is None and self.require_verifier:
            contract_errors.append("No outcome verifier configured")
        missing = [
            item.name
            for item in workflow.variables
            if item.required and variables.get(item.name) is None
        ]
        if missing:
            contract_errors.append(f"Missing workflow variables: {', '.join(missing)}")
        for variable in workflow.variables:
            if variable.name not in variables or variables[variable.name] is None:
                continue
            error = self._validate_variable(variable.name, variables[variable.name], variable.type)
            if error:
                contract_errors.append(error)
        if contract_errors:
            return self._failed(workflow, started, "; ".join(contract_errors))

        state = _ExecutionState()
        error = self._execute_steps(workflow.steps, variables, state, loop={})
        if error:
            return self._failed(
                workflow,
                started,
                error,
                outputs=state.outputs,
                traces=state.traces,
                executors=state.executors,
                input_tokens=state.input_tokens,
                output_tokens=state.output_tokens,
                reasoning_calls=state.reasoning_calls,
            )

        elapsed = (time.perf_counter() - started) * 1000
        result = ExecutionResult(
            workflow_id=workflow.workflow_id,
            success=True,
            verified=False,
            outputs=state.outputs,
            steps=state.traces,
            executor_kinds=state.executors,
            latency_ms=elapsed,
            input_tokens=state.input_tokens,
            output_tokens=state.output_tokens,
            reasoning_calls=state.reasoning_calls,
        )
        if self.verifier is not None:
            try:
                result.verified = bool(
                    self.verifier.verify(
                        request=request,
                        workflow=workflow,
                        result=result,
                    )
                )
            except Exception as exc:  # verifier failures must fail closed after side effects
                result.verified = False
                result.error = f"Outcome verifier failed: {type(exc).__name__}: {exc}"
                result.escalated = True
            result.success = result.success and result.verified
            if not result.verified and result.error is None:
                result.error = "Postcondition verification failed"
                result.escalated = True
            if not result.verified:
                result.side_effects_may_have_occurred = bool(result.steps)
        return result

    def _execute_steps(
        self,
        steps: list[Any],
        variables: dict[str, Any],
        state: _ExecutionState,
        *,
        loop: dict[str, Any],
    ) -> str | None:
        for step in steps:
            if step.kind == StepKind.BRANCH:
                selected = self._condition_passes(step.condition, variables, state.outputs, loop)
                branch = step.body if selected else step.else_body
                self._record_result(
                    step,
                    {"branch": "then" if selected else "else"},
                    state,
                    arguments={"condition": step.condition or {}},
                )
                state.outputs[step.id] = selected
                error = self._execute_steps(branch, variables, state, loop=loop)
                if error:
                    return error
                continue
            if not self._condition_passes(step.condition, variables, state.outputs, loop):
                continue
            try:
                if step.kind == StepKind.PAGINATE:
                    error = self._execute_pagination(step, variables, state, loop)
                elif step.kind == StepKind.FOREACH:
                    error = self._execute_foreach(step, variables, state, loop)
                elif step.kind == StepKind.FILTER:
                    error = self._execute_filter(step, variables, state, loop)
                elif step.kind == StepKind.REDUCE:
                    error = self._execute_reduce(step, variables, state, loop)
                else:
                    arguments = self._resolve(step.arguments, variables, state.outputs, loop)
                    result = self._invoke_step(step, arguments, state)
                    error = None if result.success else f"Step {step.id} failed: {result.error}"
                    if result.success:
                        self._store_output(step, result.value, state)
            except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
                error = f"Could not execute {step.id}: {exc}"
            if error:
                return error
        return None

    def _invoke_step(
        self,
        step: Any,
        arguments: dict[str, Any],
        state: _ExecutionState,
        *,
        trace_id: str | None = None,
    ) -> ToolResult:
        if step.kind in {StepKind.TOOL, StepKind.PAGINATE, StepKind.FOREACH}:
            attempts = [
                (ComputationKind.DETERMINISTIC, self.tools.invoke(step.operation, arguments))
            ]
        elif self.compute_backend is None:
            attempts = [
                (
                    step.executor,
                    ToolResult(False, error=f"No compute backend for {step.executor}"),
                )
            ]
        else:
            attempts = []
            for candidate in [step.executor, *step.fallback_executors]:
                try:
                    attempt = self.compute_backend.run(
                        kind=candidate,
                        operation=step.operation,
                        inputs=arguments,
                    )
                except Exception as exc:  # compute providers are an external boundary
                    attempt = ToolResult(False, error=f"{type(exc).__name__}: {exc}")
                attempts.append((candidate, attempt))
                if attempt.success:
                    break

        for index, (executor, result) in enumerate(attempts, start=1):
            suffix = "" if len(attempts) == 1 else f"#attempt-{index}"
            self._record_result(
                step,
                result.value,
                state,
                arguments=arguments,
                executor=executor,
                trace_id=(trace_id or step.id) + suffix,
                success=result.success,
                error=result.error,
                latency_ms=result.latency_ms,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                reasoning_calls=result.reasoning_calls,
            )
        return attempts[-1][1]

    def _execute_pagination(
        self,
        step: Any,
        variables: dict[str, Any],
        state: _ExecutionState,
        loop: dict[str, Any],
    ) -> str | None:
        page_argument = str(step.control.get("page_argument", "page_index"))
        page = step.control.get("start", 0)
        increment = step.control.get("increment", 1)
        maximum = int(step.control.get("max_iterations", 100))
        items_path = str(step.control.get("items_path", ""))
        has_more_path = step.control.get("has_more_path")
        page_size_argument = step.control.get("page_size_argument")
        aggregate: list[Any] = []
        result_template: Any = None
        stopped = False
        for index in range(maximum):
            iteration_loop = {**loop, "page": page, "index": index}
            arguments = self._resolve(step.arguments, variables, state.outputs, iteration_loop)
            arguments[page_argument] = page
            result = self._invoke_step(
                step, arguments, state, trace_id=f"{step.id}#page-{index + 1}"
            )
            if not result.success:
                return f"Step {step.id} failed on page {page}: {result.error}"
            page_items = self._path(result.value, items_path)
            if not isinstance(page_items, list):
                return f"Step {step.id} expected a list at result path {items_path!r}"
            if result_template is None:
                result_template = result.value
            aggregate.extend(page_items)
            if not page_items:
                stopped = True
                break
            if has_more_path is not None and not bool(self._path(result.value, str(has_more_path))):
                stopped = True
                break
            if page_size_argument and isinstance(arguments.get(page_size_argument), int):
                if len(page_items) < arguments[page_size_argument]:
                    stopped = True
                    break
            page += increment
        if not stopped and not step.control.get("allow_truncation", False):
            return f"Step {step.id} reached its pagination bound without a stop signal"
        output = (
            self._replace_path(result_template, items_path, aggregate) if items_path else aggregate
        )
        self._store_output(step, output, state)
        return None

    def _execute_foreach(
        self,
        step: Any,
        variables: dict[str, Any],
        state: _ExecutionState,
        loop: dict[str, Any],
    ) -> str | None:
        items = self._resolve(step.collection, variables, state.outputs, loop)
        if not isinstance(items, list):
            return f"Step {step.id} foreach collection is not a list"
        maximum = int(step.control.get("max_iterations", 100))
        if len(items) > maximum:
            return f"Step {step.id} collection exceeds max_iterations={maximum}"
        collected: list[Any] = []
        for index, item in enumerate(items):
            iteration_loop = {**loop, "item": item, "index": index}
            if step.body:
                error = self._execute_steps(step.body, variables, state, loop=iteration_loop)
                if error:
                    return error
                collect = step.control.get("collect")
                collected.append(
                    self._resolve(collect, variables, state.outputs, iteration_loop)
                    if collect is not None
                    else item
                )
                continue
            arguments = self._resolve(step.arguments, variables, state.outputs, iteration_loop)
            result = self._invoke_step(
                step, arguments, state, trace_id=f"{step.id}#item-{index + 1}"
            )
            if not result.success:
                return f"Step {step.id} failed on item {index}: {result.error}"
            collected.append(result.value)
        self._store_output(step, collected, state)
        return None

    def _execute_filter(
        self,
        step: Any,
        variables: dict[str, Any],
        state: _ExecutionState,
        loop: dict[str, Any],
    ) -> str | None:
        items = self._resolve(step.collection, variables, state.outputs, loop)
        if not isinstance(items, list):
            return f"Step {step.id} filter collection is not a list"
        predicate = step.control.get("predicate")
        if not isinstance(predicate, dict):
            return f"Step {step.id} has no valid filter predicate"
        selected = [
            item
            for index, item in enumerate(items)
            if self._predicate_passes(
                predicate,
                item,
                variables,
                state.outputs,
                {**loop, "item": item, "index": index},
            )
        ]
        self._record_result(
            step,
            selected,
            state,
            arguments={"input_count": len(items), "predicate": predicate},
        )
        self._store_output(step, selected, state)
        return None

    def _execute_reduce(
        self,
        step: Any,
        variables: dict[str, Any],
        state: _ExecutionState,
        loop: dict[str, Any],
    ) -> str | None:
        items = self._resolve(step.collection, variables, state.outputs, loop)
        if not isinstance(items, list):
            return f"Step {step.id} reduce collection is not a list"
        operation = step.operation
        path = str(step.control.get("path", ""))
        if operation == "flatten":
            value = [child for item in items for child in item]
        elif operation == "unique":
            value = []
            seen: set[str] = set()
            for item in items:
                key = json.dumps(item, sort_keys=True, default=str)
                if key not in seen:
                    value.append(item)
                    seen.add(key)
        elif operation == "first":
            value = items[0] if items else None
        elif operation == "last":
            value = items[-1] if items else None
        elif operation == "count":
            value = len(items)
        elif operation in {"sum", "min", "max"}:
            values = [self._path(item, path) for item in items]
            value = {"sum": sum, "min": min, "max": max}[operation](values)
        elif operation in {"sort", "top_k"}:
            value = sorted(
                items,
                key=lambda item: self._path(item, path),
                reverse=bool(step.control.get("descending", False)),
            )
            if operation == "top_k":
                value = value[: int(step.control.get("k", 1))]
        else:
            return f"Step {step.id} uses unsupported deterministic reducer {operation!r}"
        self._record_result(
            step, value, state, arguments={"input_count": len(items), **step.control}
        )
        self._store_output(step, value, state)
        return None

    @staticmethod
    def _store_output(step: Any, value: Any, state: _ExecutionState) -> None:
        state.outputs[step.id] = value
        if step.output_key:
            state.outputs[step.output_key] = value

    @staticmethod
    def _record_result(
        step: Any,
        value: Any,
        state: _ExecutionState,
        *,
        arguments: dict[str, Any],
        executor: ComputationKind = ComputationKind.DETERMINISTIC,
        trace_id: str | None = None,
        success: bool = True,
        error: str | None = None,
        latency_ms: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_calls: int = 0,
    ) -> None:
        state.executors.append(executor)
        state.input_tokens += input_tokens
        state.output_tokens += output_tokens
        state.reasoning_calls += reasoning_calls
        state.traces.append(
            ToolCallTrace(
                step_id=trace_id or step.id,
                agent=step.agent,
                tool=step.operation,
                arguments=arguments,
                result=value,
                success=success,
                error=error,
                depends_on=step.depends_on,
                executor=executor,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                reasoning_calls=reasoning_calls,
            )
        )

    @staticmethod
    def _validate_variable(name: str, value: Any, declared_type: str) -> str | None:
        if declared_type.startswith("list"):
            valid = isinstance(value, list)
        elif declared_type == "object":
            valid = isinstance(value, dict)
        elif declared_type == "integer":
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif declared_type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif declared_type == "boolean":
            valid = isinstance(value, bool)
        elif declared_type in {"string", "datetime"}:
            valid = isinstance(value, str)
        else:
            valid = True
        if not valid:
            return f"Workflow variable {name} has the wrong type; expected {declared_type}"
        return None

    @staticmethod
    def _condition_passes(
        condition: dict[str, Any] | None,
        variables: dict[str, Any],
        outputs: dict[str, Any],
        loop: dict[str, Any] | None = None,
    ) -> bool:
        if not condition:
            return True
        source = condition.get("source", "input")
        key = str(condition.get("key", ""))
        expected = condition.get("equals", True)
        containers = {"input": variables, "steps": outputs, "loop": loop or {}}
        container = containers.get(source)
        if container is None:
            return False
        try:
            actual = WorkflowExecutor._path(container, key)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            return False
        return actual == expected

    def _resolve(
        self,
        value: Any,
        variables: dict[str, Any],
        outputs: dict[str, Any],
        loop: dict[str, Any] | None = None,
    ) -> Any:
        if isinstance(value, str) and value.startswith("$input."):
            return self._path(variables, value.removeprefix("$input."))
        if isinstance(value, str) and value.startswith("$steps."):
            return self._path(outputs, value.removeprefix("$steps."))
        if isinstance(value, str) and value.startswith("$loop."):
            return self._path(loop or {}, value.removeprefix("$loop."))
        if isinstance(value, dict) and set(value) == {"$template"}:
            template = str(value["$template"])

            def replace(match: re.Match[str]) -> str:
                name = match.group(1)
                if name not in variables or variables[name] is None:
                    raise KeyError(f"Missing template variable: {name}")
                return str(variables[name])

            return re.sub(
                r"\{\{([a-zA-Z0-9_]+)\}\}",
                replace,
                template,
            )
        if isinstance(value, dict):
            return {
                key: self._resolve(item, variables, outputs, loop) for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._resolve(item, variables, outputs, loop) for item in value]
        return value

    def _predicate_passes(
        self,
        predicate: dict[str, Any],
        item: Any,
        variables: dict[str, Any],
        outputs: dict[str, Any],
        loop: dict[str, Any],
    ) -> bool:
        if "all" in predicate:
            return all(
                self._predicate_passes(child, item, variables, outputs, loop)
                for child in predicate["all"]
            )
        if "any" in predicate:
            return any(
                self._predicate_passes(child, item, variables, outputs, loop)
                for child in predicate["any"]
            )
        path = str(predicate.get("path", ""))
        try:
            actual = self._path(item, path)
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            return predicate.get("op") == "not_exists"
        expected = self._resolve(predicate.get("value"), variables, outputs, loop)
        operation = predicate.get("op", "eq")
        comparisons = {
            "eq": lambda: actual == expected,
            "ne": lambda: actual != expected,
            "in": lambda: actual in expected,
            "not_in": lambda: actual not in expected,
            "contains": lambda: expected in actual,
            "gt": lambda: actual > expected,
            "gte": lambda: actual >= expected,
            "lt": lambda: actual < expected,
            "lte": lambda: actual <= expected,
            "exists": lambda: actual is not None,
            "not_exists": lambda: actual is None,
        }
        if operation not in comparisons:
            raise ValueError(f"Unsupported predicate operator: {operation}")
        return bool(comparisons[operation]())

    @staticmethod
    def _path(container: Any, path: str) -> Any:
        current = container
        for part in path.split(".") if path else []:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                current = getattr(current, part)
        return current

    @staticmethod
    def _replace_path(container: Any, path: str, value: Any) -> Any:
        result = copy.deepcopy(container)
        parts = path.split(".") if path else []
        current = result
        for part in parts[:-1]:
            current = current[int(part)] if isinstance(current, list) else current[part]
        if not parts:
            return value
        final = parts[-1]
        if isinstance(current, list):
            current[int(final)] = value
        else:
            current[final] = value
        return result

    @staticmethod
    def _failed(
        workflow: WorkflowDefinition,
        started: float,
        error: str,
        *,
        outputs: dict[str, Any] | None = None,
        traces: list[ToolCallTrace] | None = None,
        executors: list[ComputationKind] | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_calls: int = 0,
    ) -> ExecutionResult:
        return ExecutionResult(
            workflow_id=workflow.workflow_id,
            success=False,
            verified=False,
            outputs=outputs or {},
            steps=traces or [],
            executor_kinds=executors or [],
            latency_ms=(time.perf_counter() - started) * 1000,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_calls=reasoning_calls,
            escalated=True,
            error=error,
            side_effects_may_have_occurred=any(trace.success for trace in (traces or [])),
        )
