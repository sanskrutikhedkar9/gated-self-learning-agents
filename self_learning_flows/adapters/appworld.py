"""Adapter boundary for AppWorld executions.

The adapter has no hard AppWorld dependency, so collected AppWorld records can
be normalized in the main package or in a separate Python 3.11 benchmark env.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from ..execution import ToolRegistry
from ..models import ComputationKind, TaskEpisode, ToolCallTrace


class AppWorldTraceAdapter:
    OFFICIAL_EVALUATOR = "appworld_official_evaluator"
    _SENSITIVE_KEY = re.compile(
        r"(?:^|_)(?:access_token|refresh_token|api_key|password|authorization|secret|token)(?:$|_)",
        re.IGNORECASE,
    )

    def normalize(
        self,
        *,
        task: Any,
        calls: list[dict[str, Any]],
        passed: bool,
        variables: dict[str, Any] | None = None,
        token_usage: dict[str, int] | None = None,
        evaluation: dict[str, Any] | None = None,
        tool_schema_hashes: dict[str, str] | None = None,
        tool_schemas: dict[str, dict[str, Any]] | None = None,
    ) -> TaskEpisode:
        if type(passed) is not bool:
            raise ValueError("passed must be an explicit boolean from the evaluator")
        if not isinstance(calls, list):
            raise ValueError("calls must be a list")
        usage = token_usage or {}
        task_id = str(
            self._task_value(task, "task_id", None) or self._task_value(task, "id", "unknown")
        )
        instruction = str(self._task_value(task, "instruction", ""))
        if not task_id or task_id == "unknown":
            raise ValueError("task.task_id is required")
        if not instruction.strip():
            raise ValueError("task.instruction is required")
        if passed and not calls:
            raise ValueError("a passed task must contain at least one traced call")

        evaluation = evaluation or {}
        source = str(evaluation.get("source", "unverified_record"))
        evaluation_success = evaluation.get("success")
        evaluator_verified = (
            source == self.OFFICIAL_EVALUATOR
            and type(evaluation_success) is bool
            and evaluation_success is passed
        )
        if passed and not evaluator_verified:
            raise ValueError("passed records require matching AppWorld official evaluator evidence")

        secret_values: dict[tuple[type, Any], str] = {}
        traces: list[ToolCallTrace] = []
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ValueError(f"call {index + 1} must be an object")
            app = str(call.get("app", "")).strip()
            api = str(call.get("api") or call.get("tool") or "").strip()
            if not app or not api:
                raise ValueError(f"call {index + 1} requires app and api/tool")
            if "arguments" not in call or not isinstance(call["arguments"], dict):
                raise ValueError(f"call {index + 1} requires object arguments")
            if "result" not in call:
                raise ValueError(
                    f"call {index + 1} has no result; native api_calls.jsonl is "
                    "insufficient for data-flow compilation"
                )
            if type(call.get("success")) is not bool:
                raise ValueError(f"call {index + 1} requires an explicit boolean success")
            operation = api if api.startswith(f"{app}.") else f"{app}.{api}"
            traces.append(
                ToolCallTrace(
                    step_id=str(call.get("id", f"step-{index + 1}")),
                    agent=app,
                    tool=operation,
                    arguments=self._redact(call["arguments"], secret_values),
                    result=self._redact(call["result"], secret_values),
                    success=call["success"],
                    error=call.get("error"),
                    executor=ComputationKind.DETERMINISTIC,
                    latency_ms=float(call.get("latency_ms", 0)),
                    input_tokens=int(call.get("input_tokens", 0)),
                    output_tokens=int(call.get("output_tokens", 0)),
                    reasoning_calls=int(call.get("reasoning_calls", 0)),
                )
            )
        schema_hashes = dict(tool_schema_hashes or {})
        invalid_hashes = sorted(name for name, value in schema_hashes.items() if not str(value))
        if invalid_hashes:
            raise ValueError(
                "tool schema hashes must be non-empty for: " + ", ".join(invalid_hashes)
            )
        if passed:
            missing_hashes = sorted(
                {trace.tool for trace in traces if trace.success} - set(schema_hashes)
            )
            if missing_hashes:
                raise ValueError(
                    "passed records require tool schema hashes for: " + ", ".join(missing_hashes)
                )
        return TaskEpisode(
            task_id=task_id,
            instruction=instruction,
            scope="appworld",
            success=passed,
            verified=evaluator_verified and passed,
            steps=traces,
            variables=self._redact(variables or {}, secret_values),
            framework="appworld",
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            reasoning_calls=int(usage.get("reasoning_calls", 0)),
            metadata={
                "source": source,
                "evaluation_verified": evaluator_verified,
                "tool_schema_hashes": schema_hashes,
                "tool_schemas": {
                    name: schema
                    for name, schema in (tool_schemas or {}).items()
                    if name in {trace.tool for trace in traces if trace.success}
                },
            },
        )

    @classmethod
    def _redact(
        cls,
        value: Any,
        secret_values: dict[tuple[type, Any], str],
        key: str = "",
    ) -> Any:
        if isinstance(value, dict):
            return {
                str(name): cls._redact(item, secret_values, str(name))
                for name, item in value.items()
            }
        if isinstance(value, list):
            return [cls._redact(item, secret_values, key) for item in value]
        try:
            identity = (type(value), value)
            if identity in secret_values:
                return secret_values[identity]
        except TypeError:
            identity = None
        if key and cls._SENSITIVE_KEY.search(key) and value not in {None, ""}:
            marker = f"<redacted:{len(secret_values) + 1}>"
            if identity is not None:
                secret_values[identity] = marker
            return marker
        return value

    @staticmethod
    def _task_value(task: Any, name: str, default: Any) -> Any:
        if isinstance(task, dict):
            return task.get(name, default)
        return getattr(task, name, default)


def load_appworld_tool_schemas(directory: str | Path) -> dict[str, dict[str, Any]]:
    """Load canonical ``app.api`` input schemas from AppWorld function-call docs."""
    schemas: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(directory).glob("*.json")):
        entries = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            continue
        for entry in entries:
            function = entry.get("function", {}) if isinstance(entry, dict) else {}
            name = str(function.get("name", ""))
            parameters = function.get("parameters")
            if "__" not in name or not isinstance(parameters, dict):
                continue
            app, api = name.split("__", 1)
            schemas[f"{app}.{api}"] = parameters
    if not schemas:
        raise ValueError(f"No AppWorld function-call schemas found in {directory}")
    return schemas


def appworld_tool_schemas_from_docs(
    function_docs: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build schema contracts directly from ``world.task.api_docs.function_calling()``."""
    schemas: dict[str, dict[str, Any]] = {}
    for entry in function_docs:
        function = entry.get("function", {}) if isinstance(entry, dict) else {}
        name = str(function.get("name", ""))
        parameters = function.get("parameters")
        if "__" not in name or not isinstance(parameters, dict):
            continue
        app, api = name.split("__", 1)
        schemas[f"{app}.{api}"] = parameters
    if not schemas:
        raise ValueError("AppWorld function docs contained no usable schemas")
    return schemas


def appworld_runtime_variables(task: Any) -> dict[str, Any]:
    """Extract current-world bindings without ever persisting them unredacted."""
    supervisor = getattr(task, "supervisor", None)
    if supervisor is None:
        return {}
    try:
        values = dict(supervisor)
    except (TypeError, ValueError):
        values = dict(vars(supervisor))
    variables: dict[str, Any] = {}
    for key, value in values.items():
        if key == "account_passwords":
            if isinstance(value, dict):
                passwords = value
            else:
                passwords = {
                    getattr(item, "account_name", ""): getattr(item, "password", None)
                    for item in value or []
                }
            for app, password in passwords.items():
                if app and password is not None:
                    variables[f"{app}_password"] = password
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            variables[f"supervisor_{key}"] = value
    return variables


def build_appworld_tool_registry(
    apis: Any,
    schemas: dict[str, dict[str, Any]],
) -> ToolRegistry:
    """Build a workflow registry over an AppWorld world's public API collection."""
    registry = ToolRegistry()
    for operation, schema in schemas.items():
        app_name, api_name = operation.split(".", 1)
        try:
            app = apis[app_name] if isinstance(apis, dict) else getattr(apis, app_name)
            app[api_name] if isinstance(app, dict) else getattr(app, api_name)
        except (AttributeError, KeyError):
            continue

        def invoke(
            _app_name: str = app_name,
            _api_name: str = api_name,
            **arguments: Any,
        ) -> Any:
            app = apis[_app_name] if isinstance(apis, dict) else getattr(apis, _app_name)
            function = app[_api_name] if isinstance(app, dict) else getattr(app, _api_name)
            return function(**arguments)

        registry.register(operation, invoke, input_schema=schema, agent=app_name)
    return registry


class AppWorldOutcomeVerifier:
    """Duck-typed bridge to AppWorld's official database-state evaluator."""

    def __init__(self, world: Any):
        self.world = world
        self.last_evaluation: dict[str, Any] | None = None

    def verify(self, *, request: Any, workflow: Any, result: Any) -> bool:
        del request, workflow, result
        # Direct API calls mutate AppWorld's in-memory DB. A no-op public
        # execution flushes it to the experiment output DB used by evaluate().
        execute = getattr(self.world, "execute", None)
        if callable(execute):
            execute("pass")
        tracker = self.world.evaluate()
        if hasattr(tracker, "to_dict"):
            self.last_evaluation = dict(tracker.to_dict())
        elif isinstance(tracker, dict):
            self.last_evaluation = dict(tracker)
        else:
            self.last_evaluation = {"success": getattr(tracker, "success", None)}
        success = self.last_evaluation.get("success")
        if type(success) is not bool:
            raise ValueError("AppWorld evaluator did not return a boolean success")
        return success


class AppWorldCompletionVerifier:
    """Non-oracle verifier for routing; never inspects AppWorld ground truth."""

    def __init__(self, world: Any):
        self.world = world

    def verify(self, *, request: Any, workflow: Any, result: Any) -> bool:
        del request, workflow, result
        execute = getattr(self.world, "execute", None)
        if callable(execute):
            execute("pass")
        completed = getattr(self.world, "task_completed", None)
        if not callable(completed):
            raise ValueError("AppWorld world has no observable task_completed boundary")
        return bool(completed())


class AppWorldTraceRecorder:
    """Capture rich call results from an AppWorld Requester without a hard dependency."""

    _CONTROL_ARGUMENTS = {
        "_app_name",
        "_api_name",
        "client",
        "raise_on_failure",
        "show",
        "track",
    }

    def __init__(self, requester: Any):
        self.requester = requester
        self.calls: list[dict[str, Any]] = []
        self._original_request: Any = None

    def __enter__(self):
        if self._original_request is not None:
            raise RuntimeError("AppWorldTraceRecorder is already active")
        self._original_request = self.requester.request

        def recorded_request(*args: Any, **kwargs: Any) -> Any:
            app = kwargs.get("_app_name", args[0] if args else "")
            api = kwargs.get("_api_name", args[1] if len(args) > 1 else "")
            if kwargs.get("track") is False or str(app) == "api_docs":
                return self._original_request(*args, **kwargs)
            arguments = {
                key: value for key, value in kwargs.items() if key not in self._CONTROL_ARGUMENTS
            }
            started = time.perf_counter()
            call = {
                "id": f"step-{len(self.calls) + 1}",
                "app": str(app),
                "api": str(api),
                "arguments": arguments,
            }
            try:
                result = self._original_request(*args, **kwargs)
            except Exception as exc:
                call.update(
                    {
                        "result": None,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                        "latency_ms": (time.perf_counter() - started) * 1000,
                    }
                )
                self.calls.append(call)
                raise
            call.update(
                {
                    "result": result,
                    "success": True,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                }
            )
            self.calls.append(call)
            return result

        self.requester.request = recorded_request
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._original_request is not None:
            self.requester.request = self._original_request
            self._original_request = None

    def normalized_record(
        self,
        *,
        task: Any,
        evaluation: Any,
        tool_schema_hashes: dict[str, str],
        tool_schemas: dict[str, dict[str, Any]] | None = None,
        variables: dict[str, Any] | None = None,
        token_usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        if hasattr(evaluation, "to_dict"):
            evaluation_payload = dict(evaluation.to_dict())
        elif isinstance(evaluation, dict):
            evaluation_payload = dict(evaluation)
        else:
            evaluation_payload = {"success": getattr(evaluation, "success", None)}
        passed = evaluation_payload.get("success")
        if type(passed) is not bool:
            raise ValueError("AppWorld evaluator did not return a boolean success")
        episode = AppWorldTraceAdapter().normalize(
            task=task,
            calls=self.calls,
            passed=passed,
            variables=variables,
            token_usage=token_usage,
            evaluation={
                "source": AppWorldTraceAdapter.OFFICIAL_EVALUATOR,
                "success": passed,
            },
            tool_schema_hashes=tool_schema_hashes,
            tool_schemas=tool_schemas,
        )
        return {
            "task": {"task_id": episode.task_id, "instruction": episode.instruction},
            "calls": [
                {
                    "id": trace.step_id,
                    "app": trace.agent,
                    "api": trace.tool,
                    "arguments": trace.arguments,
                    "result": trace.result,
                    "success": trace.success,
                    "error": trace.error,
                    "latency_ms": trace.latency_ms,
                }
                for trace in episode.steps
            ],
            "passed": episode.success,
            "evaluation": {
                "source": AppWorldTraceAdapter.OFFICIAL_EVALUATOR,
                "success": episode.success,
            },
            "variables": episode.variables,
            "token_usage": token_usage or {},
            "tool_schema_hashes": dict(tool_schema_hashes),
            "tool_schemas": episode.metadata.get("tool_schemas", {}),
        }
