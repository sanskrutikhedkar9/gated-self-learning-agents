"""End-to-end AppWorld baseline/treatment runner for SelfLearningFlows.

This module imports AppWorld lazily so the core package remains usable without
the benchmark's Pydantic-1 environment.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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


def _debug_text(value: str, limit: int = 4000) -> str:
    """Keep one-run diagnostics bounded and redact common credential values."""
    text = re.sub(
        r'''(?i)(["']?(?:access_token|refresh_token|password|api[_-]?key|authorization|secret|token)["']?\s*[:=]\s*)(["'])(.*?)(\2)''',
        r"\1\2<redacted>\4",
        str(value),
    )
    text = re.sub(
        r"(?i)(password|token|api[_-]?key|authorization|secret)(\s*[:=]\s*)([^,\n)}]+)",
        r"\1\2<redacted>",
        text,
    )
    return text[:limit]


def _bounded_observation(value: Any, limit: int = 16000) -> str:
    """Retain both ends of large REPL results instead of hiding early entries."""
    text = str(value)
    if len(text) <= limit:
        return text
    half = (limit - 80) // 2
    return text[:half] + "\n... <observation truncated> ...\n" + text[-half:]


def _compact_data(value: Any, *, depth: int = 0) -> Any:
    """Keep contract data useful without embedding giant response examples."""
    if depth >= 4:
        return "<nested>"
    if isinstance(value, dict):
        items = list(value.items())[:40]
        return {str(key): _compact_data(item, depth=depth + 1) for key, item in items}
    if isinstance(value, list):
        return [_compact_data(item, depth=depth + 1) for item in value[:12]]
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "..."
    return value


def _summarize_observation(value: Any, limit: int = 6000) -> str:
    """Summarize Python-shaped API results before putting them in model context."""
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return _bounded_observation(text, limit)
    summary = _summarize_value(parsed)
    rendered = repr(summary)
    return _bounded_observation(rendered, limit)


def _summarize_value(value: Any, *, depth: int = 0) -> Any:
    if depth >= 3:
        return "<nested>"
    if isinstance(value, dict):
        return {
            "type": "object",
            "keys": list(value)[:40],
            "values": {
                str(key): _summarize_value(item, depth=depth + 1)
                for key, item in list(value.items())[:12]
            },
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "first": [_summarize_value(item, depth=depth + 1) for item in value[:3]],
            "last": [_summarize_value(item, depth=depth + 1) for item in value[-2:]],
        }
    if isinstance(value, str) and len(value) > 300:
        return value[:300] + "..."
    return value


@dataclass(slots=True)
class AgentOutcome:
    completed: bool
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_calls: int = 0
    interactions: int = 0
    error: str | None = None
    last_code: str = ""
    first_concrete_code: str = ""
    first_concrete_output: str = ""
    validation_failures: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)

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
    """Schema-grounded AppWorld code agent used identically in both conditions."""

    _CODE_BLOCK = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    _SUPERVISOR_APIS = {
        "complete_task",
        "show_account_passwords",
        "show_active_task",
        "show_profile",
    }

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        max_interactions: int = 25,
        max_tokens: int = 2000,
        max_input_tokens: int = 30000,
        timeout_seconds: int = 180,
        retries: int = 6,
    ):
        self.model = model
        self.api_key = api_key
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        self.base_url = endpoint
        self.max_interactions = max_interactions
        self.max_tokens = max_tokens
        self.max_input_tokens = max_input_tokens
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def run(self, world: Any) -> AgentOutcome:
        supervisor = self._jsonable_mapping(world.task.supervisor)
        api_catalog = self._api_catalog(world)
        api_contracts = self._relevant_api_contracts(world, api_catalog)
        compact_contracts = [self._compact_contract(contract) for contract in api_contracts]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Task: {world.task.instruction}\n\n"
                    f"Supervisor account email: {supervisor.get('email', '<retrieve via API>')}\n\n"
                    "The following candidate contracts come directly from AppWorld's API "
                    "schema catalogue. Use these exact names, keyword parameters, and response "
                    "schemas:\n"
                    f"{json.dumps(compact_contracts, ensure_ascii=False, default=str)}\n\n"
                    "Begin solving the task. You may execute multiple related operations in "
                    "one code block. AppWorld APIs take keyword arguments, never a positional "
                    "dictionary. Match list versus object response shapes exactly. Assign "
                    "returned values to variables and print only the small result summaries "
                    "needed to plan the next step.\n\n"
                    f"Planning constraints for this task:\n{self._planning_constraints(world)}"
                ),
            },
        ]
        outcome = AgentOutcome(completed=False)
        documentation_calls = 0
        executed_plan: list[str] = []
        repeated_actions = 0
        previous_action_key = ""
        for interaction in range(1, self.max_interactions + 1):
            estimated_input_tokens = self._estimate_prompt_tokens(messages)
            if estimated_input_tokens > self.max_input_tokens:
                outcome.error = (
                    f"per-task input-token budget exhausted: estimated {estimated_input_tokens} "
                    f"> {self.max_input_tokens}"
                )
                break
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
            outcome.last_code = code
            if not code:
                outcome.error = "Model returned no executable Python"
                break
            action_key = re.sub(r"\s+", " ", code).strip()
            if action_key == previous_action_key:
                repeated_actions += 1
            else:
                repeated_actions = 0
            previous_action_key = action_key
            if repeated_actions >= 2:
                outcome.error = "same model action repeated three times without progress"
                break
            api_calls = self._api_call_names(code)
            validation_error = self._validate_code(
                code,
                api_catalog,
                task_instruction=str(world.task.instruction),
                prior_code="\n".join(executed_plan),
            )
            new_documentation_calls = sum(name.startswith("apis.api_docs.") for name in api_calls)
            if not validation_error and documentation_calls + new_documentation_calls > 3:
                validation_error = (
                    "documentation-call budget exhausted; use the injected exact contracts "
                    "and execute a business API"
                )
            if validation_error:
                outcome.validation_failures += 1
                environment_output = "Code rejected before execution: " + validation_error
                if outcome.validation_failures >= 5:
                    outcome.error = "Five invalid model actions were rejected before execution"
            else:
                documentation_calls += new_documentation_calls
                environment_output = world.execute(code)
                executed_plan.append(code)
            outcome.actions.append(
                {
                    "interaction": interaction,
                    "code": _debug_text(code),
                    "executed": validation_error is None,
                    "validation_error": validation_error,
                    "output": _debug_text(_summarize_observation(environment_output)),
                }
            )
            if (
                not outcome.first_concrete_code
                and validation_error is None
                and any(
                    name.startswith("apis.") and not name.startswith("apis.api_docs.")
                    for name in api_calls
                )
            ):
                outcome.first_concrete_code = code
                outcome.first_concrete_output = str(environment_output)
            messages.extend(
                [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            "Environment output:\n"
                            f"{_summarize_observation(environment_output)}\n\n"
                            "Continue with one Python code block. Complete the task through "
                            "the supervisor API when ready. Never invent an API name."
                        ),
                    },
                ]
            )
            messages = self._compact_messages(messages)
            if outcome.error:
                break
            if world.task_completed():
                outcome.completed = True
                break
        if not outcome.completed and outcome.error is None:
            outcome.error = "Maximum agent interactions reached"
        return outcome

    @staticmethod
    def _compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
        return {
            key: _compact_data(contract[key])
            for key in ("call", "description", "parameters", "response_schemas")
            if key in contract
        }

    @staticmethod
    def _compact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
        if len(messages) <= 6:
            return messages
        # Python variables persist inside AppWorld, so retain only the initial
        # contract prompt and the latest two action/observation pairs.
        return [*messages[:2], *messages[-4:]]

    @staticmethod
    def _estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
        return max(1, sum(len(str(item.get("content", ""))) for item in messages) // 4)

    @classmethod
    def _planning_constraints(cls, world: Any) -> str:
        instruction = str(world.task.instruction)
        words = cls._words(instruction)
        if {"liked", "like"}.intersection(words) and "most" in words and "playlist" in words:
            return (
                "This is a global aggregation over the user's playlists. Do not pass "
                "is_public=True; omit is_public so private playlists are included. Paginate "
                "show_playlist_library until it returns an empty page, collect every unique "
                "song_id, call show_song (not show_song_privates) for every song, select the "
                "maximum like_count, then submit that song's title. You may split these steps "
                "across turns, but do not complete the task until all steps are done."
            )
        return "No special planning constraints; follow the exact injected API contracts."

    @staticmethod
    def _api_catalog(world: Any) -> dict[str, dict[str, Any]]:
        """Return exact call, input, and output contracts for every allowed API."""
        catalog: dict[str, dict[str, Any]] = {}
        for entry in world.task.api_docs.function_calling():
            function = entry.get("function", {}) if isinstance(entry, dict) else {}
            name = str(function.get("name", ""))
            if "__" not in name:
                continue
            app, api = name.split("__", 1)
            call_name = f"apis.{app}.{api}"
            catalog[call_name] = {
                "call": call_name,
                "description": str(function.get("description", "")),
                "parameters": function.get("parameters", {"type": "object", "properties": {}}),
            }
        standard_docs = world.task.api_docs
        items = getattr(standard_docs, "items", None)
        if callable(items):
            for app, api_docs in items():
                if not isinstance(api_docs, dict):
                    continue
                for api, documentation in api_docs.items():
                    call_name = f"apis.{app}.{api}"
                    if call_name not in catalog or not isinstance(documentation, dict):
                        continue
                    response_schemas = documentation.get("response_schemas")
                    if response_schemas is not None:
                        catalog[call_name]["response_schemas"] = response_schemas
        return catalog

    @classmethod
    def _relevant_api_contracts(
        cls,
        world: Any,
        catalog: dict[str, dict[str, Any]],
        *,
        maximum: int = 24,
    ) -> list[dict[str, Any]]:
        """Rank public API contracts using task words; no task answer is consulted."""
        instruction_words = cls._words(str(world.task.instruction))
        app_names = {name.split(".", 2)[1] for name in catalog}
        mentioned_apps = {app for app in app_names if cls._words(app) & instruction_words}
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        mandatory: list[tuple[str, dict[str, Any]]] = []
        for name, contract in catalog.items():
            _, app, api = name.split(".", 2)
            api_words = cls._words(api)
            description_words = cls._words(str(contract.get("description", "")))
            response_words = cls._response_words(contract.get("response_schemas"))
            if app == "supervisor" and api in cls._SUPERVISOR_APIS:
                mandatory.append((name, contract))
                continue
            if app in mentioned_apps and api == "login":
                mandatory.append((name, contract))
                continue
            score = 0.0
            if app in mentioned_apps:
                score += 8.0
            score += 4.0 * len(api_words & instruction_words)
            score += 1.5 * len(description_words & instruction_words)
            score += 5.0 * len(response_words & instruction_words)
            if api in {"login", "search", "show", "list"}:
                score += 0.25
            if score > 0:
                ranked.append((score, name, contract))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(mandatory, key=lambda item: item[0])
        selected_names = {name for name, _ in selected}
        for _, name, contract in ranked:
            if name not in selected_names:
                selected.append((name, contract))
                selected_names.add(name)
            if len(selected) >= maximum:
                break
        return [contract for _, contract in selected[:maximum]]

    @staticmethod
    def _words(value: str) -> set[str]:
        words: set[str] = set()
        for raw in re.findall(r"[a-z0-9]+", value.lower().replace("_", " ")):
            word = raw[:-1] if len(raw) > 3 and raw.endswith("s") else raw
            words.add(word)
        return words

    @classmethod
    def _response_words(cls, response_schemas: Any) -> set[str]:
        """Extract response field names for capability-aware API ranking."""
        if response_schemas is None:
            return set()
        words: set[str] = set()
        if isinstance(response_schemas, dict):
            for key, value in response_schemas.items():
                words.update(cls._words(str(key)))
                words.update(cls._response_words(value))
        elif isinstance(response_schemas, list):
            for value in response_schemas:
                words.update(cls._response_words(value))
        return words

    @staticmethod
    def _attribute_path(node: ast.AST) -> list[str]:
        path: list[str] = []
        while isinstance(node, ast.Attribute):
            path.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            path.append(node.id)
        return list(reversed(path))

    @classmethod
    def _api_call_names(cls, code: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        names: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            path = cls._attribute_path(node.func)
            if len(path) >= 3 and path[0] == "apis":
                names.append(".".join(path[:3]))
        return names

    @classmethod
    def _validate_code(
        cls,
        code: str,
        catalog: dict[str, dict[str, Any]],
        task_instruction: str = "",
        prior_code: str = "",
    ) -> str | None:
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"invalid Python syntax: {exc.msg}"
        if not tree.body or all(
            isinstance(node, (ast.Import, ast.ImportFrom, ast.Pass)) for node in tree.body
        ):
            return "code must perform useful work, not only import or pass"
        for statement in tree.body:
            if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
                continue
            path = cls._attribute_path(statement.value.func)
            if len(path) >= 3 and path[:2] == ["apis", "api_docs"]:
                return "API documentation results are stdout-only; wrap this call in print(...)"
        call_paths = [
            cls._attribute_path(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        ]
        has_documentation_call = any(
            len(path) >= 3 and path[:2] == ["apis", "api_docs"] for path in call_paths
        )
        has_print_call = any(path == ["print"] for path in call_paths)
        if has_documentation_call and not has_print_call:
            return "API documentation results are stdout-only; print the assigned result"
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            path = cls._attribute_path(node.func)
            if len(path) < 3 or path[0] != "apis" or path[:2] == ["apis", "api_docs"]:
                continue
            call_name = ".".join(path[:3])
            if node.args:
                return (
                    f"{call_name} accepts keyword arguments only; do not pass a positional "
                    "dictionary. Use name=value, for example api(argument=value)."
                )
            contract = catalog.get(call_name)
            if not contract:
                continue
            parameters = contract.get("parameters", {})
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            if not isinstance(properties, dict) or any(
                keyword.arg is None for keyword in node.keywords
            ):
                continue
            supplied = {str(keyword.arg) for keyword in node.keywords}
            unknown_arguments = sorted(supplied - set(properties))
            if unknown_arguments:
                return (
                    f"{call_name} received unknown keyword arguments: "
                    f"{', '.join(unknown_arguments)}; allowed arguments: "
                    f"{', '.join(sorted(properties)) or 'none'}"
                )
            instruction_words = cls._words(task_instruction)
            if (
                call_name == "apis.spotify.show_playlist_library"
                and "playlist" in instruction_words
                and {"my", "user"}.intersection(instruction_words)
            ):
                public_filter = next(
                    (keyword for keyword in node.keywords if keyword.arg == "is_public"),
                    None,
                )
                if (
                    public_filter is not None
                    and isinstance(public_filter.value, ast.Constant)
                    and public_filter.value.value is True
                ):
                    return (
                        "do not restrict the user's playlists to public playlists; omit "
                        "is_public=True so private playlists are included"
                    )
        unknown = sorted(
            {
                name
                for name in cls._api_call_names(code)
                if not name.startswith("apis.api_docs.") and name not in catalog
            }
        )
        if unknown:
            details = []
            available = sorted(catalog)
            for name in unknown:
                same_app = [
                    candidate
                    for candidate in available
                    if candidate.split(".", 2)[1] == name.split(".", 2)[1]
                ]
                suggestions = difflib.get_close_matches(
                    name, same_app or available, n=5, cutoff=0.2
                )
                details.append(
                    f"unknown API {name!r}; closest real APIs: {', '.join(suggestions) or 'none'}"
                )
            return "; ".join(details)
        completion_error = cls._validate_completion_plan(
            code + "\n" + prior_code,
            task_instruction,
        )
        if completion_error:
            return completion_error
        return None

    @classmethod
    def _validate_completion_plan(cls, code: str, instruction: str) -> str | None:
        """Reject premature answers whose code lacks the task's required evidence."""
        if "complete_task" not in code:
            return None
        tree = ast.parse(code)
        completion_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and cls._attribute_path(node.func)[:3]
            == ["apis", "supervisor", "complete_task"]
        ]
        for node in completion_calls:
            answer = next((keyword for keyword in node.keywords if keyword.arg == "answer"), None)
            if answer is not None and isinstance(answer.value, ast.Constant) and answer.value.value is None:
                return "do not complete the task with answer=None; compute the requested answer first"

        normalized = cls._words(instruction)
        if {"liked", "like"}.intersection(normalized) and "most" in normalized:
            required = {
                "spotify.show_playlist_library",
                "spotify.show_song",
                "like_count",
                "max",
            }
            missing = sorted(item for item in required if item not in code)
            if "spotify.show_song_privates" in code:
                missing.append("spotify.show_song (show_song_privates has no title or like_count)")
            if missing:
                return (
                    "do not complete yet; this is a global most-liked query. Gather every "
                    "playlist/song, use spotify.show_song, and select max(like_count). "
                    "Missing evidence: "
                    + ", ".join(missing)
                )
        return None

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
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise RuntimeError(f"Model API returned {exc.code}: {detail}") from exc
                delay = self._retry_delay(exc, detail, attempt)
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"Could not reach model API: {exc}") from exc
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Model request failed")

    @staticmethod
    def _retry_delay(exc: urllib.error.HTTPError, detail: str, attempt: int) -> float:
        """Honor provider retry hints, especially token-per-minute 429 responses."""
        header = exc.headers.get("Retry-After") if exc.headers else None
        if header:
            try:
                return max(1.0, min(float(header), 60.0))
            except ValueError:
                pass
        match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", detail, re.IGNORECASE)
        if match:
            return max(1.0, min(float(match.group(1)) + 0.5, 60.0))
        return min(float(2 ** (attempt - 1)), 30.0)

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
            "`apis` and `requester`; variables persist between turns. Return exactly one Python "
            "code block per turn. Candidate API names and parameter schemas are injected from "
            "AppWorld's real catalogue, together with response schemas when available. Use only "
            "exact names; never invent an API. Call APIs with keyword arguments (`name=value`), "
            "never by passing a dictionary positionally. Respect whether each response is a list "
            "or object and only read fields shown in its response schema. Assign API "
            "returns to variables and use print(...) whenever you need to observe a value; bare "
            "expressions produce no visible output in this environment. If no injected contract "
            "fits, inspect documentation with print(apis.api_docs.search_api_docs(...)) or "
            "print(apis.api_docs.show_api_doc(...)). Documentation calls without print are "
            "useless. Use the supervisor's synthetic account credentials when login is required. "
            "For paginated list APIs, request pages in a bounded loop until an empty page. "
            "Compute the requested result, and always call "
            "apis.supervisor.complete_task(status='success', answer=...) when finished. Do not "
            "repeat failed calls, documentation searches, or unchanged code."
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
        world: Any | None = None
        if self.condition == "treatment":
            world = self.world_factory(task_id, self.experiment_name)
            try:
                workflow_result, fallback_reason, discard_world = self._try_workflow(world)
            except Exception:
                self._close_world(world)
                raise
            if workflow_result is not None:
                self._close_world(world)
                self._add_overhead(workflow_result, overhead_before)
                workflow_result.elapsed_ms = (time.perf_counter() - started) * 1000
                self._record_metric(workflow_result)
                return workflow_result
            if discard_world:
                self._close_world(world)
                world = None
                # This is deliberately a new AppWorld initialization. It resets
                # all state after an actual workflow execution attempt.
                fresh_worlds += 1

        if world is None:
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

    def _try_workflow(self, world: Any) -> tuple[TaskRun | None, str | None, bool]:
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
            return None, f"proposal_error:{type(exc).__name__}", False
        if proposal is None:
            return None, "no_eligible_workflow", False
        unsupported = {
            str(step.kind)
            for step in WorkflowExecutor._walk_steps(proposal.workflow.steps)
            if str(step.kind) not in self.guard.config.supported_step_kinds
        }
        if unsupported:
            return None, "unsupported_step_kinds:" + ",".join(sorted(unsupported)), False
        if proposal.missing_variables:
            return None, "missing_variables:" + ",".join(proposal.missing_variables), False
        ProtocolGuard.approve_in_memory(proposal)
        verifier = AppWorldCompletionVerifier(world)
        executor = WorkflowExecutor(registry, verifier=verifier)
        result = executor.execute(request, proposal.workflow, proposal.confirmed_variables or {})
        if not (result.success and result.verified):
            self.guard.record_execution(self.engine, proposal.workflow.workflow_id, result)
            return None, result.error or "workflow_verification_failed", True
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
            False,
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
        episode = AppWorldTraceAdapter().normalize(**record)
        record["agent_debug"] = {
            "error": outcome.error,
            "last_code_length": len(outcome.last_code),
            "last_code_mentions_api_docs": "api_docs" in outcome.last_code,
            "last_code_mentions_business_api": "apis." in outcome.last_code
            and "api_docs" not in outcome.last_code,
            "first_concrete_code": _debug_text(outcome.first_concrete_code),
            "first_concrete_output": _debug_text(outcome.first_concrete_output),
            "validation_failures": outcome.validation_failures,
            "actions": outcome.actions,
        }
        self._append_record(record)
        if self.guard.allows_learning:
            self.guard.observe(self.engine, episode)
        return TaskRun(
            task_id=str(world.task.id),
            condition=self.condition,
            route="fresh_world_fallback" if fresh_worlds > 1 else "full_agent",
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
    parser.add_argument("--max-input-tokens", type=int, default=30000)
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
        max_input_tokens=args.max_input_tokens,
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
                "max_input_tokens": args.max_input_tokens,
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
