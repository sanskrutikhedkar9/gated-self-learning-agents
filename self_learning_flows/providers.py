"""Optional structured-model providers for workflow synthesis and SLM tasks."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from .models import ComputationKind, ToolResult


class ProviderError(RuntimeError):
    pass


class AnthropicStructuredModel:
    """Minimal dependency-free client for Claude structured outputs."""

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1/messages",
        timeout_seconds: int = 120,
    ):
        self._model_name = model or os.getenv("SELF_LEARNING_FLOWS_LLM", "claude-sonnet-4-6")
        # ANTROPIC_API_KEY supports the misspelling in the original prototype's .env.
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTROPIC_API_KEY")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    @property
    def model_name(self) -> str:
        return self._model_name

    @classmethod
    def from_env(cls, *, dotenv_path: str = ".env", **kwargs: Any):
        load_dotenv(dotenv_path)
        return cls(**kwargs)

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured")
        if self.api_key.startswith("hf_"):
            raise ProviderError(
                "ANTHROPIC_API_KEY contains a Hugging Face token; configure an "
                "Anthropic key or use an OpenAI-compatible Hugging Face endpoint"
            )
        payload = {
            "model": self.model_name,
            "max_tokens": 2048,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            },
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ProviderError(f"Anthropic API returned {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ProviderError(f"Could not reach Anthropic API: {exc}") from exc
        text_blocks = [
            item.get("text", "") for item in body.get("content", []) if item.get("type") == "text"
        ]
        if not text_blocks:
            raise ProviderError("Anthropic response contained no JSON text block")
        try:
            return json.loads("".join(text_blocks))
        except json.JSONDecodeError as exc:
            raise ProviderError("Anthropic structured response was not valid JSON") from exc


class OpenAICompatibleStructuredModel:
    """Adapter for local SLM servers such as vLLM or an OpenAI-compatible Ollama endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434/v1/chat/completions",
        api_key: str = "local",
        timeout_seconds: int = 120,
    ):
        self._model_name = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_calls = 0
        self.last_usage: dict[str, int] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    @classmethod
    def from_env(cls, *, dotenv_path: str = ".env", **kwargs: Any):
        load_dotenv(dotenv_path)
        model = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not model:
            raise ProviderError("OPENAI_COMPATIBLE_MODEL is not configured")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        return cls(
            model,
            base_url=endpoint,
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "local"),
            **kwargs,
        )

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "workflow_output", "strict": True, "schema": schema},
            },
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            usage = body.get("usage", {})
            self.last_usage = {
                "input_tokens": int(usage.get("prompt_tokens", 0)),
                "output_tokens": int(usage.get("completion_tokens", 0)),
                "reasoning_calls": 1,
            }
            self.input_tokens += self.last_usage["input_tokens"]
            self.output_tokens += self.last_usage["output_tokens"]
            self.reasoning_calls += 1
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ProviderError(f"OpenAI-compatible structured request failed: {exc}") from exc


class OpenAICompatibleComputeBackend:
    """Run bounded semantic steps on Ollama, vLLM, or another compatible server."""

    def __init__(
        self,
        model: str,
        *,
        operation_prompts: dict[str, str],
        base_url: str = "http://localhost:11434/v1/chat/completions",
        api_key: str = "local",
        timeout_seconds: int = 120,
    ):
        self.model = model
        self.operation_prompts = dict(operation_prompts)
        self.base_url = base_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(
        cls,
        *,
        operation_prompts: dict[str, str],
        dotenv_path: str = ".env",
        **kwargs: Any,
    ):
        load_dotenv(dotenv_path)
        model = os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not model:
            raise ProviderError("OPENAI_COMPATIBLE_MODEL is not configured")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:11434/v1")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        return cls(
            model,
            operation_prompts=operation_prompts,
            base_url=endpoint,
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY", "local"),
            **kwargs,
        )

    def run(
        self,
        *,
        kind: ComputationKind,
        operation: str,
        inputs: dict[str, Any],
    ) -> ToolResult:
        prompt = self.operation_prompts.get(operation)
        if prompt is None:
            return ToolResult(False, error=f"No prompt registered for {operation}")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are the {kind.value} executor for one bounded workflow step. "
                        "Return only the requested result; do not choose tools or alter "
                        "the workflow."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\nInputs:\n"
                        f"{json.dumps(inputs, ensure_ascii=False, default=str)}"
                    ),
                },
            ],
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
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            usage = body.get("usage", {})
            return ToolResult(
                True,
                value=content,
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)),
                reasoning_calls=1,
            )
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            return ToolResult(False, error=f"OpenAI-compatible compute request failed: {exc}")


def load_dotenv(path: str = ".env") -> None:
    """Load simple KEY=VALUE entries without taking a dependency on python-dotenv."""
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
