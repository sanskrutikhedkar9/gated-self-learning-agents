"""Composable execution backends for deterministic, SLM, and LLM computation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import ComputationKind, ToolResult
from .protocols import ComputeBackend


class FunctionComputeBackend:
    """Model-free computation registry for parsers, templates, and business rules."""

    def __init__(self):
        self._operations: dict[str, Callable[..., Any]] = {}

    def register(self, operation: str, function: Callable[..., Any]) -> None:
        self._operations[operation] = function

    def run(
        self,
        *,
        kind: ComputationKind,
        operation: str,
        inputs: dict[str, Any],
    ) -> ToolResult:
        if kind != ComputationKind.DETERMINISTIC:
            return ToolResult(False, error=f"Function backend cannot execute {kind}")
        function = self._operations.get(operation)
        if function is None:
            return ToolResult(False, error=f"Unknown deterministic operation: {operation}")
        try:
            return ToolResult(True, value=function(**inputs))
        except Exception as exc:
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}")


class TieredComputeBackend:
    """Routes each requested compute tier to an independently replaceable backend."""

    def __init__(self, backends: dict[ComputationKind, ComputeBackend]):
        self.backends = dict(backends)

    def run(
        self,
        *,
        kind: ComputationKind,
        operation: str,
        inputs: dict[str, Any],
    ) -> ToolResult:
        backend = self.backends.get(kind)
        if backend is None:
            return ToolResult(False, error=f"No backend configured for {kind}")
        return backend.run(kind=kind, operation=operation, inputs=inputs)
