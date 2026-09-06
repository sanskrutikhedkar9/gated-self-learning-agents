"""Bounded semantic computation backends for the reference environment."""

from __future__ import annotations

import json
from typing import Any

from self_learning_flows.models import ComputationKind, ToolResult


def render_digest(date: str, events: list[dict[str, Any]], tickets: list[dict[str, Any]]) -> str:
    event_lines = ", ".join(event["title"] for event in events) or "no meetings"
    ticket_lines = ", ".join(ticket["title"] for ticket in tickets) or "no matching tickets"
    return f"{date}: meetings — {event_lines}; priority tickets — {ticket_lines}."


class ReferenceComputeBackend:
    """Repeatable SLM stand-in used for CI; it reports realistic call accounting.

    This is deliberately labelled a stand-in, not a model-quality result. Swap it
    for an OpenAI-compatible local server or hosted provider in a deployment.
    """

    def run(
        self,
        *,
        kind: ComputationKind,
        operation: str,
        inputs: dict[str, Any],
    ) -> ToolResult:
        if operation != "compute.render_digest":
            return ToolResult(False, error=f"Unsupported computation: {operation}")
        value = render_digest(**inputs)
        input_tokens = max(1, len(json.dumps(inputs, default=str)) // 4)
        return ToolResult(
            True,
            value=value,
            input_tokens=input_tokens,
            output_tokens=max(1, len(value) // 4),
            reasoning_calls=1 if kind != ComputationKind.DETERMINISTIC else 0,
        )
