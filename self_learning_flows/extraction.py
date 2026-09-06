"""Bounded variable extraction from natural-language task requests."""

from __future__ import annotations

import json
from typing import Any

from .models import TaskRequest, VariableSpec, WorkflowDefinition
from .protocols import StructuredModel


def _json_type(variable: VariableSpec) -> str:
    if variable.type.startswith("list"):
        return "array"
    if variable.type == "object":
        return "object"
    if variable.type == "integer":
        return "integer"
    if variable.type == "number":
        return "number"
    if variable.type == "boolean":
        return "boolean"
    return "string"


class StructuredVariableExtractor:
    """Use one constrained SLM/LLM call; never invent values for missing inputs."""

    def __init__(self, model: StructuredModel):
        self.model = model

    def extract(
        self,
        request: TaskRequest,
        workflow: WorkflowDefinition,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for variable in workflow.variables:
            properties[variable.name] = {
                "type": [_json_type(variable), "null"],
                "description": variable.description,
            }
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }
        result = self.model.generate_json(
            system=(
                "Extract workflow variables that are explicitly present in the request. "
                "Use null when a value is absent or ambiguous. Do not infer private data, "
                "choose defaults, or change the requested workflow. Return strict JSON."
            ),
            prompt=json.dumps(
                {
                    "instruction": request.instruction,
                    "workflow": workflow.name,
                    "description": workflow.description,
                    "variables": [
                        {
                            "name": variable.name,
                            "type": variable.type,
                            "required": variable.required,
                            "description": variable.description,
                        }
                        for variable in workflow.variables
                    ],
                },
                ensure_ascii=False,
            ),
            schema=schema,
        )
        if not isinstance(result, dict):
            raise TypeError("Structured variable extraction must return an object")
        return {
            name: result[name] for name in properties if name in result and result[name] is not None
        }
