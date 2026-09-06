"""Dataset loader for the versioned Mini Office task stream."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PACKAGED_DATASET = Path(__file__).resolve().parent / "data" / "mini_office_v1.jsonl"
_SOURCE_DATASET = Path(__file__).resolve().parents[1] / "data" / "mini_office_v1.jsonl"
DEFAULT_DATASET = _PACKAGED_DATASET if _PACKAGED_DATASET.exists() else _SOURCE_DATASET


@dataclass(slots=True)
class OfficeTask:
    task_id: str
    pattern: str
    instruction: str
    variables: dict[str, Any]
    tags: list[str] = field(default_factory=list)

    @property
    def workflow_variables(self) -> dict[str, Any]:
        """Public request inputs, excluding hidden world setup/evaluation data."""
        hidden = {"original_start", "original_end", "seed_tickets"}
        if self.pattern == "reschedule_and_notify":
            hidden.add("attendee_emails")
        if self.pattern == "priority_digest":
            hidden.add("event_title")
        return {key: value for key, value in self.variables.items() if key not in hidden}


def load_tasks(path: str | Path | None = None) -> list[OfficeTask]:
    dataset = Path(path) if path else DEFAULT_DATASET
    tasks: list[OfficeTask] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                tasks.append(OfficeTask(**json.loads(line)))
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid task at {dataset}:{line_number}: {exc}") from exc
    return tasks
