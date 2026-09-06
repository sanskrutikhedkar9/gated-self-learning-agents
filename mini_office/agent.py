"""Reference full-agent policy used to generate successful demonstration traces."""

from __future__ import annotations

import time
from typing import Any

from self_learning_flows.execution import ToolRegistry
from self_learning_flows.models import ComputationKind, TaskEpisode, ToolCallTrace

from .compute import ReferenceComputeBackend
from .dataset import OfficeTask
from .world import MiniOfficeWorld


class ReferenceAgent:
    """An oracle policy for measuring routing, not a claimed LLM-quality baseline.

    It performs real tool calls while assigning conservative token/call costs to
    the planning pass. Provider-backed planners can implement the same contract.
    """

    def __init__(self, world: MiniOfficeWorld):
        self.world = world
        self.tools: ToolRegistry = world.tool_registry()
        self.compute = ReferenceComputeBackend()
        self.traces: list[ToolCallTrace] = []

    def run(self, task: OfficeTask) -> TaskEpisode:
        started = time.perf_counter()
        variables = task.variables
        success = True
        error: str | None = None
        try:
            if task.pattern == "reschedule_and_notify":
                self._reschedule(variables)
            elif task.pattern == "create_assign_ticket":
                self._ticket(variables)
            elif task.pattern == "onboard_employee":
                self._onboard(variables)
            elif task.pattern == "priority_digest":
                self._digest(variables)
            else:
                raise ValueError(f"Unsupported reference pattern: {task.pattern}")
        except Exception as exc:
            success = False
            error = f"{type(exc).__name__}: {exc}"
        prompt_tokens = max(120, len(task.instruction) * 3)
        return TaskEpisode(
            task_id=task.task_id,
            instruction=task.instruction,
            scope="mini-office",
            success=success,
            verified=success,
            steps=self.traces,
            outcome={"error": error} if error else {"completed": True},
            variables=task.workflow_variables,
            framework="reference-agent",
            completed_ms=(time.perf_counter() - started) * 1000,
            input_tokens=prompt_tokens,
            output_tokens=80 + 25 * len(self.traces),
            reasoning_calls=1,
            metadata={
                "pattern_name": task.pattern,
                "environment": "mini-office-v1",
                "cost_model": "simulated_reference_accounting",
                "tool_schema_hashes": self.tools.schema_hashes(),
            },
        )

    def _call(self, operation: str, arguments: dict[str, Any]) -> Any:
        result = self.tools.invoke(operation, arguments)
        trace = ToolCallTrace(
            step_id=f"step-{len(self.traces) + 1}",
            agent=operation.split(".", 1)[0],
            tool=operation,
            arguments=arguments,
            result=result.value,
            success=result.success,
            error=result.error,
            executor=ComputationKind.DETERMINISTIC,
            latency_ms=result.latency_ms,
        )
        self.traces.append(trace)
        if not result.success:
            raise RuntimeError(result.error)
        return result.value

    def _compute(self, operation: str, arguments: dict[str, Any]) -> Any:
        result = self.compute.run(
            kind=ComputationKind.SLM,
            operation=operation,
            inputs=arguments,
        )
        self.traces.append(
            ToolCallTrace(
                step_id=f"step-{len(self.traces) + 1}",
                agent="compute",
                tool=operation,
                arguments=arguments,
                result=result.value,
                success=result.success,
                error=result.error,
                executor=ComputationKind.SLM,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                reasoning_calls=result.reasoning_calls,
            )
        )
        if not result.success:
            raise RuntimeError(result.error)
        return result.value

    def _reschedule(self, value: dict[str, Any]) -> None:
        event = self._call("calendar.find_event", {"title": value["event_title"]})
        check = self._call(
            "calendar.check_conflict",
            {
                "start": value["new_start"],
                "end": value["new_end"],
                "exclude_event_id": event["id"],
            },
        )
        if not check["available"]:
            raise RuntimeError("The requested calendar slot is unavailable")
        emails = self._call("contacts.resolve_emails", {"names": value["attendee_names"]})
        self._call(
            "calendar.update_event",
            {"event_id": event["id"], "start": value["new_start"], "end": value["new_end"]},
        )
        self._call(
            "email.send",
            {"to": emails, "subject": value["email_subject"], "body": value["email_body"]},
        )

    def _ticket(self, value: dict[str, Any]) -> None:
        ticket = self._call(
            "ticket.create",
            {
                "title": value["ticket_title"],
                "description": value["ticket_description"],
                "priority": value["priority"],
            },
        )
        self._call("ticket.assign", {"ticket_id": ticket["id"], "owner": value["owner"]})
        self._call(
            "email.send",
            {
                "to": [value["requester_email"]],
                "subject": value["email_subject"],
                "body": value["email_body"],
            },
        )

    def _onboard(self, value: dict[str, Any]) -> None:
        self._call(
            "contacts.create",
            {
                "name": value["employee_name"],
                "email": value["employee_email"],
                "team": value["team"],
            },
        )
        self._call(
            "calendar.create_event",
            {
                "title": value["orientation_title"],
                "start": value["orientation_start"],
                "end": value["orientation_end"],
                "attendees": [value["employee_email"]],
            },
        )
        self._call(
            "email.send",
            {
                "to": [value["employee_email"]],
                "subject": value["welcome_subject"],
                "body": value["welcome_body"],
            },
        )

    def _digest(self, value: dict[str, Any]) -> None:
        events = self._call(
            "calendar.events_between",
            {"start": value["range_start"], "end": value["range_end"]},
        )
        tickets = self._call("ticket.open_by_priority", {"priority": value["priority"]})
        digest = self._compute(
            "compute.render_digest",
            {"date": value["digest_date"], "events": events, "tickets": tickets},
        )
        self._call(
            "email.send",
            {"to": value["recipient_emails"], "subject": value["digest_subject"], "body": digest},
        )
