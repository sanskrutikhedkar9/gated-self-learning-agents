"""State-based task verifiers; traces alone are never treated as correctness."""

from __future__ import annotations

from self_learning_flows.models import ExecutionResult, TaskRequest, WorkflowDefinition

from .dataset import OfficeTask
from .world import MiniOfficeWorld


def verify_task(task: OfficeTask, world: MiniOfficeWorld) -> bool:
    value = task.variables
    if task.pattern == "reschedule_and_notify":
        events = [item for item in world.events.values() if item["title"] == value["event_title"]]
        return bool(
            len(events) == 1
            and events[0]["start"] == value["new_start"]
            and events[0]["end"] == value["new_end"]
            and world.sent_emails
            and world.sent_emails[-1]["to"] == value["attendee_emails"]
        )
    if task.pattern == "create_assign_ticket":
        tickets = [
            item for item in world.tickets.values() if item["title"] == value["ticket_title"]
        ]
        return bool(
            len(tickets) == 1
            and tickets[0]["owner"] == value["owner"]
            and world.sent_emails
            and world.sent_emails[-1]["to"] == [value["requester_email"]]
        )
    if task.pattern == "onboard_employee":
        contact = world.contacts.get(value["employee_name"].lower())
        meetings = [
            item for item in world.events.values() if item["title"] == value["orientation_title"]
        ]
        return bool(
            contact
            and contact["email"] == value["employee_email"]
            and len(meetings) == 1
            and world.sent_emails
            and world.sent_emails[-1]["to"] == [value["employee_email"]]
        )
    if task.pattern == "priority_digest":
        body = world.sent_emails[-1]["body"] if world.sent_emails else ""
        expected_ticket = value["seed_tickets"][0]["title"]
        return (
            value["digest_date"] in body
            and value["event_title"] in body
            and expected_ticket in body
            and world.sent_emails[-1]["to"] == value["recipient_emails"]
        )
    return False


class OfficeVerifier:
    def __init__(self, task: OfficeTask, world: MiniOfficeWorld):
        self.task = task
        self.world = world

    def verify(
        self,
        *,
        request: TaskRequest,
        workflow: WorkflowDefinition,
        result: ExecutionResult,
    ) -> bool:
        del request, workflow, result
        return verify_task(self.task, self.world)
