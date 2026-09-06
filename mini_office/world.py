"""A deterministic, stateful office environment with real side effects in memory."""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from self_learning_flows.execution import ToolRegistry


def _overlaps(left_start: str, left_end: str, right_start: str, right_end: str) -> bool:
    return datetime.fromisoformat(left_start) < datetime.fromisoformat(
        right_end
    ) and datetime.fromisoformat(right_start) < datetime.fromisoformat(left_end)


@dataclass
class MiniOfficeWorld:
    """Fresh world per task so executions are isolated and reproducible."""

    contacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    tickets: dict[str, dict[str, Any]] = field(default_factory=dict)
    sent_emails: list[dict[str, Any]] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1), repr=False)

    def clone(self) -> MiniOfficeWorld:
        clone = MiniOfficeWorld(
            contacts=copy.deepcopy(self.contacts),
            events=copy.deepcopy(self.events),
            tickets=copy.deepcopy(self.tickets),
            sent_emails=copy.deepcopy(self.sent_emails),
        )
        return clone

    def seed(self, variables: dict[str, Any]) -> None:
        names = variables.get("attendee_names", [])
        emails = variables.get("attendee_emails", [])
        for index, name in enumerate(names):
            email = (
                emails[index]
                if index < len(emails)
                else f"{name.lower().replace(' ', '.')}@example.com"
            )
            self.contacts[name.lower()] = {"name": name, "email": email, "team": "general"}
        event_title = variables.get("event_title")
        if event_title:
            event_id = f"event-{next(self._ids)}"
            self.events[event_id] = {
                "id": event_id,
                "title": event_title,
                "start": variables.get("original_start", "2026-09-07T09:00:00"),
                "end": variables.get("original_end", "2026-09-07T09:30:00"),
                "attendees": list(emails),
                "cancelled": False,
            }
        for ticket in variables.get("seed_tickets", []):
            ticket_id = f"ticket-{next(self._ids)}"
            self.tickets[ticket_id] = {"id": ticket_id, "status": "open", **ticket}

    # Contact tools
    def contacts_resolve_emails(self, names: list[str]) -> list[str]:
        result = []
        for name in names:
            contact = self.contacts.get(name.lower())
            if contact is None:
                raise KeyError(f"Unknown contact: {name}")
            result.append(contact["email"])
        return result

    def contacts_create(self, name: str, email: str, team: str) -> dict[str, Any]:
        if name.lower() in self.contacts:
            raise ValueError(f"Contact already exists: {name}")
        contact = {"name": name, "email": email, "team": team}
        self.contacts[name.lower()] = contact
        return copy.deepcopy(contact)

    # Calendar tools
    def calendar_find_event(self, title: str) -> dict[str, Any]:
        matches = [
            event for event in self.events.values() if event["title"].lower() == title.lower()
        ]
        if len(matches) != 1:
            raise LookupError(f"Expected one event titled {title!r}, found {len(matches)}")
        return copy.deepcopy(matches[0])

    def calendar_check_conflict(
        self,
        start: str,
        end: str,
        exclude_event_id: str = "",
    ) -> dict[str, Any]:
        conflicting = [
            event["id"]
            for event in self.events.values()
            if not event.get("cancelled")
            and event["id"] != exclude_event_id
            and _overlaps(start, end, event["start"], event["end"])
        ]
        return {"available": not conflicting, "conflicting_event_ids": conflicting}

    def calendar_update_event(self, event_id: str, start: str, end: str) -> dict[str, Any]:
        event = self.events[event_id]
        conflict = self.calendar_check_conflict(start, end, event_id)
        if not conflict["available"]:
            raise ValueError("Requested time conflicts with another event")
        event.update({"start": start, "end": end})
        return copy.deepcopy(event)

    def calendar_create_event(
        self,
        title: str,
        start: str,
        end: str,
        attendees: list[str],
    ) -> dict[str, Any]:
        if not self.calendar_check_conflict(start, end)["available"]:
            raise ValueError("Requested time conflicts with another event")
        event_id = f"event-{next(self._ids)}"
        event = {
            "id": event_id,
            "title": title,
            "start": start,
            "end": end,
            "attendees": list(attendees),
            "cancelled": False,
        }
        self.events[event_id] = event
        return copy.deepcopy(event)

    def calendar_events_between(self, start: str, end: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(event)
            for event in self.events.values()
            if not event.get("cancelled") and _overlaps(start, end, event["start"], event["end"])
        ]

    # Ticket tools
    def ticket_create(self, title: str, description: str, priority: str) -> dict[str, Any]:
        ticket_id = f"ticket-{next(self._ids)}"
        ticket = {
            "id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "owner": None,
            "status": "open",
        }
        self.tickets[ticket_id] = ticket
        return copy.deepcopy(ticket)

    def ticket_assign(self, ticket_id: str, owner: str) -> dict[str, Any]:
        ticket = self.tickets[ticket_id]
        ticket["owner"] = owner
        return copy.deepcopy(ticket)

    def ticket_open_by_priority(self, priority: str) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(ticket)
            for ticket in self.tickets.values()
            if ticket.get("status") == "open" and ticket.get("priority") == priority
        ]

    # Email tool
    def email_send(self, to: list[str], subject: str, body: str) -> dict[str, Any]:
        if not to:
            raise ValueError("At least one recipient is required")
        email = {
            "id": f"email-{next(self._ids)}",
            "to": list(to),
            "subject": subject,
            "body": body,
        }
        self.sent_emails.append(email)
        return copy.deepcopy(email)

    def tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()
        tools = {
            "contacts.resolve_emails": self.contacts_resolve_emails,
            "contacts.create": self.contacts_create,
            "calendar.find_event": self.calendar_find_event,
            "calendar.check_conflict": self.calendar_check_conflict,
            "calendar.update_event": self.calendar_update_event,
            "calendar.create_event": self.calendar_create_event,
            "calendar.events_between": self.calendar_events_between,
            "ticket.create": self.ticket_create,
            "ticket.assign": self.ticket_assign,
            "ticket.open_by_priority": self.ticket_open_by_priority,
            "email.send": self.email_send,
        }
        for name, function in tools.items():
            registry.register(name, function, agent=name.split(".", 1)[0])
        return registry
