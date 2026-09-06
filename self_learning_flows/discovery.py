"""Online pattern discovery from verified agent trajectories."""

from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any

from .models import TaskEpisode

_STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "for",
    "from",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "please",
    "the",
    "then",
    "this",
    "to",
    "with",
}

_SYNONYMS = {
    "shift": "reschedule",
    "move": "reschedule",
    "change": "update",
    "meeting": "event",
    "appointment": "event",
    "tell": "notify",
    "inform": "notify",
    "notifying": "notify",
    "notified": "notify",
    "message": "notify",
    "mail": "email",
    "duplicate": "duplicates",
    "dupe": "duplicates",
    "book": "create",
    "schedule": "create",
    "sending": "send",
    "assigned": "assign",
    "assigning": "assign",
    "give": "assign",
    "route": "assign",
    "issue": "ticket",
    "file": "create",
    "log": "create",
    "record": "create",
    "acknowledgement": "notify",
    "confirmation": "notify",
    "confirm": "notify",
    "receipt": "notify",
    "setup": "onboard",
    "register": "onboard",
    "invite": "event",
}

_NEGATION = re.compile(r"(?:do not|don't|never|without)\s+([^,.;]+)", re.IGNORECASE)
_PAGINATION_ARGUMENTS = ("page_index", "page", "offset")


def text_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return {
        _SYNONYMS.get(token, token)
        for token in tokens
        if token not in _STOP_WORDS and len(token) > 1
    }


def intent_similarity(left: str, right: str) -> float:
    left_tokens = text_tokens(left)
    right_tokens = text_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    sequence = SequenceMatcher(
        None, " ".join(sorted(left_tokens)), " ".join(sorted(right_tokens))
    ).ratio()
    return 0.7 * jaccard + 0.3 * sequence


def negated_concepts(text: str) -> set[str]:
    """Extract coarse forbidden-action concepts for conservative matching."""
    concepts: set[str] = set()
    for match in _NEGATION.finditer(text):
        concepts.update(text_tokens(match.group(1)))
    return concepts


def action_path(episode: TaskEpisode) -> list[str]:
    successful = [step for step in episode.steps if step.success]
    path: list[str] = []
    index = 0
    while index < len(successful):
        step = successful[index]
        operation = (
            step.tool if step.tool.startswith(f"{step.agent}.") else f"{step.agent}.{step.tool}"
        )
        run = [step]
        cursor = index + 1
        while cursor < len(successful):
            candidate = successful[cursor]
            candidate_operation = (
                candidate.tool
                if candidate.tool.startswith(f"{candidate.agent}.")
                else f"{candidate.agent}.{candidate.tool}"
            )
            if candidate_operation != operation:
                break
            run.append(candidate)
            cursor += 1
        page_argument = pagination_argument(run)
        if page_argument:
            path.append(f"paginate:{operation}:{page_argument}")
            index = cursor
        else:
            path.append(operation)
            index += 1
    return path


def pagination_argument(steps: list[Any]) -> str | None:
    if len(steps) < 2:
        return None
    for name in _PAGINATION_ARGUMENTS:
        values = [step.arguments.get(name) for step in steps]
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            continue
        increments = {right - left for left, right in zip(values, values[1:], strict=False)}
        if len(increments) != 1 or 0 in increments:
            continue
        base_arguments = [
            {key: value for key, value in step.arguments.items() if key != name} for step in steps
        ]
        if all(arguments == base_arguments[0] for arguments in base_arguments[1:]):
            return name
    return None


def structural_signature(episode: TaskEpisode) -> str:
    canonical = ">".join(action_path(episode))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def structural_similarity(left: TaskEpisode, right: TaskEpisode) -> float:
    return SequenceMatcher(None, action_path(left), action_path(right)).ratio()


def episode_similarity(left: TaskEpisode, right: TaskEpisode) -> float:
    return 0.75 * structural_similarity(left, right) + 0.25 * intent_similarity(
        left.instruction, right.instruction
    )
