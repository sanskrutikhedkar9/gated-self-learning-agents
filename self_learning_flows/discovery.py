"""Online pattern discovery from verified agent trajectories."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .models import TaskEpisode, WorkflowDefinition, WorkflowStatus
from .protocols import StructuredModel

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
    "add": "create",
    "remove": "delete",
    "cancel": "delete",
    "edit": "update",
    "modify": "update",
    "deliver": "send",
    "post": "send",
    "forward": "send",
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


_READ_ACTIONS = {
    "check",
    "count",
    "download",
    "find",
    "get",
    "list",
    "lookup",
    "query",
    "read",
    "resolve",
    "search",
    "show",
    "view",
}


def action_category(operation: str) -> str:
    """Return a coarse action category without treating an embedding as authority."""
    leaf = operation.rsplit(".", 1)[-1]
    words = re.findall(r"[a-z0-9]+", leaf.lower())
    return _SYNONYMS.get(words[0], words[0]) if words else leaf.lower()


def side_effect_categories(operations: list[str]) -> set[str]:
    return {
        category
        for operation in operations
        if (category := action_category(operation)) not in _READ_ACTIONS
        and not operation.startswith("compute.")
    }


def workflow_family_id(workflow: WorkflowDefinition) -> str:
    return str(workflow.metadata.get("family_id") or workflow.workflow_id)


@dataclass(slots=True)
class FamilyCandidate:
    workflow: WorkflowDefinition
    score: float
    tool_overlap: float
    path_similarity: float
    intent_similarity: float


class HybridEpisodeFamilyDiscoverer:
    """Join trace families using hard action gates plus a bounded model review.

    Exact signatures are accepted without a model. Very dissimilar side effects
    are rejected before a model. Only the uncertain middle is sent for semantic
    adjudication, so the LLM can improve recall but cannot erase capability gates.
    """

    _SCHEMA_BASE = {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_workflow_id", "same_task_family", "confidence", "reason"],
        "properties": {
            "selected_workflow_id": {"type": "string"},
            "same_task_family": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    }

    def __init__(
        self,
        model: StructuredModel | None = None,
        *,
        automatic_threshold: float = 0.76,
        review_threshold: float = 0.38,
        model_confidence: float = 0.82,
        maximum_candidates: int = 4,
    ):
        self.model = model
        self.automatic_threshold = automatic_threshold
        self.review_threshold = review_threshold
        self.model_confidence = model_confidence
        self.maximum_candidates = maximum_candidates

    def find(
        self,
        episode: TaskEpisode,
        workflows: list[WorkflowDefinition],
    ) -> WorkflowDefinition | None:
        signature = structural_signature(episode)
        eligible = [
            workflow
            for workflow in workflows
            if workflow.scope in {episode.scope, "*"}
            and workflow.status not in {WorkflowStatus.QUARANTINED, WorkflowStatus.RETIRED}
        ]
        for workflow in eligible:
            signatures = set(workflow.metadata.get("structural_signatures", []))
            if workflow.structural_signature == signature or signature in signatures:
                return workflow

        candidates = [
            candidate
            for workflow in eligible
            if (candidate := self._score(episode, workflow)) is not None
            and candidate.score >= self.review_threshold
        ]
        candidates.sort(key=lambda item: item.score, reverse=True)
        candidates = candidates[: self.maximum_candidates]
        if not candidates:
            return None
        top = candidates[0]
        margin = top.score - candidates[1].score if len(candidates) > 1 else top.score
        if top.score >= self.automatic_threshold and margin >= 0.12:
            return top.workflow
        if self.model is None:
            return None
        try:
            return self._model_select(episode, candidates)
        except Exception:
            # Discovery is an optimization boundary. Provider failures create
            # a conservative new family instead of stopping the agent stream.
            return None

    def _score(self, episode: TaskEpisode, workflow: WorkflowDefinition) -> FamilyCandidate | None:
        episode_path = action_path(episode)
        episode_operations = [trace.tool for trace in episode.steps if trace.success]
        episode_tools = set(episode_operations)
        workflow_paths = workflow.metadata.get("source_action_paths") or [
            workflow.metadata.get("action_path") or workflow.required_tools
        ]
        normalized_paths = [list(path) for path in workflow_paths if isinstance(path, list)]
        workflow_tools = set(workflow.required_tools)
        left_effects = side_effect_categories(episode_operations)
        right_effects = side_effect_categories(workflow.required_tools)
        # An LLM is not allowed to group clearly different mutations.
        if left_effects and right_effects and not (left_effects & right_effects):
            return None
        union = episode_tools | workflow_tools
        tool_overlap = len(episode_tools & workflow_tools) / len(union) if union else 1.0
        path_score = max(
            (SequenceMatcher(None, episode_path, path).ratio() for path in normalized_paths),
            default=0.0,
        )
        intent_score = max(
            (
                intent_similarity(episode.instruction, example)
                for example in workflow.intent_examples or [workflow.description]
            ),
            default=0.0,
        )
        score = 0.42 * tool_overlap + 0.23 * path_score + 0.35 * intent_score
        if left_effects & right_effects:
            score += 0.08
        return FamilyCandidate(
            workflow,
            min(score, 1.0),
            tool_overlap,
            path_score,
            intent_score,
        )

    def _model_select(
        self, episode: TaskEpisode, candidates: list[FamilyCandidate]
    ) -> WorkflowDefinition | None:
        allowed_ids = [candidate.workflow.workflow_id for candidate in candidates]
        schema = copy_schema(self._SCHEMA_BASE)
        schema["properties"]["selected_workflow_id"]["enum"] = ["NEW", *allowed_ids]
        response = self.model.generate_json(
            system=(
                "Decide whether a verified agent trace belongs to an existing reusable task "
                "family. Equivalent goals may use different read/discovery steps, but mutations "
                "and user-visible outcomes must mean the same thing. Select NEW when uncertain. "
                "Do not use argument values; return strict JSON."
            ),
            prompt=json.dumps(
                {
                    "new_trace": {
                        "instruction": episode.instruction,
                        "operations": [trace.tool for trace in episode.steps if trace.success],
                        "side_effect_categories": sorted(
                            side_effect_categories(
                                [trace.tool for trace in episode.steps if trace.success]
                            )
                        ),
                    },
                    "candidates": [
                        {
                            "workflow_id": item.workflow.workflow_id,
                            "name": item.workflow.name,
                            "description": item.workflow.description,
                            "intent_examples": item.workflow.intent_examples[-5:],
                            "operations": item.workflow.required_tools,
                            "side_effect_categories": sorted(
                                side_effect_categories(item.workflow.required_tools)
                            ),
                            "deterministic_score": round(item.score, 4),
                        }
                        for item in candidates
                    ],
                },
                ensure_ascii=False,
            ),
            schema=schema,
        )
        selected = str(response.get("selected_workflow_id", "NEW"))
        confidence = float(response.get("confidence", 0))
        if (
            not response.get("same_task_family")
            or confidence < self.model_confidence
            or selected not in allowed_ids
        ):
            return None
        return next(item.workflow for item in candidates if item.workflow.workflow_id == selected)


def copy_schema(value: dict[str, Any]) -> dict[str, Any]:
    """JSON round-trip keeps dynamically specialized provider schemas detached."""
    return json.loads(json.dumps(value))
