"""Conservative workflow retrieval and eligibility checks."""

from __future__ import annotations

import json

from .discovery import intent_similarity, negated_concepts, text_tokens
from .models import TaskRequest, WorkflowDefinition, WorkflowMatch, WorkflowProposal
from .protocols import StructuredModel, VariableExtractor


class WorkflowMatcher:
    def __init__(
        self,
        threshold: float = 0.48,
        allow_shadow: bool = True,
        *,
        min_intent_similarity: float = 0.20,
        ambiguity_margin: float = 0.08,
    ):
        self.threshold = threshold
        self.allow_shadow = allow_shadow
        self.min_intent_similarity = min_intent_similarity
        self.ambiguity_margin = ambiguity_margin

    def rank(
        self,
        request: TaskRequest,
        workflows: list[WorkflowDefinition],
    ) -> list[WorkflowMatch]:
        matches = [self.score(request, workflow) for workflow in workflows]
        return sorted(matches, key=lambda item: item.score, reverse=True)

    def score(self, request: TaskRequest, workflow: WorkflowDefinition) -> WorkflowMatch:
        examples = workflow.intent_examples or [workflow.description]
        intent = max(intent_similarity(request.instruction, example) for example in examples)
        if request.available_tools:
            available = set(request.available_tools)
            required = set(workflow.required_tools)
            tool_compatibility = len(required & available) / len(required) if required else 1.0
        else:
            tool_compatibility = 1.0

        required_variables = {item.name for item in workflow.variables if item.required}
        supplied = set(request.variables)
        if not required_variables:
            variable_coverage = 1.0
        elif not supplied:
            # Unknown until the bounded extractor runs; do not confuse missing
            # structured preprocessing with an incompatible contract.
            variable_coverage = 0.5
        else:
            variable_coverage = len(required_variables & supplied) / len(required_variables)
        reliability = workflow.stats.reliability
        negative_similarity = max(
            (intent_similarity(request.instruction, item) for item in workflow.negative_examples),
            default=0.0,
        )
        score = (
            0.45 * intent
            + 0.15 * tool_compatibility
            + 0.25 * variable_coverage
            + 0.15 * reliability
            - 0.30 * negative_similarity
        )
        reasons: list[str] = []
        forbidden = negated_concepts(request.instruction)
        workflow_concepts = text_tokens(" ".join([workflow.description, *workflow.required_tools]))
        contradicted = forbidden & workflow_concepts
        eligible_statuses = {"active"}
        if self.allow_shadow:
            eligible_statuses.add("shadow")
        if str(workflow.status) not in eligible_statuses:
            reasons.append(f"workflow status is {workflow.status}")
        if workflow.scope != "*" and request.scope != workflow.scope:
            reasons.append("workflow scope does not match the request")
        if workflow.environment != "*" and request.environment != workflow.environment:
            reasons.append("workflow environment does not match the request")
        if tool_compatibility < 1:
            reasons.append("required tools are unavailable")
        if intent < self.min_intent_similarity:
            reasons.append("intent similarity is too low")
        if contradicted:
            reasons.append("request negates workflow actions: " + ", ".join(sorted(contradicted)))
        eligible = not reasons and score >= self.threshold
        if score < self.threshold:
            reasons.append(f"score {score:.3f} is below threshold {self.threshold:.3f}")
        return WorkflowMatch(
            workflow_id=workflow.workflow_id,
            score=max(0.0, min(1.0, score)),
            intent_score=intent,
            tool_compatibility=tool_compatibility,
            variable_coverage=variable_coverage,
            reliability=reliability,
            eligible=eligible,
            reasons=reasons,
        )

    def propose(
        self,
        request: TaskRequest,
        workflow: WorkflowDefinition,
        match: WorkflowMatch,
        extractor: VariableExtractor | None = None,
    ) -> WorkflowProposal:
        variables = dict(request.variables)
        if extractor is not None:
            extracted = extractor.extract(request, workflow)
            variables = {**extracted, **variables}
        missing = [
            item.name
            for item in workflow.variables
            if item.required and variables.get(item.name) is None
        ]
        return WorkflowProposal(
            workflow=workflow,
            match=match,
            variables=variables,
            missing_variables=missing,
        )


class ModelSemanticReranker:
    """Use an LLM only to resolve semantic retrieval, never capability checks."""

    def __init__(
        self,
        model: StructuredModel,
        *,
        confidence_threshold: float = 0.84,
        maximum_candidates: int = 6,
        skip_model_intent: float = 0.72,
        skip_model_margin: float = 0.18,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.maximum_candidates = maximum_candidates
        self.skip_model_intent = skip_model_intent
        self.skip_model_margin = skip_model_margin

    def rerank(
        self,
        request: TaskRequest,
        workflows: list[WorkflowDefinition],
        matches: list[WorkflowMatch],
        matcher: WorkflowMatcher,
    ) -> list[WorkflowMatch]:
        if not matches:
            return matches
        margin = matches[0].score - matches[1].score if len(matches) > 1 else matches[0].score
        if (
            matches[0].eligible
            and matches[0].intent_score >= self.skip_model_intent
            and margin >= self.skip_model_margin
        ):
            return matches
        by_id = {workflow.workflow_id: workflow for workflow in workflows}
        candidates = [match for match in matches if self._capability_safe(match)][
            : self.maximum_candidates
        ]
        if not candidates:
            return matches
        candidate_ids = [match.workflow_id for match in candidates]
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["selected_workflow_id", "same_intent", "confidence", "reason"],
            "properties": {
                "selected_workflow_id": {"type": "string", "enum": ["NONE", *candidate_ids]},
                "same_intent": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
        }
        response = self.model.generate_json(
            system=(
                "Select an existing workflow only when it has the same requested outcome and "
                "actions as the user's task. Different wording and entities are allowed; omitted, "
                "negated, or additional mutations are not. Select NONE when ambiguous. This is a "
                "routing decision, not permission to alter a workflow. Return strict JSON."
            ),
            prompt=json.dumps(
                {
                    "request": request.instruction,
                    "candidates": [
                        {
                            "workflow_id": match.workflow_id,
                            "name": by_id[match.workflow_id].name,
                            "description": by_id[match.workflow_id].description,
                            "intent_examples": by_id[match.workflow_id].intent_examples[-5:],
                            "required_tools": by_id[match.workflow_id].required_tools,
                            "variables": [
                                {"name": item.name, "description": item.description}
                                for item in by_id[match.workflow_id].variables
                            ],
                        }
                        for match in candidates
                    ],
                },
                ensure_ascii=False,
            ),
            schema=schema,
        )
        selected = str(response.get("selected_workflow_id", "NONE"))
        confidence = float(response.get("confidence", 0))
        if (
            selected not in candidate_ids
            or not response.get("same_intent")
            or confidence < self.confidence_threshold
        ):
            return matches
        selected_match = next(match for match in candidates if match.workflow_id == selected)
        selected_match.semantic_score = confidence
        selected_match.matcher = "model_semantic"
        selected_match.intent_score = confidence
        selected_match.score = min(
            1.0,
            0.45 * confidence
            + 0.15 * selected_match.tool_compatibility
            + 0.25 * selected_match.variable_coverage
            + 0.15 * selected_match.reliability,
        )
        selected_match.reasons = [
            reason
            for reason in selected_match.reasons
            if not reason.startswith("intent similarity") and not reason.startswith("score ")
        ]
        selected_match.eligible = (
            not selected_match.reasons and selected_match.score >= matcher.threshold
        )
        if not selected_match.eligible and selected_match.score < matcher.threshold:
            selected_match.reasons.append(
                f"score {selected_match.score:.3f} is below threshold {matcher.threshold:.3f}"
            )
        return sorted(
            matches,
            key=lambda item: (item.workflow_id == selected, item.score),
            reverse=True,
        )

    @staticmethod
    def _capability_safe(match: WorkflowMatch) -> bool:
        unsafe_prefixes = (
            "workflow status",
            "workflow scope",
            "workflow environment",
            "required tools",
            "request negates",
        )
        return not any(reason.startswith(unsafe_prefixes) for reason in match.reasons)
