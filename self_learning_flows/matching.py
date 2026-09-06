"""Conservative workflow retrieval and eligibility checks."""

from __future__ import annotations

from .discovery import intent_similarity, negated_concepts, text_tokens
from .models import TaskRequest, WorkflowDefinition, WorkflowMatch, WorkflowProposal
from .protocols import VariableExtractor


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
