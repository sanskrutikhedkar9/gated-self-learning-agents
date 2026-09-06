from __future__ import annotations

import json
import unittest

from mini_office.agent import ReferenceAgent
from mini_office.dataset import load_tasks
from mini_office.world import MiniOfficeWorld
from self_learning_flows.compiler import EvidenceGatedWorkflowCompiler
from self_learning_flows.discovery import HybridEpisodeFamilyDiscoverer
from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.execution import ToolRegistry
from self_learning_flows.matching import ModelSemanticReranker, WorkflowMatcher
from self_learning_flows.models import (
    ExecutionResult,
    StepKind,
    TaskEpisode,
    TaskRequest,
    ToolCallTrace,
    WorkflowDefinition,
    WorkflowStats,
    WorkflowStatus,
)
from self_learning_flows.synthesis import ValidatedWorkflowCompiler


class QueueModel:
    model_name = "fake-structured-model"

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate_json(self, *, system, prompt, schema):
        del system, schema
        self.prompts.append(prompt)
        return self.responses.pop(0)


def _node(
    node_id,
    operation,
    *,
    order,
    kind="tool",
    arguments=None,
    parent_id="",
    branch="main",
    condition=None,
    depends_on=None,
):
    return {
        "id": node_id,
        "parent_id": parent_id,
        "branch": branch,
        "order": order,
        "kind": kind,
        "operation": operation,
        "agent": operation.split(".", 1)[0],
        "arguments_json": json.dumps(arguments or {}),
        "executor": "deterministic",
        "output_key": "",
        "condition_json": json.dumps(condition),
        "depends_on": depends_on or [],
        "description": operation,
        "on_error": "escalate",
        "collection_json": "null",
        "control_json": "{}",
    }


def _program(nodes):
    return {
        "name": "Search, optionally audit, and notify",
        "description": "Process a record and send the requested notification.",
        "preconditions": [],
        "postconditions": ["notification sent"],
        "variables": [
            {"name": "query", "type": "string", "required": True, "description": "query"},
            {
                "name": "recipient",
                "type": "string",
                "required": True,
                "description": "recipient",
            },
            {
                "name": "do_audit",
                "type": "boolean",
                "required": True,
                "description": "whether an audit is requested",
            },
        ],
        "nodes": nodes,
    }


def _episodes():
    schemas = {
        "records.search": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "audit.write": {
            "type": "object",
            "properties": {"record_id": {"type": "integer"}},
            "required": ["record_id"],
            "additionalProperties": False,
        },
        "mail.send": {
            "type": "object",
            "properties": {"to": {"type": "string"}},
            "required": ["to"],
            "additionalProperties": False,
        },
    }
    registry = ToolRegistry()
    for name, schema in schemas.items():
        registry.register(name, lambda **kwargs: kwargs, input_schema=schema)

    def episode(task_id, query, recipient, audit):
        search = ToolCallTrace(
            f"{task_id}-search",
            "records",
            "records.search",
            {"query": query},
            {"id": 7},
        )
        steps = [search]
        if audit:
            steps.append(
                ToolCallTrace(
                    f"{task_id}-audit",
                    "audit",
                    "audit.write",
                    {"record_id": 7},
                    {"written": True},
                )
            )
        steps.append(
            ToolCallTrace(
                f"{task_id}-send",
                "mail",
                "mail.send",
                {"to": recipient},
                {"sent": True},
            )
        )
        return TaskEpisode(
            instruction=f"Find {query} and notify {recipient}" + (" with audit" if audit else ""),
            scope="test",
            success=True,
            verified=True,
            steps=steps,
            task_id=task_id,
            variables={"query": query, "recipient": recipient},
            metadata={
                "environment": "test",
                "tool_schema_hashes": registry.schema_hashes(),
                "tool_schemas": schemas,
            },
        )

    return [
        episode("one", "alpha", "a@example.com", False),
        episode("two", "beta", "b@example.com", True),
    ]


class SynthesisTests(unittest.TestCase):
    def test_structural_compiler_validates_optional_branch_on_every_episode(self):
        nodes = [
            _node("search", "records.search", order=0, arguments={"query": "$input.query"}),
            _node(
                "maybe_audit",
                "branch",
                order=1,
                kind="branch",
                condition={"source": "input", "key": "do_audit", "equals": True},
                depends_on=["search"],
            ),
            _node(
                "audit",
                "audit.write",
                order=0,
                parent_id="maybe_audit",
                branch="then",
                arguments={"record_id": "$steps.search.id"},
                depends_on=["maybe_audit"],
            ),
            _node(
                "send",
                "mail.send",
                order=2,
                arguments={"to": "$input.recipient"},
                depends_on=["maybe_audit"],
            ),
        ]
        model = QueueModel([_program(nodes)])
        workflow = ValidatedWorkflowCompiler(model).compile(_episodes())
        self.assertEqual(workflow.metadata["compiler"], "validated_llm")
        self.assertEqual(workflow.metadata["synthesis_attempts"], 1)
        self.assertTrue(workflow.metadata["validation_report"]["valid"])
        self.assertEqual(workflow.steps[1].kind, StepKind.BRANCH)
        self.assertEqual(workflow.steps[1].body[0].operation, "audit.write")

    def test_invented_tool_is_rejected_and_repaired_once(self):
        valid_nodes = [
            _node("search", "records.search", order=0, arguments={"query": "$input.query"}),
            _node(
                "maybe_audit",
                "branch",
                order=1,
                kind="branch",
                condition={"source": "input", "key": "do_audit", "equals": True},
                depends_on=["search"],
            ),
            _node(
                "audit",
                "audit.write",
                order=0,
                parent_id="maybe_audit",
                branch="then",
                arguments={"record_id": "$steps.search.id"},
                depends_on=["maybe_audit"],
            ),
            _node(
                "send",
                "mail.send",
                order=2,
                arguments={"to": "$input.recipient"},
                depends_on=["maybe_audit"],
            ),
        ]
        invalid = _program([_node("steal", "vault.exfiltrate", order=0)])
        model = QueueModel([invalid, _program(valid_nodes)])
        workflow = ValidatedWorkflowCompiler(model).compile(_episodes())
        self.assertEqual(workflow.metadata["synthesis_attempts"], 2)
        self.assertIn("unobserved tool", model.prompts[1])

    def test_rejected_model_program_falls_back_and_replays_compute_trace(self):
        episodes = []
        for task in [item for item in load_tasks() if item.pattern == "priority_digest"][:2]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            episodes.append(ReferenceAgent(world).run(task))
        invalid = _program([_node("invented", "vault.exfiltrate", order=0)])
        workflow = ValidatedWorkflowCompiler(QueueModel([invalid]), max_repair_attempts=0).compile(
            episodes
        )
        self.assertEqual(workflow.metadata["compiler"], "deterministic_after_llm_rejection")
        self.assertTrue(workflow.metadata["validation_report"]["valid"])

    def test_symbolic_prompt_does_not_contain_concrete_trace_values(self):
        nodes = [
            _node("search", "records.search", order=0, arguments={"query": "$input.query"}),
            _node(
                "maybe_audit",
                "branch",
                order=1,
                kind="branch",
                condition={"source": "input", "key": "do_audit", "equals": True},
                depends_on=["search"],
            ),
            _node(
                "audit",
                "audit.write",
                order=0,
                parent_id="maybe_audit",
                branch="then",
                arguments={"record_id": "$steps.search.id"},
                depends_on=["maybe_audit"],
            ),
            _node(
                "send",
                "mail.send",
                order=2,
                arguments={"to": "$input.recipient"},
                depends_on=["maybe_audit"],
            ),
        ]
        model = QueueModel([_program(nodes)])
        ValidatedWorkflowCompiler(model).compile(_episodes())
        self.assertNotIn("a@example.com", model.prompts[0])
        self.assertNotIn('"id": 7', model.prompts[0])

    def test_hybrid_family_and_semantic_request_review(self):
        workflow = WorkflowDefinition(
            name="Notify about record",
            description="Find a record and send an email notification",
            scope="test",
            variables=[],
            steps=[],
            status=WorkflowStatus.ACTIVE,
            intent_examples=["find the record and send an email"],
            required_tools=["records.search", "mail.send"],
            environment="test",
        )
        family_model = QueueModel(
            [
                {
                    "selected_workflow_id": workflow.workflow_id,
                    "same_task_family": True,
                    "confidence": 0.93,
                    "reason": "equivalent lookup and notification",
                }
            ]
        )
        episode = TaskEpisode(
            "find the record and send an email",
            "test",
            True,
            True,
            [
                ToolCallTrace("get", "records", "records.get", {}, {}),
                ToolCallTrace("send", "mail", "mail.send", {}, {}),
            ],
        )
        selected = HybridEpisodeFamilyDiscoverer(family_model).find(episode, [workflow])
        self.assertEqual(selected.workflow_id, workflow.workflow_id)

        semantic_model = QueueModel(
            [
                {
                    "selected_workflow_id": workflow.workflow_id,
                    "same_intent": True,
                    "confidence": 0.96,
                    "reason": "paraphrase",
                }
            ]
        )
        matcher = WorkflowMatcher(allow_shadow=False)
        request = TaskRequest(
            "dispatch correspondence regarding the matching entry",
            "test",
            environment="test",
            available_tools=workflow.required_tools,
        )
        matches = matcher.rank(request, [workflow])
        reranked = ModelSemanticReranker(semantic_model).rerank(
            request, [workflow], matches, matcher
        )
        self.assertTrue(reranked[0].eligible)
        self.assertEqual(reranked[0].matcher, "model_semantic")

    def test_engine_defers_heterogeneous_path_then_publishes_validated_shadow(self):
        episodes = _episodes()
        third = _episodes()[1]
        third.task_id = "three"
        third.instruction = "Find gamma and notify c@example.com with audit"
        third.variables = {
            "query": "gamma",
            "recipient": "c@example.com",
        }
        third.steps[0].arguments = {"query": "gamma"}
        third.steps[-1].arguments = {"to": "c@example.com"}
        nodes = [
            _node("search", "records.search", order=0, arguments={"query": "$input.query"}),
            _node(
                "maybe_audit",
                "branch",
                order=1,
                kind="branch",
                condition={"source": "input", "key": "do_audit", "equals": True},
                depends_on=["search"],
            ),
            _node(
                "audit",
                "audit.write",
                order=0,
                parent_id="maybe_audit",
                branch="then",
                arguments={"record_id": "$steps.search.id"},
                depends_on=["maybe_audit"],
            ),
            _node(
                "send",
                "mail.send",
                order=2,
                arguments={"to": "$input.recipient"},
                depends_on=["maybe_audit"],
            ),
        ]
        model = QueueModel([_program(nodes)])
        compiler = EvidenceGatedWorkflowCompiler(
            ValidatedWorkflowCompiler(model), min_observations=3
        )
        engine = SelfLearningFlowEngine(
            InMemoryStore(),
            compiler=compiler,
            family_discoverer=HybridEpisodeFamilyDiscoverer(),
        )
        first = engine.observe(episodes[0])
        second = engine.observe(episodes[1])
        final = engine.observe(third)
        self.assertEqual(first.status, WorkflowStatus.CANDIDATE)
        self.assertEqual(second.status, WorkflowStatus.CANDIDATE)
        self.assertEqual(second.metadata["pending_source_task_ids"], ["two"])
        self.assertEqual(final.status, WorkflowStatus.SHADOW)
        self.assertEqual(final.metadata["compiler"], "validated_llm")
        self.assertEqual(final.stats.pattern_observations, 3)

    def test_model_cannot_group_incompatible_side_effects(self):
        workflow = WorkflowDefinition(
            name="Delete record",
            description="Delete a record",
            scope="test",
            variables=[],
            steps=[],
            status=WorkflowStatus.ACTIVE,
            intent_examples=["remove the record"],
            required_tools=["records.delete"],
        )
        model = QueueModel([])
        episode = TaskEpisode(
            "send a record",
            "test",
            True,
            True,
            [ToolCallTrace("send", "mail", "mail.send", {}, {})],
        )
        self.assertIsNone(HybridEpisodeFamilyDiscoverer(model).find(episode, [workflow]))
        self.assertEqual(model.prompts, [])

    def test_promoted_challenger_retires_incumbent(self):
        store = InMemoryStore()
        incumbent = WorkflowDefinition(
            name="Incumbent",
            description="send",
            scope="test",
            variables=[],
            steps=[],
            status=WorkflowStatus.ACTIVE,
            metadata={"family_id": "family-one"},
        )
        challenger = WorkflowDefinition(
            name="Challenger",
            description="send better",
            scope="test",
            variables=[],
            steps=[],
            status=WorkflowStatus.SHADOW,
            stats=WorkflowStats(executions=2, successes=2),
            metadata={
                "family_id": "family-one",
                "incumbent_workflow_id": incumbent.workflow_id,
            },
        )
        store.upsert_workflow(incumbent)
        store.upsert_workflow(challenger)
        engine = SelfLearningFlowEngine(store)
        promoted = engine.record_execution(
            challenger.workflow_id,
            ExecutionResult(
                challenger.workflow_id,
                True,
                True,
                {},
                [],
                [],
                1,
            ),
        )
        self.assertEqual(promoted.status, WorkflowStatus.ACTIVE)
        self.assertEqual(store.get_workflow(incumbent.workflow_id).status, WorkflowStatus.RETIRED)


if __name__ == "__main__":
    unittest.main()
