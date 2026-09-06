from __future__ import annotations

import unittest

from experiments.appworld.treatment_runner import (
    AgentOutcome,
    AppWorldTreatmentRunner,
)
from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.execution import ToolRegistry
from self_learning_flows.models import (
    StepKind,
    WorkflowDefinition,
    WorkflowStats,
    WorkflowStatus,
    WorkflowStep,
)
from self_learning_flows.research_protocol import (
    ExperimentPhase,
    ProtocolConfig,
    ProtocolGuard,
)


class Docs:
    def function_calling(self):
        return [
            {
                "function": {
                    "name": "mail__send",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ]


class Requester:
    def __init__(self, world):
        self.world = world

    def request(self, _app_name, _api_name, **kwargs):
        del _app_name, _api_name, kwargs
        self.world.passed = True
        return {"sent": True}


class World:
    def __init__(self, identity, *, complete_on_tool=False):
        self.identity = identity
        self.passed = False
        self.completed = False
        self.complete_on_tool = complete_on_tool
        self.closed = False
        self.task = type(
            "Task",
            (),
            {
                "id": "task-1",
                "instruction": "send a message",
                "supervisor": {},
                "api_docs": Docs(),
            },
        )()
        self.requester = Requester(self)
        self.apis = {"mail": {"send": self.send}}

    def send(self):
        self.completed = self.complete_on_tool
        return {"sent": True}

    def task_completed(self):
        return self.completed

    def evaluate(self):
        return {"success": self.passed}

    def close(self):
        self.closed = True


class Agent:
    def run(self, world):
        world.requester.request(_app_name="mail", _api_name="send")
        return AgentOutcome(completed=True, reasoning_calls=1, interactions=1)


class AppWorldTreatmentRunnerTests(unittest.TestCase):
    def test_failed_workflow_falls_back_in_a_new_world(self):
        schema = {"type": "object", "properties": {}}
        registry = ToolRegistry()
        registry.register("mail.send", lambda: None, input_schema=schema)
        workflow = WorkflowDefinition(
            name="Send message",
            description="send a message",
            scope="appworld",
            variables=[],
            steps=[WorkflowStep("send", StepKind.TOOL, "mail.send")],
            status=WorkflowStatus.ACTIVE,
            intent_examples=["send a message"],
            required_tools=["mail.send"],
            tool_schema_hashes=registry.schema_hashes(),
            environment="appworld",
            stats=WorkflowStats(executions=10, successes=10),
        )
        store = InMemoryStore()
        store.upsert_workflow(workflow)
        worlds = []

        def factory(task_id, experiment_name):
            del task_id, experiment_name
            world = World(len(worlds) + 1)
            worlds.append(world)
            return world

        runner = AppWorldTreatmentRunner(
            engine=SelfLearningFlowEngine(store),
            guard=ProtocolGuard(ProtocolConfig(), ExperimentPhase.DEV, "dev"),
            baseline_agent=Agent(),
            world_factory=factory,
            experiment_name="test",
            condition="treatment",
        )
        result = runner.run_task("task-1")
        self.assertTrue(result.success)
        self.assertEqual(result.route, "fresh_world_fallback")
        self.assertEqual(result.fresh_worlds, 2)
        self.assertEqual([world.identity for world in worlds], [1, 2])
        self.assertTrue(all(world.closed for world in worlds))

    def test_official_failure_after_completion_does_not_get_oracle_fallback(self):
        schema = {"type": "object", "properties": {}}
        registry = ToolRegistry()
        registry.register("mail.send", lambda: None, input_schema=schema)
        workflow = WorkflowDefinition(
            name="Send message",
            description="send a message",
            scope="appworld",
            variables=[],
            steps=[WorkflowStep("send", StepKind.TOOL, "mail.send")],
            status=WorkflowStatus.ACTIVE,
            intent_examples=["send a message"],
            required_tools=["mail.send"],
            tool_schema_hashes=registry.schema_hashes(),
            environment="appworld",
            stats=WorkflowStats(executions=10, successes=10),
        )
        store = InMemoryStore()
        store.upsert_workflow(workflow)
        worlds = []

        def factory(task_id, experiment_name):
            del task_id, experiment_name
            world = World(len(worlds) + 1, complete_on_tool=True)
            worlds.append(world)
            return world

        runner = AppWorldTreatmentRunner(
            engine=SelfLearningFlowEngine(store),
            guard=ProtocolGuard(ProtocolConfig(), ExperimentPhase.TEST, "test_normal"),
            baseline_agent=Agent(),
            world_factory=factory,
            experiment_name="test",
            condition="treatment",
        )
        result = runner.run_task("task-1")
        self.assertFalse(result.success)
        self.assertEqual(result.route, "workflow")
        self.assertEqual(len(worlds), 1)


if __name__ == "__main__":
    unittest.main()
