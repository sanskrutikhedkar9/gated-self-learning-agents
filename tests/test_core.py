from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_office.agent import ReferenceAgent
from mini_office.compute import ReferenceComputeBackend
from mini_office.dataset import load_tasks
from mini_office.verifier import OfficeVerifier, verify_task
from mini_office.world import MiniOfficeWorld
from self_learning_flows.compiler import DeterministicWorkflowCompiler
from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.execution import WorkflowExecutor
from self_learning_flows.models import (
    ComputationKind,
    ConfirmationDecision,
    TaskEpisode,
    TaskRequest,
    ToolCallTrace,
    WorkflowStatus,
)
from self_learning_flows.storage import SQLiteStore


class CompilerAndExecutionTests(unittest.TestCase):
    def test_compiler_filters_failed_calls_and_generalizes_nested_arguments(self):
        episodes = []
        for index, value in enumerate(("alpha", "beta"), start=1):
            episodes.append(
                TaskEpisode(
                    task_id=f"nested-{index}",
                    instruction="Store a value",
                    scope="test",
                    success=True,
                    verified=True,
                    steps=[
                        ToolCallTrace(
                            step_id="failed",
                            agent="records",
                            tool="records.invalid",
                            success=False,
                        ),
                        ToolCallTrace(
                            step_id="store",
                            agent="records",
                            tool="records.store",
                            arguments={"payload": {"value": value}},
                            result={"ok": True},
                            executor=ComputationKind.DETERMINISTIC,
                        ),
                    ],
                )
            )
        workflow = DeterministicWorkflowCompiler().compile(episodes)
        self.assertEqual([step.operation for step in workflow.steps], ["records.store"])
        self.assertEqual(
            workflow.steps[0].arguments["payload"]["value"],
            "$input.store_payload_value",
        )

    def test_raw_request_can_match_then_extract_variables(self):
        class Extractor:
            def extract(self, request, workflow):
                del request, workflow
                return {"record_id": "42"}

        from self_learning_flows.models import (
            VariableSpec,
            WorkflowDefinition,
            WorkflowStats,
        )

        workflow = WorkflowDefinition(
            name="Get record",
            description="Look up a record",
            scope="test",
            variables=[VariableSpec("record_id")],
            steps=[],
            intent_examples=["Look up a record"],
            status=WorkflowStatus.ACTIVE,
            stats=WorkflowStats(executions=10, successes=10),
        )
        store = InMemoryStore()
        store.upsert_workflow(workflow)
        engine = SelfLearningFlowEngine(store, variable_extractor=Extractor())
        proposal = engine.propose(TaskRequest("Please look up the record", "test"))
        self.assertIsNotNone(proposal)
        self.assertEqual(proposal.variables, {"record_id": "42"})
        self.assertEqual(proposal.missing_variables, [])

    def test_compiles_dataflow_and_replays_without_full_agent(self):
        tasks = [task for task in load_tasks() if task.pattern == "reschedule_and_notify"]
        episodes = []
        for task in tasks[:3]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            episode = ReferenceAgent(world).run(task)
            episode.verified = verify_task(task, world)
            episodes.append(episode)

        workflow = DeterministicWorkflowCompiler().compile(episodes)
        self.assertEqual(workflow.steps[3].arguments["event_id"], "$steps.find_event.id")
        self.assertTrue(workflow.steps[4].arguments["to"].startswith("$steps."))

        target = tasks[3]
        world = MiniOfficeWorld()
        world.seed(target.variables)
        result = WorkflowExecutor(
            world.tool_registry(),
            compute_backend=ReferenceComputeBackend(),
            verifier=OfficeVerifier(target, world),
        ).execute(
            TaskRequest(target.instruction, "mini-office", target.variables),
            workflow,
            target.variables,
        )
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        self.assertEqual(result.reasoning_calls, 0)

    def test_only_verified_successes_are_learned(self):
        store = InMemoryStore()
        engine = SelfLearningFlowEngine(store)
        task = load_tasks()[0]
        world = MiniOfficeWorld()
        world.seed(task.variables)
        episode = ReferenceAgent(world).run(task)
        episode.verified = False
        self.assertIsNone(engine.observe(episode))
        self.assertEqual(store.list_workflows(), [])

    def test_duplicate_episode_is_idempotent(self):
        store = InMemoryStore()
        engine = SelfLearningFlowEngine(store)
        task = load_tasks()[0]
        world = MiniOfficeWorld()
        world.seed(task.variables)
        episode = ReferenceAgent(world).run(task)
        first = engine.observe(episode)
        duplicate = engine.observe(episode)
        self.assertEqual(duplicate.version, first.version)
        self.assertEqual(len(store.list_workflow_versions(first.workflow_id)), 1)

    def test_engine_requires_confirmation_and_confirmed_values(self):
        store = InMemoryStore()
        engine = SelfLearningFlowEngine(store)
        tasks = [task for task in load_tasks() if task.pattern == "create_assign_ticket"]
        for task in tasks[:3]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            engine.observe(ReferenceAgent(world).run(task))
        task = tasks[3]
        world = MiniOfficeWorld()
        world.seed(task.variables)
        request = TaskRequest(
            task.instruction,
            "mini-office",
            task.workflow_variables,
            environment="mini-office-v1",
            available_tools=world.tool_registry().names(),
        )
        proposal = engine.propose(request)
        self.assertIsNotNone(proposal)
        executor = WorkflowExecutor(world.tool_registry(), verifier=OfficeVerifier(task, world))
        with self.assertRaises(PermissionError):
            engine.execute(request, proposal, executor)
        engine.record_feedback(proposal, ConfirmationDecision.APPROVE)
        with self.assertRaises(PermissionError):
            engine.execute(request, proposal, executor, variables={})


class LifecycleAndStorageTests(unittest.TestCase):
    def test_candidate_shadow_active_lifecycle(self):
        store = InMemoryStore()
        engine = SelfLearningFlowEngine(store)
        tasks = [task for task in load_tasks() if task.pattern == "create_assign_ticket"]
        for index, task in enumerate(tasks[:3]):
            world = MiniOfficeWorld()
            world.seed(task.variables)
            episode = ReferenceAgent(world).run(task)
            episode.verified = verify_task(task, world)
            workflow = engine.observe(episode)
            self.assertEqual(
                workflow.status,
                WorkflowStatus.SHADOW if index == 2 else WorkflowStatus.CANDIDATE,
            )

        for task in tasks[3:6]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            request = TaskRequest(
                task.instruction,
                "mini-office",
                task.variables,
                environment="mini-office-v1",
                available_tools=world.tool_registry().names(),
            )
            proposal = engine.propose(request)
            self.assertIsNotNone(proposal)
            result = WorkflowExecutor(
                world.tool_registry(),
                verifier=OfficeVerifier(task, world),
            ).execute(request, proposal.workflow, proposal.variables)
            workflow = engine.record_execution(proposal.workflow.workflow_id, result)
        self.assertEqual(workflow.status, WorkflowStatus.ACTIVE)

    def test_sqlite_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "learning.db")
            task = load_tasks()[0]
            world = MiniOfficeWorld()
            world.seed(task.variables)
            episode = ReferenceAgent(world).run(task)
            episode.verified = verify_task(task, world)
            workflow = SelfLearningFlowEngine(store).observe(episode)
            self.assertEqual(store.get_episode(task.task_id).instruction, task.instruction)
            self.assertEqual(store.get_workflow(workflow.workflow_id).name, workflow.name)
            self.assertEqual(
                [item.version for item in store.list_workflow_versions(workflow.workflow_id)],
                [1],
            )


if __name__ == "__main__":
    unittest.main()
