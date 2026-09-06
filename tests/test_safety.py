from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_office.agent import ReferenceAgent
from mini_office.dataset import load_tasks
from mini_office.world import MiniOfficeWorld
from self_learning_flows.compiler import (
    EvidenceGatedWorkflowCompiler,
    ModelAssistedWorkflowCompiler,
)
from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.execution import ToolRegistry, WorkflowExecutor
from self_learning_flows.models import (
    ComputationKind,
    StepKind,
    TaskEpisode,
    TaskRequest,
    ToolCallTrace,
    ToolResult,
    VariableSpec,
    WorkflowDefinition,
    WorkflowStep,
)
from self_learning_flows.providers import (
    AnthropicStructuredModel,
    OpenAICompatibleStructuredModel,
    ProviderError,
)


class FailingThenPassingBackend:
    def run(self, *, kind, operation, inputs):
        del operation, inputs
        if kind == ComputationKind.SLM:
            return ToolResult(False, error="SLM confidence too low")
        return ToolResult(True, value="safe result", reasoning_calls=1)


class FakeStructuredModel:
    model_name = "fake-compiler"

    def __init__(self):
        self.calls = 0
        self.prompts = []

    def generate_json(self, *, system, prompt, schema):
        del system, schema
        self.calls += 1
        self.prompts.append(prompt)
        return {
            "name": "Ticket workflow",
            "description": "Create, assign, and acknowledge a ticket.",
            "preconditions": [],
            "postconditions": ["ticket is assigned"],
            "step_executors": [],
        }


class SafetyTests(unittest.TestCase):
    def test_anthropic_provider_accepts_correct_and_legacy_key_names(self):
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "correct", "ANTROPIC_API_KEY": "legacy"},
            clear=True,
        ):
            self.assertEqual(AnthropicStructuredModel().api_key, "correct")
        with patch.dict("os.environ", {"ANTROPIC_API_KEY": "legacy"}, clear=True):
            self.assertEqual(AnthropicStructuredModel().api_key, "legacy")

    def test_provider_env_factory_builds_chat_completions_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "OPENAI_COMPATIBLE_MODEL=test-model\n"
                "OPENAI_COMPATIBLE_BASE_URL=http://localhost:9999/v1\n"
                "OPENAI_COMPATIBLE_API_KEY=local\n",
                encoding="utf-8",
            )
            with patch.dict("os.environ", {}, clear=True):
                model = OpenAICompatibleStructuredModel.from_env(dotenv_path=str(path))
        self.assertEqual(model.base_url, "http://localhost:9999/v1/chat/completions")

    def test_anthropic_provider_rejects_hugging_face_token(self):
        model = AnthropicStructuredModel(api_key="hf_not_an_anthropic_key")
        with self.assertRaisesRegex(ProviderError, "Hugging Face token"):
            model.generate_json(system="x", prompt="x", schema={"type": "object"})

    def test_compute_falls_up_to_more_capable_executor(self):
        workflow = WorkflowDefinition(
            name="Bounded compute",
            description="test",
            scope="test",
            variables=[VariableSpec("text")],
            steps=[
                WorkflowStep(
                    id="summarize",
                    kind=StepKind.COMPUTE,
                    operation="compute.summarize",
                    arguments={"text": "$input.text"},
                    executor=ComputationKind.SLM,
                    fallback_executors=[ComputationKind.LLM],
                )
            ],
        )
        result = WorkflowExecutor(
            ToolRegistry(),
            compute_backend=FailingThenPassingBackend(),
            require_verifier=False,
        ).execute(TaskRequest("summarize", "test"), workflow, {"text": "hello"})
        self.assertTrue(result.success)
        self.assertFalse(result.verified)
        self.assertEqual(result.executor_kinds, [ComputationKind.SLM, ComputationKind.LLM])
        self.assertEqual([step.success for step in result.steps], [False, True])

    def test_missing_verifier_fails_before_side_effects(self):
        calls = []
        registry = ToolRegistry()
        registry.register("record.update", lambda value: calls.append(value))
        workflow = WorkflowDefinition(
            name="Update record",
            description="test",
            scope="test",
            variables=[VariableSpec("value")],
            steps=[
                WorkflowStep(
                    id="update",
                    kind=StepKind.TOOL,
                    operation="record.update",
                    arguments={"value": "$input.value"},
                )
            ],
            required_tools=["record.update"],
            tool_schema_hashes=registry.schema_hashes(),
        )
        result = WorkflowExecutor(registry).execute(
            TaskRequest("update", "test"), workflow, {"value": "unsafe"}
        )
        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertEqual(calls, [])
        self.assertIn("No outcome verifier", result.error)

    def test_schema_drift_prevents_execution(self):
        registry = ToolRegistry()
        registry.register("record.get", lambda record_id: {"id": record_id})
        workflow = WorkflowDefinition(
            name="Get record",
            description="test",
            scope="test",
            variables=[VariableSpec("record_id")],
            steps=[
                WorkflowStep(
                    id="get",
                    kind=StepKind.TOOL,
                    operation="record.get",
                    arguments={"record_id": "$input.record_id"},
                )
            ],
            required_tools=["record.get"],
            tool_schema_hashes={"record.get": "old-contract"},
        )
        result = WorkflowExecutor(registry).execute(
            TaskRequest("get record", "test"), workflow, {"record_id": "42"}
        )
        self.assertFalse(result.success)
        self.assertTrue(result.escalated)
        self.assertIn("Tool schema changed", result.error)

    def test_observed_schema_drift_quarantines_before_refinement(self):
        store = InMemoryStore()
        engine = SelfLearningFlowEngine(store)
        tasks = [task for task in load_tasks() if task.pattern == "reschedule_and_notify"]
        first_world = MiniOfficeWorld()
        first_world.seed(tasks[0].variables)
        workflow = engine.observe(ReferenceAgent(first_world).run(tasks[0]))
        second_world = MiniOfficeWorld()
        second_world.seed(tasks[1].variables)
        episode = ReferenceAgent(second_world).run(tasks[1])
        episode.metadata["tool_schema_hashes"]["calendar.find_event"] = "changed"
        workflow = engine.observe(episode)
        from self_learning_flows.models import WorkflowStatus

        self.assertEqual(workflow.status, WorkflowStatus.QUARANTINED)
        self.assertEqual(workflow.version, 1)

    def test_paid_compiler_is_deferred_until_recurrence(self):
        tasks = [task for task in load_tasks() if task.pattern == "create_assign_ticket"]
        episodes = []
        for task in tasks[:3]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            episodes.append(ReferenceAgent(world).run(task))
        model = FakeStructuredModel()
        compiler = EvidenceGatedWorkflowCompiler(ModelAssistedWorkflowCompiler(model))
        workflow = compiler.compile(episodes[:1])
        self.assertEqual(model.calls, 0)
        compiler.refine(workflow, episodes)
        self.assertEqual(model.calls, 1)

    def test_model_compiler_prompt_omits_raw_variable_and_tool_values(self):
        secret = "DO_NOT_SEND_THIS_VALUE"
        episode = TaskEpisode(
            instruction="Store the supplied credential",
            scope="test",
            success=True,
            verified=True,
            variables={"credential": secret},
            steps=[
                ToolCallTrace(
                    step_id="store",
                    agent="vault",
                    tool="vault.store",
                    arguments={"credential": secret},
                    result={"stored": secret},
                )
            ],
        )
        model = FakeStructuredModel()
        ModelAssistedWorkflowCompiler(model).compile([episode])
        self.assertNotIn(secret, model.prompts[0])

    def test_repeated_failures_quarantine_workflow(self):
        store = InMemoryStore()
        engine = SelfLearningFlowEngine(store)
        tasks = [task for task in load_tasks() if task.pattern == "create_assign_ticket"]
        for task in tasks[:3]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            engine.observe(ReferenceAgent(world).run(task))
        workflow = store.list_workflows()[0]
        from self_learning_flows.models import ExecutionResult, WorkflowStatus

        failure = ExecutionResult(
            workflow_id=workflow.workflow_id,
            success=False,
            verified=False,
            outputs={},
            steps=[],
            executor_kinds=[],
            latency_ms=1,
            escalated=True,
            error="postcondition failed",
        )
        engine.record_execution(workflow.workflow_id, failure)
        workflow = engine.record_execution(workflow.workflow_id, failure)
        self.assertEqual(workflow.status, WorkflowStatus.QUARANTINED)


if __name__ == "__main__":
    unittest.main()
