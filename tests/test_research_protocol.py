from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.models import (
    ExecutionResult,
    TaskEpisode,
    WorkflowDefinition,
    WorkflowStatus,
)
from self_learning_flows.research_protocol import (
    ExperimentPhase,
    ProtocolConfig,
    ProtocolGuard,
    ProtocolViolation,
)


class ResearchProtocolTests(unittest.TestCase):
    def test_test_phase_rejects_learning_and_detects_state_mutation(self):
        store = InMemoryStore()
        workflow = WorkflowDefinition(
            name="Frozen",
            description="frozen workflow",
            scope="test",
            variables=[],
            steps=[],
            status=WorkflowStatus.ACTIVE,
        )
        store.upsert_workflow(workflow)
        config = ProtocolConfig()
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "freeze.json"
            ProtocolGuard(config, ExperimentPhase.DEV, "dev").seal(
                store,
                manifest_path,
                source_commit="abc123",
                run_config={"model": "test"},
            )
            test_guard = ProtocolGuard(config, ExperimentPhase.TEST, "test_normal")
            test_guard.validate_frozen(store, manifest_path)
            with self.assertRaises(ProtocolViolation):
                test_guard.observe(
                    SelfLearningFlowEngine(store),
                    TaskEpisode("x", "test", True, True, []),
                )
            SelfLearningFlowEngine(store).record_execution(
                workflow.workflow_id,
                ExecutionResult(
                    workflow_id=workflow.workflow_id,
                    success=True,
                    verified=True,
                    outputs={},
                    steps=[],
                    executor_kinds=[],
                    latency_ms=1,
                ),
            )
            with self.assertRaisesRegex(ProtocolViolation, "state"):
                test_guard.validate_frozen(store, manifest_path)

    def test_phase_split_mapping_is_fail_closed(self):
        with self.assertRaisesRegex(ProtocolViolation, "not declared"):
            ProtocolGuard(ProtocolConfig(), ExperimentPhase.TEST, "dev")


if __name__ == "__main__":
    unittest.main()
