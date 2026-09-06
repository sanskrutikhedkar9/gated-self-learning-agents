from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.appworld.preflight import (
    audit_dataset,
    audit_native_outputs,
    audit_protocol,
    audit_records,
)


class AppWorldPreflightTests(unittest.TestCase):
    def test_protocol_audit_rejects_wrong_phase_split(self):
        protocol = {
            "train_splits": ["train"],
            "dev_splits": ["dev"],
            "test_splits": ["test_normal"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            result = audit_protocol(path, phase="test", split="dev")
        self.assertFalse(result["valid"])
        self.assertIn("not declared", result["error"])

    def test_dataset_audit_exposes_threshold_dead_zone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dev.txt"
            path.write_text("scenario_1\nscenario_2\nscenario_3\n", encoding="utf-8")
            result = audit_dataset(path, min_observations=3)
        self.assertEqual(result["scenarios"], 1)
        self.assertEqual(result["maximum_same_scenario_online_routes"], 0)

    def test_native_logs_are_not_mistaken_for_replayable_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "verification"
            logs = root / "tasks" / "task_1" / "logs"
            logs.mkdir(parents=True)
            (logs / "api_calls.jsonl").write_text(
                json.dumps(
                    {
                        "method": "post",
                        "url": "/mail/send",
                        "data": {"access_token": "dummy-secret"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = audit_native_outputs(root, min_observations=3)
        self.assertFalse(result["ingestible"])
        self.assertEqual(result["calls_missing_result_or_success"], 1)
        self.assertEqual(result["calls_with_sensitive_fields"], 1)
        self.assertTrue(result["is_appworld_ground_truth_verification"])

    def test_normalized_records_report_real_reuse_opportunity(self):
        record = {
            "task": {"task_id": "task", "instruction": "send a message"},
            "calls": [
                {
                    "app": "mail",
                    "api": "send",
                    "arguments": {"to": "person@example.com"},
                    "result": {"sent": True},
                    "success": True,
                }
            ],
            "passed": True,
            "evaluation": {
                "source": "appworld_official_evaluator",
                "success": True,
            },
            "tool_schema_hashes": {"mail.send": "schema-v1"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            lines = []
            for index in range(4):
                item = json.loads(json.dumps(record))
                item["task"]["task_id"] = f"task-{index}"
                lines.append(json.dumps(item))
            path.write_text("\n".join(lines), encoding="utf-8")
            result = audit_records(path, min_observations=3)
        self.assertTrue(result["ingestible"])
        self.assertEqual(result["potential_online_routes_after_threshold"], 1)


if __name__ == "__main__":
    unittest.main()
