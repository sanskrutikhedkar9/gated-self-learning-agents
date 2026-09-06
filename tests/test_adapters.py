from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from self_learning_flows.adapters.appworld import (
    AppWorldOutcomeVerifier,
    AppWorldTraceAdapter,
    AppWorldTraceRecorder,
    build_appworld_tool_registry,
    load_appworld_tool_schemas,
)
from self_learning_flows.adapters.langgraph import LangGraphTraceAdapter


class AdapterTests(unittest.TestCase):
    def test_appworld_schema_registry_and_rich_recorder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mail.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "function": {
                                "name": "mail__send",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"access_token": {"type": "string"}},
                                    "required": ["access_token"],
                                    "additionalProperties": False,
                                },
                            }
                        }
                    ]
                ),
                encoding="utf-8",
            )
            schemas = load_appworld_tool_schemas(directory)
        apis = {"mail": {"send": lambda access_token: {"used": access_token}}}
        registry = build_appworld_tool_registry(apis, schemas)
        self.assertTrue(registry.invoke("mail.send", {"access_token": "x"}).success)

        class Requester:
            def request(self, _app_name, _api_name, **kwargs):
                return {"access_token": kwargs["access_token"]}

        requester = Requester()
        with AppWorldTraceRecorder(requester) as recorder:
            requester.request(_app_name="mail", _api_name="send", access_token="secret-value")
        record = recorder.normalized_record(
            task={"id": "aw-rich", "instruction": "send mail"},
            evaluation={"success": True},
            tool_schema_hashes=registry.schema_hashes(),
        )
        self.assertNotIn("secret-value", str(record))
        self.assertEqual(record["calls"][0]["api"], "mail.send")

    def test_appworld_outcome_verifier_uses_official_tracker_success(self):
        class Tracker:
            def to_dict(self):
                return {"success": True}

        class World:
            def evaluate(self):
                return Tracker()

        verifier = AppWorldOutcomeVerifier(World())
        self.assertTrue(verifier.verify(request=None, workflow=None, result=None))

    def test_langgraph_explicit_trace(self):
        episode = LangGraphTraceAdapter().normalize(
            {
                "task_id": "lg-1",
                "instruction": "look up a record",
                "success": True,
                "verified": True,
                "tool_trace": [
                    {
                        "tool": "records.get",
                        "arguments": {"id": "42"},
                        "result": {"name": "Ada"},
                    }
                ],
            }
        )
        self.assertEqual(episode.framework, "langgraph")
        self.assertEqual(episode.steps[0].tool, "records.get")

    def test_appworld_record(self):
        episode = AppWorldTraceAdapter().normalize(
            task={"task_id": "aw-1", "instruction": "send a message"},
            calls=[
                {
                    "app": "gmail",
                    "api": "send_email",
                    "arguments": {"to": "a@b.com", "access_token": "secret"},
                    "result": {"sent": True},
                    "success": True,
                }
            ],
            passed=True,
            evaluation={"source": "appworld_official_evaluator", "success": True},
            tool_schema_hashes={"gmail.send_email": "schema-v1"},
        )
        self.assertTrue(episode.verified)
        self.assertEqual(episode.steps[0].tool, "gmail.send_email")
        self.assertNotIn("secret", str(episode.to_dict()))

    def test_appworld_rejects_unverified_or_incomplete_success_records(self):
        adapter = AppWorldTraceAdapter()
        with self.assertRaisesRegex(ValueError, "official evaluator"):
            adapter.normalize(
                task={"task_id": "aw-1", "instruction": "send a message"},
                calls=[
                    {
                        "app": "gmail",
                        "api": "send_email",
                        "arguments": {},
                        "result": None,
                        "success": True,
                    }
                ],
                passed=True,
                tool_schema_hashes={"gmail.send_email": "schema-v1"},
            )
        with self.assertRaisesRegex(ValueError, "no result"):
            adapter.normalize(
                task={"task_id": "aw-1", "instruction": "send a message"},
                calls=[
                    {
                        "app": "gmail",
                        "api": "send_email",
                        "arguments": {},
                        "success": True,
                    }
                ],
                passed=False,
                evaluation={"source": "appworld_official_evaluator", "success": False},
            )

    def test_appworld_redaction_preserves_secret_dataflow_identity(self):
        episode = AppWorldTraceAdapter().normalize(
            task={"task_id": "aw-2", "instruction": "send a message"},
            calls=[
                {
                    "app": "gmail",
                    "api": "login",
                    "arguments": {"password": "credential-value"},
                    "result": {"access_token": "token-value"},
                    "success": True,
                },
                {
                    "app": "gmail",
                    "api": "send_email",
                    "arguments": {"access_token": "token-value"},
                    "result": {"sent": True},
                    "success": True,
                },
            ],
            passed=True,
            evaluation={"source": "appworld_official_evaluator", "success": True},
            tool_schema_hashes={
                "gmail.login": "schema-v1",
                "gmail.send_email": "schema-v1",
            },
        )
        login_token = episode.steps[0].result["access_token"]
        send_token = episode.steps[1].arguments["access_token"]
        self.assertEqual(login_token, send_token)
        self.assertNotIn("token-value", str(episode.to_dict()))


if __name__ == "__main__":
    unittest.main()
