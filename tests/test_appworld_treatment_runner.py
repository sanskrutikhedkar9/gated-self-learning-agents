from __future__ import annotations

import unittest

from experiments.appworld.treatment_runner import (
    AgentOutcome,
    AppWorldTreatmentRunner,
    OpenAICompatibleCodeAgent,
    _debug_text,
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


class ScriptedCodeAgent(OpenAICompatibleCodeAgent):
    def __init__(self, responses):
        super().__init__("fake-model", api_key="unused", max_interactions=len(responses))
        self.responses = list(responses)
        self.initial_prompt = ""

    def _completion(self, messages):
        if not self.initial_prompt:
            self.initial_prompt = messages[1]["content"]
        return self.responses.pop(0), {"prompt_tokens": 10, "completion_tokens": 5}


class CodeAgentDocs:
    def items(self):
        return {
            "mail": {
                "send": {
                    "response_schemas": {
                        "success": {"message_id": "integer", "sent": "boolean"}
                    }
                }
            },
            "supervisor": {
                "complete_task": {
                    "response_schemas": {"success": {"message": "string"}}
                }
            },
        }.items()

    def function_calling(self):
        return [
            {
                "function": {
                    "name": "mail__send",
                    "description": "Send a message.",
                    "parameters": {
                        "type": "object",
                        "properties": {"body": {"type": "string"}},
                    },
                }
            },
            {
                "function": {
                    "name": "supervisor__complete_task",
                    "description": "Complete the active task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string"},
                            "answer": {"type": "string"},
                        },
                    },
                }
            },
        ]


class CodeAgentWorld:
    def __init__(self):
        self.task = type(
            "Task",
            (),
            {
                "instruction": "send a message",
                "supervisor": {},
                "api_docs": CodeAgentDocs(),
            },
        )()
        self.executed = []
        self.completed = False

    def execute(self, code):
        self.executed.append(code)
        if "complete_task" in code:
            self.completed = True
        return "{'sent': true}" if "mail.send" in code else "Execution successful."

    def task_completed(self):
        return self.completed


class AppWorldTreatmentRunnerTests(unittest.TestCase):
    def test_agent_diagnostics_redact_quoted_credentials(self):
        text = _debug_text('{"access_token": "provider-secret", "value": 3}')
        self.assertNotIn("provider-secret", text)
        self.assertIn("<redacted>", text)

    def test_schema_grounded_agent_rejects_invented_api_before_execution(self):
        agent = ScriptedCodeAgent(
            [
                "```python\napis.mail.sned(body='hello')\n```",
                "```python\nresult = apis.mail.send(body='hello')\nprint(result)\n```",
                "```python\napis.supervisor.complete_task(status='success', answer='sent')\n```",
            ]
        )
        world = CodeAgentWorld()
        outcome = agent.run(world)
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.validation_failures, 1)
        self.assertEqual(len(world.executed), 2)
        self.assertNotIn("sned", "\n".join(world.executed))
        self.assertIn("apis.mail.send", agent.initial_prompt)
        self.assertIn("parameters", agent.initial_prompt)
        self.assertIn("response_schemas", agent.initial_prompt)
        self.assertIn("message_id", agent.initial_prompt)
        self.assertFalse(outcome.actions[0]["executed"])
        self.assertIn("closest real APIs", outcome.actions[0]["validation_error"])

    def test_unprinted_documentation_call_is_rejected(self):
        catalog = {"apis.mail.send": {}}
        error = OpenAICompatibleCodeAgent._validate_code(
            "apis.api_docs.show_api_descriptions(app_name='mail')", catalog
        )
        self.assertIn("print", error)

        assigned_error = OpenAICompatibleCodeAgent._validate_code(
            "docs = apis.api_docs.show_api_descriptions(app_name='mail')", catalog
        )
        self.assertIn("print", assigned_error)

        visible_error = OpenAICompatibleCodeAgent._validate_code(
            "docs = apis.api_docs.show_api_descriptions(app_name='mail')\nprint(docs)", catalog
        )
        self.assertIsNone(visible_error)

    def test_api_calls_require_keyword_arguments_and_known_parameter_names(self):
        catalog = {
            "apis.mail.send": {
                "parameters": {
                    "type": "object",
                    "properties": {"body": {"type": "string"}},
                }
            }
        }
        positional_error = OpenAICompatibleCodeAgent._validate_code(
            "apis.mail.send({'body': 'hello'})", catalog
        )
        self.assertIn("keyword arguments only", positional_error)

        unknown_argument_error = OpenAICompatibleCodeAgent._validate_code(
            "apis.mail.send(message='hello')", catalog
        )
        self.assertIn("unknown keyword arguments", unknown_argument_error)

        valid_error = OpenAICompatibleCodeAgent._validate_code(
            "apis.mail.send(body='hello')", catalog
        )
        self.assertIsNone(valid_error)

    def test_completion_gate_rejects_incomplete_most_liked_plan(self):
        catalog = {
            "apis.spotify.show_song_privates": {
                "parameters": {"properties": {"song_id": {}, "access_token": {}}}
            },
            "apis.supervisor.complete_task": {
                "parameters": {"properties": {"status": {}, "answer": {}}}
            },
        }
        error = OpenAICompatibleCodeAgent._validate_code(
            "liked = apis.spotify.show_song_privates(song_id=1, access_token='x')\n"
            "apis.supervisor.complete_task(status='success', answer='Silver Lining')",
            catalog,
            task_instruction="What is the title of the most-liked song in my Spotify playlists.",
        )
        self.assertIn("do not complete yet", error)
        self.assertIn("spotify.show_song", error)

    def test_completion_gate_accepts_complete_most_liked_plan(self):
        catalog = {
            "apis.spotify.show_playlist_library": {
                "parameters": {"properties": {"page_index": {}}}
            },
            "apis.spotify.show_song": {"parameters": {"properties": {"song_id": {}}}},
            "apis.supervisor.complete_task": {
                "parameters": {"properties": {"status": {}, "answer": {}}}
            },
        }
        error = OpenAICompatibleCodeAgent._validate_code(
            "playlists = apis.spotify.show_playlist_library(page_index=0)\n"
            "songs = [apis.spotify.show_song(song_id=1)]\n"
            "best = max(songs, key=lambda song: song['like_count'])\n"
            "apis.supervisor.complete_task(status='success', answer=best['title'])",
            catalog,
            task_instruction="What is the title of the most-liked song in my Spotify playlists.",
        )
        self.assertIsNone(error)

    def test_completion_gate_accumulates_evidence_across_turns(self):
        catalog = {
            "apis.spotify.show_playlist_library": {
                "parameters": {"properties": {"page_index": {}}}
            },
            "apis.spotify.show_song": {"parameters": {"properties": {"song_id": {}}}},
            "apis.supervisor.complete_task": {
                "parameters": {"properties": {"status": {}, "answer": {}}}
            },
        }
        error = OpenAICompatibleCodeAgent._validate_code(
            "apis.supervisor.complete_task(status='success', answer=best['title'])",
            catalog,
            task_instruction="What is the title of the most-liked song in my Spotify playlists.",
            prior_code=(
                "playlists = apis.spotify.show_playlist_library(page_index=0)\n"
                "songs = [apis.spotify.show_song(song_id=1)]\n"
                "best = max(songs, key=lambda song: song['like_count'])"
            ),
        )
        self.assertIsNone(error)

    def test_user_playlist_query_rejects_public_only_filter(self):
        catalog = {
            "apis.spotify.show_playlist_library": {
                "parameters": {"properties": {"is_public": {}}}
            }
        }
        error = OpenAICompatibleCodeAgent._validate_code(
            "apis.spotify.show_playlist_library(is_public=True)",
            catalog,
            task_instruction="What is in my Spotify playlists?",
        )
        self.assertIn("private playlists", error)

    def test_cold_start_reuses_world_and_is_not_a_fallback(self):
        worlds = []

        def factory(task_id, experiment_name):
            del task_id, experiment_name
            world = World(len(worlds) + 1)
            worlds.append(world)
            return world

        runner = AppWorldTreatmentRunner(
            engine=SelfLearningFlowEngine(InMemoryStore()),
            guard=ProtocolGuard(ProtocolConfig(), ExperimentPhase.TRAIN, "train"),
            baseline_agent=Agent(),
            world_factory=factory,
            experiment_name="test",
            condition="treatment",
        )
        result = runner.run_task("task-1")
        self.assertTrue(result.success)
        self.assertEqual(result.route, "full_agent")
        self.assertEqual(result.fresh_worlds, 1)
        self.assertEqual(len(worlds), 1)
        self.assertTrue(worlds[0].closed)

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
