from __future__ import annotations

import unittest

from self_learning_flows.compiler import DeterministicWorkflowCompiler
from self_learning_flows.execution import ToolRegistry, WorkflowExecutor
from self_learning_flows.models import (
    StepKind,
    TaskEpisode,
    TaskRequest,
    ToolCallTrace,
    WorkflowDefinition,
    WorkflowStep,
)


class AlwaysValid:
    def verify(self, *, request, workflow, result):
        del request, workflow, result
        return True


class ControlFlowTests(unittest.TestCase):
    def test_compiler_induces_pagination_from_repeated_verified_calls(self):
        episodes = []
        for index, query in enumerate(("alpha", "beta"), start=1):
            calls = [
                ToolCallTrace(
                    f"call-{page}",
                    "records",
                    "records.search",
                    {"query": query, "page_index": page, "page_limit": 2},
                    {"items": [{"id": page}]},
                )
                for page in (0, 1)
            ]
            episodes.append(
                TaskEpisode(
                    f"search {query}",
                    "test",
                    True,
                    True,
                    calls,
                    task_id=f"page-{index}",
                    variables={"query": query},
                )
            )
        workflow = DeterministicWorkflowCompiler().compile(episodes)
        self.assertEqual(len(workflow.steps), 1)
        self.assertEqual(workflow.steps[0].kind, StepKind.PAGINATE)
        self.assertEqual(workflow.steps[0].arguments["query"], "$input.query")
        self.assertEqual(workflow.steps[0].control["page_argument"], "page_index")

    def test_bounded_pagination_filter_foreach_reduce_and_branch(self):
        notifications: list[int] = []
        audits: list[int] = []
        pages = {
            0: {"items": [{"id": 1, "active": True}, {"id": 2, "active": False}]},
            1: {"items": [{"id": 3, "active": True}]},
        }
        registry = ToolRegistry()
        registry.register(
            "records.search",
            lambda page_index, page_limit: pages.get(page_index, {"items": []}),
            input_schema={
                "type": "object",
                "properties": {
                    "page_index": {"type": "integer"},
                    "page_limit": {"type": "integer"},
                },
                "required": ["page_index", "page_limit"],
                "additionalProperties": False,
            },
        )
        registry.register("mail.notify", lambda record_id: notifications.append(record_id))
        registry.register("audit.write", lambda count: audits.append(count))
        workflow = WorkflowDefinition(
            name="Process active records",
            description="Search and process active records",
            scope="test",
            variables=[],
            steps=[
                WorkflowStep(
                    id="list_records",
                    kind=StepKind.PAGINATE,
                    operation="records.search",
                    arguments={"page_limit": 2},
                    control={
                        "page_argument": "page_index",
                        "page_size_argument": "page_limit",
                        "items_path": "items",
                        "max_iterations": 5,
                    },
                ),
                WorkflowStep(
                    id="active_records",
                    kind=StepKind.FILTER,
                    operation="filter",
                    collection="$steps.list_records.items",
                    control={"predicate": {"path": "active", "op": "eq", "value": True}},
                    depends_on=["list_records"],
                ),
                WorkflowStep(
                    id="notify",
                    kind=StepKind.FOREACH,
                    operation="mail.notify",
                    collection="$steps.active_records",
                    arguments={"record_id": "$loop.item.id"},
                    control={"max_iterations": 10},
                    depends_on=["active_records"],
                ),
                WorkflowStep(
                    id="count",
                    kind=StepKind.REDUCE,
                    operation="count",
                    collection="$steps.active_records",
                    depends_on=["active_records"],
                ),
                WorkflowStep(
                    id="nonempty",
                    kind=StepKind.BRANCH,
                    operation="branch",
                    condition={"source": "steps", "key": "count", "equals": 2},
                    depends_on=["count"],
                    body=[
                        WorkflowStep(
                            id="audit",
                            kind=StepKind.TOOL,
                            operation="audit.write",
                            arguments={"count": "$steps.count"},
                            depends_on=["nonempty"],
                        )
                    ],
                ),
            ],
            required_tools=["records.search", "mail.notify", "audit.write"],
            tool_schema_hashes=registry.schema_hashes(),
        )
        result = WorkflowExecutor(registry, verifier=AlwaysValid()).execute(
            TaskRequest("process records", "test"), workflow, {}
        )
        self.assertTrue(result.success)
        self.assertTrue(result.verified)
        item_ids = [item["id"] for item in result.outputs["list_records"]["items"]]
        self.assertEqual(item_ids, [1, 2, 3])
        self.assertEqual(notifications, [1, 3])
        self.assertEqual(audits, [2])
        self.assertEqual(result.reasoning_calls, 0)

    def test_pagination_fails_instead_of_silently_truncating(self):
        registry = ToolRegistry()
        registry.register("records.search", lambda page_index: [page_index])
        workflow = WorkflowDefinition(
            name="Unsafe pagination",
            description="test",
            scope="test",
            variables=[],
            steps=[
                WorkflowStep(
                    id="search",
                    kind=StepKind.PAGINATE,
                    operation="records.search",
                    control={"page_argument": "page_index", "max_iterations": 2},
                )
            ],
            required_tools=["records.search"],
            tool_schema_hashes=registry.schema_hashes(),
        )
        result = WorkflowExecutor(registry, verifier=AlwaysValid()).execute(
            TaskRequest("search", "test"), workflow, {}
        )
        self.assertFalse(result.success)
        self.assertIn("pagination bound", result.error)


if __name__ == "__main__":
    unittest.main()
