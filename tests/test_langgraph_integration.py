from __future__ import annotations

import importlib.util
import unittest

from mini_office.agent import ReferenceAgent
from mini_office.dataset import load_tasks
from mini_office.world import MiniOfficeWorld
from self_learning_flows.adapters.langgraph import build_self_learning_graph
from self_learning_flows.engine import InMemoryStore, SelfLearningFlowEngine

HAS_LANGGRAPH = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(HAS_LANGGRAPH, "LangGraph optional dependency is not installed")
class LiveLangGraphTests(unittest.TestCase):
    def test_confirmation_interrupt_and_resume_routes_to_workflow(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.types import Command

        tasks = [task for task in load_tasks() if task.pattern == "reschedule_and_notify"]
        engine = SelfLearningFlowEngine(InMemoryStore())
        for task in tasks[:3]:
            world = MiniOfficeWorld()
            world.seed(task.variables)
            engine.observe(ReferenceAgent(world).run(task))

        graph = build_self_learning_graph(
            engine,
            full_agent_node=lambda state: {"selected": "full_agent"},
            workflow_node=lambda state: {"selected": "workflow"},
            checkpointer=InMemorySaver(),
        )
        target = tasks[3]
        world = MiniOfficeWorld()
        world.seed(target.variables)
        state = {
            "instruction": target.instruction,
            "scope": "mini-office",
            "environment": "mini-office-v1",
            "variables": target.workflow_variables,
            "available_tools": world.tool_registry().names(),
        }
        config = {"configurable": {"thread_id": "langgraph-test"}}
        paused = graph.invoke(state, config)
        self.assertIn("__interrupt__", paused)
        completed = graph.invoke(Command(resume={"decision": "approve"}), config)
        self.assertEqual(completed["selected"], "workflow")


if __name__ == "__main__":
    unittest.main()
