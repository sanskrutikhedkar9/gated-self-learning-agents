"""Minimal LangGraph wiring example; application nodes are intentionally tiny."""

from self_learning_flows import InMemoryStore, SelfLearningFlowEngine
from self_learning_flows.adapters.langgraph import build_self_learning_graph


def full_agent_node(state: dict) -> dict:
    # Run the agent and an external state verifier; never derive `verified` from
    # the agent's own success claim.
    return {"success": True, "verified": True, "tool_trace": state.get("tool_trace", [])}


def workflow_node(state: dict) -> dict:
    # Fetch state["_self_learning_flow_id"], execute it against your ToolRegistry,
    # and verify the resulting state. Use engine.execute so lifecycle statistics
    # are recorded. The Mini Office benchmark contains the full version.
    return {"success": True, "verified": True}


engine = SelfLearningFlowEngine(InMemoryStore())
graph = build_self_learning_graph(
    engine,
    full_agent_node=full_agent_node,
    workflow_node=workflow_node,
)
