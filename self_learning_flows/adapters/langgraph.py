"""LangGraph integration without coupling the learning core to LangGraph types."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from ..engine import SelfLearningFlowEngine
from ..models import (
    ComputationKind,
    ConfirmationDecision,
    TaskEpisode,
    TaskRequest,
    ToolCallTrace,
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _content(value: Any) -> Any:
    content = _get(value, "content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


class LangGraphTraceAdapter:
    """Normalizes a completed LangGraph state into a framework-neutral episode."""

    def normalize(self, state: dict[str, Any]) -> TaskEpisode:
        explicit = state.get("tool_trace") or state.get("steps")
        traces = (
            self._explicit_traces(explicit)
            if explicit
            else self._message_traces(state.get("messages", []))
        )
        return TaskEpisode(
            task_id=state.get("task_id")
            or state.get("thread_id")
            or TaskEpisode.__dataclass_fields__["task_id"].default_factory(),
            instruction=state.get("instruction") or state.get("input", ""),
            scope=state.get("scope", "langgraph"),
            success=bool(state.get("success", False)),
            verified=bool(state.get("verified", False)),
            steps=traces,
            outcome=dict(state.get("outcome", {})),
            variables=dict(state.get("variables", {})),
            framework="langgraph",
            input_tokens=int(state.get("input_tokens", 0)),
            output_tokens=int(state.get("output_tokens", 0)),
            reasoning_calls=int(state.get("reasoning_calls", 0)),
            metadata=dict(state.get("metadata", {})),
        )

    @staticmethod
    def _explicit_traces(items: list[Any]) -> list[ToolCallTrace]:
        traces = []
        for index, item in enumerate(items):
            if isinstance(item, ToolCallTrace):
                traces.append(item)
                continue
            value = dict(item)
            value.setdefault("step_id", f"step-{index + 1}")
            value.setdefault("agent", str(value.get("tool", "tool")).split(".", 1)[0])
            traces.append(ToolCallTrace.from_dict(value))
        return traces

    @staticmethod
    def _message_traces(messages: list[Any]) -> list[ToolCallTrace]:
        pending: dict[str, tuple[str, dict[str, Any]]] = {}
        traces: list[ToolCallTrace] = []
        for message in messages:
            for call in _get(message, "tool_calls", []) or []:
                call_id = str(_get(call, "id", f"call-{len(pending) + 1}"))
                pending[call_id] = (
                    str(_get(call, "name", "unknown")),
                    dict(_get(call, "args", {}) or {}),
                )
            call_id = _get(message, "tool_call_id")
            if call_id is not None and str(call_id) in pending:
                name, arguments = pending.pop(str(call_id))
                traces.append(
                    ToolCallTrace(
                        step_id=str(call_id),
                        agent=name.split(".", 1)[0],
                        tool=name,
                        arguments=arguments,
                        result=_content(message),
                        executor=ComputationKind.DETERMINISTIC,
                    )
                )
        return traces


def build_self_learning_graph(
    engine: SelfLearningFlowEngine,
    *,
    full_agent_node: Callable[[dict[str, Any]], dict[str, Any]],
    workflow_node: Callable[[dict[str, Any]], dict[str, Any]],
    observer_node: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    checkpointer: Any = None,
):
    """Build a confirmation-gated LangGraph router.

    ``workflow_node`` receives ``_self_learning_flow_id`` and
    ``_self_learning_flow_variables`` in state. The caller remains responsible for executing
    the workflow against its own tool registry.
    """

    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:  # optional integration remains optional
        raise ImportError(
            "Install the LangGraph extra: pip install 'self-learning-flows[langgraph]'"
        ) from exc

    def gate(state: dict[str, Any]) -> dict[str, Any]:
        request = TaskRequest(
            instruction=state.get("instruction") or state.get("input", ""),
            scope=state.get("scope", "langgraph"),
            variables=dict(state.get("variables", {})),
            user_id=state.get("user_id", "anonymous"),
            environment=state.get("environment", "default"),
            available_tools=list(state.get("available_tools", [])),
        )
        proposal = engine.propose(request)
        if proposal is None:
            return {**state, "_self_learning_flow_route": "full_agent"}
        decision = interrupt(proposal.confirmation_payload())
        if isinstance(decision, str):
            decision = {"decision": decision}
        selected = ConfirmationDecision(decision.get("decision", "reject"))
        variables = {**proposal.variables, **dict(decision.get("variables", {}))}
        engine.record_feedback(
            proposal,
            selected,
            final_variables=variables,
            task_instruction=request.instruction,
            user_id=request.user_id,
        )
        if selected not in {ConfirmationDecision.APPROVE, ConfirmationDecision.EDIT}:
            return {**state, "_self_learning_flow_route": "full_agent"}
        still_missing = [
            variable.name
            for variable in proposal.workflow.variables
            if variable.required and variables.get(variable.name) is None
        ]
        if still_missing:
            return {
                **state,
                "_self_learning_flow_route": "full_agent",
                "_self_learning_flow_error": "Missing workflow variables: "
                + ", ".join(still_missing),
            }
        return {
            **state,
            "_self_learning_flow_route": "workflow",
            "_self_learning_flow_id": proposal.workflow.workflow_id,
            "_self_learning_flow_variables": variables,
        }

    def observe(state: dict[str, Any]) -> dict[str, Any]:
        if observer_node is not None:
            state = {**state, **(observer_node(state) or {})}
        if state.get("_self_learning_flow_route") == "full_agent":
            engine.observe(LangGraphTraceAdapter().normalize(state))
        return state

    def run_full_agent(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, **(full_agent_node(state) or {})}

    def run_workflow(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, **(workflow_node(state) or {})}

    graph = StateGraph(dict)
    graph.add_node("gate", gate)
    graph.add_node("full_agent", run_full_agent)
    graph.add_node("workflow", run_workflow)
    graph.add_node("observe", observe)
    graph.add_edge(START, "gate")
    graph.add_conditional_edges(
        "gate",
        lambda state: state["_self_learning_flow_route"],
        {"full_agent": "full_agent", "workflow": "workflow"},
    )
    graph.add_edge("full_agent", "observe")
    graph.add_edge("workflow", "observe")
    graph.add_edge("observe", END)
    return graph.compile(checkpointer=checkpointer)
