"""In-memory LangGraph round budgets and workflow routing."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
import csv
import io
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph


@dataclass
class BudgetState:
    rounds_used: int = 0
    limit_reached: bool = False
    runtime_status: str = "pending"


class RoundLimitReached(Exception):
    """Control signal caught at a stage boundary, never a retryable model error."""


def state_values(state):
    """Keep native state objects in memory, without serializing message history."""
    if isinstance(state, dict):
        return dict(state)
    return {f.name: getattr(state, f.name) for f in fields(state)
            if not f.name.startswith("_")}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def csv_text(header, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue()


@dataclass
class RunSession:
    limits: dict[str, int]
    used: dict = field(default_factory=dict)
    total_used: dict = field(default_factory=dict)
    interactions: dict = field(default_factory=dict)
    statuses: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    exhausted: set = field(default_factory=set)
    lock: Any = field(default_factory=RLock)

    def __post_init__(self):
        for name, limit in self.limits.items():
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError(f"AGENT_MAX_ROUNDS[{name!r}] must be a positive integer")

    @property
    def recursion_limit(self):
        # Tool, interaction, import and finalization nodes also consume graph steps.
        return max(100, 12 * sum(self.limits.values()) + 100)

    def available(self, name):
        return self.used.get(name, 0) < self.limits[name]

    def reserve(self, name):
        with self.lock:
            if not self.available(name):
                self.exhausted.add(name)
                raise RoundLimitReached(f"{name}: {self.used[name]}/{self.limits[name]}")
            self.used[name] = self.used.get(name, 0) + 1
            self.total_used[name] = self.total_used.get(name, 0) + 1
            return self.used[name]

    def human_replied(self, name):
        with self.lock:
            self.used[name] = 0
            self.interactions[name] = self.interactions.get(name, 0) + 1
            self.exhausted.discard(name)

    def begin(self, name):
        """Start a new subgraph invocation with its own autonomous budget."""
        with self.lock:
            self.used[name] = 0
            self.exhausted.discard(name)
            self.statuses[name] = "running"
            self.results.pop(name, None)

    def update(self, name, state, status):
        with self.lock:
            self.statuses[name] = status
            self.results[name] = state_values(state)

    def ready(self, name):
        # Exhausted agents have saved their current outputs for the next stage.
        return self.statuses.get(name) in {"completed", "exhausted"}


_session = ContextVar("agent_run_session", default=None)


def current_session():
    session = _session.get()
    if session is None:
        raise RuntimeError("Agent must run inside a managed LangGraph run")
    return session


@contextmanager
def run_session(session=None):
    if _session.get() is not None and session is None:
        yield _session.get()
        return
    if session is None:
        from dependencies import AGENT_MAX_ROUNDS
        session = RunSession(dict(AGENT_MAX_ROUNDS))
    token = _session.set(session)
    try:
        yield session
    finally:
        _session.reset(token)


def managed_graph(compiled, name=None):
    """Give each invocation a fresh budget; retain it within its autonomous loop."""
    def invoke(value, config):
        with run_session() as session:
            config = {**(config or {}), "recursion_limit": session.recursion_limit}
            if name is not None:
                session.begin(name)
                value = {**state_values(value), "rounds_used": 0,
                         "limit_reached": False, "runtime_status": "pending"}
            while True:
                result = compiled.invoke(value, config=config)
                if name is None or result.get("runtime_status") != "human_replied":
                    return result
                # Continue with exactly the returned state, without re-importing files.
                # A fresh graph execution also renews LangGraph's step guard after
                # real human input, so long conversations do not hit a lifetime cap.
                value = result
    return RunnableLambda(invoke)


def announce(text):
    """Print runtime notices without adding them to the model's message history."""
    print(text)


def announce_limit(session, name):
    announce(f"⚠️ {name}: Round limit reached ({session.used.get(name, 0)}/{session.limits[name]}). "
             "Continuing the workflow.")


def build_agent_graph(namespace, name, state_type, complete, outputs, coordinator=False):
    """Use explicit check/finish/exhausted nodes for all interactive agent loops.

    The last agent response may execute its tool before the budget is checked.
    Both normal completion and exhaustion save current products to configured paths.
    """
    graph = StateGraph(state_type)

    def update_state(state, status="running"):
        session = current_session()
        state.rounds_used = session.used.get(name, 0)
        state.runtime_status = status
        session.update(name, state, status)
        return state

    def call(function):
        def node(state):
            # LangGraph reconstructs dataclasses between nodes; bind tools to this state.
            if hasattr(type(state), "_current_instance"):
                type(state)._current_instance = state
            result = function(state)
            return update_state(result)
        return node

    def agent_node(state):
        state.rounds_used = current_session().reserve(name)
        return call(namespace["agent_node"])(state)

    def human_chat(state):
        history = getattr(state, "messages", getattr(state, "msg", []))
        previous_length = len(history)
        result = namespace["user_chat_node"](state)
        history = getattr(result, "messages", getattr(result, "msg", []))
        if any(message.startswith("user: ") for message in history[previous_length:]):
            current_session().human_replied(name)
            result.limit_reached = False
            if coordinator:
                result = namespace["chatEnd_node"](result)
            announce(f"✅ {name}: User reply received; autonomous round counter reset to 0.")
            return update_state(result, "human_replied")
        # A node that returns without receiving input must not renew the budget.
        return update_state(result)

    def finish(state, exhausted=False):
        session = current_session()
        state.limit_reached = exhausted
        for path, content in outputs(state).items():
            atomic_text(path, content)
        if exhausted:
            session.exhausted.add(name)
            announce_limit(session, name)
        status = "exhausted" if exhausted else "completed"
        return update_state(state, status)

    def blocked(state):
        announce(f"⚠️ {name}: A child agent did not return usable results. Continuing the workflow.")
        return update_state(state, "blocked")

    def completed(state):
        if not complete(state):
            return False
        if coordinator:
            return all(current_session().ready(child) for child in
                       ("structuredProposalAgent", "structuredCriticAgent"))
        return True

    def route_check(state):
        if completed(state):
            return "finish"
        return "agent" if current_session().available(name) else "exhausted"

    def route_agent(state):
        history = getattr(state, "messages", getattr(state, "msg", []))
        if history and "[tool_call]" in history[-1]:
            return "tool"
        if completed(state):
            return "finish"
        if "rest_node" not in namespace:
            return "interact"
        return "interact" if current_session().available(name) else "exhausted"

    def route_tool(state):
        if completed(state):
            return "finish"
        if coordinator and state.isCallLoop:
            return "proposal"
        if coordinator and state.isConflict:
            return "user_chat"
        if not current_session().available(name):
            return "exhausted"
        return "check"

    graph.add_node("check", lambda state: state)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", call(namespace["tool_node"]))
    graph.add_node("interact", call(namespace["rest_node"]) if "rest_node" in namespace else human_chat)
    graph.add_node("finish", lambda state: finish(state))
    graph.add_node("exhausted", lambda state: finish(state, exhausted=True))
    if "import_node" in namespace:
        graph.add_node("import", call(namespace["import_node"]))
        graph.set_conditional_entry_point(
            lambda state: "check" if state.runtime_status == "human_replied" else "import",
            ["check", "import"],
        )
        graph.add_edge("import", "check")
    else:
        graph.set_entry_point("check")
    graph.add_conditional_edges("check", route_check, ["agent", "finish", "exhausted"])
    graph.add_conditional_edges("agent", route_agent, ["tool", "interact", "finish", "exhausted"])
    graph.add_conditional_edges("tool", route_tool,
                                ["check", "finish", "exhausted"] + (["proposal", "user_chat"] if coordinator else []))
    if "rest_node" in namespace:
        graph.add_edge("interact", "check")
    else:
        graph.add_conditional_edges("interact", lambda state: END if state.runtime_status == "human_replied" else "check", [END, "check"])
    for end in ("finish", "exhausted"):
        graph.add_edge(end, END)
    if coordinator:
        def child_node(key, function):
            def node(state):
                function.invoke({})
                announce(f"ℹ️ {key}: {current_session().statuses[key]}.")
                return update_state(state)
            return node
        graph.add_node("proposal", child_node("structuredProposalAgent", namespace["structuredProposalAgentGraph"]))
        graph.add_node("critic", child_node("structuredCriticAgent", namespace["structuredCriticAgentGraph"]))
        graph.add_node("loop_end", call(namespace["loopEnd_node"]))
        graph.add_node("user_chat", human_chat)
        graph.add_node("blocked", blocked)
        graph.add_conditional_edges("proposal", lambda state: "critic" if current_session().ready("structuredProposalAgent") else "blocked", ["critic", "blocked"])
        graph.add_conditional_edges("critic", lambda state: "loop_end" if current_session().ready("structuredCriticAgent") else "blocked", ["loop_end", "blocked"])
        graph.add_edge("loop_end", "check")
        graph.add_conditional_edges("user_chat", lambda state: END if state.runtime_status == "human_replied" else "check", [END, "check"])
        graph.add_edge("blocked", END)
    return managed_graph(graph.compile(), name)




def managed_main_graph(compiled):
    def invoke(value, config):
        with run_session() as session:
            result = compiled.invoke(value, config={**(config or {}), "recursion_limit": session.recursion_limit})
            announce("✅ Workflow finished.")
            return result
    return RunnableLambda(invoke)


def stage_node(name, graph):
    """Run each stage independently; record failures without blocking later stages."""
    def node(state, config):
        session = current_session()
        try:
            result = graph.invoke({}, config=config)
            if name not in session.statuses:
                session.update(name, result, result.get("construction_status", "completed"))
        except Exception as error:
            announce(f"❌ {name}: Failed ({type(error).__name__}: {error}). Continuing the workflow.")
            previous = session.results.get(name, {})
            session.update(name, previous, "failed")
        statuses = dict(state.stages)
        statuses[name] = session.statuses[name]
        return {"stages": statuses}
    return node

