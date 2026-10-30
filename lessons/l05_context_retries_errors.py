"""Lesson 5: runtime context, retry policies and error handlers.

WHAT YOU LEARN
- RUNTIME CONTEXT: `StateGraph(State, context_schema=Context)` and
  `app.invoke(..., context=Context(...))`. Nodes take `runtime: Runtime[Context]`;
  tools take `runtime: ToolRuntime[Context, ...]`. Dependencies (Fabric client, model,
  settings) travel here, NOT in state. State is data and gets checkpointed; a client
  object is not data. This is the fix for lesson 4's globals, and for putting live
  objects in state (which blocks you from adding a checkpointer).
- RETRY POLICY per node: `add_node(..., retry_policy=RetryPolicy(max_attempts=4,
  retry_on=is_transient))`. Only transient errors are retried (503, timeouts), never
  401/403/404.
- ERROR HANDLER per node: `add_node(..., error_handler=fn)`. After retries are
  exhausted, `fn(state, error: NodeError)` runs and returns a Command to recover.
- ONE RETRY LAYER PER KIND OF FAILURE: the Fabric client honours 429 Retry-After itself,
  the OpenAI SDK retries model calls, and the graph retries transient node failures.
  Never stack two layers on the same error, or 4 retries become 16.

GOTCHA
`add_node(..., timeout=30)` exists, but only for async nodes: a sync Python function
can't be cancelled safely. Use client timeouts (httpx, the OpenAI SDK) for sync code.

WHAT TO WATCH
Three runs: a clean one, one where Fabric fails twice and the retry policy recovers,
and one where Fabric keeps failing and the error handler takes over so the agent still
answers (with less to go on).

INTERVIEW LINE
"Dependencies go in runtime context, data goes in state. Retries are per node and only
on transient errors, with an error handler for when they run out, and I keep exactly one
retry layer per failure type."

RUN
    python -m lessons.l05_context_retries_errors
"""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.errors import NodeError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy

from triage_lab.contracts import Failure
from triage_lab.fabric import FabricError, FakeFabric, get_fabric, pick_failure, recent_failures
from triage_lab.models import get_model
from triage_lab.settings import get_settings
from triage_lab.team import Context
from triage_lab.texts import definition_text, failure_text, items_text, tables_text


class State(MessagesState, total=False):
    failure: Failure
    notes: Annotated[list[str], operator.add]


# --- tools: dependencies come from runtime.context, the workspace from state ---


@tool
def list_workspace_items(runtime: ToolRuntime[Context, dict]) -> str:
    """List every item in the failing workspace: id | type | name."""
    return items_text(runtime.context.fabric.list_items(runtime.state["failure"].workspace_id))


@tool
def read_item_definition(item_id: str, runtime: ToolRuntime[Context, dict]) -> str:
    """Read the full definition of one item by item id."""
    ws = runtime.state["failure"].workspace_id
    return definition_text(runtime.context.fabric.read_definition(ws, item_id))


@tool
def list_lakehouse_tables(lakehouse_id: str, runtime: ToolRuntime[Context, dict]) -> str:
    """List the tables that exist in one lakehouse, by lakehouse item id."""
    ws = runtime.state["failure"].workspace_id
    return tables_text(lakehouse_id, runtime.context.fabric.list_lakehouse_tables(ws, lakehouse_id))


TOOLS = [list_workspace_items, read_item_definition, list_lakehouse_tables]


# --- nodes ----------------------------------------------------------------------


def is_transient(exc: Exception) -> bool:
    """Retry only what could succeed next time."""
    return isinstance(exc, FabricError) and exc.transient


def preload(state: State, runtime: Runtime[Context]) -> dict:
    """Read the failing item's definition up front, so the agent starts informed.

    Fabric reads can fail transiently. The retry policy on this node handles that.
    """
    failure = state["failure"]
    definition = runtime.context.fabric.read_definition(failure.workspace_id, failure.item.id)
    return {
        "messages": [HumanMessage("Here is the failing item's definition:\n\n" + definition_text(definition))],
        "notes": ["preload: definition read"],
    }


def preload_failed(state: State, error: NodeError) -> Command:
    """Runs only after the retries ran out. Recover instead of crashing the run."""
    return Command(
        update={"notes": [f"preload gave up: {error.error}. The agent will read what it needs itself."]},
        goto="agent",
    )


SYSTEM = (
    "You are the notebook specialist on a Microsoft Fabric support team. Use your read-only "
    "tools to find the root cause, then answer in two or three sentences with evidence."
)


def agent(state: State, runtime: Runtime[Context]) -> dict:
    reply = runtime.context.model.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [reply]}


def build_graph():
    builder = StateGraph(State, context_schema=Context)
    builder.add_node(
        "preload",
        preload,
        retry_policy=RetryPolicy(max_attempts=4, initial_interval=0.2, backoff_factor=2.0, retry_on=is_transient),
        error_handler=preload_failed,
    )
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "preload")
    builder.add_edge("preload", "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile()


def run(app, failure: Failure, fabric, label: str) -> dict:
    settings = get_settings()
    context = Context(fabric=fabric, model=get_model(settings), settings=settings)
    start = {
        "failure": failure,
        "messages": [SystemMessage(SYSTEM), HumanMessage(failure_text(failure))],
    }
    final = app.invoke(start, {"recursion_limit": 15}, context=context)
    print(f"\n--- {label}")
    for note in final.get("notes", []):
        print("   note:", note)
    print("   answer:", str(final["messages"][-1].content)[:200])
    return final


def main() -> list[dict]:
    app = build_graph()
    fabric = get_fabric()
    failure = pick_failure("Notebook", fabric)

    results = [run(app, failure, fabric, "clean run")]

    # The next two runs use the fake workspace with injected 503s, so you can watch
    # retries happen without breaking a real workspace.
    fake_failure = next(f for f in recent_failures(FakeFabric()) if f.item.type == "Notebook")
    flaky = FakeFabric(flaky_reads=2)
    results.append(run(app, fake_failure, flaky, "fake Fabric fails twice, retry policy recovers"))
    print("   Fabric calls made:", sum(c.startswith("read_definition") for c in flaky.calls), "reads")

    broken = FakeFabric(flaky_reads=4)  # more failures than max_attempts
    results.append(run(app, fake_failure, broken, "fake Fabric keeps failing, error handler recovers"))
    return results


if __name__ == "__main__":
    main()
