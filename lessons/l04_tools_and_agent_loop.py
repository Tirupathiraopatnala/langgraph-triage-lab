"""Lesson 4: tools and the agent loop (ReAct).

WHAT YOU LEARN
- A TOOL is a function with a docstring and typed arguments. `@tool` turns it into a
  schema the model can call. The docstring IS the tool description the model reads.
- `model.bind_tools(TOOLS)` lets the model answer with tool calls instead of text.
- `ToolNode` runs whatever tool calls the last AI message asked for and appends
  ToolMessages. `tools_condition` routes to "tools" if there are calls, else to END.
- `MessagesState` is a ready-made state: a `messages` list with the `add_messages`
  reducer (appends, and replaces a message if the id matches).
- `recursion_limit` caps the number of steps, so a confused model can't loop forever.

WHAT TO WATCH
The model asks for items, then the failing notebook's code, then the lakehouse tables,
then answers. Every step is a message you can read.

THE SMELL (fixed in lesson 5)
The tools reach the Fabric client and workspace through module GLOBALS. That makes them
hard to test, impossible to run for two workspaces at once, and tempting to stuff into
graph state instead. Hold that thought.

INTERVIEW LINE
"The agent loop is two nodes and a conditional edge: the model proposes tool calls,
a ToolNode executes them, and we loop until the model stops calling tools or we hit
the recursion limit. Tools are read-only here; anything that writes goes behind
human approval."

(Prebuilt versions of this loop exist: `create_agent` in the `langchain` package, which
replaced the now-deprecated `langgraph.prebuilt.create_react_agent`. Building it by hand
once is how you learn what they do.)

TRY THIS
1. Set recursion_limit to 3 and read the GraphRecursionError.
2. Rename a tool's docstring to something vague and watch the model use it worse
   (with MODEL_MODE=azure).

RUN
    python -m lessons.l04_tools_and_agent_loop
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from triage_lab.fabric import get_fabric, pick_failure
from triage_lab.models import get_model
from triage_lab.texts import definition_text, failure_text, items_text, tables_text

# --- globals: the smell lesson 5 removes ---------------------------------------
FABRIC = get_fabric()
WORKSPACE_ID = ""  # set in main() before the graph runs
MODEL = get_model()


@tool
def list_workspace_items() -> str:
    """List every item in the failing workspace: id | type | name."""
    return items_text(FABRIC.list_items(WORKSPACE_ID))


@tool
def read_item_definition(item_id: str) -> str:
    """Read the full definition of one item (notebook code, pipeline JSON...) by item id."""
    return definition_text(FABRIC.read_definition(WORKSPACE_ID, item_id))


@tool
def list_lakehouse_tables(lakehouse_id: str) -> str:
    """List the tables that exist in one lakehouse, by lakehouse item id."""
    return tables_text(lakehouse_id, FABRIC.list_lakehouse_tables(WORKSPACE_ID, lakehouse_id))


TOOLS = [list_workspace_items, read_item_definition, list_lakehouse_tables]

SYSTEM = (
    "You are the notebook specialist on a Microsoft Fabric support team. Use your read-only "
    "tools to find the root cause of the failure, then answer in two or three sentences with "
    "the evidence. Stop calling tools once you know."
)


def agent(state: MessagesState) -> dict:
    reply = MODEL.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [reply]}


def build_graph():
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")
    return builder.compile()


def main() -> dict:
    global WORKSPACE_ID
    failure = pick_failure("Notebook", FABRIC)
    WORKSPACE_ID = failure.workspace_id

    app = build_graph()
    start = {"messages": [SystemMessage(SYSTEM), HumanMessage(failure_text(failure))]}
    final = app.invoke(start, {"recursion_limit": 12})

    for message in final["messages"][1:]:
        kind = type(message).__name__.replace("Message", "")
        if getattr(message, "tool_calls", None):
            for call in message.tool_calls:
                print(f"[{kind}] calls {call['name']}({call['args']})")
        else:
            text = str(message.content)
            print(f"[{kind}] {text[:300]}{'...' if len(text) > 300 else ''}")
    return final


if __name__ == "__main__":
    main()
