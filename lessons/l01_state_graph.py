"""Lesson 1: a graph is a state machine.

WHAT YOU LEARN
- STATE is a TypedDict: the shared notebook every node reads and writes.
- A NODE is a plain function: state in, a dict of ONLY the keys it changes out.
- EDGES are fixed arrows between nodes. START and END are built-in.
- compile() checks the wiring and gives you a Runnable app: invoke, stream, batch.
- stream(stream_mode="values") shows the whole state after every step.

No model and no Fabric yet. Just the mechanics, on one Fabric error string.

WHAT TO WATCH WHEN YOU RUN IT
The printed state grows step by step. `parse` adds error_class and table; `summarise`
adds summary. Neither node returns the keys it did not touch.

INTERVIEW LINE
"In LangGraph a node is a function from state to a partial update. The graph merges
each update into shared state and follows the edges. compile() validates the wiring
and returns a Runnable, so invoke, stream and batch all work."

TRY THIS
1. Make `parse` return {**state, ...}. It still works, but now the node "owns" every
   key, which bites you once two nodes run in parallel (lesson 2).
2. Delete the edge into END and run. Read the error message.

RUN
    python -m lessons.l01_state_graph
"""

from __future__ import annotations

import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

SAMPLE_ERROR = (
    "NotebookExecutionFailed | [TABLE_OR_VIEW_NOT_FOUND] The table or view "
    "`lh_sales`.`sales_raw` cannot be found."
)


class State(TypedDict, total=False):
    error: str          # input
    error_class: str    # set by parse
    table: str          # set by parse
    summary: str        # set by summarise


def parse(state: State) -> dict:
    """Pull the error class and the table name out of the raw error."""
    error_class = re.search(r"\[([A-Z_]+)\]", state["error"])
    table = re.search(r"`(\w+)`\.`(\w+)`", state["error"])
    return {
        "error_class": error_class.group(1) if error_class else "UNKNOWN",
        "table": f"{table.group(1)}.{table.group(2)}" if table else "",
    }


def summarise(state: State) -> dict:
    if state["error_class"] == "TABLE_OR_VIEW_NOT_FOUND":
        text = f"Code reads `{state['table']}`, which does not exist in the lakehouse."
    else:
        text = f"Unrecognised error class {state['error_class']}."
    return {"summary": text}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("parse", parse)
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "parse")
    builder.add_edge("parse", "summarise")
    builder.add_edge("summarise", END)
    return builder.compile()


def main() -> dict:
    app = build_graph()
    print("The graph, as Mermaid (paste into https://mermaid.live):\n")
    print(app.get_graph().draw_mermaid())

    print("State after each step:")
    for step, state in enumerate(app.stream({"error": SAMPLE_ERROR}, stream_mode="values")):
        print(f"  step {step}: {state}")

    final = app.invoke({"error": SAMPLE_ERROR})
    print("\nFinal summary:", final["summary"])
    return final


if __name__ == "__main__":
    main()
