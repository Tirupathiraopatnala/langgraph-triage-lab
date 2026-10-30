"""Lesson 2: reducers, parallel branches and conditional edges.

WHAT YOU LEARN
- PARALLEL BRANCHES: two edges out of START run two nodes in the same step.
  `add_edge([a, b], c)` makes c wait for BOTH (a join).
- REDUCERS: when two nodes write the same key in one step, LangGraph needs to know how
  to combine them. `Annotated[list[str], operator.add]` appends; a custom function can
  do anything (here, merge dicts). Without a reducer you get InvalidUpdateError.
- CONDITIONAL EDGES: a routing function reads the state and returns the name of the
  next node. The list you pass to add_conditional_edges is what makes the drawing right.
- GOTCHA: LangGraph reads each node's type hint (`state: State`) and registers that
  schema. Reusing typed nodes in a graph with a different state class fails with
  "Channel already exists with a different type".

WHAT TO WATCH
Each of the three Fabric failures takes a different route. The "notes" list has
entries from both parallel nodes. Then the demo at the end shows the error you get
when the reducer is missing.

INTERVIEW LINE
"Reducers decide how concurrent updates merge. It's what makes fan-out safe: two
branches can both append to a list in the same superstep. Routing is a plain function
on state, so it's easy to unit test."

TRY THIS
1. Change `operator.add` to a reducer that keeps only unique notes.
2. Add a fourth route for throttling errors (429 / capacity) to the general handler.

RUN
    python -m lessons.l02_reducers_and_routing
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

from triage_lab.contracts import Failure
from triage_lab.fabric import recent_failures


def merge_facts(left: dict | None, right: dict | None) -> dict:
    """A custom reducer: keep every key from both sides."""
    return {**(left or {}), **(right or {})}


class State(TypedDict, total=False):
    failure: Failure
    notes: Annotated[list[str], operator.add]
    facts: Annotated[dict[str, str], merge_facts]
    action: str


# --- two checks that run in parallel -----------------------------------------


def check_history(state: State) -> dict:
    runs = state["failure"].runs
    repeated = "repeating" if len(runs) > 1 else "one-off"
    return {"notes": [f"history: {len(runs)} failed run(s), {repeated}"], "facts": {"pattern": repeated}}


def check_error(state: State) -> dict:
    error = state["failure"].error_text.lower()
    if "table_or_view_not_found" in error:
        kind = "missing_table"
    elif "credential" in error:
        kind = "credentials"
    else:
        kind = "other"
    return {"notes": [f"error: looks like {kind}"], "facts": {"kind": kind}}


# --- the router and the three handlers ----------------------------------------


def choose_handler(state: State) -> str:
    """Conditional edge. Reads state, returns the next node's name."""
    kind = state["facts"]["kind"]
    if kind == "missing_table":
        return "fix_code"
    if kind == "credentials":
        return "fix_credentials"
    return "escalate"


def fix_code(state: State) -> dict:
    return {"action": f"Hand {state['failure'].item.name} to the notebook/pipeline specialist for a code fix."}


def fix_credentials(state: State) -> dict:
    return {"action": f"Ask the owner of {state['failure'].item.name} to update data source credentials."}


def escalate(state: State) -> dict:
    return {"action": "Escalate to a human: no rule matched."}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("check_history", check_history)
    builder.add_node("check_error", check_error)
    builder.add_node("route", lambda state: {})  # a join point; does no work itself
    builder.add_node("fix_code", fix_code)
    builder.add_node("fix_credentials", fix_credentials)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "check_history")  # two edges out of START:
    builder.add_edge(START, "check_error")    # both run in the same step
    builder.add_edge(["check_history", "check_error"], "route")  # wait for both
    builder.add_conditional_edges("route", choose_handler, ["fix_code", "fix_credentials", "escalate"])
    for handler in ("fix_code", "fix_credentials", "escalate"):
        builder.add_edge(handler, END)
    return builder.compile()


class NoReducer(TypedDict, total=False):
    notes: list[str]  # no reducer: two parallel writers in one step will clash


def demo_missing_reducer() -> str:
    """Two parallel nodes write `notes` in the same step, with nothing to merge them."""
    builder = StateGraph(NoReducer)
    builder.add_node("a", lambda state: {"notes": ["from a"]})
    builder.add_node("b", lambda state: {"notes": ["from b"]})
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge(["a", "b"], END)
    try:
        builder.compile().invoke({})
    except InvalidUpdateError as exc:
        return str(exc).splitlines()[0]
    return "no error (unexpected)"


def main() -> list[dict]:
    app = build_graph()
    results = []
    for failure in recent_failures():
        result = app.invoke({"failure": failure})
        results.append(result)
        print(f"\n{failure.item.name} ({failure.item.type})")
        for note in result["notes"]:
            print("   ", note)
        print("    ->", result["action"])

    print("\nTwo parallel writers without a reducer:")
    print("    InvalidUpdateError:", demo_missing_reducer())
    return results


if __name__ == "__main__":
    main()
