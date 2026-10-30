"""Lesson 8: Send for map-reduce, and subgraphs.

WHAT YOU LEARN
- SEND: a conditional edge can return a LIST of `Send(node, payload)`. LangGraph runs
  that node once per Send, in parallel, each with its own payload. This is map-reduce:
  one investigation per failure, however many failures there are.
- FAN-IN: the node after them runs once, when all branches are done. A reducer on the
  shared key collects every branch's result.
- SUBGRAPH: a compiled graph can be a node of another graph. Here each investigation is
  `build_investigator()` from triage_lab/team.py: a full tool loop with its own state
  (messages, definitions read, report). Its private keys stay private; only keys the
  parent ALSO declares (here `entry`) flow back up.
- The subgraph shares the parent's runtime context automatically.
- `get_graph(xray=True)` draws subgraphs inside the parent.

ALSO IN team.py (read it now)
- Tools that take `runtime: ToolRuntime[...]` get context and state injected.
- `read_item_definition` returns a `Command(update=...)`: a tool updating graph state,
  so the definitions a specialist read are in state (and checkpointed) instead of in a
  hidden cache. The patch check reads them from there.
- A tool-call budget enforced in a conditional edge, and patches accepted by code checks,
  not by the model.

OTHER WAY TO CALL A SUBGRAPH
If parent and child share no keys, call it inside a normal node:
`result = INVESTIGATOR.invoke({...})` and return what you need. You lose the xray
drawing, but you get full control of the mapping.

INTERVIEW LINE
"For N failures I fan out with Send: one subgraph run per failure in parallel, results
merged by a reducer. Each specialist is a subgraph with its own private state, so their
message histories never mix."

RUN
    python -m lessons.l08_send_and_subgraphs
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from triage_lab.contracts import CaseEntry, Failure
from triage_lab.fabric import get_fabric, recent_failures
from triage_lab.models import get_model
from triage_lab.settings import get_settings
from triage_lab.team import Context, build_investigator

FIRST_SPECIALIST = {"Notebook": "notebook", "DataPipeline": "pipeline", "SemanticModel": "semantic_model"}


def collect(left: list | None, right: CaseEntry | list | None) -> list:
    """Reducer: each investigator writes ONE CaseEntry; keep them all."""
    items = right if isinstance(right, list) else [right]
    return (left or []) + [i for i in items if i is not None]


class State(TypedDict, total=False):
    failures: list[Failure]
    entry: Annotated[list[CaseEntry], collect]  # the only key shared with the subgraph
    summary: str


def load(state: State, runtime: Runtime[Context]) -> dict:
    return {"failures": recent_failures(runtime.context.fabric, runtime.context.settings)}


def fan_out(state: State) -> list[Send]:
    """One Send per failure. Each payload becomes the subgraph's input state."""
    return [
        Send("investigate", {
            "failure": f,
            "specialist": FIRST_SPECIALIST.get(f.item.type, "general"),
            "briefing": f"Find out why {f.item.name} failed.",
        })
        for f in state["failures"]
    ]


def summarise(state: State) -> dict:
    lines = [f"- {e.specialist}: {e.outcome}. {e.root_cause}" for e in state["entry"]]
    return {"summary": "\n".join(lines)}


def build_graph():
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("load", load)
    builder.add_node("investigate", build_investigator())  # a compiled graph as a node
    builder.add_node("summarise", summarise)
    builder.add_edge(START, "load")
    builder.add_conditional_edges("load", fan_out, ["investigate"])
    builder.add_edge("investigate", "summarise")
    builder.add_edge("summarise", END)
    return builder.compile()


def main() -> dict:
    settings = get_settings()
    context = Context(fabric=get_fabric(settings), model=get_model(settings), settings=settings)
    app = build_graph()
    print(app.get_graph(xray=True).draw_mermaid())
    final = app.invoke({}, context=context)
    print(f"Investigated {len(final['entry'])} failures in parallel:\n{final['summary']}")
    return final


if __name__ == "__main__":
    main()
