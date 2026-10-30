"""Lesson 9: a supervisor and specialists, wired with Command.

READ triage_lab/supervisor.py ALONGSIDE THIS. The code lives there because lesson 12
reuses it; this file only builds and runs it.

WHAT YOU LEARN
- COMMAND: a node returns `Command(goto="notebook", update={...})` to choose where to
  go AND update state in one value. No conditional edge needed.
- `destinations=(...)` on add_node declares where a node may go, so the graph can be
  drawn and checked even though there are no fixed edges.
- MULTI-AGENT HANDOFFS: the pipeline specialist answers "not_mine, hand to notebook";
  the supervisor reroutes; the notebook specialist fixes it; the supervisor accepts.
  Everyone reads and writes one shared CASE FILE (a list with a reducer).
- MODEL VS CODE: the model judges (route, accept, reroute, the narrative); code enforces
  the rules (roster, hop budget, no self-handoff, not_mine can't close a case).
- Each specialist is the lesson 8 investigator subgraph, called inside a node.

WHAT TO WATCH
The pipeline failure goes to the pipeline specialist first. It hands off to the
notebook specialist, who patches the notebook. Two hops, then a conclusion.

INTERVIEW LINE
"It's a supervisor pattern: nodes route with Command, specialists share a case file,
and every loop is bounded by a hop budget checked in code, not by the model. The model
makes judgements; code enforces invariants."

TRY THIS
1. Set SUPERVISOR_MAX_HOPS=1 in .env and watch the case stop with a reason.
2. Run it on the semantic model failure: one hop, manual steps, no patch.

RUN
    python -m lessons.l09_supervisor_command
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from triage_lab.fabric import get_fabric, pick_failure
from triage_lab.models import get_model
from triage_lab.settings import get_settings
from triage_lab.supervisor import TeamState, add_team
from triage_lab.team import Context
from triage_lab.texts import case_file_text


def build_graph():
    builder = StateGraph(TeamState, context_schema=Context)
    add_team(builder)
    builder.add_edge(START, "supervisor")
    builder.add_edge("conclude", END)
    return builder.compile()


def main(item_type: str = "DataPipeline") -> dict:
    settings = get_settings()
    fabric = get_fabric(settings)
    context = Context(fabric=fabric, model=get_model(settings), settings=settings)
    failure = pick_failure(item_type, fabric, settings)

    app = build_graph()
    print("Route taken (node, then the keys it updated):")
    final: dict = {}
    # Two stream modes at once: "updates" to watch the route, "values" to keep the final state.
    for mode, chunk in app.stream({"failure": failure}, context=context, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, change in chunk.items():
                print(f"  {node:<15} {', '.join(sorted(change or {}))}")
        else:
            final = chunk

    print("\nCase file:\n" + case_file_text(final["case_file"]))
    c = final["conclusion"]
    print(f"\nConclusion after {final['hops']} hop(s): {c.root_cause}")
    for patch in final.get("patches", []):
        print(f"  patch: {patch.item_name}/{patch.part_path}: {patch.explanation}")
    for step in c.manual_steps:
        print("  manual step:", step)
    return final


if __name__ == "__main__":
    main()
