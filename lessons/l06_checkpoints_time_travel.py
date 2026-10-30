"""Lesson 6: checkpointers, threads and time travel.

WHAT YOU LEARN
- A CHECKPOINTER saves the full state after every step (a "superstep").
  `compile(checkpointer=...)`. InMemorySaver for tests, SqliteSaver for one machine,
  PostgresSaver for production.
- A THREAD is one conversation/run history, picked by
  `config={"configurable": {"thread_id": ...}}`. Same thread_id = same history.
  Here the thread is one incident.
- `get_state(config)` = the latest snapshot (values, next nodes, checkpoint id).
  `get_state_history(config)` = every snapshot, newest first.
- TIME TRAVEL: take an old checkpoint's config, `update_state(...)` to change what a
  node produced (this FORKS a new branch), then `invoke(None, new_config)` re-runs
  only the steps after it. The original branch is untouched.
- DURABILITY: state survives a restart. A new process, a new app object, the same
  SQLite file and thread_id, and the run is still there.
- Checkpointed state must be serialisable, and newer LangGraph refuses to load classes
  it wasn't told about. Two ways to handle that: keep state JSON-native (this lesson
  stores `triage` as `model_dump()`, a plain dict), or allowlist your types
  (triage_lab/checkpoints.py does that for the shared ones like Failure). Dependencies
  never go in state at all; they live in runtime context (lesson 5).

WHAT TO WATCH
The history lists each step. Then we pretend the operator corrects the triage
(the model said "notebook", the operator says "lakehouse"), fork from that checkpoint,
and only the `plan` node runs again.

INTERVIEW LINE
"The checkpointer gives me persistence per thread: resume after a crash, human-in-the-
loop pauses, and time travel to replay from any step with corrected state. thread_id
is the unit of memory; I key it by incident."

RUN
    python -m lessons.l06_checkpoints_time_travel
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from triage_lab.checkpoints import sqlite_checkpointer
from triage_lab.contracts import Failure
from triage_lab.fabric import get_fabric, pick_failure
from triage_lab.models import get_model
from triage_lab.settings import REPO_ROOT, get_settings
from triage_lab.team import Context
from triage_lab.texts import failure_text

DB = REPO_ROOT / "data" / "lesson06.sqlite"


class Triage(BaseModel):
    category: Literal["missing_table", "credentials", "throttling", "other"]
    specialist: Literal["notebook", "pipeline", "lakehouse", "semantic_model", "general"]
    summary: str


class State(TypedDict, total=False):
    failure: Failure
    triage: dict  # Triage.model_dump(): plain data checkpoints well
    plan: str


def triage(state: State, runtime: Runtime[Context]) -> dict:
    structured = runtime.context.model.with_structured_output(Triage)
    result = structured.invoke([
        SystemMessage("Classify this Fabric failure and pick the first specialist."),
        HumanMessage(failure_text(state["failure"])),
    ])
    return {"triage": result.model_dump()}


def plan(state: State) -> dict:
    t = state["triage"]
    return {"plan": f"Assign {state['failure'].item.name} to the {t['specialist']} specialist ({t['category']})."}


def build_graph(checkpointer=None):
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("triage", triage)
    builder.add_node("plan", plan)
    builder.add_edge(START, "triage")
    builder.add_edge("triage", "plan")
    builder.add_edge("plan", END)
    return builder.compile(checkpointer=checkpointer or sqlite_checkpointer(DB))


def main() -> dict:
    settings = get_settings()
    fabric = get_fabric(settings)
    context = Context(fabric=fabric, model=get_model(settings), settings=settings)
    failure = pick_failure("Notebook", fabric, settings)

    checkpointer = sqlite_checkpointer(DB)
    thread = {"configurable": {"thread_id": f"lesson06:{failure.key}"}}
    checkpointer.delete_thread(thread["configurable"]["thread_id"])  # start clean each run
    app = build_graph(checkpointer)

    first = app.invoke({"failure": failure}, thread, context=context)
    print("1. First run:", first["plan"])

    print("\n2. History, newest first:")
    history = list(app.get_state_history(thread))
    for snap in history:
        print(f"   step {snap.metadata.get('step'):>2}  next={snap.next or '(done)'}  "
              f"keys={sorted(snap.values)}")

    # Time travel: the checkpoint taken right after `triage`, i.e. the one whose next node is `plan`.
    after_triage = next(s for s in history if s.next == ("plan",))
    corrected = {**after_triage.values["triage"], "specialist": "lakehouse"}
    forked = app.update_state(after_triage.config, {"triage": corrected}, as_node="triage")
    second = app.invoke(None, forked, context=context)  # None = continue from this checkpoint
    print("\n3. After the operator's correction, replayed from the fork:", second["plan"])

    # Durability: a brand-new app object reading the same file sees the latest state.
    reopened = build_graph(sqlite_checkpointer(DB))
    print("\n4. A fresh process would see:", reopened.get_state(thread).values["plan"])
    print("   Checkpoints on this thread now:", len(list(reopened.get_state_history(thread))))
    return second


if __name__ == "__main__":
    main()
