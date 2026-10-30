"""Lesson 10: long-term memory with the Store.

WHAT YOU LEARN
- CHECKPOINTER vs STORE. The checkpointer is short-term memory: the state of ONE thread
  (one incident's run). The Store is long-term memory shared ACROSS threads: facts that
  should outlive any single run. Here: "last time a notebook failed with
  TABLE_OR_VIEW_NOT_FOUND in this workspace, this was the cause and this was the fix".
- `compile(store=...)`, then `runtime.store.put(namespace, key, value)`, `.get(...)`,
  `.search(namespace_prefix)` inside nodes. Namespaces are tuples, like folders:
  ("fixes", workspace_id).
- Backends: InMemoryStore (tests), SqliteStore (one machine, used here, so memory
  survives between runs), PostgresStore (production).
- SEMANTIC SEARCH: give the store an embeddings index
  (`InMemoryStore(index={"embed": ..., "dims": 1536})`) and `search(..., query="...")`
  finds similar errors, not only identical ones. We use an exact error signature here
  to keep it offline.
- Memory is a HINT, not an answer. A remembered fix goes into the briefing; the
  specialist still verifies against today's code.

WHAT TO WATCH
First run: no memory, a full investigation, then the fix is remembered. Second run of the
same failure: memory is recalled and handed to the specialist as a lead.

INTERVIEW LINE
"Checkpointer is per-thread state; Store is cross-thread memory. I namespace memories
by workspace and key them by an error signature, and I treat recalled fixes as leads
the agent must re-verify, not as answers."

RUN
    python -m lessons.l10_store_memory
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.store.sqlite import SqliteStore

from triage_lab.contracts import CaseEntry, Failure
from triage_lab.fabric import get_fabric, pick_failure
from triage_lab.models import get_model
from triage_lab.settings import REPO_ROOT, get_settings
from triage_lab.team import Context, build_investigator

STORE_DB = REPO_ROOT / "data" / "lesson10_store.sqlite"
INVESTIGATOR = build_investigator()


class State(TypedDict, total=False):
    failure: Failure
    recalled: dict | None
    entry: CaseEntry


def namespace(failure: Failure) -> tuple[str, ...]:
    return ("fixes", failure.workspace_id)


def recall(state: State, runtime: Runtime[Context]) -> dict:
    memory = runtime.store.get(namespace(state["failure"]), state["failure"].signature)
    return {"recalled": memory.value if memory else None}


def investigate(state: State, runtime: Runtime[Context]) -> dict:
    briefing = f"Find out why {state['failure'].item.name} failed."
    if state.get("recalled"):
        r = state["recalled"]
        briefing += (
            f"\nLead from memory (seen {r['times_seen']} time(s), last {r['last_seen'][:10]}): "
            f"{r['root_cause']} Fix then: {r['fix']}. Verify it still applies."
        )
    result = INVESTIGATOR.invoke(
        {"failure": state["failure"], "specialist": "notebook", "briefing": briefing},
        context=runtime.context,
    )
    return {"entry": result["entry"]}


def remember(state: State, runtime: Runtime[Context]) -> dict:
    entry = state["entry"]
    if entry.outcome != "resolved":
        return {}  # only remember what actually closed a case
    previous = state.get("recalled") or {}
    runtime.store.put(
        namespace(state["failure"]),
        state["failure"].signature,
        {
            "root_cause": entry.root_cause,
            "fix": ", ".join(entry.patched) or " | ".join(entry.manual_steps) or "none recorded",
            "item": state["failure"].item.name,
            "times_seen": previous.get("times_seen", 0) + 1,
            "last_seen": datetime.now(UTC).isoformat(),
        },
    )
    return {}


def open_store() -> SqliteStore:
    STORE_DB.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteStore(sqlite3.connect(str(STORE_DB), check_same_thread=False, isolation_level=None))
    store.setup()  # creates the tables the first time
    return store


def build_graph(store=None):
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("recall", recall)
    builder.add_node("investigate", investigate)
    builder.add_node("remember", remember)
    builder.add_edge(START, "recall")
    builder.add_edge("recall", "investigate")
    builder.add_edge("investigate", "remember")
    builder.add_edge("remember", END)
    return builder.compile(store=store or open_store())


def main() -> list[dict]:
    settings = get_settings()
    fabric = get_fabric(settings)
    context = Context(fabric=fabric, model=get_model(settings), settings=settings)
    failure = pick_failure("Notebook", fabric, settings)

    store = open_store()
    store.delete(namespace(failure), failure.signature)  # start clean so run 1 has no memory
    app = build_graph(store)

    results = []
    for run in (1, 2):
        result = app.invoke({"failure": failure}, context=context)
        results.append(result)
        print(f"\nRun {run}: memory {'HIT' if result['recalled'] else 'miss'} for {failure.signature}")
        if result["recalled"]:
            print("   recalled:", result["recalled"]["root_cause"])
        print("   outcome:", result["entry"].outcome, "-", result["entry"].root_cause)

    print("\nEverything remembered for this workspace:")
    for item in store.search(namespace(failure)):
        print(f"   {item.key}: seen {item.value['times_seen']}x, fix: {item.value['fix']}")
    return results


if __name__ == "__main__":
    main()
