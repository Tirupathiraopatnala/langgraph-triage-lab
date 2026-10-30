"""Lesson 7: human-in-the-loop with interrupt() and Command(resume=...).

WHAT YOU LEARN
- `interrupt(payload)` inside a node PAUSES the run and hands `payload` to whoever
  called the graph. The state is checkpointed, so a checkpointer is required. The run
  can wait seconds or days: nothing is held in memory.
- `app.invoke(Command(resume=answer), same_thread)` continues. `interrupt()` returns
  `answer` inside the node, and the node carries on from there.
- THE RULE: on resume, the interrupted node runs again FROM THE TOP. So a node that
  interrupts must do nothing with side effects before the `interrupt()` call. Do the
  reading in an earlier node (`gather`), the writing in a later node (`apply`), and
  keep the approval node to "show, ask, record".
- `invoke(..., version="v2")` returns a GraphOutput with `.value` and `.interrupts`, so
  you can see the pause without poking at internals.
- OPTIMISTIC CONCURRENCY: `apply` re-reads the live definition and refuses if it no
  longer matches what the model read. Somebody may have edited it while the approval
  waited.

WHAT TO WATCH
The run stops with a diff and a question. You answer in the terminal. Approve, and the
patch is written (in the fake workspace writes are allowed; in a real one, only if
FABRIC_WRITES_ENABLED and the workspace is on FABRIC_WRITE_WORKSPACE_IDS).

INTERVIEW LINE
"Approval is an interrupt inside the graph, backed by the checkpointer, so the run can
wait for a human indefinitely and resume exactly where it stopped. The approval node has
no side effects before the interrupt because it re-executes on resume, and the apply
step re-checks the source before writing."

TRY THIS
1. Answer "reject" and read the final state.
2. Resume with Command(resume={"action": "edit", "after": "..."}) to apply your own text.
3. Between the pause and the approval, change the fake notebook (FakeFabric._definitions)
   and watch apply refuse.

RUN
    python -m lessons.l07_human_approval            (asks you)
    python -m lessons.l07_human_approval approve    (answers for you)
"""

from __future__ import annotations

import dataclasses
import difflib
import sys
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt
from pydantic import BaseModel, Field

from triage_lab.checkpoints import memory_checkpointer
from triage_lab.contracts import Definition, Failure, Patch
from triage_lab.fabric import FAKE_WS, FabricError, get_fabric, pick_failure, with_patched_part
from triage_lab.models import get_model
from triage_lab.settings import get_settings
from triage_lab.team import Context, ProposedPatch, accept_patch
from triage_lab.texts import definition_text, failure_text, tables_text


class ProposedFix(BaseModel):
    root_cause: str
    patches: list[ProposedPatch] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)


class State(TypedDict, total=False):
    failure: Failure
    definition: Definition
    tables: str
    root_cause: str
    patches: list[Patch]
    decision: dict
    outcome: str


def gather(state: State, runtime: Runtime[Context]) -> dict:
    """All the reading, before any interrupt."""
    fabric, failure = runtime.context.fabric, state["failure"]
    definition = fabric.read_definition(failure.workspace_id, failure.item.id)
    lakehouses = [i for i in fabric.list_items(failure.workspace_id) if i.type == "Lakehouse"]
    tables = "\n".join(
        tables_text(lh.id, fabric.list_lakehouse_tables(failure.workspace_id, lh.id)) for lh in lakehouses
    )
    return {"definition": definition, "tables": tables}


def propose(state: State, runtime: Runtime[Context]) -> dict:
    prompt = "\n\n".join([failure_text(state["failure"]), definition_text(state["definition"]), state["tables"]])
    fix: ProposedFix = runtime.context.model.with_structured_output(ProposedFix).invoke([
        SystemMessage(
            "You are the notebook specialist. Find the root cause. If a code change fixes it, "
            "give the COMPLETE new content of the part as a patch."
        ),
        HumanMessage(prompt),
    ])
    definitions = {state["definition"].item.id: state["definition"]}
    patches = [p for p in (accept_patch(definitions, proposed) for proposed in fix.patches) if isinstance(p, Patch)]
    return {"root_cause": fix.root_cause, "patches": patches}


def approve(state: State) -> dict:
    """Show, ask, record. Nothing before interrupt() has side effects: this re-runs on resume."""
    answer = interrupt({
        "question": "Apply this fix? Answer approve, reject, or edit (with 'after').",
        "root_cause": state["root_cause"],
        "diffs": [_diff(p) for p in state.get("patches", [])],
    })
    return {"decision": answer}


def after_approval(state: State) -> Literal["apply", "__end__"]:
    return "apply" if state["decision"].get("action") in {"approve", "edit"} else END


def apply(state: State, runtime: Runtime[Context]) -> dict:
    fabric, ws = runtime.context.fabric, state["failure"].workspace_id
    results = []
    for n, patch in enumerate(state.get("patches", [])):
        after = state["decision"].get("after") if n == 0 and state["decision"].get("action") == "edit" else patch.after
        live = fabric.read_definition(ws, patch.item_id)
        if live.parts.get(patch.part_path) != patch.before:
            results.append(f"{patch.item_name}: refused, it changed since it was read. Analyse again.")
            continue
        if not runtime.context.settings.writes_allowed_for(ws):
            results.append(f"{patch.item_name}: dry run, writes are disabled for this workspace.")
            continue
        fabric.update_definition(ws, patch.item_id, with_patched_part(live.raw, patch.part_path, after))
        results.append(f"{patch.item_name}/{patch.part_path}: written to Fabric.")
    return {"outcome": " ".join(results) or "Nothing to apply."}


def _diff(patch: Patch) -> str:
    return "".join(difflib.unified_diff(
        patch.before.splitlines(keepends=True), patch.after.splitlines(keepends=True),
        fromfile=f"{patch.item_name}/{patch.part_path} (now)", tofile="(proposed)",
    ))


def build_graph(checkpointer=None):
    transient = RetryPolicy(max_attempts=3, retry_on=lambda e: isinstance(e, FabricError) and e.transient)
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("gather", gather, retry_policy=transient)
    builder.add_node("propose", propose)
    builder.add_node("approve", approve)
    builder.add_node("apply", apply, retry_policy=transient)
    builder.add_edge(START, "gather")
    builder.add_edge("gather", "propose")
    builder.add_edge("propose", "approve")
    builder.add_conditional_edges("approve", after_approval, ["apply", END])
    builder.add_edge("apply", END)
    return builder.compile(checkpointer=checkpointer or memory_checkpointer())


def main(answer: str | None = None) -> dict:
    settings = get_settings()
    if settings.fabric_mode == "fake":
        # The fake workspace is pretend, so let this lesson actually write to it.
        settings = dataclasses.replace(settings, writes_enabled=True, write_workspace_ids=[FAKE_WS])
    fabric = get_fabric(settings)
    context = Context(fabric=fabric, model=get_model(settings), settings=settings)
    failure = pick_failure("Notebook", fabric, settings)

    app = build_graph()
    thread = {"configurable": {"thread_id": f"lesson07:{failure.key}"}}

    paused = app.invoke({"failure": failure}, thread, context=context, version="v2")
    question = paused.interrupts[0].value
    print("PAUSED. The graph is waiting for a human.\n")
    print("Root cause:", question["root_cause"])
    for diff in question["diffs"]:
        print(diff)
    print("Waiting at node:", app.get_state(thread).next)

    answer = answer or input("\napprove / reject? ").strip().lower() or "reject"
    final = app.invoke(Command(resume={"action": answer, "by": "you"}), thread, context=context)
    print("\nDecision:", final["decision"])
    print("Outcome:", final.get("outcome", "rejected, nothing written"))
    if settings.fabric_mode == "fake" and answer == "approve":
        print("Failed runs left on the notebook:", len(fabric.list_job_instances(FAKE_WS, failure.item.id)))
    return final


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
