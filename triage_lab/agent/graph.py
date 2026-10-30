"""The full triage agent: every capability from lessons 1 to 11 in one graph.

    START -> recall --> supervisor <-> specialists (Command handoffs, hop budget)
                            |
                            v
                         conclude -> approve --interrupt()--> apply -> remember -> END
                                        |                        ^
                                        +-- no patch ------------+-- (remember) / reject -> END

Where each lesson shows up:
  1  StateGraph, nodes return partial updates               (everywhere)
  2  reducers on case_file and patches                      triage_lab/supervisor.py
  3  structured output: Routing, Review, Conclusion,         supervisor.py, team.py
     SpecialistReport
  4  tools + tool loop with a budget                        team.py (investigator)
  5  runtime context; retry policy + error handler           Context; apply node below
  6  checkpointer: one thread per plan version               build_agent(checkpointer=...)
  7  interrupt() for approval, apply re-checks `before`      approve, apply below
  8  investigator subgraph; tools updating state via Command team.py
  9  supervisor routing with Command                        supervisor.py
  10 Store: recall a past fix as a lead, remember new ones   recall, remember below
  11 streaming progress (custom events from tools)           cli.py

The graph never touches the system-of-record database (store.py). The CLI does that,
around the graph: it records the plan when the graph pauses, and it flips the plan's
status with a compare-and-swap BEFORE resuming an approval. Workflow in the graph,
records in the store.
"""

from __future__ import annotations

import difflib
from datetime import UTC, datetime
from typing import Literal

from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, RetryPolicy, interrupt

from triage_lab.contracts import Patch
from triage_lab.fabric import FabricError, with_patched_part
from triage_lab.supervisor import TeamState, add_team
from triage_lab.team import Context


class AgentState(TeamState, total=False):
    decision: dict   # what the human answered at the approval interrupt
    outcome: dict    # what apply did: {"ok": bool, "dry_run": bool, "lines": [...]}


def memory_namespace(state: AgentState) -> tuple[str, ...]:
    return ("fixes", state["failure"].workspace_id)


# --- before the team -----------------------------------------------------------


def recall(state: AgentState, runtime: Runtime[Context]) -> dict:
    """Look up a past fix for the same error signature. It becomes a lead, not an answer."""
    if runtime.store is None:
        return {"memory": None}
    hit = runtime.store.get(memory_namespace(state), state["failure"].signature)
    if hit is None:
        return {"memory": None}
    v = hit.value
    return {"memory": f"{v['root_cause']} Fixed then by: {v['fix']} (seen {v['times_seen']}x, last {v['last_seen'][:10]})."}


# --- after the team ------------------------------------------------------------


def diff_text(patch: Patch) -> str:
    return "".join(difflib.unified_diff(
        patch.before.splitlines(keepends=True), patch.after.splitlines(keepends=True),
        fromfile=f"{patch.item_name}/{patch.part_path} (live)", tofile="(proposed)",
    ))


def approve(state: AgentState) -> dict:
    """Show, ask, record. NOTHING with side effects before interrupt(): this node re-runs on resume."""
    if not state.get("patches"):
        return {"decision": {"action": "none", "why": "no code change proposed"}}
    conclusion = state["conclusion"]
    answer = interrupt({
        "root_cause": conclusion.root_cause,
        "diffs": [diff_text(p) for p in state["patches"]],
        "manual_steps": conclusion.manual_steps,
        "unresolved": conclusion.unresolved,
        "question": "approve, reject, or edit (with 'after' for the first patch)?",
    })
    return {"decision": answer}


def after_approve(state: AgentState) -> Literal["apply", "remember", "__end__"]:
    action = state["decision"].get("action")
    if action in {"approve", "edit"}:
        return "apply"
    if action == "none":
        return "remember"  # manual-steps-only cases are still worth remembering
    return END


def apply(state: AgentState, runtime: Runtime[Context]) -> dict:
    """Write each patch, re-checking first that the live part is what the specialist read."""
    fabric, settings = runtime.context.fabric, runtime.context.settings
    ws = state["failure"].workspace_id
    decision = state["decision"]
    lines, ok, dry_run = [], True, False
    for n, patch in enumerate(state["patches"]):
        after = decision["after"] if n == 0 and decision.get("action") == "edit" and decision.get("after") else patch.after
        live = fabric.read_definition(ws, patch.item_id)
        if live.parts.get(patch.part_path) != patch.before:
            lines.append(f"{patch.item_name}/{patch.part_path}: REFUSED, it changed since it was read.")
            ok = False
            break  # Fabric has no transaction across items; stop at the first problem
        if not settings.writes_allowed_for(ws):
            lines.append(f"{patch.item_name}/{patch.part_path}: DRY RUN, writes are disabled for this workspace.")
            dry_run = True
            continue
        fabric.update_definition(ws, patch.item_id, with_patched_part(live.raw, patch.part_path, after))
        lines.append(f"{patch.item_name}/{patch.part_path}: written.")
    return {"outcome": {"ok": ok and not dry_run, "dry_run": dry_run and ok, "lines": lines}}


def apply_failed(state: AgentState, error: NodeError) -> Command:
    """Retries ran out (or a permanent error). Record it instead of crashing the thread."""
    return Command(update={"outcome": {"ok": False, "dry_run": False, "lines": [f"apply failed: {error.error}"]}}, goto=END)


def remember(state: AgentState, runtime: Runtime[Context]) -> dict:
    """Store what fixed it, keyed by error signature, for next time (lesson 10)."""
    applied = state.get("outcome", {}).get("ok")
    manual_only = state["decision"].get("action") == "none" and not state["conclusion"].unresolved
    if runtime.store is None or not (applied or manual_only):
        return {}
    namespace, key = memory_namespace(state), state["failure"].signature
    previous = runtime.store.get(namespace, key)
    fix = ", ".join(f"{p.item_name}/{p.part_path}: {p.explanation}" for p in state.get("patches", [])) or " | ".join(
        state["conclusion"].manual_steps
    )
    runtime.store.put(namespace, key, {
        "root_cause": state["conclusion"].root_cause,
        "fix": fix or "none recorded",
        "times_seen": (previous.value["times_seen"] + 1) if previous else 1,
        "last_seen": datetime.now(UTC).isoformat(),
    })
    return {}


def build_agent(checkpointer=None, store=None):
    fabric_retry = RetryPolicy(max_attempts=3, retry_on=lambda e: isinstance(e, FabricError) and e.transient)
    builder = StateGraph(AgentState, context_schema=Context)
    builder.add_node("recall", recall)
    add_team(builder)  # supervisor, one node per specialist, conclude
    builder.add_node("approve", approve)
    builder.add_node("apply", apply, retry_policy=fabric_retry, error_handler=apply_failed)
    builder.add_node("remember", remember)

    builder.add_edge(START, "recall")
    builder.add_edge("recall", "supervisor")
    builder.add_edge("conclude", "approve")
    builder.add_conditional_edges("approve", after_approve, ["apply", "remember", END])
    builder.add_edge("apply", "remember")
    builder.add_edge("remember", END)
    return builder.compile(checkpointer=checkpointer, store=store)
