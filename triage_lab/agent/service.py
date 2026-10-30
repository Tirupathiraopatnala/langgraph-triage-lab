"""The application layer: what a CLI or a FastAPI app calls.

It owns the three persistent things and keeps them apart:
  - records       TriageStore (incidents, plans, actions)    the system of record
  - checkpointer  LangGraph checkpoints, one thread per plan  the workflow's memory
  - memory        LangGraph Store of past fixes               long-term memory

and it does the ordering that keeps them consistent:
  analyze:  new plan row -> run the graph until it pauses -> write the plan body once
  approve:  compare-and-swap proposed->approved FIRST -> resume the graph -> record result
  reject:   compare-and-swap proposed->rejected -> resume the graph so the thread ends

Putting this in FastAPI would mean: analyze returns 202 and runs in a worker (it takes
minutes), progress goes out as SSE from the stream, and approve/reject are quick POSTs.
"""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.store.sqlite import SqliteStore
from langgraph.types import Command

from triage_lab.agent.graph import build_agent, diff_text
from triage_lab.agent.store import StoreError, TriageStore
from triage_lab.checkpoints import sqlite_checkpointer
from triage_lab.contracts import Failure
from triage_lab.fabric import FAKE_WS, FabricApi, get_fabric, recent_failures
from triage_lab.models import get_model
from triage_lab.settings import REPO_ROOT, Settings, get_settings
from triage_lab.team import Context

Event = Callable[[str, str], None]  # (who, what)


class TriageService:
    def __init__(self, settings: Settings | None = None, *, fabric: FabricApi | None = None, data_dir: Path | None = None):
        settings = settings or get_settings()
        if settings.fabric_mode == "fake":
            # The fake workspace is pretend, so the agent may write to it.
            settings = dataclasses.replace(settings, writes_enabled=True, write_workspace_ids=[FAKE_WS])
        self.settings = settings
        self.fabric = fabric or get_fabric(settings)
        db_path = Path(settings.db_path)
        if data_dir:
            db_path = data_dir / db_path.name
        elif not db_path.is_absolute():
            db_path = REPO_ROOT / db_path
        self.records = TriageStore(db_path)
        self.checkpointer = sqlite_checkpointer(db_path.with_name("agent_checkpoints.sqlite"))
        self.memory = SqliteStore(sqlite3.connect(
            str(db_path.with_name("agent_memory.sqlite")), check_same_thread=False, isolation_level=None
        ))
        self.memory.setup()
        self.app = build_agent(checkpointer=self.checkpointer, store=self.memory)
        self.context = Context(fabric=self.fabric, model=get_model(settings), settings=settings)

    def close(self) -> None:
        for conn in (self.records.db, self.checkpointer.conn, self.memory.conn):
            conn.close()
        if hasattr(self.fabric, "close"):
            self.fabric.close()

    # -- scanning ---------------------------------------------------------------

    def scan(self) -> list[dict[str, Any]]:
        failures = recent_failures(self.fabric, self.settings)
        incidents = [self.records.upsert_incident(f) for f in failures]
        self.records.resolve_missing({i["id"] for i in incidents})
        return incidents

    def _failure_for(self, incident: dict[str, Any]) -> Failure:
        for failure in recent_failures(self.fabric, self.settings):
            if (failure.workspace_id, failure.item.id) == (incident["workspace_id"], incident["item_id"]):
                return failure
        raise StoreError(f"{incident['id']} has no failed runs in the lookback window any more. Run scan.")

    # -- analysis -----------------------------------------------------------------

    def analyze(self, incident_id: str, *, actor: str = "cli", note: str | None = None,
                on_event: Event | None = None) -> dict[str, Any]:
        incident = self.records.incident(incident_id)
        failure = self._failure_for(incident)
        plan = self.records.new_plan(incident_id, actor, note)
        thread = {"configurable": {"thread_id": plan["thread_id"]}}
        emit = on_event or (lambda who, what: None)
        try:
            for namespace, mode, chunk in self.app.stream(
                {"failure": failure, "operator_note": note},
                thread,
                context=self.context,
                stream_mode=["updates", "custom"],
                subgraphs=True,
            ):
                if mode == "custom":
                    emit(namespace[0].split(":")[0] if namespace else "agent", chunk.get("progress", str(chunk)))
                elif not namespace:
                    for node in chunk:
                        if node != "__interrupt__":
                            emit("agent", f"finished {node}")
        except Exception as exc:
            self.records.finish_analysis(plan["id"], "failed", f"{type(exc).__name__}: {exc}", {"error": str(exc)})
            raise

        snapshot = self.app.get_state(thread)
        values = snapshot.values
        conclusion = values["conclusion"]
        patches = values.get("patches", [])
        body = {
            "root_cause": conclusion.root_cause,
            "evidence": conclusion.evidence,
            "manual_steps": conclusion.manual_steps,
            "unresolved": conclusion.unresolved,
            "patches": [p.model_dump() | {"diff": diff_text(p)} for p in patches],
            "case_file": [e.model_dump() for e in values.get("case_file", [])],
            "hops": values.get("hops", 0),
            "memory_used": values.get("memory"),
        }
        waiting = bool(snapshot.interrupts)  # paused at approve = there is something to apply
        self.records.finish_analysis(plan["id"], "proposed" if waiting else "manual", conclusion.root_cause, body)
        return self.records.plan(plan["id"])

    # -- decisions ----------------------------------------------------------------

    def approve(self, plan_id: str, *, actor: str, after: str | None = None) -> dict[str, Any]:
        plan = self.records.plan(plan_id)
        # Compare-and-swap FIRST. If two people approve at once, only one gets past here.
        if not self.records.transition(plan_id, "proposed", "approved", actor, "edited first patch" if after else ""):
            raise StoreError(f"{plan_id} is {plan['status']}; only a proposed plan can be approved.")
        final = self.app.invoke(
            Command(resume={"action": "edit" if after else "approve", "after": after, "by": actor}),
            {"configurable": {"thread_id": plan["thread_id"]}},
            context=self.context,
        )
        outcome = final.get("outcome") or {"ok": False, "lines": ["no outcome recorded"]}
        detail = " ".join(outcome.get("lines", []))
        if outcome.get("ok"):
            self.records.transition(plan_id, "approved", "applied", "agent", detail)
        elif outcome.get("dry_run"):
            self.records.log(plan["incident_id"], plan_id, "dry_run", "agent", detail)
        else:
            self.records.transition(plan_id, "approved", "apply_failed", "agent", detail)
        return self.records.plan(plan_id) | {"outcome": outcome}

    def reject(self, plan_id: str, *, actor: str, reason: str = "") -> dict[str, Any]:
        plan = self.records.plan(plan_id)
        if not self.records.transition(plan_id, "proposed", "rejected", actor, reason):
            raise StoreError(f"{plan_id} is {plan['status']}; only a proposed plan can be rejected.")
        # Resume so the LangGraph thread finishes cleanly instead of waiting forever.
        self.app.invoke(Command(resume={"action": "reject", "by": actor, "reason": reason}),
                        {"configurable": {"thread_id": plan["thread_id"]}}, context=self.context)
        return self.records.plan(plan_id)

    def show(self, incident_id: str) -> dict[str, Any]:
        return {
            "incident": self.records.incident(incident_id),
            "plans": self.records.plans(incident_id),
            "history": self.records.history(incident_id),
        }
