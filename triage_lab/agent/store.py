"""The system of record: incidents, plans and actions in one SQLite file.

This is the "simple store" the original triage agent was missing. Three tables, each
with one job:

  incidents  One row per item that keeps failing. The id (INC-0001) is assigned ONCE, the
             first time we see the item fail, and never changes. (The original derived the
             id from the first failed run inside a sliding time window, so the same failure
             got a new id as the window moved, and its plans were orphaned.)

  plans      One row per analysis. Written once, then never edited: root cause, patches,
             manual steps. A second analysis is a new version; older proposed plans are
             marked superseded so two plans can't both be applied.

  actions    Append-only log of who did what: analysed, approved, rejected, applied.

Plan status changes go through `transition()`, a single conditional UPDATE:

    UPDATE plans SET status = 'approved' WHERE id = ? AND status = 'proposed'

If two people click approve at once, exactly one UPDATE changes a row; the other sees
rowcount 0 and is refused. That's what prevents a double apply. (The original checked the
status and saved the new one in two separate steps, so both clicks could pass the check.)

Plan lifecycle:
    analysing -> proposed -> approved -> applied
                          |           -> apply_failed
                          -> rejected
                          -> superseded   (a newer analysis replaced it)
    analysing -> manual     (no code fix; a person must follow the manual steps)
    analysing -> failed     (the analysis itself crashed)

In production this would be Postgres (or Cosmos DB with ETags for the same
compare-and-swap). The SQL is deliberately boring so it ports.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from triage_lab.contracts import Failure

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id            TEXT PRIMARY KEY,
    workspace_id  TEXT NOT NULL,
    item_id       TEXT NOT NULL,
    item_name     TEXT NOT NULL,
    item_type     TEXT NOT NULL,
    signature     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'open',      -- open | resolved
    failed_runs   INTEGER NOT NULL DEFAULT 0,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    latest_error  TEXT NOT NULL DEFAULT ''
);
-- At most one OPEN incident per item.
CREATE UNIQUE INDEX IF NOT EXISTS one_open_incident_per_item
    ON incidents (workspace_id, item_id) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS plans (
    id           TEXT PRIMARY KEY,                  -- INC-0001/v1
    incident_id  TEXT NOT NULL REFERENCES incidents(id),
    version      INTEGER NOT NULL,
    thread_id    TEXT NOT NULL,                     -- the LangGraph thread for this analysis
    status       TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    summary      TEXT,
    body         TEXT,                              -- JSON, written once
    UNIQUE (incident_id, version)
);

CREATE TABLE IF NOT EXISTS actions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id  TEXT NOT NULL,
    plan_id      TEXT,
    kind         TEXT NOT NULL,
    actor        TEXT NOT NULL,
    detail       TEXT NOT NULL DEFAULT '',
    at           TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StoreError(RuntimeError):
    pass


class TriageStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # -- incidents ------------------------------------------------------------

    def upsert_incident(self, failure: Failure) -> dict[str, Any]:
        """The open incident for this item, created if there is none. Its id never changes."""
        with self.db:
            row = self.db.execute(
                "SELECT * FROM incidents WHERE workspace_id = ? AND item_id = ? AND status = 'open'",
                (failure.workspace_id, failure.item.id),
            ).fetchone()
            latest = failure.latest.started_at.isoformat(timespec="seconds")
            if row is None:
                count = self.db.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
                incident_id = f"INC-{count + 1:04d}"
                self.db.execute(
                    "INSERT INTO incidents (id, workspace_id, item_id, item_name, item_type, signature, "
                    "failed_runs, first_seen, last_seen, latest_error) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (incident_id, failure.workspace_id, failure.item.id, failure.item.name, failure.item.type,
                     failure.signature, len(failure.runs), min(r.started_at for r in failure.runs).isoformat(
                         timespec="seconds"), latest, failure.error_text),
                )
                self._log(incident_id, None, "opened", "scanner", failure.error_text[:300])
            else:
                incident_id = row["id"]
                self.db.execute(
                    "UPDATE incidents SET failed_runs = MAX(failed_runs, ?), last_seen = MAX(last_seen, ?), "
                    "latest_error = ?, signature = ? WHERE id = ?",
                    (len(failure.runs), latest, failure.error_text, failure.signature, incident_id),
                )
        return self.incident(incident_id)

    def resolve_missing(self, seen_ids: set[str]) -> list[str]:
        """Open incidents whose item no longer fails in the window are marked resolved."""
        resolved = []
        with self.db:
            for row in self.db.execute("SELECT id FROM incidents WHERE status = 'open'").fetchall():
                if row["id"] not in seen_ids:
                    self.db.execute("UPDATE incidents SET status = 'resolved' WHERE id = ?", (row["id"],))
                    self._log(row["id"], None, "resolved", "scanner", "no failed runs in the lookback window")
                    resolved.append(row["id"])
        return resolved

    def incident(self, incident_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
        if row is None:
            raise StoreError(f"No incident {incident_id}")
        return dict(row)

    def incidents(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute("SELECT * FROM incidents ORDER BY last_seen DESC")]

    # -- plans ------------------------------------------------------------------

    def new_plan(self, incident_id: str, actor: str, note: str | None = None) -> dict[str, Any]:
        """Start a new analysis: next version, its own LangGraph thread. Older proposals are superseded."""
        with self.db:
            version = self.db.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM plans WHERE incident_id = ?", (incident_id,)
            ).fetchone()[0]
            plan_id = f"{incident_id}/v{version}"
            for old in self.db.execute(
                "SELECT id FROM plans WHERE incident_id = ? AND status = 'proposed'", (incident_id,)
            ).fetchall():
                self.db.execute("UPDATE plans SET status = 'superseded' WHERE id = ?", (old["id"],))
                self._log(incident_id, old["id"], "superseded", actor, f"replaced by {plan_id}")
            self.db.execute(
                "INSERT INTO plans (id, incident_id, version, thread_id, status, created_at) VALUES (?,?,?,?,?,?)",
                (plan_id, incident_id, version, f"triage:{plan_id}", "analysing", _now()),
            )
            self._log(incident_id, plan_id, "analysis_started", actor, note or "")
        return self.plan(plan_id)

    def finish_analysis(self, plan_id: str, status: str, summary: str, body: dict[str, Any]) -> None:
        """Write the plan's body exactly once."""
        with self.db:
            changed = self.db.execute(
                "UPDATE plans SET status = ?, summary = ?, body = ? WHERE id = ? AND status = 'analysing'",
                (status, summary, json.dumps(body, default=str), plan_id),
            ).rowcount
            if changed != 1:
                raise StoreError(f"{plan_id} is not being analysed; plans are written once.")
            self._log(self.plan(plan_id)["incident_id"], plan_id, f"analysis_{status}", "agent", summary[:300])

    def transition(self, plan_id: str, from_status: str, to_status: str, actor: str, detail: str = "") -> bool:
        """Compare-and-swap on status. True if THIS call made the change."""
        with self.db:
            changed = self.db.execute(
                "UPDATE plans SET status = ? WHERE id = ? AND status = ?", (to_status, plan_id, from_status)
            ).rowcount
            if changed == 1:
                self._log(self.plan(plan_id)["incident_id"], plan_id, to_status, actor, detail)
        return changed == 1

    def plan(self, plan_id: str) -> dict[str, Any]:
        row = self.db.execute("SELECT * FROM plans WHERE id = ?", (plan_id,)).fetchone()
        if row is None:
            raise StoreError(f"No plan {plan_id}")
        plan = dict(row)
        plan["body"] = json.loads(plan["body"]) if plan["body"] else None
        return plan

    def plans(self, incident_id: str) -> list[dict[str, Any]]:
        ids = [r["id"] for r in self.db.execute(
            "SELECT id FROM plans WHERE incident_id = ? ORDER BY version", (incident_id,))]
        return [self.plan(i) for i in ids]

    # -- actions ----------------------------------------------------------------

    def log(self, incident_id: str, plan_id: str | None, kind: str, actor: str, detail: str = "") -> None:
        """Record an action that is not a status change (e.g. a dry run)."""
        with self.db:
            self._log(incident_id, plan_id, kind, actor, detail)

    def _log(self, incident_id: str, plan_id: str | None, kind: str, actor: str, detail: str) -> None:
        self.db.execute(
            "INSERT INTO actions (incident_id, plan_id, kind, actor, detail, at) VALUES (?,?,?,?,?,?)",
            (incident_id, plan_id, kind, actor, detail, _now()),
        )

    def history(self, incident_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.db.execute(
            "SELECT * FROM actions WHERE incident_id = ? ORDER BY id", (incident_id,))]
