"""Lesson 12: the full triage agent.

Everything from lessons 1 to 11, put together the way you would ship it. The code lives
in triage_lab/agent/. Read it in this order:

  1. triage_lab/agent/graph.py    the graph: recall -> team -> conclude -> approve -> apply -> remember
  2. triage_lab/supervisor.py     the team: supervisor + specialists with Command (lesson 9)
  3. triage_lab/team.py           one specialist: tools + investigator subgraph (lessons 4, 5, 8)
  4. triage_lab/agent/store.py    the system of record: incidents, plans, actions
  5. triage_lab/agent/service.py  the application layer that keeps graph and records in step
  6. triage_lab/agent/cli.py      the command line

THE DESIGN DECISIONS (each one is an interview answer)

- Three kinds of persistence, kept apart:
    records (store.py)    incidents, plans, actions. What happened, for people and audit.
    checkpointer          the graph's own per-thread state, so it can pause and resume.
    Store (memory)        fixes that worked, keyed by error signature, across all threads.
- An incident's id is assigned once, when the item first fails, and never changes.
  A plan is written once and never edited; re-analysis makes v2 and supersedes v1.
- Approval is a compare-and-swap on the plan's status BEFORE the graph resumes, so two
  approvals can't both apply. The interrupt keeps the run paused for as long as the human
  needs.
- The graph proposes; acting is a separate, recorded step. apply re-reads the live item
  and refuses if it changed since the specialist read it. Writes fail closed (switch +
  workspace allowlist).
- Dependencies live in runtime context, so state is pure data and checkpoints cleanly.
- The model judges (route, accept, root cause, patch). Code enforces invariants: roster,
  hop budget, tool budget, patch checks, no self-handoff, not_mine can't close a case.
- Retries are one layer per failure type: the Fabric client handles 429, the OpenAI SDK
  handles model errors, the graph retries transient node failures, and an error handler
  records what's left.

WHAT THIS STILL DOESN'T DO (good "what would you improve" answers)
- No web API. In FastAPI: POST /incidents/{id}/analyses returns 202 and runs in a worker;
  progress goes out as SSE; approve/reject are POSTs that call the service. Auth with Entra
  (Easy Auth or MSAL) and the approver's identity recorded on the action.
- SQLite is single-machine. Postgres for records, PostgresSaver and PostgresStore for the
  graph, and the compare-and-swap SQL stays the same.
- No evaluation set. Save real incidents with their accepted fixes and replay them on every
  prompt or model change: did it pick the right specialist, the right root cause, the same
  patch?
- Fabric credentials: move from a client secret to managed identity.

RUN
    python -m lessons.l12_triage_agent          (same as: python -m triage_lab.agent demo)
    python -m triage_lab.agent --help           (the individual commands)
"""

from __future__ import annotations

from triage_lab.agent.cli import demo


def main() -> None:
    demo()


if __name__ == "__main__":
    main()
