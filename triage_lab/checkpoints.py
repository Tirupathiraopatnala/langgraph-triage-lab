"""Checkpointers and stores with a serializer that knows our types.

LangGraph saves graph state with msgpack. Newer versions refuse to load Python
classes they were not told about (a security measure: loading arbitrary classes from
a database is how deserialisation attacks work). So we list ours explicitly.

The alternative design is to keep checkpointed state JSON-native (plain dicts and
strings). Both are fine; the important thing is to know why the warning appears.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

OUR_TYPES = [
    ("triage_lab.contracts", name)
    for name in ("Item", "FailedRun", "Failure", "Definition", "Patch", "CaseEntry")
] + [
    ("triage_lab.team", name) for name in ("SpecialistReport", "ProposedPatch")
] + [
    ("triage_lab.supervisor", name) for name in ("Conclusion",)
]


def serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=OUR_TYPES)


def memory_checkpointer() -> InMemorySaver:
    return InMemorySaver(serde=serializer())


def sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    """A checkpointer that survives restarts. One file, no server."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn, serde=serializer())
