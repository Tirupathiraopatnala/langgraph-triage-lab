"""The handful of data shapes every lesson shares.

Kept deliberately small. A shape lives here only if more than one part of the code
passes it around. Anything only a model returns (a classification, a report) is
defined next to the graph that asks for it, so you can read the prompt and the
schema together.
"""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field


class Item(BaseModel):
    """One item in a Fabric workspace: a notebook, pipeline, lakehouse, semantic model..."""

    id: str
    name: str
    type: str


class FailedRun(BaseModel):
    """One failed job instance, as Fabric's job history reported it."""

    job_instance_id: str
    job_type: str | None = None
    started_at: datetime
    ended_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None


class Failure(BaseModel):
    """One item that failed, with its failed runs in the lookback window."""

    workspace_id: str
    workspace_name: str
    item: Item
    runs: list[FailedRun]

    @property
    def key(self) -> str:
        """Stable identity for 'this item keeps failing'. Used by the store in lesson 12."""
        return f"{self.workspace_id}/{self.item.id}"

    @property
    def latest(self) -> FailedRun:
        return max(self.runs, key=lambda run: run.started_at)

    @property
    def error_text(self) -> str:
        latest = self.latest
        return " | ".join(p for p in (latest.error_code, latest.error_message) if p)

    @property
    def signature(self) -> str:
        """A short fingerprint of the error, used to look up past fixes (lesson 10).

        Spark and Fabric errors usually carry an upper-case error class such as
        TABLE_OR_VIEW_NOT_FOUND. If there is none, fall back to the error code.
        """
        match = re.search(r"\b[A-Z][A-Z0-9_]{5,}\b", self.latest.error_message or "")
        core = match.group(0) if match else (self.latest.error_code or "UNKNOWN")
        return f"{self.item.type}:{core}"


class Definition(BaseModel):
    """An item's definition, decoded to text per part (e.g. notebook-content.py)."""

    item: Item
    parts: dict[str, str] = Field(default_factory=dict)
    # The raw Fabric payload, needed to write a patched definition back.
    raw: dict = Field(default_factory=dict, repr=False)


class Patch(BaseModel):
    """A complete replacement for one part of one item's definition.

    `before` is the exact text the agent read. Applying re-reads the live item and
    refuses if it no longer matches (optimistic concurrency against Fabric).
    """

    item_id: str
    item_name: str
    part_path: str
    before: str
    after: str
    explanation: str


class CaseEntry(BaseModel):
    """What one specialist concluded, kept in the case file for the next one."""

    specialist: str
    outcome: str  # resolved | not_mine | partial
    root_cause: str
    evidence: list[str] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)
    patched: list[str] = Field(default_factory=list)  # "item_name/part_path" of accepted patches
    handoff_to: str | None = None
    handoff_reason: str | None = None
    notes: list[str] = Field(default_factory=list)  # e.g. a dropped patch and why
