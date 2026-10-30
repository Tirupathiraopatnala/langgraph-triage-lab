"""How Fabric facts are written out for a model to read.

Every lesson formats failures, definitions and tool results the same way, so the
prompts stay consistent (and the scripted fake model can read them too).
"""

from __future__ import annotations

from triage_lab.contracts import Definition, Failure, Item


def failure_text(failure: Failure) -> str:
    lines = [
        f"Workspace: {failure.workspace_name} ({failure.workspace_id})",
        f"Item: {failure.item.name} ({failure.item.type}, id {failure.item.id})",
        f"Failed runs: {len(failure.runs)}, latest at {failure.latest.started_at:%Y-%m-%d %H:%M} UTC",
        f"Error: {failure.error_text or '(Fabric gave no error text)'}",
    ]
    return "\n".join(lines)


def items_text(items: list[Item]) -> str:
    return "\n".join(f"{i.id} | {i.type} | {i.name}" for i in items) or "(no items)"


def definition_text(definition: Definition) -> str:
    head = f"## {definition.item.name} ({definition.item.type}, id {definition.item.id})"
    if not definition.parts:
        return head + "\n(no text parts in this definition)"
    blocks = [f"=== {path} ===\n{text}" for path, text in definition.parts.items()]
    # The end marker lets anything reading this (including the fake model) know where
    # the last part stops when the definition sits in the middle of a longer prompt.
    return head + "\n\n" + "\n\n".join(blocks) + f"\n=== end of {definition.item.name} ==="


def tables_text(lakehouse_id: str, tables: list[str]) -> str:
    return f"Tables in lakehouse {lakehouse_id}: " + (", ".join(tables) or "(none)")


def case_file_text(entries: list) -> str:
    """The shared case file, numbered, as the supervisor and specialists read it."""
    if not entries:
        return "Nobody has looked at this yet."
    lines = []
    for n, e in enumerate(entries, 1):
        lines.append(f"{n}. {e.specialist} answered {e.outcome}: {e.root_cause}")
        if e.evidence:
            lines.append("   evidence: " + " | ".join(e.evidence))
        if e.patched:
            lines.append("   patched: " + ", ".join(e.patched))
        if e.manual_steps:
            lines.append("   manual steps: " + " | ".join(e.manual_steps))
        if e.handoff_to:
            lines.append(f"   handed to {e.handoff_to}: {e.handoff_reason or ''}")
        lines.extend(f"   note: {note}" for note in e.notes)
    return "\n".join(lines)
