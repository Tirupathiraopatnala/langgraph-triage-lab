"""The specialist team as a reusable module. Lessons 8 to 12 import from here.

This is lessons 4 and 5 tidied up into something you would ship:

- Fabric tools get the client from the RUNTIME CONTEXT (`ToolRuntime`), never from a
  global, and the failing workspace from the graph STATE.
- `read_item_definition` returns a `Command` that updates state as well as answering
  the model. So the definitions a specialist read live in state, where the patch check
  can see them (and the checkpointer can save them). No hidden caches.
- `build_investigator()` is a SUBGRAPH: one specialist's tool loop with a budget,
  ending in a structured `SpecialistReport` whose patches are checked by code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, ToolRuntime
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from triage_lab.contracts import CaseEntry, Definition, Failure, Patch
from triage_lab.fabric import FabricApi, FabricError
from triage_lab.settings import Settings
from triage_lab.texts import definition_text, failure_text, items_text, tables_text

# A part longer than this is shown to the model truncated, and no patch is accepted
# for it: the model must have seen the whole thing to replace the whole thing.
MAX_PART_CHARS = 60_000


@dataclass
class Context:
    """Run-time dependencies. Passed with `context=` at invoke time, never stored in state.

    State is data that gets checkpointed. A Fabric client, a model or settings are
    not data, so they go here. (Putting them in state is what stops you adding a
    checkpointer later.)
    """

    fabric: FabricApi
    model: BaseChatModel
    settings: Settings


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

ROSTER: dict[str, str] = {
    "notebook": "notebooks and Spark job definitions: PySpark/SQL code, Spark sessions, "
    "libraries, notebook parameters, reads and writes of lakehouse tables from code.",
    "pipeline": "data pipelines: activities, their settings and parameters, dependencies, "
    "copy activities, and which item an activity runs.",
    "lakehouse": "lakehouses: which tables and files exist, schemas, shortcuts, and "
    "whether data a job expects is actually there.",
    "semantic_model": "semantic models: refreshes, data source connections and credentials, "
    "Direct Lake and storage mode issues, TMDL definitions.",
    "general": "anything that does not clearly belong to another specialist: capacity and "
    "throttling, permissions, gateways, workspace settings.",
}


def charter(name: str) -> str:
    return (
        f"You are the {name} specialist on a Microsoft Fabric support team. You own {ROSTER[name]}\n\n"
        "Investigate with your read-only tools, then report. Rules:\n"
        "- Quote evidence exactly: lines from the error or from the definitions you read.\n"
        "- If the fix is a code or definition change you own, write a patch: the COMPLETE new "
        "content of one part you have read in full. Change only what the evidence points at.\n"
        "- If the fix needs a person (credentials, permissions, capacity), give manual steps.\n"
        "- If the root cause sits in an item another specialist owns, answer not_mine and name "
        "them in handoff_to. Never hand a case to yourself.\n"
        f"- The roster is: {', '.join(ROSTER)}."
    )


# ---------------------------------------------------------------------------
# What a specialist returns
# ---------------------------------------------------------------------------


class ProposedPatch(BaseModel):
    item_id: str
    part_path: str = Field(description="Definition part path, e.g. notebook-content.py")
    after: str = Field(description="The complete new content of that part")
    explanation: str = Field(description="What changed and why, for the reviewer")


class SpecialistReport(BaseModel):
    outcome: Literal["resolved", "not_mine", "partial"]
    root_cause: str
    evidence: list[str] = Field(default_factory=list, description="Exact quoted lines")
    patches: list[ProposedPatch] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)
    handoff_to: str | None = Field(default=None, description="Specialist name, for not_mine or partial")
    handoff_reason: str | None = None
    needs_from_other: str | None = None


# ---------------------------------------------------------------------------
# Tools. The `runtime` parameter is injected by LangGraph and hidden from the model.
# ---------------------------------------------------------------------------


def _workspace(runtime: ToolRuntime[Context, dict]) -> str:
    return runtime.state["failure"].workspace_id


def _progress(runtime: ToolRuntime[Context, dict], text: str) -> None:
    """A custom stream event (lesson 11). Harmless when nobody is streaming."""
    if runtime.stream_writer:
        runtime.stream_writer({"progress": text})


@tool
def list_workspace_items(runtime: ToolRuntime[Context, dict]) -> str:
    """List every item in the failing workspace: id | type | name."""
    _progress(runtime, "listing workspace items")
    try:
        return items_text(runtime.context.fabric.list_items(_workspace(runtime)))
    except FabricError as exc:
        return f"Fabric refused that read: {exc}"


@tool
def read_item_definition(item_id: str, runtime: ToolRuntime[Context, dict]) -> Command | str:
    """Read the full definition of one item (notebook code, pipeline JSON, TMDL...) by item id."""
    _progress(runtime, f"reading definition {item_id[:8]}")
    try:
        definition = runtime.context.fabric.read_definition(_workspace(runtime), item_id)
    except FabricError as exc:
        return f"Fabric refused that read: {exc}"
    shown = definition.model_copy(
        update={
            "parts": {
                path: text if len(text) <= MAX_PART_CHARS else text[:MAX_PART_CHARS] + "\n[TRUNCATED: do not patch]"
                for path, text in definition.parts.items()
            }
        }
    )
    # A tool can return a Command to update graph state, not just answer the model.
    return Command(
        update={
            "definitions": {item_id: definition},
            "messages": [ToolMessage(definition_text(shown), tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def list_lakehouse_tables(lakehouse_id: str, runtime: ToolRuntime[Context, dict]) -> str:
    """List the tables that exist in one lakehouse, by lakehouse item id."""
    _progress(runtime, f"listing tables in {lakehouse_id[:8]}")
    try:
        return tables_text(lakehouse_id, runtime.context.fabric.list_lakehouse_tables(_workspace(runtime), lakehouse_id))
    except FabricError as exc:
        return f"Fabric refused that read: {exc}"


TOOLS = [list_workspace_items, read_item_definition, list_lakehouse_tables]


# ---------------------------------------------------------------------------
# Patch acceptance: mechanical checks only, no model judgement
# ---------------------------------------------------------------------------


def accept_patch(definitions: dict[str, Definition], proposed: ProposedPatch) -> Patch | str:
    definition = definitions.get(proposed.item_id)
    if definition is None:
        return f"patch for {proposed.item_id} dropped: that definition was never read"
    before = definition.parts.get(proposed.part_path)
    if before is None:
        return f"patch for {definition.item.name}/{proposed.part_path} dropped: no such part"
    if len(before) > MAX_PART_CHARS:
        return f"patch for {definition.item.name}/{proposed.part_path} dropped: part was truncated"
    if not proposed.after.strip() or proposed.after.strip() == before.strip():
        return f"patch for {definition.item.name}/{proposed.part_path} dropped: empty or unchanged"
    return Patch(
        item_id=proposed.item_id,
        item_name=definition.item.name,
        part_path=proposed.part_path,
        before=before,
        after=proposed.after,
        explanation=proposed.explanation,
    )


# ---------------------------------------------------------------------------
# The investigator subgraph
# ---------------------------------------------------------------------------


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer: new keys are added, existing keys are replaced."""
    return {**(left or {}), **(right or {})}


class InvestigationState(TypedDict, total=False):
    # inputs
    failure: Failure
    specialist: str
    briefing: str
    case_file: str
    operator_note: str | None
    # working state
    messages: Annotated[list[AnyMessage], add_messages]
    definitions: Annotated[dict[str, Definition], merge_dicts]
    # outputs
    report: SpecialistReport
    patches: list[Patch]
    entry: CaseEntry


def _brief(state: InvestigationState) -> dict[str, Any]:
    parts = [
        "# Briefing", state.get("briefing") or "Find the root cause of this failure.",
        "", "# Failure", failure_text(state["failure"]),
        "", "# Case file so far", state.get("case_file") or "Nobody has looked at this yet.",
    ]
    if state.get("operator_note"):
        parts += ["", "# The operator said", state["operator_note"]]
    parts += ["", "Use your tools to read what you need, then stop calling tools."]
    return {"messages": [SystemMessage(charter(state["specialist"])), HumanMessage("\n".join(parts))]}


def _agent(state: InvestigationState, runtime: Runtime[Context]) -> dict[str, Any]:
    reply = runtime.context.model.bind_tools(TOOLS).invoke(state["messages"])
    return {"messages": [reply]}


def _after_agent(state: InvestigationState, runtime: Runtime[Context]) -> str:
    last = state["messages"][-1]
    used = sum(isinstance(m, ToolMessage) for m in state["messages"])
    if isinstance(last, AIMessage) and last.tool_calls and used < runtime.context.settings.max_tool_calls:
        return "tools"
    return "report"


def _report(state: InvestigationState, runtime: Runtime[Context]) -> dict[str, Any]:
    messages = list(state["messages"])
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        # Budget ran out mid-request. Every tool call needs an answer before we continue.
        messages += [
            ToolMessage("Read budget exhausted. Report from what you have.", tool_call_id=c["id"])
            for c in last.tool_calls
        ]
    messages.append(HumanMessage(
        "Write your report now. Quote evidence exactly. A patch must be the complete new "
        "content of a part you have read in full."
    ))
    structured = runtime.context.model.with_structured_output(SpecialistReport)
    report: SpecialistReport = structured.invoke(messages)

    me = state["specialist"]
    if report.outcome == "not_mine" and (report.handoff_to not in ROSTER or report.handoff_to == me):
        # A specialist cannot hand a case to itself or to nobody. Ask once more, with the rule spelled out.
        messages.append(HumanMessage(
            f"You answered not_mine but named {report.handoff_to!r}. Name another specialist from "
            f"{[n for n in ROSTER if n != me]}, or answer resolved/partial yourself."
        ))
        report = structured.invoke(messages)

    patches, notes = [], []
    for proposed in report.patches:
        accepted = accept_patch(state.get("definitions", {}), proposed)
        if isinstance(accepted, Patch):
            patches.append(accepted)
        else:
            notes.append(accepted)
    entry = CaseEntry(
        specialist=me,
        outcome=report.outcome,
        root_cause=report.root_cause,
        evidence=report.evidence,
        manual_steps=report.manual_steps,
        patched=[f"{p.item_name}/{p.part_path}" for p in patches],
        handoff_to=report.handoff_to,
        handoff_reason=report.handoff_reason,
        notes=notes,
    )
    return {"report": report, "patches": patches, "entry": entry}


def build_investigator():
    """One specialist's investigation as a compiled graph, usable as a subgraph.

        START -> brief -> agent <-> tools
                           |
                           v
                         report -> END
    """
    builder = StateGraph(InvestigationState, context_schema=Context)
    builder.add_node("brief", _brief)
    builder.add_node("agent", _agent)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("report", _report)
    builder.add_edge(START, "brief")
    builder.add_edge("brief", "agent")
    builder.add_conditional_edges("agent", _after_agent, ["tools", "report"])
    builder.add_edge("tools", "agent")
    builder.add_edge("report", END)
    return builder.compile(name="investigator")
