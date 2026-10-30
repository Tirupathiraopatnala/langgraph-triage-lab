"""A supervisor and its specialists, wired with Command. Lesson 9 teaches it; lesson 12 reuses it.

    START -> supervisor --Command(goto=X)--> specialist X --Command(goto=supervisor)--+
                ^                                                                      |
                +----------------------------------------------------------------------+
             supervisor --Command(goto="conclude")--> conclude

The graph has almost no fixed edges. Each node DECIDES where to go next by returning
`Command(goto=..., update=...)`: routing and state update in one return value.
`destinations=` on add_node tells LangGraph (and the drawing) where a node may go.

Who decides what:
- The MODEL decides routing and judgement: which specialist, accept or reroute, the
  final narrative. Always as structured output, never parsed free text.
- CODE enforces the rules the model must not be trusted with: the specialist must be on
  the roster, a not_mine report can't close a case, nobody reroutes to themselves, and
  the hop budget is checked before the model is even asked.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command
from pydantic import BaseModel, Field

from triage_lab.contracts import CaseEntry, Failure, Patch
from triage_lab.team import ROSTER, Context, build_investigator
from triage_lab.texts import case_file_text, failure_text


def collect(left: list | None, right: CaseEntry | list | None) -> list:
    items = right if isinstance(right, list) else [right]
    return (left or []) + [i for i in items if i is not None]


class TeamState(TypedDict, total=False):
    failure: Failure
    operator_note: str | None
    memory: str | None  # a remembered fix for this error, as a lead (lesson 10 / 12)
    case_file: Annotated[list[CaseEntry], collect]
    patches: Annotated[list[Patch], operator.add]
    hops: int
    briefing: str
    stop_reason: str | None
    conclusion: Conclusion


class Routing(BaseModel):
    specialist: str = Field(description=f"One of: {', '.join(ROSTER)}")
    reason: str
    briefing: str = Field(description="What the specialist should know before starting")


class Review(BaseModel):
    decision: Literal["accept", "reroute", "stop"]
    reason: str
    next_specialist: str | None = None
    briefing: str | None = None


class Conclusion(BaseModel):
    root_cause: str = Field(description="One narrative for the operator")
    evidence: list[str] = Field(default_factory=list)
    manual_steps: list[str] = Field(default_factory=list)
    unresolved: str | None = Field(default=None, description="What is still open, if anything")


SUPERVISOR = (
    "You supervise a team of Microsoft Fabric specialists. You route a failure to the "
    "specialist who owns its root cause, review what they report, and decide whether the "
    "case is closed or someone else must look. You do not diagnose yourself.\n"
    f"Roster: {', '.join(f'{k} ({v})' for k, v in ROSTER.items())}\n"
    "Accept only a report whose root cause is backed by quoted evidence. A not_mine report "
    "is a request for another specialist, never a closed case."
)


def _context(state: TeamState) -> str:
    parts = [failure_text(state["failure"]), "", "# Case file", case_file_text(state.get("case_file", []))]
    if state.get("operator_note"):
        parts += ["", "# The operator said", state["operator_note"]]
    if state.get("memory"):
        parts += ["", "# A similar failure before (a lead to verify, not an answer)", state["memory"]]
    return "\n".join(parts)


def supervisor(state: TeamState, runtime: Runtime[Context]) -> Command:
    model, settings = runtime.context.model, runtime.context.settings
    case_file = state.get("case_file", [])

    # First visit: pick a specialist.
    if not case_file:
        routing: Routing = model.with_structured_output(Routing).invoke(
            [SystemMessage(SUPERVISOR), HumanMessage(_context(state) + "\n\nWho should look first?")]
        )
        target = routing.specialist if routing.specialist in ROSTER else "general"  # code guardrail
        return Command(goto=target, update={"hops": 1, "briefing": routing.briefing})

    # Later visits: the budget is checked in code, before asking the model anything.
    last, hops = case_file[-1], state.get("hops", 1)
    if hops >= settings.max_hops:
        return Command(goto="conclude", update={"stop_reason": f"Hop budget of {settings.max_hops} reached."})

    review: Review = model.with_structured_output(Review).invoke(
        [SystemMessage(SUPERVISOR),
         HumanMessage(_context(state) + f"\n\n{last.specialist} just reported. Accept, reroute, or stop?")]
    )
    decision, target = review.decision, review.next_specialist

    # Code guardrails on the model's judgement.
    if decision == "accept" and last.outcome == "not_mine":
        decision, target = ("reroute", last.handoff_to) if last.handoff_to in ROSTER else ("stop", None)
    if decision == "reroute" and (target not in ROSTER or target == last.specialist):
        decision = "stop"

    if decision == "reroute":
        briefing = review.briefing or last.handoff_reason or last.root_cause
        return Command(goto=target, update={"hops": hops + 1, "briefing": briefing})
    stop_reason = None if decision == "accept" else review.reason
    return Command(goto="conclude", update={"stop_reason": stop_reason})


INVESTIGATOR = build_investigator()


def make_specialist(name: str):
    """A node that runs the investigator subgraph as `name`, then reports to the supervisor."""

    def specialist(state: TeamState, runtime: Runtime[Context]) -> Command:
        result = INVESTIGATOR.invoke(
            {
                "failure": state["failure"],
                "specialist": name,
                "briefing": state.get("briefing", "")
                + (f"\nLead from memory, verify it: {state['memory']}" if state.get("memory") else ""),
                "case_file": case_file_text(state.get("case_file", [])),
                "operator_note": state.get("operator_note"),
            },
            context=runtime.context,
        )
        return Command(goto="supervisor", update={"case_file": result["entry"], "patches": result.get("patches", [])})

    specialist.__name__ = f"{name}_specialist"
    return specialist


def conclude(state: TeamState, runtime: Runtime[Context]) -> dict:
    prompt = _context(state)
    if state.get("patches"):
        prompt += "\n\n# Patches on the table\n" + "\n".join(
            f"- {p.item_name}/{p.part_path}: {p.explanation}" for p in state["patches"]
        )
    if state.get("stop_reason"):
        prompt += f"\n\n# Why the case stopped\n{state['stop_reason']}"
    conclusion: Conclusion = runtime.context.model.with_structured_output(Conclusion).invoke(
        [SystemMessage(SUPERVISOR), HumanMessage(prompt + "\n\nWrite the conclusion for the operator.")]
    )
    return {"conclusion": conclusion}


def add_team(builder: StateGraph) -> None:
    """Add the supervisor, one node per specialist, and conclude. The caller wires START and conclude's exit."""
    builder.add_node("supervisor", supervisor, destinations=(*ROSTER, "conclude"))
    for name in ROSTER:
        builder.add_node(name, make_specialist(name), destinations=("supervisor",))
    builder.add_node("conclude", conclude)
