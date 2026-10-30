"""Lesson 3: the first model call, with structured output.

WHAT YOU LEARN
- One place creates the model: `triage_lab.models.get_model()`. MODEL_MODE=azure gives
  you your Foundry deployment; MODEL_MODE=fake gives a scripted stand-in.
- `model.with_structured_output(Triage)` makes the model return a validated Pydantic
  object instead of free text. With OpenAI models this uses JSON-schema structured
  outputs; the result is still validated by Pydantic on the way back.
- `Literal[...]` fields constrain the answer to values your code can route on.
- `app.batch([...])` runs the graph over many inputs (in parallel threads).

WHAT TO WATCH
The model's answer drives the route. Nothing in the routing function parses free text.

INTERVIEW LINE
"I never route on free text. The model returns a schema with Literal fields, Pydantic
validates it, and the conditional edge reads a field. If validation fails, it fails
loudly at that node, not three steps later."

TRY THIS
1. Set MODEL_MODE=azure in .env and compare the fake's answers with your deployment's.
2. Add a `confidence` threshold: low-confidence triage routes to `escalate`.
3. Pass include_raw=True to with_structured_output and print the token usage.

RUN
    python -m lessons.l03_structured_output
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from triage_lab.contracts import Failure
from triage_lab.fabric import recent_failures
from triage_lab.models import get_model
from triage_lab.texts import failure_text


class Triage(BaseModel):
    """What the model must return. Field descriptions become part of the schema it sees."""

    category: Literal["missing_table", "credentials", "throttling", "other"]
    specialist: Literal["notebook", "pipeline", "lakehouse", "semantic_model", "general"] = Field(
        description="Who should investigate first"
    )
    summary: str = Field(description="One sentence for an on-call engineer")
    confidence: Literal["high", "medium", "low"]


SYSTEM = (
    "You triage Microsoft Fabric job failures. Classify the failure, say which specialist "
    "should look first, and summarise it in one sentence. Use only the facts given."
)


class State(TypedDict, total=False):
    failure: Failure
    triage: Triage
    next_step: str


MODEL = get_model()


def triage(state: State) -> dict:
    structured = MODEL.with_structured_output(Triage)
    result = structured.invoke([SystemMessage(SYSTEM), HumanMessage(failure_text(state["failure"]))])
    return {"triage": result}


def route(state: State) -> str:
    return "needs_person" if state["triage"].category == "credentials" else "needs_specialist"


def needs_person(state: State) -> dict:
    return {"next_step": "Notify the item owner: a person must fix credentials."}


def needs_specialist(state: State) -> dict:
    return {"next_step": f"Send to the {state['triage'].specialist} specialist (lesson 4 onwards)."}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("triage", triage)
    builder.add_node("needs_person", needs_person)
    builder.add_node("needs_specialist", needs_specialist)
    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", route, ["needs_person", "needs_specialist"])
    builder.add_edge("needs_person", END)
    builder.add_edge("needs_specialist", END)
    return builder.compile()


def main() -> list[dict]:
    app = build_graph()
    failures = recent_failures()
    results = app.batch([{"failure": f} for f in failures])  # one run per failure
    for result in results:
        t = result["triage"]
        print(f"\n{result['failure'].item.name}: {t.category} / {t.specialist} ({t.confidence})")
        print("   ", t.summary)
        print("    ->", result["next_step"])
    return results


if __name__ == "__main__":
    main()
