"""Lesson 11: streaming what the agents are doing, live.

WHAT YOU LEARN
- `app.stream(..., stream_mode=...)` yields events while the graph runs. The modes:
  - "values":   the full state after each step (lesson 1)
  - "updates":  just what each node changed (lesson 9)
  - "messages": LLM tokens as they're generated, plus messages written to state,
                each with metadata saying which node produced it
  - "custom":   anything a node or tool sends with `get_stream_writer()` /
                `runtime.stream_writer`; team.py sends {"progress": ...} from every tool
  - "checkpoints", "tasks", "debug": for deep inspection
- Pass a LIST of modes and each event arrives as (mode, chunk).
- `subgraphs=True` includes events from inside subgraphs, even ones called inside a node.
  Each event then comes as (namespace, mode, chunk); an empty namespace means the parent.
- In a web app you forward these as Server-Sent Events, so the user watches the
  investigation instead of staring at a spinner for three minutes.

WHAT TO WATCH
The supervisor's route at the top level, each specialist's tool progress, and the
model's final answer for each specialist arriving token by token.

INTERVIEW LINE
"I stream 'updates' for the step timeline, 'custom' for tool progress and 'messages'
for tokens, with subgraphs=True so nested specialists show up too. In FastAPI that
becomes an SSE endpoint, which also avoids gateway timeouts on long runs."

RUN
    python -m lessons.l11_streaming
"""

from __future__ import annotations

from langchain_core.messages import AIMessageChunk, ToolMessage

from lessons.l09_supervisor_command import build_graph
from triage_lab.fabric import get_fabric, pick_failure
from triage_lab.models import get_model
from triage_lab.settings import get_settings
from triage_lab.team import Context


def main() -> dict:
    settings = get_settings()
    fabric = get_fabric(settings)
    context = Context(fabric=fabric, model=get_model(settings), settings=settings)
    failure = pick_failure("DataPipeline", fabric, settings)
    app = build_graph()

    counts: dict[str, int] = {}
    streaming_text = False
    for namespace, mode, chunk in app.stream(
        {"failure": failure},
        context=context,
        stream_mode=["updates", "custom", "messages"],
        subgraphs=True,
    ):
        counts[mode] = counts.get(mode, 0) + 1
        where = namespace[0].split(":")[0] if namespace else "team"

        if mode == "updates" and not namespace:  # top-level route only
            for node in chunk:
                print(f"\n>> {node}")
        elif mode == "custom":
            print(f"   [{where}] {chunk.get('progress', chunk)}")
        elif mode == "messages":
            message, meta = chunk
            if isinstance(message, AIMessageChunk) and meta.get("langgraph_node") == "agent":
                if message.tool_call_chunks:
                    for call in message.tool_call_chunks:
                        print(f"   [{where}] model asks for {call.get('name')}")
                elif message.content:
                    if not streaming_text:
                        print(f"   [{where}] says: ", end="")
                        streaming_text = True
                    print(message.content, end="", flush=True)
                    continue
            elif isinstance(message, ToolMessage):
                print(f"   [{where}] tool returned {len(str(message.content))} chars")
        if streaming_text:
            print()
            streaming_text = False

    print("\nEvents seen by mode:", counts)
    return counts


if __name__ == "__main__":
    main()
