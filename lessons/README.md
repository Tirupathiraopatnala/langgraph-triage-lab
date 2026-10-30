# The lessons, and what to say about each in an interview

Run them in order: `python -m lessons.l01_state_graph`, then `l02`, and so on. Read the
guide at the top of each file first, run it, then read the code. Each lesson is under
200 lines.

## How LangGraph thinks, in five sentences

A graph is a state machine over a shared, typed **state**. **Nodes** are functions that
return partial updates; **reducers** decide how updates merge. **Edges** (fixed,
conditional, or a node's own `Command`) decide what runs next, and everything that runs
in the same step is a **superstep**. A **checkpointer** saves state after every superstep,
which is what makes pausing, resuming, human approval and time travel possible. Things
that aren't data (clients, models, settings) travel in **runtime context**, not in state.

## Lesson by lesson

**1. State graph.** A node is a function from state to a partial update; the graph merges
updates and follows edges. `compile()` validates the wiring and returns a Runnable.

**2. Reducers and routing.** Reducers decide how concurrent writes to one key merge, which
is what makes parallel branches safe. Without one: `InvalidUpdateError`. Routing is a plain
function on state, so it's easy to unit test.

**3. Structured output.** Never route on free text. The model returns a Pydantic schema
with `Literal` fields; the conditional edge reads a field. Validation fails loudly at the
node that produced the bad output.

**4. Tools and the agent loop.** The ReAct loop is two nodes and a conditional edge: the
model proposes tool calls, a `ToolNode` runs them, repeat until no calls or the recursion
limit. Tool docstrings are the descriptions the model reads.

**5. Context, retries, errors.** Dependencies go in runtime context, data goes in state.
Retry policies are per node and only on transient errors; an error handler takes over when
they run out. One retry layer per failure type, never stacked. (Node timeouts need async
nodes.)

**6. Checkpoints and time travel.** The checkpointer saves state per thread after every
step: resume after a crash, pause for a human, replay from any step. `update_state` on an
old checkpoint forks a branch; `invoke(None, forked_config)` re-runs from there. Checkpointed
state must be serialisable, so keep it JSON-native or allowlist your types.

**7. Human approval.** `interrupt(payload)` pauses the run (checkpointed, so it can wait for
days); `Command(resume=answer)` continues it. On resume, the node re-runs from the top, so
nothing with side effects goes before the `interrupt()`. Apply re-checks the live source
before writing.

**8. Send and subgraphs.** A conditional edge returning a list of `Send`s runs a node once
per item, in parallel (map-reduce); a reducer collects the results. A compiled graph can be
a node; only the keys parent and child share flow between them. A tool can return a
`Command` to update graph state.

**9. Supervisor with Command.** Nodes route with `Command(goto=..., update=...)`.
Specialists share a case file. The model makes judgements (route, accept, reroute); code
enforces invariants (roster, hop budget, no self-handoff).

**10. Store.** Checkpointer is per-thread short-term state; Store is cross-thread
long-term memory, namespaced like folders. A recalled fix is a lead to verify, not an
answer. Semantic search needs an embeddings index.

**11. Streaming.** `updates` for the timeline, `custom` for progress you emit yourself,
`messages` for tokens, `subgraphs=True` for nested graphs. In a web app it becomes
Server-Sent Events, which also avoids gateway timeouts on long runs.

**12. The agent.** Three kinds of persistence kept apart (records, checkpoints, memory);
ids assigned once; plans written once; approval as a compare-and-swap before resuming;
the graph proposes, acting is a separate recorded step. Read `l12_triage_agent.py` for
the full list and for "what I'd improve next".

## Questions you should now be able to answer

- What is a superstep, and why does it matter for reducers?
- Why shouldn't a Fabric client or a model object be in graph state?
- What happens to code before `interrupt()` when the run resumes?
- Checkpointer vs Store: when do you use which?
- How do you stop an agent loop from running forever? (Three answers: recursion limit,
  tool budget, hop budget.)
- How does the graph know where a `Command`-routing node can go, for drawing?
- How would you make approval safe if two people click at once?
- Where would you put retries for a 429 from Fabric, a 503 from Fabric, and a 429 from
  Azure OpenAI?
- How would you serve this from FastAPI if one analysis takes three minutes?
