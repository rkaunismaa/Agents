# Module C — Multi-Agent + MCP (Design Spec)

**Date:** 2026-05-05
**Owner:** rkaunismaa
**Parent spec:** [`2026-05-02-ai-agents-curriculum-design.md`](2026-05-02-ai-agents-curriculum-design.md)
**Branch:** `feature/module-c` (off `main` at `18ac202`)

## Goal

Implement Module C (NB 09-12) of the curriculum: take a learner from a
single-agent + RAG-aware research assistant (end of Module B) to a
multi-agent system with parallel subagent fan-out, MCP-backed memory,
and an extracted spine library that NB 15 can wrap as a CLI.

## Decisions (from brainstorming, 2026-05-05)

| # | Dimension | Decision |
|---|---|---|
| 1 | MCP server purpose | Notes server — replaces NB 06's in-process `KeyValueMemory` with out-of-process state. Tools: `add_note`, `get_note`, `list_notes`, `delete_note`. Resources: each note as `notes://<key>`. |
| 2 | NB 09 build progression | Two-stage: low-level `Server` with `@server.list_tools()` / `@server.call_tool()` handlers first, then graduate to `FastMCP` decorators. Same surface, scratch-first match. |
| 3 | MCP transport | Stdio for the main flow; Streamable HTTP as appendix in NB 09 (mirrors NB 08's "ConsoleSpanExporter primary, Jaeger appendix" pattern). |
| 4 | NB 11 orchestration | Two-stage: scratch nested-Claude orchestrator (manual dispatch tool calls + sequential worker runs) → rebuild with `claude-agent-sdk`. |
| 5 | NB 11 parallelism | Sequential only; parallelism is reserved for NB 12. |
| 6 | NB 12 spine extraction | Extract orchestrator + workers to `src/agentlab/spine.py`. NB 12 imports + wires + traces + evals. NB 15 capstone reduces to CLI/env wrapping. |
| 7 | Eval coverage | Extend `data/eval_tasks.jsonl` with 5 parallel-friendly tasks (multi-source comparison, ranked summarization). Existing pytest harness picks them up. |
| 8 | Spine NBs | ✦ at NB 12 only (matches curriculum spec). NB 09/10/11 standalone. |
| 9 | NB 12 demos | Parallel subagent fan-out (researcher / summarizer / fact-checker), asyncio cancellation, JSONL checkpointing for resumability. |
| 10 | Branch | `feature/module-c` (new branch off `main` at `18ac202`). |

## Notebooks

### NB 09 — Building an MCP server from scratch

**Goal:** show how MCP works at the protocol level, then graduate to the
idiomatic API.

**Demo artifact:** a notes server backed by `data/notes.json` (single-file
JSON persistence, no DB).

**Structure:**

1. **Step 1 — Design the notes server.** Define the on-disk shape
   (`{"key": str, "content": str, "created_at": iso8601}`), the four
   tools, and the resource URI scheme (`notes://<key>`). No code yet —
   just the contract.
2. **Step 2 — Low-level `Server`.** Implement with the SDK's primitive
   layer:
   - `@server.list_tools()` returns tool definitions
   - `@server.call_tool()` dispatches by name (`add_note`, `get_note`,
     `list_notes`, `delete_note`)
   - `@server.list_resources()` enumerates each note as `notes://<key>`
   - `@server.read_resource()` returns content for a URI
   - Run via `stdio_server()` async context manager

   Approx 80–100 lines including persistence helpers. The point: you
   see the protocol shape — list/call/read handlers wired to a
   transport.
3. **Step 3 — Hit it with the MCP Inspector.** `mcp dev mcp_servers/notes_server.py`
   opens a browser-based inspector. Add a note, list notes, read a
   resource. No agent involved yet.
4. **Step 4 — Refactor to `FastMCP`.** Same server, same data file:
   - `@mcp.tool()` per tool function (signature → schema, like NB 02's
     `ToolRegistry`)
   - `@mcp.resource("notes://{key}")` for parameterized resources
   - Side-by-side line count: ~30 lines vs ~100. Show what FastMCP is
     hiding.
5. **Step 5 — Add a Prompt** (third MCP primitive). `@mcp.prompt()`
   `summarize_notes` returns a prompt template ingesting all notes and
   asking for a digest. Demonstrates that MCP exposes more than tools.
6. **Step 6 — Streamable HTTP appendix.** Show the one-line transport
   swap and `mcp dev` over HTTP. Pedagogy only — main flow stays stdio.
7. **Reflect:** when FastMCP is enough; when you'd reach for the
   low-level layer (custom transports, request middleware, dynamic
   tool lists).

**Cost target:** ~$0.01 (one Claude call to demo the prompt template).
Mostly local.

### NB 10 — Consuming MCP in an agent

**Goal:** wire an MCP client into the agent loop. Auto-discover tools.
Walk through the tool/resource/prompt distinction from the consumer
side.

**Structure:**

1. **Step 1 — Connect.** Spawn the NB 09 notes server as a subprocess
   via `mcp.client.stdio.stdio_client`, open a `ClientSession`. Show
   the handshake: `await session.initialize()` returns server
   capabilities + protocol version.
2. **Step 2 — Auto-discover tools → Anthropic schemas.** Iterate
   `await session.list_tools()`, convert each `Tool` to an Anthropic
   tool dict (`{name, description, input_schema}`). MCP's
   `inputSchema` is already JSON Schema, so it maps almost 1:1. Lands
   as a helper `mcp_tools_to_anthropic(session) -> list[dict]`.
3. **Step 3 — Wire into `run_agent_loop`.** Adapt NB 02's loop: when
   the model emits a `tool_use` block whose name is one of the
   MCP-discovered tools, route through `await session.call_tool(name,
   arguments)` and feed the result back as `tool_result`. Local tools
   (e.g., `web_search`) work alongside MCP tools — same registry shape.
4. **Step 4 — Demo conversation.**
   - Turn 1: *"Remember that my favorite Python testing library is
     pytest, key 'fav_test_lib'."* — agent calls `add_note` via MCP.
   - Turn 2: *"What was my favorite testing library?"* — agent calls
     `get_note` via MCP.
   - Turn 3 (cold start, fresh session): same question — same answer,
     because notes outlive the process. **MCP earning its keep.**
5. **Step 5 — Resources walkthrough.** `session.list_resources()` and
   `session.read_resource("notes://fav_test_lib")`. Tools are
   agent-driven (the model decides to call them); resources are
   app-driven (your harness fetches them and stuffs the content into
   the context).
6. **Step 6 — Prompts walkthrough.** `session.list_prompts()` +
   `session.get_prompt("summarize_notes")`. Prompts are template
   fixtures the host app can offer to the user (e.g. as slash
   commands), not things the agent calls.
7. **Step 7 — Library landing.** Helpers (`mcp_tools_to_anthropic`,
   `MCPToolRouter`) extract to `src/agentlab/mcp_helpers.py`. ~40
   lines.
8. **Reflect:** tool vs resource vs prompt mental model; MCP's
   separation of "what the agent decides" from "what the app decides"
   maps cleanly to which primitive to reach for.

**Cost target:** ~$0.02–0.03 (3 conversation turns + a small follow-up
demo).

### NB 11 — Orchestrator + sequential subagents

**Goal:** introduce orchestrator-worker. Show what `claude-agent-sdk`
is hiding by building it from scratch first, then rebuilding the same
shape with the SDK. Sequential dispatch only — parallelism is NB 12.

**Demo task across both stages:** *"Compare httpx, aiohttp, and
requests. For each, summarize what it's good at and call out one
gotcha. Then rank them for an async-first project."* Three independent
subtasks (one per library) + synthesis. Different from NB 05's
planning demo: NB 05 produced a plan that one agent executed; NB 11
dispatches separate subagent runs with their own context.

**Structure:**

1. **Step 1 — Why subagents at all?** One paragraph. Token isolation
   (each worker has its own context window, no pollution from earlier
   subtasks), role specialization (different system prompts), clean
   failure boundaries (one fails, others continue). Foreshadow NB 12's
   parallelism + cancellation argument.
2. **Step 2 — Stage A: scratch nested-Claude orchestrator.**
   - Outer agent has two tools: `dispatch_subagent(role: str, task:
     str)` and `submit_final(answer: Answer)`. Reuse NB 03's `Answer`.
   - Three role definitions as system prompts (`RESEARCHER_SYSTEM`,
     `SUMMARIZER_SYSTEM`, `RANKER_SYSTEM`).
   - When the orchestrator calls `dispatch_subagent`, the harness runs
     a fresh `messages.create` loop with the role's system prompt and
     the task as the user message; returns the worker's final text as
     the tool result.
   - Sequential — one subagent at a time.
   - Print the trace: *orchestrator → dispatch researcher → result →
     dispatch researcher → result → dispatch researcher → result →
     dispatch ranker → result → submit_final*.
   - Approx 80–100 lines.
3. **Step 3 — Stage B: rebuild with `claude-agent-sdk`.**
   - Same demo task. Same `Answer` shape.
   - Use the SDK's subagent definition mechanism with role-specific
     system prompts; let the SDK orchestrate.
   - Side-by-side cell: line count, orchestration code visible vs
     hidden, total tokens, wall-clock time.
4. **Step 4 — Reflect.** Where the SDK earns its keep (multi-tool
   subagents, longer running flows, hooks/permission needs). Where
   manual is fine (you control dispatch, debug-friendly traces, no
   extra dep). Foreshadow NB 12: "now we add parallelism +
   cancellation, and the manual loop gets harder fast."

**Cost target:** ~$0.04–0.06 (two full multi-agent runs).

**Notes:**
- No MCP in NB 11. Adding MCP + orchestration + framework swap in one
  NB blurs the lesson; NB 12 wires MCP in.
- Both stages use the same `Answer` model so output shape is
  comparable.

### NB 12 — ✦ Parallel + durable spine

**Goal:** combine MCP-backed memory + parallel subagent fan-out +
cancellation + checkpointing into the extracted spine library. Close
the ✦ Module-C spine progression.

**Architectural call:** NB 12 extends NB 11 **Stage A** (the manual
orchestrator), not Stage B (Agent SDK). Reason: parallel/cancellation/
checkpointing want explicit control over worker task handles, which
the SDK manages internally. The reflect section names this tradeoff
explicitly so the learner knows we're not retracting NB 11's lesson —
we're picking the tool that fits the constraints.

**Demo task:** *"For each of these three open-source projects (httpx,
FastAPI, Pydantic), summarize the README, fetch the latest GitHub
release, and rank them by recent activity."* Three sources × three
subtasks each = a fan-out that benefits from parallelism. Plus a
synthesis step. Reuses NB 11's `Answer` shape.

**Structure:**

1. **Step 0 — Extract NB 11 Stage A to `src/agentlab/spine.py`.** Move
   the orchestrator + workers + `Subagent` protocol + dispatch logic
   into the library. NB 12 imports `from agentlab.spine import
   Orchestrator, Subagent`. ~150–200 lines. TDD: `tests/test_spine.py`
   first, sequential-dispatch behavior pinned before async lands.
2. **Step 1 — Async workers + `asyncio.gather`.** Convert dispatch to
   parallel fan-out. Three workers run concurrently. Show the
   wall-clock difference: sequential (~30s) → parallel (~12s).
3. **Step 2 — Cancellation.** Wrap each worker in `asyncio.wait_for(task,
   timeout=60)`. If one times out, others continue; the synthesizer is
   told which workers completed. Explicit `CancelledError` handling in
   the worker. Demo: artificially slow one worker; show graceful
   degradation.
4. **Step 3 — Checkpointing.** Each worker, on completion, appends
   `{run_id, role, source, result}` to `data/checkpoints/<run_id>.jsonl`.
   `Orchestrator.resume(run_id)` skips completed workers and
   re-dispatches the rest. Demo: kill mid-run; resume; observe cached
   results.
5. **Step 4 — MCP integration.** Start the NB 09 notes server as a
   subprocess at orchestrator startup. Workers write intermediate
   findings via `add_note(key=f"{run_id}:{role}", content=...)`. The
   synthesizer pulls them via MCP. After the run, the notes outlive
   the process — a follow-up question (in a new orchestrator session)
   can recall them. **The spine's second MCP-earning-its-keep moment**
   (the first was NB 10's cross-session demo).
6. **Step 5 — OTel traces.** Instrument the spine: `agent.run` →
   `subagent.{role}` (one span per worker, parallel siblings under the
   run) → `mcp.call_tool` + `llm.complete` as children.
   `ConsoleSpanExporter` per NB 08; Jaeger appendix recapped.
7. **Step 6 — Eval extension.** Add 5 parallel-friendly tasks to
   `data/eval_tasks.jsonl` (multi-source comparisons, "rank these N",
   etc.). The existing pytest harness in
   `tests/eval/test_research_assistant.py` parametrizes over the JSONL
   — new rows get tested automatically. Add one `@pytest.mark.eval`
   test that asserts the spine emits a `parallel_dispatch=true` trace
   attribute on parallel-friendly tasks (verifies the architecture
   *is* fanning out, not just producing answers).
8. **Reflect:**
   - **Spine ✦ progression complete.** Memory (NB 06) → retrieval (NB
     07) → measurable (NB 08) → multi-agent + durable + MCP-backed
     (this NB).
   - **What NB 15 will do.** No new architecture — wrap `Orchestrator`
     as a CLI script with env-driven config, tracing on by default,
     README, and a `Makefile`-or-shell entrypoint.
   - **Why we picked manual over SDK in this NB.** Explicit task
     handles + cancellation + checkpointing are hard to thread through
     the SDK's lifecycle. Constraint-fit, not framework-rejection.

**Cost target:** **$0.10–0.15** (3-4 full spine runs across the
section + eval extension run).

## Library additions

### `src/agentlab/mcp_helpers.py` (new in NB 10)

```python
async def mcp_tools_to_anthropic(session) -> list[dict]: ...

class MCPToolRouter:
    """Wraps an MCP ClientSession; dispatches tool calls back through it."""
    def __init__(self, session): ...
    async def call(self, name: str, arguments: dict) -> str: ...
```

~40 lines. TDD: `tests/test_mcp_helpers.py` first.

### `src/agentlab/spine.py` (new in NB 12)

```python
@dataclass
class Subagent:
    role: str
    system_prompt: str

class Orchestrator:
    def __init__(self, workers: list[Subagent], mcp_session=None,
                 checkpoint_dir: Path | None = None): ...
    async def run(self, question: str, run_id: str | None = None) -> Answer: ...
    def resume(self, run_id: str) -> Answer: ...
```

~150–200 lines. TDD: `tests/test_spine.py` first, sequential-dispatch
behavior pinned before async lands.

### `mcp_servers/notes_server.py` (new in NB 09)

The FastMCP version is the canonical, runnable file. The low-level
version lives inline in NB 09 only as a teaching artifact.

## Dependency changes

None new. The `[module-c]` extra is already declared in
`pyproject.toml`:

```toml
module-c = [
    "mcp>=1.0.0",
    "claude-agent-sdk>=0.1.0",
]
```

Quickstart command becomes
`uv sync --extra dev --extra module-b --extra module-c`. (Module C
imports `agentlab.memory.SemanticMemory` only if NB 12 chooses to demo
it; otherwise `module-b` extra is technically unneeded — but
quickstart pins both for the curriculum-complete path.)

After first successful NB 11 run, pin `claude-agent-sdk` to a specific
minor in `pyproject.toml` to insulate the curriculum from API drift.

## File map

```
src/agentlab/
  ├ mcp_helpers.py                    # NEW (NB 10)
  └ spine.py                          # NEW (NB 12)

notebooks/_src/
  ├ 09_building_an_mcp_server.py      # NEW
  ├ 10_consuming_mcp_in_an_agent.py   # NEW
  ├ 11_orchestrator_and_subagents.py  # NEW
  └ 12_parallel_and_durable.py        # NEW (✦ spine)

mcp_servers/
  └ notes_server.py                   # NEW (FastMCP; canonical)

tests/
  ├ test_mcp_helpers.py               # NEW (TDD)
  └ test_spine.py                     # NEW (TDD)

data/
  ├ eval_tasks.jsonl                  # MODIFIED (+5 parallel-friendly rows)
  ├ notes.json                        # runtime; gitignored
  └ checkpoints/                      # runtime dir; gitignored

docs/superpowers/
  ├ specs/2026-05-05-module-c-multi-agent-mcp-design.md   # this file
  └ plans/2026-05-XX-module-c-implementation.md           # next step (writing-plans)
```

`.gitignore` additions: `data/notes.json`, `data/checkpoints/`.

## Implementation discipline

- **TDD for `mcp_helpers.py` and `spine.py`.** Tests first, watch
  fail, implement, watch pass. Same RGB cycle as Module A and
  `memory.py`.
- **Confirm before each notebook.** Persistent memory rule — never
  auto-run notebooks; ask before NB 09, 10, 11, 12.
- **Commit per NB + library bump pair.** Each library file gets its
  own commit ahead of the NB that consumes it. Each NB then ships in
  its own commit.
- **Branch:** `feature/module-c` off `main` at `18ac202`. Fast-forward
  merge when complete.
- **Conventional Commits + `Co-Authored-By: Claude Sonnet 4.6`**
  trailer per project history.

## Cost estimate

| Notebook | Estimate |
|---|---|
| NB 09 | $0.01 (prompt-template demo only) |
| NB 10 | $0.02–0.03 |
| NB 11 | $0.04–0.06 (two full multi-agent runs) |
| NB 12 | $0.10–0.15 (multi-source spine + eval extension) |
| **Module C end-to-end** | **$0.17–0.25** |

## Out of scope (deferred or owned by a later module)

- **CLI wrapping + env config + Makefile entry point** → NB 15
  capstone.
- **Multi-server MCP composition** (one client, multiple servers) →
  not a curriculum goal; mention in passing in NB 10 reflect.
- **OAuth / authenticated MCP HTTP transport** → enterprise-shaped,
  beyond curriculum scope.
- **Distributed checkpointing (across hosts)** → out of scope;
  single-host JSONL is sufficient for the lesson.
- **Subagent retry / circuit-breaker patterns** → noted in NB 12
  reflect as a real-world concern, not implemented.
- **Computer-use subagents** → NB 13 (Module D).
- **Local-model orchestration** → NB 14 (Module D); subagents stay on
  Claude Sonnet for Module C.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `claude-agent-sdk` API drift (early-version dep) | Pin to a specific minor in `pyproject.toml` after first successful NB 11 run; document the version in NB 11's cost banner. |
| MCP stdio subprocess management is platform-finicky | Wrap subprocess startup in a context manager (`async with notes_server_process()`); document Linux-only testing in the NB. |
| Async + cancellation race conditions in NB 12 | Cover with TDD in `tests/test_spine.py` before adding async; pin worker timeout to a value larger than typical web_search latency (default 60s). |
| Checkpoint files accumulate in `data/checkpoints/` | Gitignore the directory; commit a `.gitkeep` only. NB 12 demonstrates manual cleanup. |
| Eval suite cost balloons with 15 tasks × 3 styles | Already opt-in via `@pytest.mark.eval`; document the new total ($0.10–0.15 per full eval pass) in NB 08 cost banner update + Module C README section. |
| Orchestrator-with-SDK (NB 11 stage B) leaks SDK-internal logs into NB output | Configure SDK logging to WARN in NB 11 setup cell; leave the manual stage's verbose trace alone. |
| Notes server schema evolves; old `data/notes.json` files break | Server validates on read; on schema mismatch, refuses to start and prints a migration message naming the offending file. Never silently overwrites learner data. Document in NB 09. |

## Success criteria

Module C is complete when:

1. All 4 notebooks execute end-to-end against the live Anthropic API
   from a fresh clone (after `uv sync --extra dev --extra module-b
   --extra module-c`).
2. `pytest tests/test_mcp_helpers.py tests/test_spine.py` passes (real
   library code, green tests).
3. `pytest -m eval tests/eval/` passes against the extended 15-task
   spine eval set (deterministic + reference; LLM-judge informational).
4. The MCP notes server runs standalone via `mcp dev
   mcp_servers/notes_server.py` and via `python
   mcp_servers/notes_server.py` (subprocess for spine).
5. NB 12 spine demonstrates: parallel fan-out (visible wall-clock
   improvement), graceful cancellation (slow worker times out, others
   complete), checkpoint resume (kill mid-run, resume picks up cached
   results), MCP-backed memory (notes outlive the orchestrator
   process).
6. `README.md` Module C row flips from ⏳ planned to ✅ shipped, with
   per-notebook descriptions and links.
7. Cost banners on each NB match the targets in this spec within
   ±$0.02.

## Next steps

After this spec is approved, invoke the `superpowers:writing-plans`
skill to produce a detailed implementation plan with task breakdown
mirroring Module B's plan structure.
