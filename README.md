# Agents — building AI agents on Claude, from scratch

A 15-notebook progressive curriculum that goes from a hand-rolled agent
loop in `while True: ...` form to multi-agent systems with MCP servers,
retrieval, evals, and local-model fallbacks.

The bias is **scratch-first**: build the bare loop yourself before any
framework shows up, so you understand what frameworks are hiding from
you. Frameworks (the Claude Agent SDK, MCP) earn their keep before they
appear.

## Status

| Module | Notebooks | State |
|---|---|---|
| **A · Foundations** | [01 bare agent loop](notebooks/01_bare_agent_loop.ipynb) · [02 tool use](notebooks/02_tool_use.ipynb) · [03 structured outputs](notebooks/03_structured_outputs.ipynb) | ✅ shipped |
| **B · Workflows** | [04 ReAct & extended thinking](notebooks/04_react_and_extended_thinking.ipynb) · [05 planning](notebooks/05_planning_and_decomposition.ipynb) · [06 memory](notebooks/06_memory_primitives.ipynb) · [07 RAG](notebooks/07_rag_for_agents.ipynb) · [08 evals + observability](notebooks/08_evals_and_observability.ipynb) | ✅ shipped |
| **C · Multi-agent + MCP** | [09 build MCP server](notebooks/09_building_an_mcp_server.ipynb) · [10 consume MCP](notebooks/10_consuming_mcp_in_an_agent.ipynb) · [11 orchestrator + subagents](notebooks/11_orchestrator_and_subagents.ipynb) · [12 ✦ parallel + durable](notebooks/12_parallel_and_durable.ipynb) | ✅ shipped |
| **D · Autonomous + local** | [13 computer use](notebooks/13_computer_use.ipynb) · [14 local models](notebooks/14_running_locally.ipynb) · [15 ✦ capstone](notebooks/15_capstone_research_assistant.ipynb) | ✅ shipped |

A "research assistant" spine project recurs at NB 03, 08, 12, and 15 to
integrate the patterns. At NB 03 it returns Pydantic-validated
`Answer` objects with citations; by NB 15 it's a deployable script with
subagents and an MCP-backed corpus.

## What Module A teaches (the part that's shipped)

- **[NB 01 — bare agent loop](notebooks/01_bare_agent_loop.ipynb)**
  A hand-rolled agent loop using only the bare Anthropic SDK. One
  `while`, one `add` tool, no abstractions. Demystifies what "agent"
  actually means at the API level.
- **[NB 02 — tool use, properly](notebooks/02_tool_use.ipynb)**
  Graduates to a reusable `ToolRegistry` (function signature → JSON
  schema) and `run_agent_loop`. Runs three tools (`read_file`,
  `fetch_url`, optional Tavily `web_search`), shows Anthropic's
  *managed* `web_search_20250305` side-by-side with a local tool, and
  demonstrates how tool errors are fed back to the model so it can
  recover instead of crashing.
- **[NB 03 — structured outputs](notebooks/03_structured_outputs.ipynb)**
  Stops parsing free-form text. Uses tool-use coercion
  (`Answer.model_json_schema()` *as* a tool's input schema) for
  near-100% schema compliance, then runs the same task with a
  prompt-only "please return JSON" approach to show why you don't go
  back. The recurring research-assistant project debuts here.

Full design: [`docs/superpowers/specs/2026-05-02-ai-agents-curriculum-design.md`](docs/superpowers/specs/2026-05-02-ai-agents-curriculum-design.md).

## What Module B teaches

- **[NB 04 — ReAct & extended thinking](notebooks/04_react_and_extended_thinking.ipynb)**
  Three takes on the ReAct pattern: classic Yao-2022 prompt-parsed
  loop, then tool-use as ReAct (no parsing), then Claude's extended
  thinking. Same multi-hop demo task across all three so the
  tradeoffs are obvious side-by-side.
- **[NB 05 — Planning & decomposition](notebooks/05_planning_and_decomposition.ipynb)**
  Plan-then-execute with a Pydantic `Plan` model, a tool-use-coerced
  planner, an executor, a re-planner on failure, and a synthesizer
  that returns a structured `Answer`. Demo: "compare httpx, aiohttp,
  requests; recommend one."
- **[NB 06 — Memory primitives](notebooks/06_memory_primitives.ipynb)**
  Three memory shapes (`ConversationBuffer`, `KeyValueMemory`,
  `SemanticMemory`) live in `agentlab.memory`. The research-assistant
  spine gets `KeyValueMemory` so it can answer follow-ups about
  earlier-stated context.
- **[NB 07 — RAG for agents](notebooks/07_rag_for_agents.ipynb)**
  Self-referential corpus over `notebooks/_src/*.py` and
  `docs/superpowers/specs/*.md`. Compares naive top-k RAG (always
  inject context) with tool-shaped retrieval (the agent decides when
  to retrieve), showing why the tool shape preserves agency.
- **[NB 08 — ✦ Evals + observability](notebooks/08_evals_and_observability.ipynb)**
  Three eval styles on a 10-task spine eval set: deterministic asserts
  on tool calls + Pydantic shape, LLM-as-judge with rubric,
  reference-based exact/substring. Pytest suite at
  `tests/eval/test_research_assistant.py`. OpenTelemetry traces via
  `ConsoleSpanExporter`, with a Jaeger appendix.

## What Module C teaches

- **[NB 09 — Building an MCP server](notebooks/09_building_an_mcp_server.ipynb)**
  Builds the same notes server twice: first using the low-level `Server`
  class (~80 lines, protocol fully visible), then using `FastMCP`
  (decorator-based, ~30 lines). Adds a Prompt to show the third MCP
  primitive. Appendix covers Streamable HTTP transport.
- **[NB 10 — Consuming MCP in an agent](notebooks/10_consuming_mcp_in_an_agent.ipynb)**
  Wires an MCP client into the agent loop via `MCPToolRouter`
  (extracted to `agentlab.mcp_helpers`). Auto-discovers tools from the
  server, routes Claude's tool calls through `ClientSession`, and
  demonstrates the MCP-earning-its-keep moment: notes written in one
  session are still there after a cold-started server.
- **[NB 11 — Orchestrator + sequential subagents](notebooks/11_orchestrator_and_subagents.ipynb)**
  Builds orchestrator-worker from scratch (Stage A) then rebuilds the
  same shape with the Claude Agent SDK (Stage B) so the abstraction is
  visible. Three roles — researcher, summarizer, ranker — with
  token-isolated contexts. The `Orchestrator` + `Subagent` types are
  extracted to `agentlab.spine`.
- **[NB 12 — ✦ Parallel + durable spine](notebooks/12_parallel_and_durable.ipynb)**
  Closes the spine progression. Extends `Orchestrator` with
  `asyncio.gather` fan-out, per-worker `asyncio.wait_for` cancellation,
  and JSONL checkpointing (success-only, so failed workers retry on
  re-run). Demonstrates MCP-backed persistence (checkpoint → notes
  server survives kernel restart) and OTel span trees
  (`agent.run → subagent.{role} → llm.complete`).

**Design decisions made in Module C:**
- `MCPToolRouter` discovered from `ClientSession` rather than hard-coded
  tool lists — keeps the agent loop MCP-server-agnostic.
- Manual orchestrator preferred over the Agent SDK for NB 12 because
  explicit task handles make cancellation and checkpointing
  straightforward; constraint-fit, not framework-rejection.
- Checkpoint semantics: only successful results are cached, so partial
  failures auto-retry on the next `run_async` with the same `run_id`.
- Rate-limit backoff uses jitter (`delay + uniform(0, 0.5×delay)`) to
  desynchronise parallel workers and avoid synchronized retry storms.

## What Module D teaches

- **[NB 13 — Computer use](notebooks/13_computer_use.ipynb)**
  Starts Anthropic's reference Docker container as a sandboxed virtual
  desktop. The `claude_cu_loop` function handles the screenshot→action
  cycle: Claude receives PNG images, returns `computer` tool-use blocks,
  and the loop executes click/type/key via `docker exec xdotool`. Three
  tool types covered: `computer_20241022`, `text_editor_20241022`,
  `bash_20241022`.
- **[NB 14 — Running locally](notebooks/14_running_locally.ipynb)**
  Swaps `get_client()` for an `openai.OpenAI` client pointed at
  `localhost:11434` (Ollama). Compares tool-use compliance, structured
  output reliability, and response quality on three eval tasks side by
  side (Claude vs `llama3.2`). LM Studio shown as a one-line endpoint
  swap — the code is identical.
- **[NB 15 — ✦ Capstone](notebooks/15_capstone_research_assistant.ipynb)**
  Graduates the spine to `research_assistant.py` — a `typer` CLI at the
  repo root. `--run-id` resumes from JSONL checkpoint; `--trace` wires
  OTel spans; `--model` and `--checkpoint-dir` are env-overridable. The
  notebook runs the eval suite via subprocess as the final green-check.

## Quickstart

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv) ≥ 0.11.

```bash
git clone git@github.com:rkaunismaa/Agents.git
cd Agents

# Pin uv to the bundled .agents venv (the project's .envrc does this if you use direnv).
export UV_PROJECT_ENVIRONMENT=.agents

uv sync --extra dev --extra module-b --extra module-c --extra module-d

cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY.
# Optional: TAVILY_API_KEY enables NB 02 Step 3's third-party web search.

# Register the kernel so VS Code's notebook UI can find it by name:
.agents/bin/python -m ipykernel install --user --name agents --display-name "Python (.agents)"
```

Then open `notebooks/01_bare_agent_loop.ipynb` in VS Code and pick the
**Python (.agents)** kernel.

## Repo layout

```
notebooks/                 # .ipynb files you execute
notebooks/_src/            # jupytext-format .py source — canonical content
notebooks/_common.py       # shared helpers: load_env, cost_banner, chdir_to_repo_root
src/agentlab/              # the small library imported by every notebook
  ├ llm.py                 # Anthropic client wrapper + run_agent_loop
  ├ tools.py               # ToolRegistry + schema generation
  ├ types.py               # Pydantic models (Answer, Citation)
  ├ memory.py              # ConversationBuffer + KeyValueMemory + SemanticMemory
  ├ mcp_helpers.py         # MCPToolRouter — auto-discovers + routes MCP tool calls
  └ spine.py               # Orchestrator + Subagent + WorkerResult (multi-agent spine)
research_assistant.py      # capstone CLI — typer wrapper around agentlab.spine
tests/                     # pytest tests for agentlab + eval suite (tests/eval/)
data/                      # small seed files + eval_tasks.jsonl
mcp_servers/               # MCP servers (added in Module C)
docs/superpowers/specs/    # design specs
docs/superpowers/plans/    # implementation plans
```

## Editing notebooks

Notebook content is canonical in `notebooks/_src/*.py` (jupytext
"percent" format). The `.ipynb` files in `notebooks/` are generated.
After editing a `.py` source, regenerate its `.ipynb`:

```bash
.agents/bin/jupytext --sync notebooks/_src/02_tool_use.py
```

The pairing is configured in `pyproject.toml` under `[tool.jupytext]`.
If you edit a notebook in VS Code first, run `--sync` against the
`.ipynb` to push changes back to the `.py`.

## Tests

```bash
.agents/bin/pytest -v
```

Tests cover the `agentlab` library. The notebooks are verified by
running them in VS Code.

## Costs

Each notebook prints a cost banner at the top with an estimate.
Module A end-to-end is roughly **$0.05–0.10**; Module B end-to-end is
**$0.10–0.20** (the eval suite drives most of that — opt in with
`pytest -m eval`); Module C end-to-end is **$0.15–0.25** (the parallel
fan-out runs in NB 12 are the expensive steps). Module D end-to-end is **$0.20–0.40** (computer use image tokens in NB 13
and the eval suite in NB 15 drive most of that).

## License

MIT — see [LICENSE](LICENSE).
