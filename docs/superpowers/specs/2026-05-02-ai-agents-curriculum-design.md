# AI Agents Curriculum — Design Spec

**Date:** 2026-05-02
**Owner:** rkaunismaa
**Repo:** `git@github.com:rkaunismaa/Agents.git`
**Working dir:** `/home/rob/PythonEnvironments/Agents`

## Goal

A 15-notebook progressive curriculum that takes a Python-comfortable learner with prior toy-agent experience from "curious" to "capable of building production-shaped, multi-agent, MCP-integrated systems on Claude — and porting them to local models when needed."

## Decisions (from brainstorming)

| Dimension | Decision |
|---|---|
| Learner level | Intermediate (built toy agents, wants depth + know-when-to-use-what) |
| Primary provider | Anthropic (Claude) |
| Local model coverage | Late-curriculum module (Ollama / LM Studio swap-in) |
| Scope ceiling | Full ladder: single-purpose → workflows → multi-agent → autonomous |
| Framework strategy | Scratch-first; frameworks introduced only once they earn their keep (NB 11+) |
| Notebook spine | Hybrid — mostly standalone, plus 4 "research assistant" integration points |
| MCP coverage | First-class — build a server (NB 9), consume one (NB 10) |
| Runtime | JupyterLab `.ipynb` files, executed in VS Code |
| Repo visibility | Public (default; portfolio-friendly) |

## Repo layout

```
/home/rob/PythonEnvironments/Agents/
├── .agents/                  # existing uv venv (Python 3.13.12) — reused
├── pyproject.toml            # uv-managed, dependency groups per module
├── uv.lock
├── .env.example              # ANTHROPIC_API_KEY, TAVILY_API_KEY (optional)
├── .gitignore                # ignores .env, .ipynb_checkpoints, data/cache, .venv noise
├── README.md                 # curriculum index + setup instructions
├── LICENSE                   # MIT
├── notebooks/
│   ├── 01_bare_agent_loop.ipynb
│   ├── 02_tool_use.ipynb
│   ├── 03_structured_outputs_and_prompts.ipynb
│   ├── 04_react_and_extended_thinking.ipynb
│   ├── 05_planning_and_decomposition.ipynb
│   ├── 06_memory_primitives.ipynb
│   ├── 07_rag_for_agents.ipynb
│   ├── 08_evals_and_observability.ipynb
│   ├── 09_building_an_mcp_server.ipynb
│   ├── 10_consuming_mcp_in_an_agent.ipynb
│   ├── 11_orchestrator_and_subagents.ipynb
│   ├── 12_parallel_and_durable.ipynb
│   ├── 13_computer_use.ipynb
│   ├── 14_running_locally.ipynb
│   └── 15_capstone_research_assistant.ipynb
├── src/agentlab/             # shared utilities imported by notebooks
│   ├── __init__.py
│   ├── llm.py                # Anthropic client wrapper, retry/backoff
│   ├── tools.py              # reusable tool definitions + registry pattern
│   ├── memory.py             # memory primitives (introduced in NB 6)
│   ├── mcp_helpers.py        # MCP client helpers (introduced in NB 10)
│   └── tracing.py            # OTel helpers (introduced in NB 8)
├── mcp_servers/
│   └── notes_server/         # built in NB 9
├── data/                     # tiny seed corpora for RAG/eval notebooks
├── research_assistant.py     # capstone deliverable (NB 15 produces this)
└── docs/
    └── superpowers/specs/2026-05-02-ai-agents-curriculum-design.md  # this file
```

## Tooling

uv-managed Python project pinned to the existing `.agents` venv via
`UV_PROJECT_ENVIRONMENT=.agents` (set in `.envrc` or documented in README).
ipykernel registered so VS Code can pick the kernel:

```bash
uv run python -m ipykernel install --user --name agents --display-name "Python (.agents)"
```

### Dependency groups (`pyproject.toml`)

- **Core (always installed):** `anthropic`, `python-dotenv`, `pydantic`, `httpx`, `jupyterlab`, `ipykernel`, `rich`.
- **`module-b`:** `chromadb`, `pytest`, `pytest-asyncio`, `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`.
- **`module-c`:** `mcp` (official Python SDK), `claude-agent-sdk`.
- **`module-d`:** `openai` (used as a generic OpenAI-compatible client
  pointed at Ollama or LM Studio's local server — same code works for
  both). NB 13 (computer use) uses Anthropic's reference Docker
  container as the sandboxed virtual desktop — no extra Python deps
  required on the host.
- **`web-search` (optional):** `tavily-python` — alternative to Anthropic's built-in web-search tool; both shown in NB 2.

A learner can install only what's needed for the module they're on:
`uv sync --group module-b`.

## Curriculum

✦ marks the four "research assistant" spine integration points.

### Module A — Foundations (NB 1-3)

**NB 1 — The bare agent loop, from scratch.**
Build a ~30-line agent: `messages.create` in a loop, parse `tool_use`,
run the tool, feed back `tool_result`. One hardcoded calculator tool.
No SDK abstractions beyond the raw client. Endpoint: a working agent
that demystifies what every framework hides.

**NB 2 — Tool use, properly.**
Multi-tool agents. Pydantic-validated schemas, the tool-registry pattern
(lands in `src/agentlab/tools.py`), tool errors and recovery, parallel
tool calls. Tools introduced: `web_search` (Tavily AND Anthropic's
built-in web search, side by side), `fetch_url`, `read_file`.

**NB 3 — ✦ Structured outputs & prompt design.**
System prompts for agents (vs. chat), Pydantic-validated structured
outputs, when to use tool-shaped output vs. JSON output, prompt
patterns. *Spine debut:* research assistant takes a question, returns
`Answer{ summary: str, citations: list[Citation] }`.

### Module B — Agentic workflows (NB 4-8)

**NB 4 — ReAct & extended thinking.**
Implement ReAct on top of NB 1's loop. Compare with Claude's extended
thinking. When explicit reasoning helps; when it just adds latency.

**NB 5 — Planning & decomposition.**
Plan-then-execute, subgoal generation, re-planning on error. Build a
tiny planner that produces a TODO list the agent then executes.

**NB 6 — Memory primitives.**
Short-term (conversation truncation, summarization), long-term
(key-value store), semantic recall. Built small, no framework.
Lands in `src/agentlab/memory.py`.

**NB 7 — RAG for agents.**
Tool-shaped retrieval: the agent *decides* to retrieve. Chunk → embed
→ Chroma → retrieve. Why naive RAG underperforms compared to giving
the agent a search tool it can call selectively.

**NB 8 — ✦ Evaluation + observability.**
Pytest-based agent evals (golden tasks, judge-with-LLM,
deterministic asserts on tool calls). OpenTelemetry traces of agent
runs. *Spine integration:* research assistant gets a 10-task eval set;
traces viewed in a local OTel collector.

### Module C — Multi-agent + MCP (NB 9-12)

**NB 9 — Building an MCP server from scratch.**
Use the official `mcp` Python SDK. Build a "notes server" exposing
tools + resources + prompts. Run it locally; hit it with the MCP CLI.

**NB 10 — Consuming MCP in an agent.**
Wire an MCP client into your agent loop. Auto-discover tools from a
server. Spec walkthrough: tools vs. resources vs. prompts.

**NB 11 — Orchestrator + subagents.**
Claude Agent SDK enters. Orchestrator-worker patterns. Subagent
dispatch. Where multi-agent earns its keep vs. just adding latency.

**NB 12 — ✦ Parallel work & durable execution.**
Parallel subagents, cancellation, simple checkpointing.
*Spine integration:* research assistant fans out to subagents
(researcher / summarizer / fact-checker), uses your MCP notes server.

### Module D — Autonomous & local (NB 13-15)

**NB 13 — Computer use.**
Claude's computer-use API: screenshots, mouse/keyboard tools, safety
considerations. Demo: open a browser, search, copy a result, in a
recommended sandbox.

**NB 14 — Running locally with Ollama / LM Studio.**
Swap the Anthropic client for an OpenAI-compatible local endpoint.
Which patterns translate cleanly. Where local models break (tool use
quality, long context, structured output reliability) — pragmatic,
not boosterism.

**NB 15 — ✦ Capstone — research assistant as a deployable script.**
Move the spine project out of a notebook into `research_assistant.py`:
env-driven config, CLI args, tracing on, eval suite green, README.
The "from notebook to artifact" graduation.

## Spine project — capability progression

| After NB | Can do |
|---|---|
| 3 (debut)  | Web-search-backed answers; returns Pydantic-validated `Answer`. Single tool loop. |
| 8 (workflows) | Plus: memory across conversations, retrieval from notes corpus, 10-task eval set, OTel traces. |
| 12 (multi-agent) | Plus: parallel subagent fan-out (researcher/summarizer/fact-checker), retrieval via MCP notes server. |
| 15 (capstone) | Lives as `research_assistant.py` — CLI, env config, tracing on, evals green, README. |

## Conventions

- **Imports from `src/agentlab/`, not copy-paste.** Reusable code lives
  in the package; notebooks contain pedagogy + experiments. Improvements
  propagate.
- **Each notebook is re-runnable top-to-bottom.** No hidden state
  across cells.
- **Notebook structure (consistent):** Goal → Concepts → Build →
  Try it → Reflect (3-5 bullets).
- **API keys via `.env`** (`.env.example` committed, `.env` ignored).
  `python-dotenv` loaded in the first cell of each notebook.
- **Tracing off by default in NB 1-7**, on from NB 8 once the learner
  knows what to look at.
- **Cost estimate at top of each notebook.** Computed from typical
  model + token usage. Notebooks are designed to be cheap to run; the
  expensive ones (multi-agent, capstone) flag this prominently and
  default to a small Haiku-class model where it's pedagogically
  acceptable.
- **Tone:** show, don't lecture. Code-first; prose minimal; comments
  only where the *why* isn't obvious from the code.

## Non-goals

- Fine-tuning / model training.
- Production deployment infra (Docker, K8s, serverless).
- Frontend / chat UIs.
- Voice / vision agents beyond NB 13 computer use.
- Comparative framework tour (LangGraph / CrewAI / AutoGen).
- Agent safety / alignment beyond the practical guardrails of NB 13.

## Success criteria

The curriculum is "done" when:

1. All 15 notebooks run top-to-bottom with no errors against a fresh
   `.agents` venv and the documented dependency groups.
2. The capstone script `research_assistant.py` runs from CLI, passes
   its eval suite, and emits OTel traces.
3. The repo is pushed to `git@github.com:rkaunismaa/Agents.git` with a
   README that lets a stranger get from clone to NB 1 working in under
   ten minutes.
4. The MCP `notes_server` runs standalone and is consumable by the
   curriculum's agent.

**Nice-to-have:** the MCP `notes_server` is also consumable by Claude
Code (i.e. configured via `~/.claude/settings.json`) — proves the
server speaks real MCP, not a private dialect.
