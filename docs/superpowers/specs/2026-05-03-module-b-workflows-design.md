# Module B — Agentic Workflows (Design Spec)

**Date:** 2026-05-03
**Owner:** rkaunismaa
**Parent spec:** [`2026-05-02-ai-agents-curriculum-design.md`](2026-05-02-ai-agents-curriculum-design.md)
**Branch:** `feature/phase-0-module-a` (continuation; new branch on commit if desired)

## Goal

Implement Module B (NB 04-08) of the curriculum: agentic workflows that
take a learner from a single tool-using loop (end of Module A) to a
research assistant with memory, retrieval, evals, and observability.

## Decisions (from brainstorming, 2026-05-03)

| # | Dimension | Decision |
|---|---|---|
| 1 | Embedding model | Local — `sentence-transformers` (`all-MiniLM-L6-v2`). Free, offline-after-first-pull, matches Chroma's default family. |
| 2 | OTel scope (NB 08) | `ConsoleSpanExporter` primary; Jaeger-via-Docker as appendix only. NB runs end-to-end with no extra infra. |
| 3 | Memory module shape | Three small classes in `src/agentlab/memory.py`: `ConversationBuffer`, `KeyValueMemory`, `SemanticMemory`. Each ~50–80 lines. |
| 4 | Spine project location | `research()` stays inline in NBs through Module B; redefined per-NB to layer in memory (NB 06) and traces (NB 08). Formal extraction to `agentlab/research.py` deferred to NB 15. |
| 5 | Eval methodology | Three styles, side by side, inline in NB 08: deterministic asserts on tool calls + Pydantic shape, LLM-as-judge with rubric, reference-based exact/substring. Pytest suite under `tests/eval/`. |
| 6 | RAG corpus | Self-referential — chunk + embed `notebooks/_src/*.py` + `docs/superpowers/specs/*.md`. Re-indexes on each run. No external corpus, no licensing footnotes. |
| 7 | ReAct framing (NB 04) | Three-stage progression: (1) classic Yao-2022 prompt-parsed ReAct, (2) tool-use-as-ReAct (no parsing), (3) extended thinking. Tradeoff discussion at the end. |
| 8 | NB 05 demo task | "Compare 3 candidate libraries (httpx, aiohttp, requests) and recommend one for an async-first project." Re-planning triggered on tool failure / empty search. |

## Notebooks

### NB 04 — ReAct & extended thinking

**Goal:** show that "ReAct" in 2026 is a layered concept and that the
modern Claude tool-use loop *is* ReAct.

**Demo task across all three stages:** *"What's the population of the
capital of the country that won the most recent men's football World
Cup?"* — a multi-hop question that benefits from explicit
intermediate reasoning. Uses `web_search` from NB 02's tool registry.

**Structure:**
1. **Stage A — verbatim Yao-2022 ReAct.** Prompt the model to emit
   `Thought:` / `Action:` / `Observation:` text. Parse with regex.
   Drive a custom loop. ~20 lines, deliberately brittle.
2. **Stage B — tool-use as ReAct.** Same task, replace prompt-parsing
   with `tool_use` blocks. Reasoning lives in text content blocks
   between tool calls; no parsing. Use `run_agent_loop` from NB 02.
3. **Stage C — extended thinking.** Same task again, with
   `thinking={"type": "enabled", "budget_tokens": 4096}`. Show the
   server-side reasoning trace, no loop control.
4. **Reflect:** when explicit reasoning earns its tokens vs. just adds
   latency. Rule of thumb: tool-use ReAct is the default; extended
   thinking pays off on multi-step planning, not on lookups.

**Cost target:** ~$0.02 per full run.

### NB 05 — Planning & decomposition

**Goal:** plan-then-execute, with re-planning when steps fail.

**Demo task:** *"Compare httpx, aiohttp, and requests for an
async-first Python project; recommend one."*

**Structure:**
1. **Step 1 — define the plan type.** Pydantic `Plan { steps: list[Step] }`
   where `Step = { description, tool, args, status }`.
2. **Step 2 — planner.** Single Claude call with tool-use coercion
   (same trick as NB 03's `submit_answer`): the planner *must* call
   `submit_plan` with the validated `Plan`.
3. **Step 3 — executor.** Walk the plan; for each step, call the
   relevant tool via `run_agent_loop`. Mark step `done` or `failed`.
4. **Step 4 — re-planner.** On step failure (e.g. `web_search` returns
   nothing for an obscure third candidate), feed the failed plan back
   to the planner and ask for a revision. One revision pass max.
5. **Step 5 — synthesizer.** Final Claude call: take the executed plan
   and produce an `Answer` (reuse NB 03's structured output).
6. **Reflect:** when a plan helps (multi-step, parallelizable
   subgoals) vs. when the agent's natural loop is already enough.

**Cost target:** ~$0.03 per full run.

### NB 06 — Memory primitives

**Goal:** introduce the three memory shapes; layer `KeyValueMemory`
into the spine project.

**Structure:**
1. **Step 1 — `ConversationBuffer`.** Build it from scratch in the NB
   first (~10 lines), then import the library version. Methods:
   `append(message)`, `truncate(max_tokens)`,
   `summarize(client) -> str`. Show truncation losing context;
   summarization preserving it at a price.
2. **Step 2 — `KeyValueMemory`.** In-memory dict + JSON persistence.
   `set / get / save / load`. Show how the agent uses it: a tool
   `remember(key, value)` and `recall(key) -> str` that wrap the
   memory.
3. **Step 3 — `SemanticMemory`.** Chroma + sentence-transformers.
   `add(text, metadata)`, `query(text, top_k)`. Show the difference
   between recalling by exact key vs. recalling by *meaning*.
4. **Step 4 — spine integration.** Redefine `research()` inline with
   `KeyValueMemory` attached. Demo: ask a question, ask a
   follow-up that requires the previous answer ("what about its
   licensing model?"); without memory the agent doesn't remember.
5. **Reflect:** which memory shape fits which problem. Forecast NB 07
   (semantic recall as a *retrieval tool*).

**Cost target:** ~$0.03 per full run.

### NB 07 — RAG for agents

**Goal:** retrieval as a *tool the agent decides to use*, not as
unconditional context injection.

**Structure:**
1. **Step 1 — build the corpus.** Glob `notebooks/_src/*.py` and
   `docs/superpowers/specs/*.md`. Chunk by paragraphs / cells (target
   ~500 tokens). Use sentence-transformers to embed; index in Chroma.
   Print corpus stats (file count, chunk count, embedding dim).
2. **Step 2 — naive RAG.** Always inject top-3 chunks for *every*
   query. Run two questions: one where the corpus has the answer
   ("how does NB 02 register tools?"), one where it doesn't ("what's
   today's weather?"). Show the latter wastes tokens and confuses
   the model.
3. **Step 3 — retrieval as a tool.** Define `search_notes(query) -> str`
   in the tool registry, backed by `SemanticMemory.query()`. Re-run
   both questions through `run_agent_loop`. Agent calls
   `search_notes` for the first, ignores it for the second.
4. **Step 4 — citations.** Add a `Citation { source_path, chunk_id }`
   field to `search_notes` output; the agent must cite which chunk it
   used. Reuse the NB 03 `Answer` model.
5. **Reflect:** why naive RAG is the wrong default for agents.

**Cost target:** ~$0.03 per full run (plus one-time ~$0 first-run
embedding-model download cost).

### NB 08 — ✦ Evals + observability (spine integration)

**Goal:** show three eval styles + OpenTelemetry tracing on the spine
project. Make "is the agent working?" a measurable question.

**Structure:**
1. **Step 1 — eval task set.** `data/eval_tasks.jsonl` with 10 tasks.
   Each task has: `id`, `question`, `expected_tool_calls` (list of
   tool names that must appear), `reference_answer` (optional, for
   substring match), `judge_rubric` (string, for LLM-judge).
2. **Step 2 — deterministic eval.** For each task, run `research()`
   and assert: `Answer` validates, expected tools were called, at
   least one citation present. Pass / fail per task.
3. **Step 3 — LLM-as-judge eval.** A separate Haiku call rates the
   summary 1–5 against the rubric. Surface per-task scores; flag
   anything ≤ 2.
4. **Step 4 — reference-based eval.** For tasks with a reference
   answer, substring or exact match. Skip tasks without a reference.
5. **Step 5 — tracing.** Wrap `research()` with OTel spans:
   `agent.run`, `tool.web_search`, `llm.complete`. Use
   `ConsoleSpanExporter`. Show one trace pretty-printed.
6. **Step 6 — pytest suite.** `tests/eval/test_research_assistant.py`
   runs the same evals via pytest. Demonstrate `pytest -v
   tests/eval/`.
7. **Appendix — Jaeger.** One-cell instructions to swap to OTLP
   exporter pointed at a local Jaeger Docker container, with the
   `docker run jaegertracing/all-in-one` one-liner.
8. **Reflect:** each eval style's failure mode (deterministic = false
   negatives on creative answers; LLM-judge = expensive + flaky;
   reference = doesn't fit open-ended tasks). Use them in
   combination.

**Cost target:** ~$0.05–0.10 per full run (eval suite drives most of
this; uses Haiku for judges).

## Library additions

### `src/agentlab/memory.py` (new)

```python
class ConversationBuffer:
    def __init__(self, max_tokens: int = 4096): ...
    def append(self, message: dict) -> None: ...
    def truncate(self) -> list[dict]: ...
    def summarize(self, client: Anthropic) -> str: ...
    def messages(self) -> list[dict]: ...

class KeyValueMemory:
    def __init__(self): ...
    def set(self, key: str, value: Any) -> None: ...
    def get(self, key: str, default: Any = None) -> Any: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...

class SemanticMemory:
    def __init__(self, collection_name: str = "default",
                 embedder_model: str = "all-MiniLM-L6-v2"): ...
    def add(self, text: str, metadata: dict | None = None) -> str: ...
    def query(self, text: str, top_k: int = 3) -> list[Match]: ...

class Match(BaseModel):
    text: str
    score: float
    metadata: dict
```

**TDD discipline.** Same as Module A: write `tests/test_memory.py`
first, watch it fail, implement, watch it pass. Each class gets:
construction test, happy-path test, edge-case test (empty buffer,
missing key, empty corpus).

### `tests/eval/` (new directory)

- `tests/eval/__init__.py` — empty.
- `tests/eval/conftest.py` — fixtures: `eval_tasks` (loads JSONL),
  `client` (real Anthropic client gated on `ANTHROPIC_API_KEY`).
- `tests/eval/test_research_assistant.py` — three test functions, one
  per eval style, parameterised over `eval_tasks`.

Eval tests are slow + cost real money. Mark with `@pytest.mark.eval`
and exclude from default pytest run; opt in with `pytest -m eval`.

### `data/eval_tasks.jsonl` (new)

10 tasks. Mix of: factual lookups (deterministic-friendly), explainer
questions (LLM-judge friendly), known-answer questions (reference
friendly). Examples:

```jsonl
{"id":"t01","question":"What is the latest stable Python release?","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Cites a recent Python.org release; gives a version number."}
{"id":"t02","question":"What does MCP stand for in the context of Claude tool use?","expected_tool_calls":["web_search"],"reference_answer":"Model Context Protocol","judge_rubric":"Defines MCP correctly; explains it's a protocol for tool/resource exposure."}
```

(Full 10-task set drafted as part of the NB 08 implementation task in
the forthcoming Module B implementation plan.)

## Dependency changes

`pyproject.toml` `[module-b]` extra:

```toml
module-b = [
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",   # NEW
    "opentelemetry-api>=1.27.0",
    "opentelemetry-sdk>=1.27.0",
    "opentelemetry-exporter-otlp>=1.27.0",
]
```

`uv sync --extra dev --extra module-b` after the change. The
sentence-transformers dep pulls in `torch`; first install is large
(~1.5 GB). Document this in the NB 06 cost banner.

## File map

```
src/agentlab/
  ├ memory.py                    # NEW — three memory classes
  └ (existing files unchanged)

notebooks/_src/
  ├ 04_react_and_extended_thinking.py    # NEW
  ├ 05_planning_and_decomposition.py     # NEW
  ├ 06_memory_primitives.py              # NEW
  ├ 07_rag_for_agents.py                 # NEW
  └ 08_evals_and_observability.py        # NEW (✦ spine integration)

notebooks/
  └ (paired .ipynb files generated by jupytext sync)

tests/
  ├ test_memory.py                       # NEW (TDD)
  └ eval/
      ├ __init__.py                      # NEW
      ├ conftest.py                      # NEW
      └ test_research_assistant.py       # NEW

data/
  └ eval_tasks.jsonl                     # NEW (10 tasks)

docs/superpowers/
  ├ specs/2026-05-03-module-b-workflows-design.md   # this file
  └ plans/2026-05-03-module-b-implementation.md     # next step (writing-plans)
```

## Implementation discipline

- **TDD for `memory.py`.** Tests first. Same RGB cycle as Module A.
- **Subagent-driven NB implementation.** Implementer subagent writes
  the `.py` source from this spec; reviewer subagent checks against
  the spec before sync. Pattern from Module A NB 02/03.
- **Confirm before each notebook.** User preference (persistent
  memory): never auto-run notebooks; ask before kicking off NB 04,
  05, 06, 07, 08.
- **Commit per notebook.** One commit per NB+spec-source pair, plus
  separate commits for: dep bump, `memory.py` + tests, eval suite +
  data file.

## Cost estimate

Module B end-to-end: **$0.10–0.20** at current Sonnet 4.6 pricing.

| Notebook | Estimate |
|---|---|
| NB 04 | $0.02 |
| NB 05 | $0.03 |
| NB 06 | $0.03 |
| NB 07 | $0.03 |
| NB 08 | $0.05–0.10 (eval suite drives this; Haiku for judges) |

The sentence-transformers model download (~80 MB) is a one-time cost
on first NB 06 / NB 07 run; no API cost.

## Out of scope (deferred or owned by a later module)

- **Spine extraction to `agentlab/research.py`** — NB 15 (capstone).
- **Async / parallel agent execution** — NB 12 (Module C).
- **MCP integration of the notes corpus** — NB 09–12 (Module C).
- **Streaming UIs / web frontends** — never, by spec.
- **Production deployment** — NB 15 only as a CLI script.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| `sentence-transformers` install pulls torch (~1.5 GB), surprises learners | Document in NB 06 cost banner; mention `module-b` extra is opt-in. |
| Eval suite costs accumulate during dev iteration | `@pytest.mark.eval` excluded from default; document `pytest -m eval` opt-in. |
| Self-referential RAG corpus drifts as curriculum evolves | Re-index on each run; corpus is regenerated, not pinned. |
| Extended thinking adds large hidden token cost in NB 04 | Cap `budget_tokens` at 4096; show the cost in the cell output. |
| LLM-judge scores are flaky | Use Haiku (cheap), run twice, surface the variance in NB 08. |

## Success criteria

Module B is complete when:

1. All 5 notebooks execute end-to-end against the live Anthropic API
   from a fresh clone (after `uv sync --extra dev --extra module-b`
   and `.env` setup).
2. `pytest tests/test_memory.py` passes (`agentlab.memory` is real
   library code with green tests).
3. `pytest -m eval tests/eval/` passes against the spine eval set
   (deterministic + reference-based; LLM-judge is informational).
4. `README.md` Module B row flips from ⏳ planned to ✅ shipped, with
   per-notebook descriptions and links.
5. Cost banners on each NB match the targets in this spec
   within ±$0.02.

## Next steps

After this spec is approved, invoke the `superpowers:writing-plans`
skill to produce a detailed implementation plan with task breakdown
mirroring Phase 0/1's plan structure.
