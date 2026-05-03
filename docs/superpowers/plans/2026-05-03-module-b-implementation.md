# Module B — Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Module B of the curriculum (NB 04-08) — five executable JupyterLab notebooks that teach ReAct/extended thinking, planning & decomposition, memory primitives, RAG-as-tool, and evals + observability — plus the supporting `src/agentlab/memory.py` library, the `tests/eval/` pytest suite, and the `data/eval_tasks.jsonl` spine eval set.

**Architecture:** Same shape as Module A. Reusable code lands in `src/agentlab/memory.py` with TDD-driven tests. Notebooks are authored as `.py` files in `notebooks/_src/` and synced to `.ipynb` via jupytext. The spine `research()` function stays inline per-notebook (no extraction until NB 15). The eval suite has its own copy of `research()` under `tests/eval/research_under_test.py` so pytest can exercise it without forcing a premature library extraction.

**Tech Stack additions:** `chromadb` (already stubbed), `sentence-transformers` (NEW dep), `opentelemetry-api/sdk/exporter-otlp` (already stubbed). Embedding model: `all-MiniLM-L6-v2` via sentence-transformers, ~80MB local.

**Spec:** [`../specs/2026-05-03-module-b-workflows-design.md`](../specs/2026-05-03-module-b-workflows-design.md).

**Subsequent plans:** Module C (NB 09-12 — MCP + multi-agent) and Module D (NB 13-15 — autonomous + local + capstone) get their own plans after Module B ships.

---

## Files created/modified in this plan

```
# Created
src/agentlab/memory.py                            # ConversationBuffer, KeyValueMemory, SemanticMemory, Match
tests/test_memory.py                              # TDD tests for memory.py
tests/eval/__init__.py
tests/eval/conftest.py                            # pytest fixtures (eval_tasks, client)
tests/eval/research_under_test.py                 # local research() copy for the eval suite
tests/eval/test_research_assistant.py             # three eval-style tests
data/eval_tasks.jsonl                             # 10 spine eval tasks
notebooks/_src/04_react_and_extended_thinking.py
notebooks/_src/05_planning_and_decomposition.py
notebooks/_src/06_memory_primitives.py
notebooks/_src/07_rag_for_agents.py
notebooks/_src/08_evals_and_observability.py
notebooks/04_react_and_extended_thinking.ipynb    # jupytext-generated
notebooks/05_planning_and_decomposition.ipynb
notebooks/06_memory_primitives.ipynb
notebooks/07_rag_for_agents.ipynb
notebooks/08_evals_and_observability.ipynb

# Modified
pyproject.toml                                    # add sentence-transformers; bump module-b extra
README.md                                         # flip Module B row to ✅, add NB descriptions
```

---

## Task B-1: Dependency bump (sentence-transformers + module-b sync)

**Files:**
- Modify: `/home/rob/PythonEnvironments/Agents/pyproject.toml` (lines 26-31, the `module-b` extra)

- [ ] **Step 1: Add `sentence-transformers` to the `module-b` extra**

Edit `pyproject.toml`. Replace:

```toml
module-b = [
    "chromadb>=0.5.0",
    "opentelemetry-api>=1.27.0",
    "opentelemetry-sdk>=1.27.0",
    "opentelemetry-exporter-otlp>=1.27.0",
]
```

with:

```toml
module-b = [
    "chromadb>=0.5.0",
    "sentence-transformers>=3.0.0",
    "opentelemetry-api>=1.27.0",
    "opentelemetry-sdk>=1.27.0",
    "opentelemetry-exporter-otlp>=1.27.0",
]
```

- [ ] **Step 2: Sync the env**

Run from the repo root:

```bash
UV_PROJECT_ENVIRONMENT=.agents uv sync --extra dev --extra module-b
```

Expected: solver completes, installs `chromadb`, `sentence-transformers` (pulls `torch` ~1.5 GB on first install), and the OpenTelemetry packages. Some "Downloading" lines are normal. No errors.

- [ ] **Step 3: Smoke-test the imports**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/python -c "import chromadb; import sentence_transformers; from opentelemetry import trace; print('OK')"
```

Expected: prints `OK`. If sentence-transformers warns about cuDNN or TensorFlow, ignore — we use neither.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): add sentence-transformers + finalize module-b extras"
```

---

## Task B-2: `src/agentlab/memory.py` (TDD)

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_memory.py`
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/memory.py`

The library exports four names: `ConversationBuffer`, `KeyValueMemory`, `SemanticMemory`, and `Match` (a Pydantic model returned by `SemanticMemory.query`).

### Step 2.1: ConversationBuffer (TDD)

- [ ] **Step 1: Write the failing test for `ConversationBuffer.append` + `truncate`**

Create `tests/test_memory.py`:

```python
"""TDD tests for agentlab.memory."""
from __future__ import annotations

from pathlib import Path

import pytest


# ── ConversationBuffer ─────────────────────────────────────────────


def test_conversation_buffer_appends_and_returns_messages():
    from agentlab.memory import ConversationBuffer

    buf = ConversationBuffer(max_tokens=1000)
    buf.append({"role": "user", "content": "hello"})
    buf.append({"role": "assistant", "content": "hi"})

    assert buf.messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_conversation_buffer_truncates_oldest_first_to_fit_token_budget():
    from agentlab.memory import ConversationBuffer

    buf = ConversationBuffer(max_tokens=20)  # ~5 tokens per "word " * 4 = ~20
    for i in range(10):
        buf.append({"role": "user", "content": f"message-{i} " * 4})

    truncated = buf.truncate()
    # Truncation drops oldest messages until token estimate fits.
    assert len(truncated) < 10
    # Newest message must be retained.
    assert truncated[-1]["content"].startswith("message-9")
```

- [ ] **Step 2: Run the tests and verify they fail**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_memory.py -v
```

Expected: both tests fail with `ModuleNotFoundError: No module named 'agentlab.memory'`.

- [ ] **Step 3: Implement `ConversationBuffer` minimally**

Create `src/agentlab/memory.py`:

```python
"""Memory primitives for agentic workflows.

Three flavors, one file:

- ConversationBuffer — short-term, token-bounded, optional summarization.
- KeyValueMemory     — long-term, dict + JSON persistence.
- SemanticMemory     — embedding-based recall via Chroma + sentence-transformers.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _approx_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token. Good enough for buffer sizing."""
    return max(1, len(text) // 4)


def _message_tokens(message: dict) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        return _approx_tokens(content)
    # Anthropic content blocks (list of dicts).
    return sum(_approx_tokens(json.dumps(block)) for block in content)


class ConversationBuffer:
    """Token-bounded message buffer.

    Stores messages in order. `truncate()` returns the newest suffix that fits
    within `max_tokens`; `summarize()` (used in the next step) rolls older
    messages into a single system summary.
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        self.max_tokens = max_tokens
        self._messages: list[dict] = []

    def append(self, message: dict) -> None:
        self._messages.append(message)

    def messages(self) -> list[dict]:
        return list(self._messages)

    def truncate(self) -> list[dict]:
        kept: list[dict] = []
        budget = self.max_tokens
        for m in reversed(self._messages):
            cost = _message_tokens(m)
            if cost > budget and kept:
                break
            kept.append(m)
            budget -= cost
        kept.reverse()
        return kept
```

- [ ] **Step 4: Run the tests and verify they pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_memory.py -v
```

Expected: 2 passing.

### Step 2.2: ConversationBuffer.summarize (TDD)

- [ ] **Step 5: Write the failing test for `summarize`**

Append to `tests/test_memory.py`:

```python
def test_conversation_buffer_summarize_calls_client_with_buffer():
    from agentlab.memory import ConversationBuffer

    captured: dict = {}

    class _StubResponse:
        content = [type("B", (), {"text": "User asked about X. Assistant answered Y."})()]

    class _StubClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                captured.update(kwargs)
                return _StubResponse()

    buf = ConversationBuffer(max_tokens=1000)
    buf.append({"role": "user", "content": "What is X?"})
    buf.append({"role": "assistant", "content": "Y."})

    summary = buf.summarize(_StubClient())

    assert "X" in summary and "Y" in summary
    # The buffer's own messages should appear inside the prompt sent to the client.
    sent = json.dumps(captured)
    assert "What is X?" in sent
```

(Add `import json` at the top of the test file if it isn't already there.)

- [ ] **Step 6: Run and verify the test fails**

Expected: `AttributeError: 'ConversationBuffer' object has no attribute 'summarize'`.

- [ ] **Step 7: Implement `summarize`**

Append to `ConversationBuffer` in `src/agentlab/memory.py`:

```python
    def summarize(self, client) -> str:
        """Roll the buffer into a single short summary string via Claude."""
        if not self._messages:
            return ""
        rendered = "\n".join(
            f"{m['role']}: {m['content']}" if isinstance(m['content'], str) else f"{m['role']}: <blocks>"
            for m in self._messages
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="Summarize the conversation in 2-3 sentences. Preserve key facts and decisions.",
            messages=[{"role": "user", "content": rendered}],
        )
        return response.content[0].text
```

- [ ] **Step 8: Run and verify all 3 tests pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_memory.py -v
```

Expected: 3 passing.

### Step 2.3: KeyValueMemory (TDD)

- [ ] **Step 9: Write the failing tests for `KeyValueMemory`**

Append to `tests/test_memory.py`:

```python
# ── KeyValueMemory ─────────────────────────────────────────────────


def test_key_value_memory_set_and_get():
    from agentlab.memory import KeyValueMemory

    kv = KeyValueMemory()
    kv.set("user_name", "Rob")
    kv.set("favorite_color", "purple")

    assert kv.get("user_name") == "Rob"
    assert kv.get("favorite_color") == "purple"
    assert kv.get("missing") is None
    assert kv.get("missing", default="fallback") == "fallback"


def test_key_value_memory_save_and_load_roundtrip(tmp_path: Path):
    from agentlab.memory import KeyValueMemory

    kv = KeyValueMemory()
    kv.set("a", 1)
    kv.set("b", {"nested": [1, 2, 3]})

    target = tmp_path / "kv.json"
    kv.save(target)

    fresh = KeyValueMemory()
    fresh.load(target)

    assert fresh.get("a") == 1
    assert fresh.get("b") == {"nested": [1, 2, 3]}
```

- [ ] **Step 10: Run and verify both tests fail**

Expected: `ImportError` for `KeyValueMemory`.

- [ ] **Step 11: Implement `KeyValueMemory`**

Append to `src/agentlab/memory.py`:

```python
class KeyValueMemory:
    """Long-term key-value store with JSON persistence."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def keys(self) -> list[str]:
        return list(self._store.keys())

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self._store, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        self._store = json.loads(Path(path).read_text(encoding="utf-8"))
```

- [ ] **Step 12: Run and verify all 5 tests pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_memory.py -v
```

Expected: 5 passing.

### Step 2.4: SemanticMemory (TDD with stub embedder)

- [ ] **Step 13: Write the failing tests for `SemanticMemory`**

Append to `tests/test_memory.py`:

```python
# ── SemanticMemory ─────────────────────────────────────────────────


class _StubEmbedder:
    """Deterministic 4-d embedder for tests. Avoids loading sentence-transformers."""

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t) % 7), float(len(t) % 11), float(t.count("a")), float(t.count("e"))] for t in texts]


def test_semantic_memory_add_and_query_returns_match_with_metadata():
    from agentlab.memory import SemanticMemory

    sm = SemanticMemory(collection_name="test_basic", embedder=_StubEmbedder())
    sm.add("Anthropic released Claude Opus 4.7 in 2026.", metadata={"source": "doc-1"})
    sm.add("MCP is the Model Context Protocol.", metadata={"source": "doc-2"})

    matches = sm.query("Anthropic released Claude Opus 4.7 in 2026.", top_k=1)
    assert len(matches) == 1
    assert matches[0].metadata["source"] == "doc-1"
    assert "Anthropic" in matches[0].text
    assert isinstance(matches[0].score, float)


def test_semantic_memory_query_respects_top_k():
    from agentlab.memory import SemanticMemory

    sm = SemanticMemory(collection_name="test_top_k", embedder=_StubEmbedder())
    for i in range(5):
        sm.add(f"document number {i}", metadata={"i": i})

    matches = sm.query("document", top_k=3)
    assert len(matches) == 3
```

- [ ] **Step 14: Run and verify both tests fail**

Expected: `ImportError` for `SemanticMemory`.

- [ ] **Step 15: Implement `SemanticMemory` and `Match`**

Append to `src/agentlab/memory.py`:

```python
class Match(BaseModel):
    text: str
    score: float
    metadata: dict


def _default_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load sentence-transformers so import doesn't pay for it."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, convert_to_numpy=False, show_progress_bar=False)

    return embed


class SemanticMemory:
    """Embedding-backed recall via Chroma.

    Pass `embedder=callable` (taking list[str], returning list[list[float]]) to
    override the default sentence-transformers embedder — useful in tests.
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedder=None,
        embedder_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        import chromadb

        self._client = chromadb.EphemeralClient()
        self._embedder = embedder or _default_embedder(embedder_model)
        try:
            self._client.delete_collection(collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(name=collection_name)
        self._next_id = 0

    def add(self, text: str, metadata: dict | None = None) -> str:
        doc_id = f"doc-{self._next_id}"
        self._next_id += 1
        embedding = self._embedder([text])[0]
        self._collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[list(embedding)],
            metadatas=[metadata or {}],
        )
        return doc_id

    def query(self, text: str, top_k: int = 3) -> list[Match]:
        embedding = self._embedder([text])[0]
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
        )
        out: list[Match] = []
        for doc, dist, meta in zip(
            result["documents"][0],
            result["distances"][0],
            result["metadatas"][0],
        ):
            out.append(Match(text=doc, score=float(dist), metadata=meta or {}))
        return out
```

- [ ] **Step 16: Run and verify all 7 tests pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_memory.py -v
```

Expected: 7 passing.

- [ ] **Step 17: Run the full pytest suite to confirm no regressions**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -v
```

Expected: all existing Module A tests + 7 new memory tests, all green.

- [ ] **Step 18: Commit**

```bash
git add src/agentlab/memory.py tests/test_memory.py
git commit -m "feat(memory): ConversationBuffer + KeyValueMemory + SemanticMemory (TDD)"
```

---

## Task B-3: NB 04 — ReAct & extended thinking

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/04_react_and_extended_thinking.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/04_react_and_extended_thinking.ipynb`

**Demo task across all stages:** *"What's the population of the capital of the country that won the most recent men's football World Cup?"*

- [ ] **Step 1: Write the notebook source (jupytext percent format)**

Create `notebooks/_src/04_react_and_extended_thinking.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 04 — ReAct & extended thinking
#
# **Goal:** show that "ReAct" in 2026 is a layered concept and that the
# modern Claude tool-use loop *is* ReAct. We build it three ways:
#
# 1. **Verbatim Yao-2022 ReAct** — prompt-parsed `Thought:/Action:/Observation:`.
#    Brittle, token-expensive, but historically what "ReAct" meant.
# 2. **Tool-use as ReAct** — same idea via `tool_use` blocks. No parsing.
#    This is what Module A's `run_agent_loop` already does.
# 3. **Extended thinking** — server-side reasoning the model performs
#    before responding. No loop control; you pay tokens for thinking.
#
# The demo task is the same in all three stages: a multi-hop question
# that benefits from explicit intermediate reasoning.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="04 — ReAct & extended thinking",
    estimate="$0.02",
    model="claude-sonnet-4-6",
)

QUESTION = (
    "What's the population of the capital of the country that won the "
    "most recent men's football World Cup?"
)

# %% [markdown]
# ## Stage A — Verbatim Yao-2022 ReAct
#
# Prompt the model to emit `Thought: / Action: / Observation:` text. We
# parse with regex and drive a custom loop. The model never sees a
# `tools` parameter — it only knows about the contract from the prompt.

# %%
import re

from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()

REACT_SYSTEM = """You are a ReAct agent. On each turn, output ONE of:

  Thought: <your reasoning>
  Action: search[<query>]

OR, when you have the answer:

  Thought: <final reasoning>
  Final Answer: <answer>

Available action: search[query] — runs a web search and returns a short text result.

Output exactly one Thought followed by exactly one Action OR Final Answer. Never both.
"""


def fake_search(query: str) -> str:
    """A tiny stand-in for web search so the regex-driven loop is deterministic.
    Stage B uses the real managed web_search."""
    db = {
        "world cup 2022": "Argentina won the 2022 FIFA World Cup, beating France 4-2 on penalties.",
        "argentina capital": "Buenos Aires is the capital of Argentina.",
        "buenos aires population": "Buenos Aires has a population of approximately 3.1 million in the city proper.",
    }
    for key, value in db.items():
        if all(w in query.lower() for w in key.split()):
            return value
    return "No results."


ACTION_RE = re.compile(r"Action:\s*search\[(?P<q>[^\]]+)\]")
FINAL_RE = re.compile(r"Final Answer:\s*(?P<a>.+)", re.DOTALL)


def react_yao(question: str, max_turns: int = 6) -> str:
    transcript = f"Question: {question}\n"
    for turn in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=512,
            system=REACT_SYSTEM,
            messages=[{"role": "user", "content": transcript}],
        )
        text = response.content[0].text
        print(f"--- turn {turn + 1} ---\n{text}\n")
        transcript += text + "\n"

        if (m := FINAL_RE.search(text)):
            return m.group("a").strip()
        if (m := ACTION_RE.search(text)):
            obs = fake_search(m.group("q"))
            transcript += f"Observation: {obs}\n"
            continue
        raise RuntimeError(f"Couldn't parse turn {turn + 1}; brittleness in action.")

    raise RuntimeError("Out of turns.")


answer_a = react_yao(QUESTION)
print(f"\nStage A answer: {answer_a}")

# %% [markdown]
# Notes on Stage A:
# - The regex parser is fragile — if the model phrases things slightly
#   differently we crash. Modern instruction-tuned Claude is quite good
#   at following the format, but not perfect.
# - Every turn re-sends the full transcript (token-quadratic).
# - The "tools" are invisible to the model as a structured concept;
#   only the prompt enforces the contract.

# %% [markdown]
# ## Stage B — Tool-use as ReAct
#
# Same task, but reasoning lives in text content blocks between
# `tool_use` blocks. No parsing, no fake search — we use Anthropic's
# managed `web_search_20250305`. This is exactly what `run_agent_loop`
# from NB 02 does.

# %%
TOOL_USE_SYSTEM = """You are a careful research assistant. Use the web_search
tool to look up facts you don't know. Reason step-by-step in your text
content; tools handle the actions. When you have the final answer,
respond with just the answer."""


def react_tool_use(question: str, max_turns: int = 6) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=TOOL_USE_SYSTEM,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )
        for block in response.content:
            if hasattr(block, "text") and block.text.strip():
                print(f"[reasoning] {block.text[:300]}...")
        if response.stop_reason == "end_turn":
            text = "".join(getattr(b, "text", "") for b in response.content)
            return text.strip()
        messages.append({"role": "assistant", "content": response.content})
    raise RuntimeError("Out of turns.")


answer_b = react_tool_use(QUESTION)
print(f"\nStage B answer: {answer_b}")

# %% [markdown]
# Notes on Stage B:
# - No parser. The model emits structured `tool_use` blocks and we (and
#   Anthropic's web search) handle them.
# - The reasoning still happens — it's just in regular text content
#   blocks between tool calls.
# - This is what NB 02 already taught; the point is to *name* it as
#   ReAct so the lineage is clear.

# %% [markdown]
# ## Stage C — Extended thinking
#
# Same task, with `thinking` enabled. The model produces a hidden
# reasoning trace before responding. We pay tokens for it but get no
# loop control — Anthropic does the thinking server-side.

# %%
response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 4096},
    system=TOOL_USE_SYSTEM,
    tools=[{"type": "web_search_20250305", "name": "web_search"}],
    messages=[{"role": "user", "content": QUESTION}],
)

for block in response.content:
    btype = getattr(block, "type", "?")
    if btype == "thinking":
        print(f"[THINKING] {block.thinking[:400]}...\n")
    elif btype == "text":
        print(f"[ANSWER] {block.text}")
    elif btype == "tool_use":
        print(f"[TOOL_USE] {block.name}({block.input})")
    elif btype == "server_tool_use":
        print(f"[SERVER_TOOL_USE] {getattr(block, 'name', '?')}")
    elif btype == "web_search_tool_result":
        print(f"[WEB_SEARCH_RESULT] (managed)")

# %% [markdown]
# ## Reflect
#
# - **Tool-use ReAct is the modern default.** It's what `run_agent_loop`
#   does. The regex era is over for instruction-tuned models.
# - **Extended thinking pays off on planning-heavy tasks** (e.g. NB 05)
#   and on hard math/code reasoning. For lookups, it's just slower.
# - **Both reasoning channels — text content blocks AND extended
#   thinking — coexist.** You don't pick one or the other; you opt in
#   to thinking when the task needs it.
# - **Next:** NB 05 builds on this with explicit *plan-then-execute*
#   decomposition.
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/04_react_and_extended_thinking.py
```

Expected: writes `notebooks/04_react_and_extended_thinking.ipynb`.

- [ ] **Step 3: Pause for user to run the notebook**

The user runs the notebook in VS Code (kernel: Python (.agents)). Verify all cells execute without error and Stage A's parsed answer mentions Buenos Aires + ~3 million.

- [ ] **Step 4: After confirmation, commit**

```bash
git add notebooks/_src/04_react_and_extended_thinking.py notebooks/04_react_and_extended_thinking.ipynb
git commit -m "feat(nb04): ReAct three ways — Yao-style, tool-use, extended thinking"
```

---

## Task B-4: NB 05 — Planning & decomposition

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/05_planning_and_decomposition.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/05_planning_and_decomposition.ipynb`

**Demo task:** *"Compare httpx, aiohttp, and requests for an async-first Python project; recommend one."*

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/05_planning_and_decomposition.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 05 — Planning & decomposition
#
# **Goal:** plan-then-execute. A *planner* produces a TODO list; an
# *executor* walks the list; on failure, a *re-planner* revises. Then a
# *synthesizer* turns the executed plan into a structured `Answer`.
#
# **Demo task:** *"Compare httpx, aiohttp, and requests for an async-first
# Python project; recommend one."*
#
# Why this task: three subgoals (one per library), a synthesis step,
# and a natural failure mode if a search returns nothing useful.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="05 — Planning & decomposition",
    estimate="$0.03",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Define the plan type
#
# A `Plan` is a list of `Step`s. Each `Step` has a description, the
# tool to call, the args, and a status. We use Pydantic so the planner
# can't return malformed plans.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    description: str = Field(description="What this step accomplishes, in plain English.")
    tool: Literal["web_search", "synthesize"] = Field(
        description="Which tool runs this step. 'synthesize' is the final summarizer."
    )
    args: dict = Field(description="Arguments for the tool. For web_search: {'query': str}.")
    status: Literal["pending", "done", "failed"] = "pending"
    result: str | None = None


class Plan(BaseModel):
    goal: str
    steps: list[Step]


# %% [markdown]
# ## Step 2 — Planner: tool-use coercion forces a valid `Plan`
#
# We define a `submit_plan` tool whose `input_schema` is the Plan's
# JSON schema. The planner *must* call it.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()

submit_plan_tool = {
    "name": "submit_plan",
    "description": "Submit your plan. Must be called exactly once.",
    "input_schema": Plan.model_json_schema(),
}

PLANNER_SYSTEM = """You are a planner. Decompose the user's goal into 3-5
concrete steps. Each step uses one of: web_search (gather information),
synthesize (only as the final step, to produce the recommendation).
Call submit_plan exactly once."""


def make_plan(goal: str) -> Plan:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=PLANNER_SYSTEM,
        tools=[submit_plan_tool],
        messages=[{"role": "user", "content": goal}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_plan":
            return Plan.model_validate(block.input)
    raise RuntimeError("Planner did not call submit_plan.")


GOAL = (
    "Compare httpx, aiohttp, and requests for a new async-first Python "
    "project. Recommend one."
)

plan = make_plan(GOAL)
for i, step in enumerate(plan.steps, 1):
    print(f"{i}. [{step.tool}] {step.description}")

# %% [markdown]
# ## Step 3 — Executor
#
# Walk the plan. For `web_search` steps, run a single-turn agent with
# the managed web_search tool and capture the answer. For `synthesize`,
# defer to Step 5. Mark each step `done` or `failed`.

# %%
def execute_search_step(step: Step) -> str:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system="You are a research assistant. Use web_search; reply in 2-3 sentences with citations.",
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": step.args.get("query", step.description)}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content)
    if not text.strip() or "no results" in text.lower():
        raise RuntimeError("empty search result")
    return text.strip()


def execute(plan: Plan) -> Plan:
    for step in plan.steps:
        if step.tool != "web_search":
            continue
        try:
            step.result = execute_search_step(step)
            step.status = "done"
            print(f"✓ {step.description}")
        except Exception as exc:
            step.status = "failed"
            step.result = str(exc)
            print(f"✗ {step.description}  ({exc})")
    return plan


plan = execute(plan)

# %% [markdown]
# ## Step 4 — Re-planning on failure
#
# If any step failed, send the partially-executed plan back to the
# planner and ask for a revision. One revision pass max.

# %%
def replan_if_needed(plan: Plan) -> Plan:
    failed = [s for s in plan.steps if s.status == "failed"]
    if not failed:
        return plan
    print(f"\n[re-plan triggered: {len(failed)} step(s) failed]")
    revised = make_plan(
        f"Original goal: {plan.goal}\n\n"
        f"Previous plan had failures: {[s.description for s in failed]}\n"
        f"Revise the plan to work around them."
    )
    return execute(revised)


plan = replan_if_needed(plan)

# %% [markdown]
# ## Step 5 — Synthesize the final answer
#
# Reuse NB 03's `Answer` model. The synthesizer reads all the executed
# steps and produces the recommendation.

# %%
from agentlab.types import Answer

submit_answer_tool = {
    "name": "submit_answer",
    "description": "Final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}


def synthesize(plan: Plan) -> Answer:
    body = f"Goal: {plan.goal}\n\nResearch:\n"
    for step in plan.steps:
        if step.tool == "web_search" and step.status == "done":
            body += f"\n## {step.description}\n{step.result}\n"
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system="Synthesize a recommendation from the research. Cite sources.",
        tools=[submit_answer_tool],
        messages=[{"role": "user", "content": body}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_answer":
            return Answer.model_validate(block.input)
    raise RuntimeError("Synthesizer did not call submit_answer.")


answer = synthesize(plan)
print(answer.summary)
print()
for c in answer.citations:
    print(f"  - {c.title} — {c.url}")

# %% [markdown]
# ## Reflect
#
# - **Plan-then-execute beats free-form looping** when subgoals are
#   independent. The agent can't "forget" to compare candidate B
#   because it's an explicit step.
# - **Re-planning on failure** is cheap insurance against tool flakiness.
# - **Tool-use coercion is doing a lot of work here** — both the
#   planner and synthesizer use it. NB 03's pattern compounds.
# - **When NOT to plan:** single-tool tasks. NB 02's `run_agent_loop`
#   handles those without ceremony.
# - **Next:** NB 06 introduces memory so the spine project can answer
#   follow-ups.
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/05_planning_and_decomposition.py
```

- [ ] **Step 3: Pause for user to run the notebook**

The user runs in VS Code. Verify the plan has 3-4 steps, each library gets researched, the recommendation cites real sources.

- [ ] **Step 4: After confirmation, commit**

```bash
git add notebooks/_src/05_planning_and_decomposition.py notebooks/05_planning_and_decomposition.ipynb
git commit -m "feat(nb05): plan-then-execute with re-planning + Pydantic plans"
```

---

## Task B-5: NB 06 — Memory primitives (✦ spine touchpoint)

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/06_memory_primitives.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/06_memory_primitives.ipynb`

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/06_memory_primitives.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 06 — Memory primitives (✦ spine touchpoint)
#
# **Goal:** introduce three memory shapes via `agentlab.memory`:
# `ConversationBuffer`, `KeyValueMemory`, `SemanticMemory`. Then layer
# `KeyValueMemory` into the research-assistant spine so it can answer
# follow-up questions.
#
# Heads up: NB 07 will depend on `SemanticMemory`. NB 08 will trace
# everything.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="06 — Memory primitives (✦ spine + memory)",
    estimate="$0.03",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Short-term: `ConversationBuffer`
#
# Token-bounded message buffer with optional summarization. The simplest
# memory: just keep the last N tokens of conversation.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.memory import ConversationBuffer

client = get_client()

buf = ConversationBuffer(max_tokens=200)  # tiny on purpose
for i in range(8):
    buf.append({"role": "user", "content": f"This is message {i}, which has some words in it for token weight."})
    buf.append({"role": "assistant", "content": f"Acknowledged message {i}."})

print(f"All messages: {len(buf.messages())}")
truncated = buf.truncate()
print(f"After truncate: {len(truncated)} (oldest dropped to fit budget)")
print("Newest content:", truncated[-1]["content"][:60])

# %% [markdown]
# ### Summarization keeps context at a price
#
# `truncate()` *drops* old messages. `summarize()` *compresses* them
# into a single summary string. Use it when the conversation has
# load-bearing context older than your budget.

# %%
summary = buf.summarize(client)
print("Summary:", summary)

# %% [markdown]
# ## Step 2 — Long-term: `KeyValueMemory`
#
# Persistent dict. Use cases: user preferences, learned facts,
# pre-computed results. `save / load` round-trip via JSON.

# %%
from pathlib import Path

from agentlab.memory import KeyValueMemory

kv = KeyValueMemory()
kv.set("user_name", "Rob")
kv.set("project_focus", "AI agents curriculum")
kv.set("last_question", "How do I build NB 06?")

scratch = Path(".kv-demo.json")
kv.save(scratch)
print(scratch.read_text())

fresh = KeyValueMemory()
fresh.load(scratch)
assert fresh.get("user_name") == "Rob"
print(f"\nRe-loaded: user_name = {fresh.get('user_name')}")

scratch.unlink()

# %% [markdown]
# ## Step 3 — Semantic recall: `SemanticMemory`
#
# Embedding-backed retrieval. The first run downloads the
# sentence-transformers model (~80 MB). Subsequent runs are fast.

# %%
from agentlab.memory import SemanticMemory

sm = SemanticMemory(collection_name="nb06_demo")
sm.add("Anthropic's Claude API uses an HTTP-based message protocol.", metadata={"topic": "api"})
sm.add("MCP (Model Context Protocol) standardizes how tools are exposed to LLMs.", metadata={"topic": "mcp"})
sm.add("ReAct interleaves reasoning and acting in a loop.", metadata={"topic": "react"})
sm.add("Pydantic validates structured outputs from LLMs.", metadata={"topic": "structured"})

for query in ["How does the Anthropic API work?", "What is the protocol for tool exposure?"]:
    matches = sm.query(query, top_k=2)
    print(f"\nQuery: {query}")
    for m in matches:
        print(f"  [{m.score:.3f}] {m.text}  (topic={m.metadata.get('topic')})")

# %% [markdown]
# ## Step 4 — Spine: research assistant with memory
#
# We redefine `research()` from NB 03 inline (per the curriculum's
# "no library extraction until NB 15" rule), but this time the
# assistant has a `KeyValueMemory` it can read on each call. We expose
# two extra tools: `remember(key, value)` and `recall(key)`.

# %%
from agentlab.tools import ToolRegistry
from agentlab.types import Answer, Citation

registry = ToolRegistry()
session_memory = KeyValueMemory()


@registry.tool(description="Save a fact for later retrieval. Use when the user mentions a preference or fact.")
def remember(key: str, value: str) -> str:
    session_memory.set(key, value)
    return f"Saved {key}={value}"


@registry.tool(description="Recall a previously-saved fact by key. Returns the stored value or 'not found'.")
def recall(key: str) -> str:
    val = session_memory.get(key)
    return str(val) if val is not None else "not found"


submit_answer_tool = {
    "name": "submit_answer",
    "description": "Submit your final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}

SPINE_SYSTEM = """You are a research assistant with persistent memory.
- When the user mentions a preference, project, or fact about themselves, call remember(key, value).
- When relevant, call recall(key) to look up earlier facts.
- For factual questions, use web_search.
- When ready, call submit_answer exactly once."""


def research_with_memory(question: str, max_turns: int = 6) -> Answer:
    messages = [{"role": "user", "content": question}]
    tools = registry.schemas() + [
        {"type": "web_search_20250305", "name": "web_search"},
        submit_answer_tool,
    ]
    handlers = registry.handlers()
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=2048, system=SPINE_SYSTEM,
            tools=tools, messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                return Answer.model_validate(block.input)
        # Run any local tool calls; web_search is managed.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name in handlers:
                try:
                    out = handlers[block.name](**block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(out)})
                except Exception as e:
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True})
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        if response.stop_reason == "end_turn":
            raise RuntimeError("end_turn without submit_answer.")
    raise RuntimeError("Out of turns.")


# Turn 1: introduce a preference.
answer1 = research_with_memory(
    "I'm working on a Python async-first project. Remember that. "
    "What's the most popular async HTTP library?"
)
print("Q1 →", answer1.summary[:200])
print(f"\n[memory now contains: {session_memory.keys()}]")

# Turn 2: follow-up that needs the remembered context.
answer2 = research_with_memory(
    "Given my project's focus, which library do you recommend I use? "
    "(Look up what you remembered.)"
)
print("\nQ2 →", answer2.summary[:300])

# %% [markdown]
# ## Reflect
#
# - **Three memory shapes, three jobs.** Conversation buffers handle
#   token budgets; key-value stores hold facts; semantic memory finds
#   things by meaning.
# - **Memory becomes a tool the agent uses.** `remember/recall` are
#   plain `ToolRegistry` tools — the agent decides when to use them.
# - **Why `KeyValueMemory` for the spine, not `ConversationBuffer`?**
#   Because the spine is multi-turn but stateless across calls — we
#   want explicit, named facts (preferences, project context), not
#   raw transcripts.
# - **Foreshadow:** NB 07 puts `SemanticMemory` to work — but exposed as
#   a *retrieval tool*, not unconditional context injection.
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/06_memory_primitives.py
```

- [ ] **Step 3: Pause for user to run the notebook**

The user runs in VS Code. Verify Step 4's Q2 answer references the project context that was remembered in Q1.

- [ ] **Step 4: After confirmation, commit**

```bash
git add notebooks/_src/06_memory_primitives.py notebooks/06_memory_primitives.ipynb
git commit -m "feat(nb06): memory primitives + KeyValueMemory in the spine"
```

---

## Task B-6: NB 07 — RAG for agents

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/07_rag_for_agents.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/07_rag_for_agents.ipynb`

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/07_rag_for_agents.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 07 — RAG for agents
#
# **Goal:** make retrieval a tool the agent *decides* to use, not
# unconditional context injection. We chunk + embed this curriculum's
# own source (notebooks/_src + specs) into Chroma, then run two
# variants:
#
# 1. **Naive RAG** — always inject top-3 chunks. Wastes tokens when the
#    question doesn't need retrieval.
# 2. **Tool-shaped retrieval** — `search_notes(query)` is a tool the
#    agent calls when it judges it useful. The agent answers
#    weather-style questions without hitting the corpus.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="07 — RAG for agents",
    estimate="$0.03 (+ ~80 MB one-time embedding model download)",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Build the corpus
#
# Glob `notebooks/_src/*.py` and `docs/superpowers/specs/*.md`. Chunk
# by paragraphs (target ~500 tokens, ~2000 chars). Embed and index.

# %%
from pathlib import Path

from agentlab.memory import SemanticMemory

CHUNK_CHARS = 2000

def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 > max_chars and buf:
            chunks.append(buf.strip())
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


sources = sorted(
    list(Path("notebooks/_src").glob("*.py"))
    + list(Path("docs/superpowers/specs").glob("*.md"))
)
print(f"Source files: {len(sources)}")

corpus = SemanticMemory(collection_name="agents_curriculum")
chunk_count = 0
for src in sources:
    text = src.read_text(encoding="utf-8")
    for i, chunk in enumerate(chunk_text(text)):
        corpus.add(chunk, metadata={"source": str(src), "chunk_index": i})
        chunk_count += 1

print(f"Indexed {chunk_count} chunks across {len(sources)} files.")

# %% [markdown]
# ## Step 2 — Naive RAG (always inject top-3)
#
# Run two questions: one the corpus can answer ("how does NB 02
# register tools?"), one it cannot ("what's today's weather in
# Toronto?"). Watch the second waste tokens on irrelevant context.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()


def naive_rag(question: str) -> str:
    matches = corpus.query(question, top_k=3)
    context = "\n\n".join(f"[{i}] {m.text[:600]}" for i, m in enumerate(matches))
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=512,
        system="Answer using the provided context. If the context is irrelevant, say so.",
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return "".join(getattr(b, "text", "") for b in response.content)


for q in [
    "How does NB 02 register tools with the registry?",
    "What's the weather in Toronto today?",
]:
    print(f"\nQ: {q}")
    print(f"A: {naive_rag(q)[:400]}")

# %% [markdown]
# Notice the second answer wasted tokens on irrelevant chunks. We want
# the agent to *decide* whether to retrieve.

# %% [markdown]
# ## Step 3 — Retrieval as a tool
#
# Expose `search_notes(query)` via the tool registry. The agent calls
# it when it judges retrieval would help.

# %%
from agentlab.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool(description="Search the curriculum's own source code and specs for relevant context. Use when the question is about how this codebase works.")
def search_notes(query: str) -> str:
    matches = corpus.query(query, top_k=3)
    return "\n\n".join(
        f"[{m.metadata.get('source')}] {m.text[:600]}" for m in matches
    )


from agentlab.llm import run_agent_loop

for q in [
    "How does NB 02 register tools with the registry?",
    "What's the weather in Toronto today?",
]:
    print(f"\n=== Q: {q} ===")
    result = run_agent_loop(
        client=client,
        system=(
            "You are a helpful assistant. If the question is about this "
            "codebase, call search_notes. If it's a general or current-events "
            "question, answer from your own knowledge or say you can't."
        ),
        user_message=q,
        tools=registry.schemas(),
        tool_handlers=registry.handlers(),
        max_turns=4,
    )
    print(f"Turns: {result.turns}")
    print(f"Answer: {result.final_text[:400]}")

# %% [markdown]
# Look at the turn counts. The codebase question used `search_notes`;
# the weather question didn't — the agent (correctly) noted it can't
# answer that with this tool.

# %% [markdown]
# ## Step 4 — Citations
#
# When the agent retrieves, we want it to cite which chunk it used.
# Reuse NB 03's `Answer` model so the contract is consistent.

# %%
from agentlab.types import Answer

submit_answer_tool = {
    "name": "submit_answer",
    "description": "Submit final answer with citations to the chunks you used.",
    "input_schema": Answer.model_json_schema(),
}


def cited_research(question: str, max_turns: int = 6) -> Answer:
    messages = [{"role": "user", "content": question}]
    tools = registry.schemas() + [submit_answer_tool]
    handlers = registry.handlers()
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=1024,
            system=(
                "Answer questions about this codebase. Use search_notes to find "
                "relevant chunks. Cite the source path and a short quote in your "
                "Answer.citations. Always call submit_answer to finish."
            ),
            tools=tools, messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                return Answer.model_validate(block.input)
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name in handlers:
                try:
                    out = handlers[block.name](**block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(out)})
                except Exception as e:
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e), "is_error": True})
        messages.append({"role": "assistant", "content": response.content})
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("Out of turns.")


answer = cited_research("How does NB 03 use Pydantic to enforce structured output?")
print(answer.summary)
print()
for c in answer.citations:
    print(f"  • {c.title} — {c.url}")
    if c.quote:
        print(f"      \"{c.quote[:140]}\"")

# %% [markdown]
# ## Reflect
#
# - **Naive RAG injects context unconditionally — the agent has no
#   choice.** Tokens wasted on irrelevant matches; latency spent on
#   embeddings the agent didn't need.
# - **Tool-shaped retrieval restores agency.** The agent uses search
#   when the question warrants it, skips it otherwise.
# - **Citations are cheap with Pydantic.** Reusing `Answer` from NB 03
#   meant we got typed citations for free.
# - **Self-referential corpus is fun for learning** but real corpora
#   need re-indexing on document change. Production: a content hash
#   tracker; this NB just re-indexes on each run.
# - **Next:** NB 08 makes "is the agent working?" measurable with evals
#   + traces.
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/07_rag_for_agents.py
```

- [ ] **Step 3: Pause for user to run the notebook**

The user runs in VS Code. Verify the corpus indexes 30+ chunks, naive RAG wastes context on the weather question, tool-shaped retrieval skips it.

- [ ] **Step 4: After confirmation, commit**

```bash
git add notebooks/_src/07_rag_for_agents.py notebooks/07_rag_for_agents.ipynb
git commit -m "feat(nb07): RAG as a tool — self-referential corpus + naive vs tool-shaped"
```

---

## Task B-7: Eval suite scaffolding (data + fixtures + research-under-test)

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/data/eval_tasks.jsonl`
- Create: `/home/rob/PythonEnvironments/Agents/tests/eval/__init__.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/eval/conftest.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/eval/research_under_test.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/eval/test_research_assistant.py`

This task lays the pytest infrastructure NB 08 will demonstrate. The
eval tests are slow + cost real money, so they're behind the
`@pytest.mark.eval` marker and excluded from the default pytest run.

- [ ] **Step 1: Register the `eval` marker in `pyproject.toml`**

Edit `pyproject.toml`. In the `[tool.pytest.ini_options]` block, change:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra -q"
```

to:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra -q -m 'not eval'"
markers = [
    "eval: end-to-end agent evaluation tests (slow, costs money). Opt in: pytest -m eval",
]
```

- [ ] **Step 2: Create the 10-task eval set**

Create `data/eval_tasks.jsonl`:

```jsonl
{"id":"t01","question":"What is the latest stable Python release?","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Cites a recent Python.org release and gives a version number."}
{"id":"t02","question":"What does MCP stand for in the context of Claude tool use?","expected_tool_calls":["web_search"],"reference_answer":"Model Context Protocol","judge_rubric":"Defines MCP correctly and explains it's a protocol for tool/resource exposure."}
{"id":"t03","question":"What is Anthropic's web_search tool identifier (the 'type' value)?","expected_tool_calls":["web_search"],"reference_answer":"web_search_20250305","judge_rubric":"Provides the exact identifier."}
{"id":"t04","question":"In a few sentences, explain ReAct as proposed in the Yao 2022 paper.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Mentions interleaving Thought / Action / Observation; ties to LLM agents."}
{"id":"t05","question":"Name two well-known async HTTP libraries for Python.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Names two of: httpx, aiohttp, asyncio + urllib variants. Concise."}
{"id":"t06","question":"What is Pydantic, in one sentence?","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Describes Pydantic as a Python library for data validation using type hints."}
{"id":"t07","question":"What is the difference between Anthropic's extended thinking and a ReAct loop?","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Distinguishes server-side thinking (no loop control) from explicit tool-driven reasoning loops."}
{"id":"t08","question":"What is OpenTelemetry's primary purpose?","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Identifies OpenTelemetry as an observability framework for traces/metrics/logs."}
{"id":"t09","question":"Name a vector database commonly used with Python LLM applications.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Names one of: Chroma, Pinecone, Weaviate, Qdrant, Milvus, FAISS."}
{"id":"t10","question":"What is the maximum context window of Claude Sonnet 4.6 (in tokens)?","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Provides a numeric token count from a Claude documentation source."}
```

- [ ] **Step 3: Create the test infrastructure**

Create `tests/eval/__init__.py` (empty file).

Create `tests/eval/conftest.py`:

```python
"""Fixtures for the agent eval suite."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_TASKS_PATH = REPO_ROOT / "data" / "eval_tasks.jsonl"


def _load_tasks() -> list[dict]:
    if not EVAL_TASKS_PATH.exists():
        return []
    with EVAL_TASKS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def pytest_collection_modifyitems(config, items):
    """Skip eval tests if ANTHROPIC_API_KEY is missing."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    skip = pytest.mark.skip(reason="ANTHROPIC_API_KEY not set")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip)


def pytest_generate_tests(metafunc):
    """Inject the eval task list into any test that takes a `task` parameter."""
    if "task" in metafunc.fixturenames:
        tasks = _load_tasks()
        metafunc.parametrize("task", tasks, ids=[t["id"] for t in tasks])


@pytest.fixture(scope="session")
def eval_tasks() -> list[dict]:
    return _load_tasks()


@pytest.fixture(scope="session")
def client():
    from agentlab.llm import get_client
    return get_client()
```

- [ ] **Step 4: Create `research_under_test.py`** (the eval suite's local copy of the spine)

Create `tests/eval/research_under_test.py`:

```python
"""A copy of the spine `research()` function for the eval suite to exercise.

Per the curriculum design, the spine stays inline in notebooks until
NB 15 (capstone). The eval suite needs *something* to import, so it
keeps its own copy here. NB 15 will unify all of these.
"""
from __future__ import annotations

from agentlab.llm import DEFAULT_MODEL
from agentlab.types import Answer


SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}

SYSTEM_PROMPT = """You are a careful technical research assistant.

Workflow:
1. If you need facts you aren't sure of, use web_search.
2. When ready, call submit_answer with summary + citations.
3. Do not produce free-form text after submit_answer.
"""


def research(client, question: str, max_turns: int = 8) -> tuple[Answer, list[str]]:
    """Run the spine. Returns (Answer, list of tool names called)."""
    tool_calls: list[str] = []
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[
                {"type": "web_search_20250305", "name": "web_search"},
                SUBMIT_ANSWER_TOOL,
            ],
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(block.name)
                if block.name == "submit_answer":
                    return Answer.model_validate(block.input), tool_calls
            if block.type == "server_tool_use":
                tool_calls.append(getattr(block, "name", "server_tool"))
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            raise RuntimeError("end_turn without submit_answer.")
    raise RuntimeError(f"Out of turns ({max_turns}).")
```

- [ ] **Step 5: Create the three eval tests**

Create `tests/eval/test_research_assistant.py`:

```python
"""Three eval styles on the spine research assistant.

Run with: `pytest -m eval tests/eval/ -v`
"""
from __future__ import annotations

import pytest

from agentlab.types import Answer
from tests.eval.research_under_test import research


# ── Eval Style 1 — deterministic asserts ───────────────────────────


@pytest.mark.eval
def test_research_deterministic(client, task):
    """Asserts: returned object validates as Answer, expected tools were called,
    at least one citation present."""
    answer, tool_calls = research(client, task["question"])

    assert isinstance(answer, Answer)
    assert len(answer.citations) >= 1, "expected at least one citation"

    for expected in task["expected_tool_calls"]:
        assert expected in tool_calls, (
            f"expected tool '{expected}' to be called; saw {tool_calls}"
        )


# ── Eval Style 2 — LLM-as-judge ────────────────────────────────────


JUDGE_SYSTEM = """You are an evaluator. Score the answer 1-5 against the rubric:
1 = wildly wrong / off-topic
2 = partially relevant but missing key facts
3 = correct but minimal
4 = correct, well-cited, complete
5 = exemplary

Output ONLY the integer score on its own line."""


@pytest.mark.eval
def test_research_llm_judge(client, task):
    answer, _ = research(client, task["question"])

    judge_prompt = (
        f"Question: {task['question']}\n\n"
        f"Rubric: {task['judge_rubric']}\n\n"
        f"Answer summary: {answer.summary}\n\n"
        f"Citations: {[c.url for c in answer.citations]}"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": judge_prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content).strip()
    score = int(text.split()[0])

    # Treat scores ≤ 2 as failures; 3+ as passes (with note).
    assert score >= 3, f"judge gave {score}/5 for task {task['id']}"


# ── Eval Style 3 — reference-based (substring/exact match) ─────────


@pytest.mark.eval
def test_research_reference(client, task):
    if task["reference_answer"] is None:
        pytest.skip("no reference answer for this task")

    answer, _ = research(client, task["question"])
    ref = task["reference_answer"].lower()
    summary = answer.summary.lower()
    cited = " ".join(c.title.lower() + " " + (c.quote or "").lower() for c in answer.citations)

    assert ref in summary or ref in cited, (
        f"reference '{task['reference_answer']}' not found in summary or citations"
    )


```

Note: parametrization is handled exclusively by the
`pytest_generate_tests` hook above — every test that takes a `task`
fixture parameter gets the full eval set injected at collection time
(30 instances total: 10 tasks × 3 styles).

- [ ] **Step 6: Run the default suite — eval tests skipped**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -v
```

Expected: existing tests + memory tests pass. No eval tests collected
(filtered by `-m 'not eval'` from `addopts`).

- [ ] **Step 7: Run the eval suite explicitly to smoke-test**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -m eval tests/eval/ -v --maxfail=3
```

Expected: 30 test instances collected (10 tasks × 3 styles). Each
deterministic test takes 5-15s; the LLM-judge test adds another call
per task. Some flakiness on the LLM-judge tests is expected;
deterministic + reference should be stable. Cost: ~$0.05–0.10 for the
full run.

If failures look like model variance (judge gave 2/5 for a clearly
fine answer), that's the expected pedagogy. If failures look like
infrastructure (import errors, validation crashes), fix them.

- [ ] **Step 8: Commit**

```bash
git add data/eval_tasks.jsonl tests/eval/ pyproject.toml
git commit -m "feat(evals): pytest suite + 10-task spine eval set + research-under-test"
```

---

## Task B-8: NB 08 — Evals + observability (✦ spine integration)

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/08_evals_and_observability.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/08_evals_and_observability.ipynb`

This NB demonstrates the suite from Task B-7, adds OTel tracing inline,
and discusses the tradeoffs of each eval style.

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/08_evals_and_observability.py`:

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 08 — ✦ Evals + observability (spine integration)
#
# **Goal:** make "is the agent working?" measurable. We layer three
# eval styles + OpenTelemetry tracing on the research-assistant spine.
#
# Three eval styles, each with a different failure mode:
# 1. **Deterministic asserts** — fast, free, catches structural bugs;
#    misses content quality.
# 2. **LLM-as-judge** — flexible, subjective; flaky and costs money.
# 3. **Reference-based** — exact match for known-answer tasks; doesn't
#    fit open-ended questions.
#
# Plus OpenTelemetry traces so you can *see* what the agent did.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="08 — Evals + observability (✦ spine + measurement)",
    estimate="$0.05–0.10",
    model="claude-sonnet-4-6 (judges: claude-haiku-4-5)",
)

# %% [markdown]
# ## Step 1 — Load the eval tasks
#
# 10 tasks live in `data/eval_tasks.jsonl`. Mix of factual lookups,
# explainers, and known-answer questions.

# %%
import json
from pathlib import Path

tasks = [
    json.loads(line)
    for line in Path("data/eval_tasks.jsonl").read_text().splitlines()
    if line.strip()
]
print(f"Loaded {len(tasks)} eval tasks.")
for t in tasks[:3]:
    print(f"  [{t['id']}] {t['question']}")
print("  ...")

# %% [markdown]
# ## Step 2 — Trace the spine with OpenTelemetry
#
# We wrap `research()` with spans for `agent.run`, `tool.web_search`,
# and `llm.complete`. Use `ConsoleSpanExporter` so spans print to the
# notebook output as JSON.

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agentlab.spine")

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.types import Answer

client = get_client()

# Inline copy of the spine `research()` (per the curriculum's "no
# library extraction until NB 15" rule). Same logic as NB 03 and as
# tests/eval/research_under_test.py; consolidated in NB 15.
SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}
SPINE_SYSTEM = """You are a careful technical research assistant.
1. Use web_search for facts you aren't sure of.
2. Call submit_answer with summary + citations when ready.
3. Do not produce free-form text after submit_answer."""


def research(question: str, max_turns: int = 8):
    tool_calls = []
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=2048, system=SPINE_SYSTEM,
            tools=[
                {"type": "web_search_20250305", "name": "web_search"},
                SUBMIT_ANSWER_TOOL,
            ],
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(block.name)
                if block.name == "submit_answer":
                    return Answer.model_validate(block.input), tool_calls
            if block.type == "server_tool_use":
                tool_calls.append(getattr(block, "name", "server_tool"))
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            raise RuntimeError("end_turn without submit_answer.")
    raise RuntimeError("Out of turns.")


def traced_research(question: str):
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("question", question)
        answer, tool_calls = research(question)
        span.set_attribute("tool_calls", ",".join(tool_calls))
        span.set_attribute("citation_count", len(answer.citations))
        return answer, tool_calls


answer, tool_calls = traced_research(tasks[1]["question"])  # MCP question
print(f"Tool calls: {tool_calls}")
print(f"Summary: {answer.summary[:300]}")

# Force the exporter to flush so spans appear before the next cell.
provider.force_flush()

# %% [markdown]
# ## Step 3 — Eval style 1: deterministic asserts
#
# For each task: did `research()` return a valid `Answer`? Did it call
# the expected tools? Are there ≥1 citations?

# %%
from agentlab.types import Answer

deterministic_results = []
for task in tasks[:3]:  # subset for cost; full suite via pytest below
    try:
        answer, tool_calls = research(task["question"])
        ok = isinstance(answer, Answer) and len(answer.citations) >= 1
        ok &= all(t in tool_calls for t in task["expected_tool_calls"])
        deterministic_results.append((task["id"], ok))
    except Exception as e:
        deterministic_results.append((task["id"], False))
for tid, ok in deterministic_results:
    print(f"{'✓' if ok else '✗'} {tid}")

# %% [markdown]
# ## Step 4 — Eval style 2: LLM-as-judge
#
# A separate Haiku call rates the summary 1-5 against each task's
# rubric. Anything ≤ 2 is a fail.

# %%
JUDGE_SYSTEM = """You are an evaluator. Score the answer 1-5 against the rubric:
1 = wildly wrong / off-topic
2 = partially relevant but missing key facts
3 = correct but minimal
4 = correct, well-cited, complete
5 = exemplary

Output ONLY the integer score on its own line."""


def judge(question: str, rubric: str, answer: Answer) -> int:
    prompt = (
        f"Question: {question}\n\nRubric: {rubric}\n\n"
        f"Answer summary: {answer.summary}\n\n"
        f"Citations: {[c.url for c in answer.citations]}"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content).strip()
    return int(text.split()[0])


judge_results = []
for task in tasks[:3]:
    answer, _ = research(task["question"])
    score = judge(task["question"], task["judge_rubric"], answer)
    judge_results.append((task["id"], score))
for tid, score in judge_results:
    flag = "✓" if score >= 3 else "✗"
    print(f"{flag} {tid}: {score}/5")

# %% [markdown]
# ## Step 5 — Eval style 3: reference-based
#
# For tasks with a `reference_answer`, substring match against the
# summary or citations.

# %%
ref_results = []
for task in tasks[:3]:
    if task["reference_answer"] is None:
        ref_results.append((task["id"], "skip"))
        continue
    answer, _ = research(task["question"])
    ref = task["reference_answer"].lower()
    summary = answer.summary.lower()
    cited = " ".join(c.title.lower() + " " + (c.quote or "").lower() for c in answer.citations)
    ok = ref in summary or ref in cited
    ref_results.append((task["id"], "✓" if ok else "✗"))
for tid, flag in ref_results:
    print(f"{flag} {tid}")

# %% [markdown]
# ## Step 6 — Run the full suite via pytest
#
# Eval tests live in `tests/eval/` and are gated behind
# `@pytest.mark.eval`. Run them with:
#
# ```bash
# pytest -m eval tests/eval/ -v
# ```
#
# That runs all 10 tasks × 3 styles = 30 test instances. Skip pytest
# from the notebook (it's a separate command); the snippet above covers
# the demo subset.

# %% [markdown]
# ## Step 7 — Where each eval style breaks down
#
# - **Deterministic asserts** are great for structural checks
#   (validation passed? expected tools called?) but blind to whether
#   the answer is *correct*. A confidently-wrong answer with one
#   citation passes.
# - **LLM-as-judge** is flexible but flaky. The same answer can score
#   3 or 4 across runs. Use cheap models (Haiku) and run twice for
#   stability.
# - **Reference-based** is rock-solid when you can write a reference
#   answer, but most agent tasks are open-ended. Half our tasks have
#   `reference_answer: null` for that reason.
#
# **Use them in combination.** Deterministic catches bugs you don't
# even mean to introduce; reference catches regressions on known
# answers; LLM-judge surfaces quality drift.

# %% [markdown]
# ## Appendix — Swap to a real OTel collector (Jaeger)
#
# To send traces to a real UI:
#
# ```bash
# docker run -d --name jaeger \
#   -p 16686:16686 -p 4317:4317 \
#   jaegertracing/all-in-one:latest
# ```
#
# Then swap the exporter:
#
# ```python
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
#
# provider = TracerProvider()
# provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)))
# trace.set_tracer_provider(provider)
# ```
#
# Open http://localhost:16686 and click into traces.

# %% [markdown]
# ## Reflect
#
# - **Evals turn vibes into numbers.** "Feels right" doesn't survive a
#   model upgrade; pinned eval scores do.
# - **Tracing turns black boxes into something you can ask questions of.**
#   When the spine answers slowly, traces tell you whether it was the
#   model, the tool, or your own glue code.
# - **Module B is done.** The research assistant has memory (NB 06),
#   knows when to retrieve (NB 07), and is measurable (this NB).
# - **Next (Module C):** MCP. We build a server (NB 09), consume one
#   (NB 10), and start orchestrating subagents (NB 11-12).
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/08_evals_and_observability.py
```

- [ ] **Step 3: Pause for user to run the notebook**

The user runs in VS Code. Verify: traces print as JSON between cells; deterministic + judge + reference all run for the 3-task subset; pytest invocation is documented.

- [ ] **Step 4: After confirmation, commit**

```bash
git add notebooks/_src/08_evals_and_observability.py notebooks/08_evals_and_observability.ipynb
git commit -m "feat(nb08): three eval styles + OTel traces on the spine"
```

---

## Task B-9: README + Module B retrospective

**Files:**
- Modify: `/home/rob/PythonEnvironments/Agents/README.md`

- [ ] **Step 1: Update the status table**

Edit `README.md`. Replace the Module B row in the status table:

```markdown
| **B · Workflows** | 04 ReAct · 05 planning · 06 memory · 07 RAG · 08 evals + observability | ⏳ planned |
```

with:

```markdown
| **B · Workflows** | [04 ReAct & extended thinking](notebooks/04_react_and_extended_thinking.ipynb) · [05 planning](notebooks/05_planning_and_decomposition.ipynb) · [06 memory](notebooks/06_memory_primitives.ipynb) · [07 RAG](notebooks/07_rag_for_agents.ipynb) · [08 evals + observability](notebooks/08_evals_and_observability.ipynb) | ✅ shipped |
```

- [ ] **Step 2: Add a "What Module B teaches" section**

After the "What Module A teaches (the part that's shipped)" section, add:

```markdown
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
```

- [ ] **Step 3: Update the repo layout block**

In the existing "Repo layout" code block, add:

```
src/agentlab/
  ├ llm.py                 # Anthropic client wrapper + run_agent_loop
  ├ tools.py               # ToolRegistry + schema generation
  ├ types.py               # Pydantic models (Answer, Citation)
  └ memory.py              # ConversationBuffer + KeyValueMemory + SemanticMemory
tests/                     # pytest tests for agentlab + eval suite (tests/eval/)
data/                      # small seed files + eval_tasks.jsonl
```

(Leave the rest of the layout block intact.)

- [ ] **Step 4: Update the Costs section**

Replace:

```
Module A end-to-end is roughly **$0.05–0.10** at current Sonnet/Haiku
pricing.
```

with:

```
Module A end-to-end is roughly **$0.05–0.10**; Module B end-to-end is
**$0.10–0.20** (the eval suite drives most of that — opt in with
`pytest -m eval`).
```

- [ ] **Step 5: Update the Quickstart `uv sync` line**

In the Quickstart block, replace:

```bash
uv sync --extra dev
```

with:

```bash
uv sync --extra dev --extra module-b
```

(Module B is now the current frontier; learners should install both.)

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: flip Module B to ✅ shipped + add NB 04-08 descriptions"
```

- [ ] **Step 7: Push**

```bash
git push
```

---

## Self-Review Pass

Before declaring Module B done, run these checks from the repo root:

- [ ] **All notebooks execute end-to-end.** User has run NB 04, 05, 06, 07, 08 in VS Code without errors.
- [ ] **Default pytest is green:** `UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -v` (memory tests + Module A tests, no eval tests).
- [ ] **Eval suite runs:** `UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -m eval tests/eval/ -v` completes; deterministic + reference styles green; LLM-judge mostly green (occasional 2/5 acceptable).
- [ ] **Cost banners match spec.** NB 04 = $0.02; NB 05/06/07 = $0.03; NB 08 = $0.05–0.10.
- [ ] **README Module B row is ✅ shipped.**
- [ ] **All commits are on `feature/phase-0-module-a` (or whatever branch was current).** Each NB is its own commit; library + tests + eval suite are their own commits.
- [ ] **Spec link in README still resolves.**
