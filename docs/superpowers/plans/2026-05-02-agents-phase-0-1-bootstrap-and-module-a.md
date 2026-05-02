# Agents Curriculum — Phase 0 + Module A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bootstrap the `Agents` repo (uv-managed Python project pinned to the existing `.agents` venv, `src/agentlab/` shared library, README/LICENSE/.env.example/CI) and deliver Module A of the curriculum: three executable JupyterLab notebooks (NB 01-03) that teach the bare agent loop, tool use, and structured outputs. NB 03 introduces the recurring "research assistant" spine project.

**Architecture:** uv-managed `pyproject.toml` pinned to the existing `.agents` Python 3.13 venv via `UV_PROJECT_ENVIRONMENT=.agents`. Reusable code lives in `src/agentlab/` (an installable Python package); notebooks import from it. Notebooks are authored as `.py` files in `notebooks/_src/` with `jupytext` cell markers, then synced to `.ipynb` files in `notebooks/` for VS Code execution. Both `.py` (canonical source) and `.ipynb` (rendered, executable) are committed. Tests are pytest-based and target `agentlab` (notebook content is verified by execution).

**Tech Stack:** Python 3.13, uv 0.11+, Anthropic Python SDK, Pydantic v2, JupyterLab + ipykernel, pytest + pytest-asyncio, jupytext, python-dotenv, rich, httpx.

**Subsequent plans:** Module B (NB 04-08), Module C (NB 09-12), and Module D (NB 13-15) get their own plans after this one delivers and is reviewed. Each will build on the `agentlab` package and the spine project.

---

## Files created in this plan

```
pyproject.toml
.python-version
.envrc
.env.example
README.md
LICENSE
src/agentlab/__init__.py
src/agentlab/llm.py                    # Anthropic client wrapper
src/agentlab/tools.py                  # Tool registry + schema validation
src/agentlab/types.py                  # Shared Pydantic types (Citation, Answer, etc.)
tests/__init__.py
tests/conftest.py
tests/test_llm.py
tests/test_tools.py
tests/test_types.py
notebooks/_src/_common.py              # cost banner, env loader, shared imports
notebooks/_src/01_bare_agent_loop.py   # jupytext-format source
notebooks/_src/02_tool_use.py
notebooks/_src/03_structured_outputs.py
notebooks/01_bare_agent_loop.ipynb     # generated via jupytext
notebooks/02_tool_use.ipynb
notebooks/03_structured_outputs.ipynb
.github/workflows/test.yml             # CI: pytest + notebook execution
```

---

## Task 1: Initialize pyproject.toml and pin uv to the .agents venv

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/pyproject.toml`
- Create: `/home/rob/PythonEnvironments/Agents/.python-version`
- Create: `/home/rob/PythonEnvironments/Agents/.envrc`

- [ ] **Step 1: Write `.python-version`**

```
3.13
```

- [ ] **Step 2: Write `.envrc`**

This sets `UV_PROJECT_ENVIRONMENT` so uv operates on the existing `.agents` venv instead of creating a new `.venv`. The user can either `direnv allow` or rely on the README documenting that the variable must be set.

```
export UV_PROJECT_ENVIRONMENT=.agents
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[project]
name = "agentlab"
version = "0.1.0"
description = "Curriculum for building AI agents on Claude — from bare loop to multi-agent + MCP."
requires-python = ">=3.13"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "rkaunismaa"}]
dependencies = [
    "anthropic>=0.39.0",
    "pydantic>=2.9.0",
    "python-dotenv>=1.0.1",
    "httpx>=0.27.0",
    "rich>=13.9.0",
]

[project.optional-dependencies]
dev = [
    "jupyterlab>=4.3.0",
    "ipykernel>=6.29.0",
    "jupytext>=1.16.4",
    "nbclient>=0.10.0",
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
]
module-b = [
    "chromadb>=0.5.0",
    "opentelemetry-api>=1.27.0",
    "opentelemetry-sdk>=1.27.0",
    "opentelemetry-exporter-otlp>=1.27.0",
]
module-c = [
    "mcp>=1.0.0",
    "claude-agent-sdk>=0.1.0",
]
module-d = [
    "openai>=1.55.0",
]
web-search = [
    "tavily-python>=0.5.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/agentlab"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-ra -q"

[tool.jupytext]
formats = "notebooks///ipynb,notebooks/_src///py:percent"
```

- [ ] **Step 4: Sync dependencies into the .agents venv**

Run from `/home/rob/PythonEnvironments/Agents`:
```bash
UV_PROJECT_ENVIRONMENT=.agents uv sync --extra dev
```

Expected: uv installs anthropic, pydantic, jupyterlab, jupytext, pytest, etc. into `.agents/lib/python3.13/site-packages/`. The package `agentlab` itself is installed in editable mode (so notebooks can `from agentlab import ...`).

- [ ] **Step 5: Verify imports work**

```bash
.agents/bin/python -c "import anthropic, pydantic, dotenv, rich, jupytext, pytest; print('ok')"
```

Expected output: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .envrc
git commit -m "chore: bootstrap uv project pinned to .agents venv

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Create the agentlab package skeleton with empty modules and a smoke test

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/__init__.py`
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/llm.py` (stub)
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/tools.py` (stub)
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/types.py` (stub)
- Create: `/home/rob/PythonEnvironments/Agents/tests/__init__.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/conftest.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_smoke.py`

- [ ] **Step 1: Write `src/agentlab/__init__.py`**

```python
"""agentlab — shared utilities for the AI agents curriculum."""

__version__ = "0.1.0"
```

- [ ] **Step 2: Write `src/agentlab/llm.py` as a stub**

```python
"""Anthropic client wrapper. Real implementation lands in Task 4."""
```

- [ ] **Step 3: Write `src/agentlab/tools.py` as a stub**

```python
"""Tool registry and schema validation. Real implementation lands in Task 6."""
```

- [ ] **Step 4: Write `src/agentlab/types.py` as a stub**

```python
"""Shared Pydantic types used across notebooks. Populated alongside notebooks."""
```

- [ ] **Step 5: Write `tests/__init__.py`** (empty file — makes pytest discovery cleaner)

```python
```

- [ ] **Step 6: Write `tests/conftest.py`**

```python
"""Pytest fixtures shared across the test suite."""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session", autouse=True)
def load_env() -> None:
    """Load .env once at the start of the test session if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


@pytest.fixture
def has_api_key() -> bool:
    """True if ANTHROPIC_API_KEY is set; tests that need a real call check this."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
```

- [ ] **Step 7: Write `tests/test_smoke.py`**

```python
"""Smoke test: agentlab imports cleanly."""


def test_agentlab_imports():
    import agentlab
    assert agentlab.__version__ == "0.1.0"


def test_submodules_import():
    from agentlab import llm, tools, types  # noqa: F401
```

- [ ] **Step 8: Run the smoke test**

```bash
.agents/bin/python -m pytest tests/test_smoke.py -v
```

Expected output: `2 passed`.

- [ ] **Step 9: Commit**

```bash
git add src/ tests/
git commit -m "feat(agentlab): package skeleton + smoke test

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Write README, LICENSE, and .env.example

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/README.md`
- Create: `/home/rob/PythonEnvironments/Agents/LICENSE`
- Create: `/home/rob/PythonEnvironments/Agents/.env.example`

- [ ] **Step 1: Write `LICENSE`** (MIT, current year)

```
MIT License

Copyright (c) 2026 rkaunismaa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Write `.env.example`**

```bash
# Anthropic API — required for all notebooks except NB 14 (local models)
ANTHROPIC_API_KEY=sk-ant-...

# Tavily — optional; alternative web-search provider (free tier available).
# NB 02 also shows Anthropic's built-in web-search tool, which doesn't need
# this key.
TAVILY_API_KEY=

# OpenTelemetry — used from NB 08 onwards. Defaults work without a collector.
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

- [ ] **Step 3: Write `README.md`**

````markdown
# Agents — a notebook-driven curriculum for building AI agents on Claude

A 15-notebook progressive curriculum that takes you from "I've built a toy
agent" to "I can ship multi-agent, MCP-integrated systems on Claude — and
port them to local models when I need to."

The curriculum is **scratch-first**: you build the bare agent loop by hand
before any framework abstraction shows up, so you understand what
frameworks are hiding from you. Frameworks (Claude Agent SDK, MCP) appear
once they earn their keep.

## Modules

| Module | Notebooks | Status |
|---|---|---|
| **A — Foundations** | 01 bare loop · 02 tool use · 03 structured outputs | ✅ this plan |
| **B — Workflows** | 04 ReAct · 05 planning · 06 memory · 07 RAG · 08 evals + observability | ⏳ next plan |
| **C — Multi-agent + MCP** | 09 build MCP server · 10 consume MCP · 11 orchestrator + subagents · 12 parallel + durable | ⏳ |
| **D — Autonomous + local** | 13 computer use · 14 local models (Ollama / LM Studio) · 15 capstone | ⏳ |

The "research assistant" spine project recurs at NB 03, NB 08, NB 12, and
NB 15 to integrate the patterns you've learned.

Full design: [`docs/superpowers/specs/2026-05-02-ai-agents-curriculum-design.md`](docs/superpowers/specs/2026-05-02-ai-agents-curriculum-design.md)

## Quickstart

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv) ≥ 0.11.

```bash
git clone git@github.com:rkaunismaa/Agents.git
cd Agents

# Tell uv to use the included .agents venv (or set in .envrc + direnv allow):
export UV_PROJECT_ENVIRONMENT=.agents

uv sync --extra dev

cp .env.example .env
# edit .env to add your ANTHROPIC_API_KEY

# Register the kernel so VS Code's notebook UI can find it:
.agents/bin/python -m ipykernel install --user --name agents --display-name "Python (.agents)"

# Then open notebooks/01_bare_agent_loop.ipynb in VS Code.
```

## Repo layout

```
notebooks/                 # the .ipynb files you execute
notebooks/_src/            # jupytext-format .py source for the notebooks
src/agentlab/              # shared utilities imported by every notebook
tests/                     # pytest tests for agentlab
mcp_servers/               # MCP servers (added in Module C)
data/                      # tiny seed corpora for RAG/eval notebooks
docs/superpowers/specs/    # design specs
docs/superpowers/plans/    # implementation plans
```

## Editing notebooks

Notebook content is canonical in `notebooks/_src/*.py` (jupytext "percent"
format). The `.ipynb` files in `notebooks/` are generated. After editing
a `.py` source, regenerate its `.ipynb`:

```bash
.agents/bin/jupytext --sync notebooks/_src/01_bare_agent_loop.py
```

This pairing is configured in `pyproject.toml` (`[tool.jupytext]`).

## Tests

```bash
.agents/bin/python -m pytest -v
```

Tests run against `agentlab` only; notebook content is verified by
executing the notebooks (see `.github/workflows/test.yml`).

## Costs

Each notebook prints a cost estimate at the top. Most notebooks cost a
few cents to run; the multi-agent and capstone notebooks default to
Claude Haiku where pedagogically acceptable to keep costs low.

## License

MIT — see [LICENSE](LICENSE).
````

- [ ] **Step 4: Commit**

```bash
git add README.md LICENSE .env.example
git commit -m "docs: README, LICENSE, and .env.example

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Implement agentlab.types — shared Pydantic types (TDD)

The `Citation` and `Answer` types are the structured output of the
research-assistant spine project, introduced in NB 03. Defining them in
`agentlab.types` keeps notebook code clean and lets later modules
reuse them.

**Files:**
- Modify: `/home/rob/PythonEnvironments/Agents/src/agentlab/types.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_types.py`

- [ ] **Step 1: Write the failing test**

`tests/test_types.py`:

```python
"""Tests for agentlab.types — shared Pydantic types."""
import pytest
from pydantic import ValidationError

from agentlab.types import Answer, Citation


def test_citation_minimum_fields():
    c = Citation(url="https://example.com", title="Example")
    assert c.url == "https://example.com"
    assert c.title == "Example"
    assert c.quote is None


def test_citation_with_quote():
    c = Citation(url="https://example.com", title="Example", quote="lorem ipsum")
    assert c.quote == "lorem ipsum"


def test_citation_rejects_non_url():
    with pytest.raises(ValidationError):
        Citation(url="not-a-url", title="x")


def test_answer_with_citations():
    a = Answer(
        summary="Bears eat fish.",
        citations=[Citation(url="https://example.com/bears", title="Bears 101")],
    )
    assert a.summary == "Bears eat fish."
    assert len(a.citations) == 1


def test_answer_requires_at_least_one_citation():
    with pytest.raises(ValidationError):
        Answer(summary="Bears eat fish.", citations=[])


def test_answer_serializes_to_json():
    a = Answer(
        summary="Bears eat fish.",
        citations=[Citation(url="https://example.com/", title="x")],
    )
    payload = a.model_dump_json()
    assert "Bears eat fish." in payload
    assert "example.com" in payload
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.agents/bin/python -m pytest tests/test_types.py -v
```

Expected: `ImportError` or `AttributeError` because `Answer` and `Citation` don't exist yet.

- [ ] **Step 3: Implement the types**

Replace `src/agentlab/types.py`:

```python
"""Shared Pydantic types used across notebooks.

These types model the structured output of the recurring research-assistant
spine project introduced in NB 03 and reused in later modules.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class Citation(BaseModel):
    """A single source citation."""

    url: HttpUrl
    title: str = Field(..., min_length=1)
    quote: str | None = None

    model_config = {"frozen": True}


class Answer(BaseModel):
    """A researched answer with at least one citation."""

    summary: str = Field(..., min_length=1)
    citations: list[Citation] = Field(..., min_length=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.agents/bin/python -m pytest tests/test_types.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/agentlab/types.py tests/test_types.py
git commit -m "feat(agentlab.types): Citation and Answer Pydantic models

Used as the structured output of the research-assistant spine project
debuting in NB 03.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Implement agentlab.llm — Anthropic client wrapper (TDD)

A thin wrapper around `anthropic.Anthropic` that:
1. Loads `ANTHROPIC_API_KEY` from `.env`.
2. Centralises model defaults (Sonnet for "real" work, Haiku for cheap demos).
3. Provides a `run_agent_loop` helper that drives messages → tool_use → tool_result iteration. Notebooks can ignore the helper and write the loop by hand (NB 01) or use it for cleaner code (NB 02 onwards).

The Anthropic client itself isn't mocked — we test our wrapper logic, not the SDK. The agent loop is tested with a fake client that returns scripted responses.

**Files:**
- Modify: `/home/rob/PythonEnvironments/Agents/src/agentlab/llm.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_llm.py`:

```python
"""Tests for agentlab.llm — client wrapper and agent loop helper."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agentlab.llm import (
    DEFAULT_MODEL,
    HAIKU_MODEL,
    AgentLoopResult,
    get_client,
    run_agent_loop,
)


def test_default_model_is_sonnet():
    assert "sonnet" in DEFAULT_MODEL.lower()


def test_haiku_model_is_haiku():
    assert "haiku" in HAIKU_MODEL.lower()


def test_get_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_client()


def test_get_client_uses_explicit_key():
    client = get_client(api_key="sk-test-explicit")
    # The Anthropic SDK exposes the key on the client instance.
    assert client.api_key == "sk-test-explicit"


# --- Fake client used to drive run_agent_loop without hitting the API ---

@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    id: str | None = None
    tool_use_id: str | None = None


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"


@dataclass
class FakeMessages:
    scripted: list[FakeResponse]
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.scripted.pop(0)


@dataclass
class FakeClient:
    messages: FakeMessages


def test_run_agent_loop_terminates_on_end_turn():
    client = FakeClient(messages=FakeMessages(scripted=[
        FakeResponse(content=[FakeBlock(type="text", text="Hello!")]),
    ]))

    result = run_agent_loop(
        client=client,
        system="You are a helpful agent.",
        user_message="Hi",
        tools=[],
        tool_handlers={},
    )

    assert isinstance(result, AgentLoopResult)
    assert result.final_text == "Hello!"
    assert result.turns == 1
    assert len(client.messages.calls) == 1


def test_run_agent_loop_executes_tool_then_continues():
    client = FakeClient(messages=FakeMessages(scripted=[
        FakeResponse(
            content=[FakeBlock(
                type="tool_use", id="t1", name="add", input={"a": 2, "b": 3}
            )],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeBlock(type="text", text="The answer is 5.")]),
    ]))

    def add(a: int, b: int) -> int:
        return a + b

    result = run_agent_loop(
        client=client,
        system="You are a calculator.",
        user_message="What is 2+3?",
        tools=[{"name": "add", "description": "add two ints",
                "input_schema": {"type": "object",
                                 "properties": {"a": {"type": "integer"},
                                                "b": {"type": "integer"}},
                                 "required": ["a", "b"]}}],
        tool_handlers={"add": add},
    )

    assert result.final_text == "The answer is 5."
    assert result.turns == 2
    # Second call should include the tool_result.
    second_call_messages = client.messages.calls[1]["messages"]
    assert any(
        block.get("type") == "tool_result"
        for msg in second_call_messages
        for block in (msg["content"] if isinstance(msg["content"], list) else [])
    )


def test_run_agent_loop_respects_max_turns():
    # Always returns tool_use, never end_turn — should bail out.
    def loop_response():
        return FakeResponse(
            content=[FakeBlock(type="tool_use", id="t1", name="noop", input={})],
            stop_reason="tool_use",
        )

    client = FakeClient(messages=FakeMessages(scripted=[loop_response() for _ in range(20)]))

    with pytest.raises(RuntimeError, match="max_turns"):
        run_agent_loop(
            client=client,
            system="x",
            user_message="x",
            tools=[{"name": "noop", "description": "x",
                    "input_schema": {"type": "object", "properties": {}}}],
            tool_handlers={"noop": lambda: None},
            max_turns=5,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.agents/bin/python -m pytest tests/test_llm.py -v
```

Expected: ImportError on `AgentLoopResult`, `run_agent_loop`, etc.

- [ ] **Step 3: Implement `agentlab.llm`**

Replace `src/agentlab/llm.py`:

```python
"""Anthropic client wrapper and a reusable agent-loop helper.

NB 01 deliberately re-implements the loop by hand; from NB 02 onwards
notebooks can use ``run_agent_loop`` for cleaner code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def get_client(api_key: str | None = None) -> Anthropic:
    """Build an Anthropic client.

    Reads ``ANTHROPIC_API_KEY`` from the environment if ``api_key`` isn't
    given. Raises ``RuntimeError`` with a friendly message if no key is
    available — better than the SDK's late KeyError.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or pass api_key= explicitly."
        )
    return Anthropic(api_key=key)


@dataclass
class AgentLoopResult:
    """What ``run_agent_loop`` returns when the model finally says ``end_turn``."""

    final_text: str
    turns: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


class _ClientLike(Protocol):
    messages: Any  # duck-typed for the FakeClient used in tests


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an Anthropic SDK content block (or test fake) to a plain dict.

    The SDK returns Pydantic models; tests use simple dataclasses. We accept
    either by reading the common attributes.
    """
    out: dict[str, Any] = {"type": block.type}
    for attr in ("text", "name", "input", "id", "tool_use_id"):
        val = getattr(block, attr, None)
        if val is not None:
            out[attr] = val
    return out


def run_agent_loop(
    *,
    client: _ClientLike,
    system: str,
    user_message: str,
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[..., Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    max_turns: int = 10,
) -> AgentLoopResult:
    """Drive a Claude messages.create loop until end_turn.

    Each turn: call messages.create with the current transcript. If the
    response has tool_use blocks, run the matching handlers and append a
    user message containing tool_result blocks. Stop when the model
    returns end_turn (or another non-tool stop reason).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        assistant_blocks = [_block_to_dict(b) for b in response.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                b.get("text", "") for b in assistant_blocks if b["type"] == "text"
            )
            return AgentLoopResult(
                final_text=final_text, turns=turn, transcript=messages
            )

        tool_results: list[dict[str, Any]] = []
        for block in assistant_blocks:
            if block["type"] != "tool_use":
                continue
            handler = tool_handlers.get(block["name"])
            if handler is None:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Error: no handler for tool {block['name']!r}",
                    "is_error": True,
                })
                continue
            try:
                result = handler(**(block.get("input") or {}))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": str(result),
                })
            except Exception as exc:  # surface tool errors back to the model
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Error: {exc}",
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent loop did not terminate within max_turns={max_turns}.")
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.agents/bin/python -m pytest tests/test_llm.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Run all tests to confirm no regressions**

```bash
.agents/bin/python -m pytest -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/agentlab/llm.py tests/test_llm.py
git commit -m "feat(agentlab.llm): Anthropic client wrapper + run_agent_loop helper

NB 01 re-implements the loop by hand for pedagogy; NB 02+ can use
run_agent_loop for cleaner code. Tested with a duck-typed fake client.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Implement agentlab.tools — tool registry and schema validation (TDD)

The tool registry is the pattern used from NB 02 onwards: define a Python
function, decorate it with `@tool`, the registry collects the JSON schema
from the function signature + Pydantic model. Notebooks then pass
`registry.schemas()` to the API and `registry.handlers()` to
`run_agent_loop`.

**Files:**
- Modify: `/home/rob/PythonEnvironments/Agents/src/agentlab/tools.py`
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_tools.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_tools.py`:

```python
"""Tests for agentlab.tools — the tool registry."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from agentlab.tools import ToolRegistry, tool


def test_register_simple_tool():
    registry = ToolRegistry()

    @registry.tool(description="Add two integers.")
    def add(a: int, b: int) -> int:
        return a + b

    schemas = registry.schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "add"
    assert schema["description"] == "Add two integers."
    assert schema["input_schema"]["properties"]["a"]["type"] == "integer"
    assert schema["input_schema"]["properties"]["b"]["type"] == "integer"
    assert set(schema["input_schema"]["required"]) == {"a", "b"}


def test_register_tool_with_pydantic_model():
    class SearchInput(BaseModel):
        query: str = Field(..., description="The search query.")
        max_results: int = Field(5, ge=1, le=20)

    registry = ToolRegistry()

    @registry.tool(description="Search the web.", input_model=SearchInput)
    def search(query: str, max_results: int = 5) -> list[str]:
        return [f"{query}#{i}" for i in range(max_results)]

    schema = registry.schemas()[0]
    assert schema["name"] == "search"
    props = schema["input_schema"]["properties"]
    assert props["query"]["description"] == "The search query."
    assert props["max_results"]["minimum"] == 1


def test_handlers_dispatch_correctly():
    registry = ToolRegistry()

    @registry.tool(description="x")
    def add(a: int, b: int) -> int:
        return a + b

    handlers = registry.handlers()
    assert handlers["add"](a=2, b=3) == 5


def test_tool_decorator_module_level():
    """The bare ``@tool`` decorator builds a global registry-free entry."""
    @tool(description="Negate an integer.")
    def neg(x: int) -> int:
        return -x

    assert neg.tool_name == "neg"
    assert neg.tool_schema["description"] == "Negate an integer."
    assert neg(5) == -5  # original callable still works


def test_optional_parameters_are_not_required():
    registry = ToolRegistry()

    @registry.tool(description="Greet someone.")
    def greet(name: str, formal: bool = False) -> str:
        return f"{'Greetings' if formal else 'Hi'}, {name}"

    schema = registry.schemas()[0]
    required = set(schema["input_schema"].get("required", []))
    assert required == {"name"}


def test_tool_name_must_be_unique():
    registry = ToolRegistry()

    @registry.tool(description="x")
    def dup() -> int:
        return 1

    with pytest.raises(ValueError, match="already registered"):
        @registry.tool(description="y")
        def dup() -> int:  # noqa: F811
            return 2
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.agents/bin/python -m pytest tests/test_tools.py -v
```

Expected: ImportError on `ToolRegistry`, `tool`.

- [ ] **Step 3: Implement `agentlab.tools`**

Replace `src/agentlab/tools.py`:

```python
"""Tool registry and schema generation for agentlab notebooks.

The registry pattern: a notebook defines a Python function and decorates
it with ``@registry.tool(...)``. The registry inspects the signature
(or an explicit Pydantic model) to produce the JSON schema Anthropic's
API expects, and exposes both ``.schemas()`` and ``.handlers()`` for
use with ``agentlab.llm.run_agent_loop``.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, create_model

_PY_TO_JSON = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def _python_type_to_jsonschema(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a fragment of JSON schema.

    Handles the primitive types we use in Module A. Lists, dicts, and
    Optional are supported; richer cases use ``input_model=`` with a
    Pydantic model directly.
    """
    origin = typing.get_origin(annotation)
    if origin is list:
        (item_type,) = typing.get_args(annotation)
        return {"type": "array", "items": _python_type_to_jsonschema(item_type)}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_jsonschema(non_none[0])
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    return {"type": "string"}  # safe default


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        properties[param_name] = _python_type_to_jsonschema(param.annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _schema_from_pydantic_model(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    # Anthropic accepts standard JSON schema; trim Pydantic's $defs unless used.
    schema.pop("title", None)
    return schema


class ToolRegistry:
    """Holds a set of tools and produces schemas + handlers for the API."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def tool(
        self,
        *,
        description: str,
        name: str | None = None,
        input_model: type[BaseModel] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register ``fn`` as a tool.

        If ``input_model`` is given, its JSON schema is used; otherwise the
        function signature drives schema generation.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            if tool_name in self._entries:
                raise ValueError(f"Tool {tool_name!r} is already registered.")

            input_schema = (
                _schema_from_pydantic_model(input_model)
                if input_model is not None
                else _schema_from_signature(fn)
            )
            self._entries[tool_name] = {
                "schema": {
                    "name": tool_name,
                    "description": description,
                    "input_schema": input_schema,
                },
                "handler": fn,
            }
            return fn

        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        return [entry["schema"] for entry in self._entries.values()]

    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {name: entry["handler"] for name, entry in self._entries.items()}


def tool(
    *,
    description: str,
    name: str | None = None,
    input_model: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Module-level decorator: attach schema metadata to a function without a registry.

    Useful for one-off tools defined inline in a notebook.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        input_schema = (
            _schema_from_pydantic_model(input_model)
            if input_model is not None
            else _schema_from_signature(fn)
        )
        fn.tool_name = tool_name  # type: ignore[attr-defined]
        fn.tool_schema = {  # type: ignore[attr-defined]
            "name": tool_name,
            "description": description,
            "input_schema": input_schema,
        }
        return fn

    return decorator
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.agents/bin/python -m pytest tests/test_tools.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Run all tests to confirm no regressions**

```bash
.agents/bin/python -m pytest -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/agentlab/tools.py tests/test_tools.py
git commit -m "feat(agentlab.tools): tool registry with schema generation

Inspects function signatures or accepts a Pydantic input model. Produces
schemas and handler dicts compatible with run_agent_loop.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Build NB 01 — The bare agent loop, from scratch

Authored as `notebooks/_src/01_bare_agent_loop.py` in jupytext "percent"
format, then synced to `notebooks/01_bare_agent_loop.ipynb`. NB 01
deliberately does NOT use `run_agent_loop` — it implements the loop by
hand so the learner sees what's happening.

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/_common.py`
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/01_bare_agent_loop.py`
- Generated: `/home/rob/PythonEnvironments/Agents/notebooks/01_bare_agent_loop.ipynb`

- [ ] **Step 1: Write `notebooks/_src/_common.py`** — shared notebook preamble

```python
"""Shared imports + setup used at the top of every notebook.

Notebooks import this in their first cell so the boilerplate (env
loading, rich display, cost banner) is consistent.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel


def load_env() -> None:
    """Load .env from the repo root if present."""
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent.parent, here.parent.parent, here.parent]:
        env = candidate / ".env"
        if env.exists():
            load_dotenv(env)
            return


def cost_banner(notebook: str, estimate: str, model: str) -> None:
    """Render a cost-estimate banner at the top of a notebook."""
    rprint(Panel.fit(
        f"[bold]{notebook}[/bold]\n"
        f"Default model: [cyan]{model}[/cyan]\n"
        f"Estimated cost per full run: [yellow]{estimate}[/yellow]",
        title="Notebook info",
    ))
```

- [ ] **Step 2: Write `notebooks/_src/01_bare_agent_loop.py`**

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 01 — The bare agent loop, from scratch
#
# **Goal:** demystify what an "agent" actually is, mechanically, by
# building the loop by hand. By the end of this notebook you'll have a
# ~30-line agent that uses a tool to do arithmetic, and you'll be able
# to point at every part of it and say what it does.
#
# **What's an agent?** An LLM in a loop. Each turn the model decides:
# *answer the user* or *call a tool*. If it calls a tool, you run the
# tool, hand the result back, and let the model take another turn.
# That's it. Every framework you'll see in later notebooks is doing
# exactly this — they just hide more of it.
#
# **In this notebook we'll deliberately use no abstractions** beyond
# `anthropic.Anthropic`. No `agentlab.llm.run_agent_loop`. We'll build
# the loop ourselves.

# %%
from _common import cost_banner, load_env

load_env()
cost_banner(
    notebook="01 — Bare agent loop",
    estimate="< $0.01",
    model="claude-haiku-4-5-20251001",
)

# %% [markdown]
# ## Step 1 — Make a single API call (no agent yet)
#
# Before the loop, here's the simplest call: ask the model a question,
# get a string back. This is what "calling an LLM" looks like at the
# bottom of every framework.

# %%
import os
from anthropic import Anthropic

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[{"role": "user", "content": "Say hi in five words."}],
)
print(response.content[0].text)

# %% [markdown]
# ## Step 2 — Define a tool
#
# A "tool" is a function the model can call. We describe its name,
# what it does, and the JSON schema for its inputs. The model never
# runs the tool — it returns a `tool_use` block, and *we* run the
# function and pass the result back.

# %%
def add(a: int, b: int) -> int:
    """The actual implementation — what we'll run when the model picks this."""
    return a + b


add_tool_schema = {
    "name": "add",
    "description": "Add two integers.",
    "input_schema": {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "first integer"},
            "b": {"type": "integer", "description": "second integer"},
        },
        "required": ["a", "b"],
    },
}

# %% [markdown]
# ## Step 3 — One turn with a tool call
#
# Send a question that needs the tool. We'll get back a `tool_use`
# block, not text. The model didn't compute the answer — it asked us
# to.

# %%
first = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    system="You are a calculator. Use the add tool for any addition.",
    tools=[add_tool_schema],
    messages=[{"role": "user", "content": "What is 17 + 25?"}],
)
print("stop_reason:", first.stop_reason)
for block in first.content:
    print(block)

# %% [markdown]
# ## Step 4 — Run the tool, send the result back
#
# We pull the `tool_use` block out, run our `add` function with the
# arguments the model provided, then send a follow-up `messages.create`
# call. The new messages list contains:
#
# 1. The original user message.
# 2. The assistant's response (containing `tool_use`).
# 3. A new user message containing a `tool_result` block.

# %%
tool_use_block = next(b for b in first.content if b.type == "tool_use")
result_value = add(**tool_use_block.input)
print(f"add({tool_use_block.input}) = {result_value}")

second = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    system="You are a calculator. Use the add tool for any addition.",
    tools=[add_tool_schema],
    messages=[
        {"role": "user", "content": "What is 17 + 25?"},
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": str(result_value),
        }]},
    ],
)
print("stop_reason:", second.stop_reason)
print(second.content[0].text)

# %% [markdown]
# ## Step 5 — The loop
#
# Two turns happened to be enough above, but in general the model may
# need many tool calls before it's ready to answer. The loop runs
# until `stop_reason != "tool_use"`. Below is the full loop in ~25
# lines. Read it carefully — every framework you'll see in later
# notebooks is doing this, just with more abstraction on top.

# %%
def run_calculator_agent(user_message: str, max_turns: int = 10) -> str:
    messages = [{"role": "user", "content": user_message}]
    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system="You are a calculator. Use the add tool for any addition.",
            tools=[add_tool_schema],
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            value = add(**block.input)  # only one tool today
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": str(value),
            })
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent didn't finish in {max_turns} turns.")


# %%
print(run_calculator_agent("What is 17 + 25 + 100 + 1?"))

# %% [markdown]
# ## Reflect
#
# - An agent is the messages.create loop above. Nothing else.
# - The model never runs tools — *you* run them and feed results back.
# - Every framework you'll see later wraps this same loop with extra
#   conveniences: tool registries, parallel calls, streaming, etc.
# - The loop must terminate. If the model is stuck in a tool-use cycle,
#   you bail out (we used `max_turns`).
# - **Next:** NB 02 generalises this with a tool registry and multiple
#   tools (web search, fetch, file read).
```

- [ ] **Step 3: Sync the .py source to .ipynb**

```bash
.agents/bin/jupytext --sync notebooks/_src/01_bare_agent_loop.py
```

Expected: creates `notebooks/01_bare_agent_loop.ipynb`. (jupytext uses the
`formats` directive in `pyproject.toml` to know where to put it.)

- [ ] **Step 4: Verify the notebook executes top-to-bottom**

This requires `ANTHROPIC_API_KEY` set. The user runs this; if no key
is available, the executor flags this and skips Step 4 — Step 5
(static lint) still runs. Run:

```bash
.agents/bin/jupyter execute notebooks/01_bare_agent_loop.ipynb
```

Expected: completes without error; cells produce text output.

- [ ] **Step 5: Static check — Python parses cleanly**

```bash
.agents/bin/python -c "import ast; ast.parse(open('notebooks/_src/01_bare_agent_loop.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add notebooks/_src/_common.py notebooks/_src/01_bare_agent_loop.py notebooks/01_bare_agent_loop.ipynb
git commit -m "feat(nb01): bare agent loop, from scratch

A ~30-line agent built without any agentlab abstractions, so the loop
itself is visible. Calculator + add tool. Companion to NB 02 which
introduces the tool registry.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Build NB 02 — Tool use, properly

NB 02 introduces `agentlab.tools.ToolRegistry` and `agentlab.llm.run_agent_loop`,
with three tools: `web_search` (Tavily), `web_search_anthropic` (Anthropic's
built-in), `fetch_url`, and `read_file`. Demonstrates Pydantic-validated
schemas, tool errors, and parallel tool calls.

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/02_tool_use.py`
- Generated: `/home/rob/PythonEnvironments/Agents/notebooks/02_tool_use.ipynb`

- [ ] **Step 1: Write `notebooks/_src/02_tool_use.py`**

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 02 — Tool use, properly
#
# **Goal:** graduate from NB 01's hand-rolled loop to a reusable
# pattern: a *tool registry* that takes Python functions, generates
# JSON schemas from their signatures, and hands you `.schemas()` +
# `.handlers()` ready to plug into the agent loop.
#
# We'll also see:
# - **Two web-search tools:** Tavily (third-party) and Anthropic's
#   built-in web search, side by side. You'll see why "tool" can mean
#   "function I run locally" or "managed tool the model invokes."
# - **Tool errors:** what happens when a tool raises, and why surfacing
#   the error to the model is usually the right move.
# - **Parallel tool calls:** the model can issue multiple tool_use
#   blocks in one turn — `run_agent_loop` runs all of them before
#   returning the next message.

# %%
from _common import cost_banner, load_env

load_env()
cost_banner(
    notebook="02 — Tool use, properly",
    estimate="$0.01–0.03 (depends on web search count)",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Define tools as decorated Python functions
#
# Compare with NB 01, where we hand-wrote `add_tool_schema`. Now the
# registry inspects the signature and generates the schema for us.

# %%
import os
from pathlib import Path

import httpx
from agentlab.tools import ToolRegistry

registry = ToolRegistry()


@registry.tool(description="Read a UTF-8 text file from disk.")
def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


@registry.tool(description="Fetch a URL and return its text body (truncated to 8 KiB).")
def fetch_url(url: str) -> str:
    response = httpx.get(url, timeout=10.0, follow_redirects=True)
    response.raise_for_status()
    return response.text[:8192]


# %% [markdown]
# ## Step 2 — Inspect the generated schemas
#
# This is what gets sent to the API as the `tools` parameter. It's the
# same shape you wrote by hand in NB 01.

# %%
import json

print(json.dumps(registry.schemas(), indent=2))

# %% [markdown]
# ## Step 3 — Web search via Tavily (a "local" tool)
#
# Tavily is a third-party search API. From the model's perspective,
# this is just another function we run locally. The model never talks
# to Tavily — *we* do, and we hand the result back.
#
# This cell skips itself if `TAVILY_API_KEY` isn't set.

# %%
if os.environ.get("TAVILY_API_KEY"):
    from tavily import TavilyClient

    tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    @registry.tool(description="Search the web with Tavily; returns top results as a JSON string.")
    def web_search(query: str, max_results: int = 5) -> str:
        results = tavily.search(query=query, max_results=max_results)
        return json.dumps(results.get("results", []), indent=2)

    print("Tavily web_search registered.")
else:
    print("TAVILY_API_KEY not set — skipping Tavily registration.")
    print("(NB 02 still works; we'll show Anthropic's managed web search next.)")

# %% [markdown]
# ## Step 4 — Run the agent
#
# Use `agentlab.llm.run_agent_loop` instead of writing the loop by
# hand. Compare against NB 01: same shape, but the loop logic lives in
# one place now.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client, run_agent_loop

client = get_client()

result = run_agent_loop(
    client=client,
    system=(
        "You are a research assistant. Use tools when you need facts you don't "
        "have. Cite the URLs you used."
    ),
    user_message=(
        "Read the file `data/seed.txt` and summarise its main points. "
        "If it isn't there, say so."
    ),
    tools=registry.schemas(),
    tool_handlers=registry.handlers(),
    max_turns=8,
)
print(f"Turns: {result.turns}")
print(result.final_text)

# %% [markdown]
# ## Step 5 — Anthropic's managed web search
#
# Anthropic provides a *server-side* web-search tool: you declare it
# but never implement it locally. The model invokes it on Anthropic's
# infrastructure and returns results in-band. Useful when you don't
# want to manage a search API yourself.
#
# Note the schema shape: `{"type": "web_search_20250305", "name": "web_search"}` —
# this is a *managed* tool, identified by type, not a function we
# define.

# %%
managed_search_tool = {"type": "web_search_20250305", "name": "web_search"}

response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=1024,
    system="You are a careful research assistant. Use the web_search tool when you need current info.",
    tools=[managed_search_tool],
    messages=[{"role": "user", "content": "What is the latest stable Python release? Give a one-line answer."}],
)
print("stop_reason:", response.stop_reason)
for block in response.content:
    print(getattr(block, "text", block))

# %% [markdown]
# ## Step 6 — Tool errors are model-visible
#
# When a tool raises, `run_agent_loop` catches the exception and feeds
# it back to the model as `tool_result` with `is_error=True`. The model
# typically retries with corrected arguments. This is much better than
# crashing — the agent can recover from bad inputs.

# %%
result = run_agent_loop(
    client=client,
    system="You are a file-reading assistant.",
    user_message="Read /tmp/this-file-definitely-does-not-exist-xyz.txt",
    tools=registry.schemas(),
    tool_handlers=registry.handlers(),
    max_turns=4,
)
print(result.final_text)

# %% [markdown]
# ## Reflect
#
# - The tool registry lets you define tools as plain Python functions;
#   schemas are derived from signatures (or Pydantic models for richer
#   types).
# - "Tool" can mean a function you run locally OR a managed tool the
#   model runs on Anthropic's side (web search, code execution).
# - Returning errors to the model usually beats raising — the model
#   adapts. Crash only on truly unrecoverable conditions.
# - The model can issue multiple `tool_use` blocks in one turn; the
#   loop should handle them all before the next API call.
# - **Next:** NB 03 introduces structured outputs (Pydantic-validated
#   `Answer` objects) and the recurring "research assistant" project.
```

- [ ] **Step 2: Create the seed data file used by NB 02 Step 4**

Create `data/seed.txt`:

```
The agent loop is the heart of every AI agent: an LLM in a loop with
tool calls. Each turn the model decides whether to answer or to call
a tool; if it calls a tool, the runtime executes it and feeds the
result back. The loop terminates when the model returns end_turn.

This file exists so notebook 02's read_file tool has something real
to chew on.
```

- [ ] **Step 3: Sync the .py source to .ipynb**

```bash
.agents/bin/jupytext --sync notebooks/_src/02_tool_use.py
```

Expected: creates `notebooks/02_tool_use.ipynb`.

- [ ] **Step 4: Static check**

```bash
.agents/bin/python -c "import ast; ast.parse(open('notebooks/_src/02_tool_use.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 5: Verify the notebook executes (requires API key)**

```bash
.agents/bin/jupyter execute notebooks/02_tool_use.ipynb
```

Expected: completes without error.

- [ ] **Step 6: Commit**

```bash
git add notebooks/_src/02_tool_use.py notebooks/02_tool_use.ipynb data/seed.txt
git commit -m "feat(nb02): tool use with the agentlab registry

Three tools (read_file, fetch_url, optional Tavily web_search), plus
Anthropic's managed web search side-by-side. Demonstrates schema
generation from function signatures, run_agent_loop, tool error
handling.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Build NB 03 — Structured outputs & prompt design (spine debut)

NB 03 introduces:
1. System prompts for agents (vs. chat).
2. Pydantic-validated structured outputs via tool-use coercion.
3. The "research assistant" spine project: ask a technical question, get back an `Answer{ summary, citations[] }`.

The technique: define a `submit_answer` tool whose `input_schema` is
generated from `agentlab.types.Answer`. Tell the model it MUST call
`submit_answer` exactly once with the final result. The model's
"output" is then a Pydantic-validated object, not free-form text.

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/03_structured_outputs.py`
- Generated: `/home/rob/PythonEnvironments/Agents/notebooks/03_structured_outputs.ipynb`

- [ ] **Step 1: Write `notebooks/_src/03_structured_outputs.py`**

```python
# ---
# jupyter:
#   jupytext:
#     formats: notebooks///ipynb,notebooks/_src///py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (.agents)
#     language: python
#     name: agents
# ---

# %% [markdown]
# # NB 03 — Structured outputs & prompt design (✦ research assistant debuts)
#
# **Goal:** stop parsing free-form text. From now on, our agents emit
# Pydantic-validated objects we can trust at the type level.
#
# **Two ways to get structured output from Claude:**
#
# 1. **JSON-mode prompts** — ask for JSON in the prompt and parse it.
#    Cheap, but unreliable: nothing forces the model to comply, and
#    Pydantic validation might fail on the third call in a hundred.
# 2. **Tool-use coercion** — define a tool whose schema *is* your
#    output type, tell the model to call it exactly once with the
#    final answer. The model can't return free text any more — it has
#    to fill the tool's schema, which Anthropic validates.
#
# We'll use option 2 throughout the curriculum. It's nearly free
# reliability for the cost of one tool definition.
#
# **Spine debut:** this is also where the *research assistant* project
# starts. Today it answers questions using Anthropic's managed web
# search and returns `Answer{ summary, citations[] }`. By NB 15 it'll
# be a deployable script with subagents and an MCP-backed corpus.

# %%
from _common import cost_banner, load_env

load_env()
cost_banner(
    notebook="03 — Structured outputs (research assistant debuts)",
    estimate="$0.02–0.05",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Define the output type
#
# `Answer` and `Citation` live in `agentlab.types` (Task 4). We import
# them here so notebooks past Module A can reuse them too.

# %%
from agentlab.types import Answer, Citation

print(Answer.model_json_schema())

# %% [markdown]
# ## Step 2 — Make the type into a tool
#
# Anthropic's API takes a list of tools, each with an `input_schema`.
# We use `Answer.model_json_schema()` directly as that schema. The
# model "calls" `submit_answer` with our exact shape; we just collect
# the call.

# %%
import json

submit_answer_tool = {
    "name": "submit_answer",
    "description": (
        "Submit your final answer. Must be called exactly once when "
        "you're ready to respond. The summary must be at least one "
        "sentence and you must include at least one citation."
    ),
    "input_schema": Answer.model_json_schema(),
}

print(json.dumps(submit_answer_tool["input_schema"], indent=2)[:600], "...")

# %% [markdown]
# ## Step 3 — Build the research assistant
#
# It uses Anthropic's managed web search to gather sources, then calls
# `submit_answer` to deliver. We do NOT use `run_agent_loop` here
# because we want a custom termination rule: stop when the model
# returns a `submit_answer` tool_use, not on `end_turn`.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()

SYSTEM_PROMPT = """You are a careful technical research assistant.

Workflow:
1. If you need facts you aren't certain of, use the web_search tool to gather sources.
2. Cross-check at least two sources where possible.
3. When you're ready, call the submit_answer tool with:
   - summary: 2-4 sentences answering the question.
   - citations: list of {url, title, optional quote}. At least one entry; ideally 2-3.
4. Do not produce free-form text after calling submit_answer.
"""


def research(question: str, *, max_turns: int = 8) -> Answer:
    messages = [{"role": "user", "content": question}]
    for _turn in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[
                {"type": "web_search_20250305", "name": "web_search"},
                submit_answer_tool,
            ],
            messages=messages,
        )

        # Did the model submit?
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                return Answer.model_validate(block.input)

        # Otherwise this was either web_search (managed; result already inline)
        # or text reasoning. Append the assistant turn and continue.
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            raise RuntimeError(
                "Model said end_turn without calling submit_answer. "
                "Tighten the system prompt."
            )

    raise RuntimeError(f"Research didn't finish in {max_turns} turns.")


# %%
answer = research("What is the practical difference between Claude's extended thinking and a ReAct loop?")

print("Summary:", answer.summary)
print()
print("Citations:")
for c in answer.citations:
    print(f"  - {c.title} — {c.url}")
    if c.quote:
        print(f"      \"{c.quote}\"")

# %% [markdown]
# ## Step 4 — Why structured output matters
#
# Run a downstream consumer of the answer. Because `Answer` is a real
# Pydantic model, we get type checking for free.

# %%
def render_markdown(answer: Answer) -> str:
    lines = [f"## Summary\n\n{answer.summary}\n\n## Sources\n"]
    for c in answer.citations:
        line = f"- [{c.title}]({c.url})"
        if c.quote:
            line += f" — *{c.quote}*"
        lines.append(line)
    return "\n".join(lines)


print(render_markdown(answer))

# %% [markdown]
# ## Step 5 — Compare with prompt-only JSON
#
# For contrast, here's the same task done with a "please return JSON"
# prompt. Run it 5 times — you'll see the cracks.

# %%
import time

PROMPT_ONLY = """You are a research assistant. Answer the user's question.

Return ONLY a JSON object with this shape:
{
  "summary": "string",
  "citations": [{"url": "string", "title": "string", "quote": "string or null"}]
}
No prose around it. No markdown fences."""


for attempt in range(3):  # 3 not 5 to keep cost down; bump if curious
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=PROMPT_ONLY,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": "What is MCP, in two sentences?"}],
    )
    raw = "".join(b.text for b in response.content if hasattr(b, "text"))
    try:
        parsed = Answer.model_validate_json(raw)
        print(f"Attempt {attempt + 1}: ✓ parsed cleanly")
    except Exception as e:
        print(f"Attempt {attempt + 1}: ✗ {type(e).__name__}: {str(e)[:120]}")
    time.sleep(0.5)

# %% [markdown]
# ## Reflect
#
# - **Tool-use coercion gives near-100% schema compliance.** The model
#   physically can't return free text once it decides to "call"
#   submit_answer.
# - **System prompts shape behaviour more than user prompts.** Use them
#   to define the agent's role, workflow, and termination rules.
# - **Validate at the boundary.** Pydantic catches drift the moment it
#   happens; downstream code can assume types.
# - **Spine project debut:** the `research` function above will grow.
#   In NB 06 it'll get memory; in NB 07 it'll retrieve from a notes
#   corpus; in NB 12 it'll fan out to subagents; in NB 15 it leaves
#   the notebook entirely as `research_assistant.py`.
# - **Next (Module B):** ReAct, planning, memory, RAG, evals,
#   observability — the production-shaping module.
```

- [ ] **Step 2: Sync the .py source to .ipynb**

```bash
.agents/bin/jupytext --sync notebooks/_src/03_structured_outputs.py
```

Expected: creates `notebooks/03_structured_outputs.ipynb`.

- [ ] **Step 3: Static check**

```bash
.agents/bin/python -c "import ast; ast.parse(open('notebooks/_src/03_structured_outputs.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Verify the notebook executes (requires API key)**

```bash
.agents/bin/jupyter execute notebooks/03_structured_outputs.ipynb
```

Expected: completes without error. The "compare with prompt-only JSON"
cell may produce some failures — that's the lesson; it's not an error.

- [ ] **Step 5: Commit**

```bash
git add notebooks/_src/03_structured_outputs.py notebooks/03_structured_outputs.ipynb
git commit -m "feat(nb03): structured outputs + research assistant spine debut

Tool-use coercion for schema compliance. The research assistant project
debuts here, returning Pydantic-validated Answer objects. Will grow
across notebooks; finishes as a deployable script in NB 15.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 10: GitHub Actions CI — pytest + notebook static check

A minimal CI that:
1. Sets up Python 3.13 with uv.
2. Runs the pytest suite.
3. Statically parses each notebook source `.py` file (no API calls).

Notebook *execution* is not run in CI because it would require an
Anthropic API key as a secret and tokens cost real money on every push.
Local pre-merge execution is documented in the README.

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/.github/workflows/test.yml`

- [ ] **Step 1: Write `.github/workflows/test.yml`**

```yaml
name: test

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "0.11.7"

      - name: Set up Python
        run: uv python install 3.13

      - name: Sync dependencies
        run: uv sync --extra dev

      - name: Run pytest
        run: uv run pytest -v

      - name: Static check notebook sources
        run: |
          for f in notebooks/_src/*.py; do
            echo "checking $f"
            uv run python -c "import ast; ast.parse(open('$f').read())"
          done
```

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/test.yml
git commit -m "ci: pytest + static notebook check on push/PR

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 11: Final integration — run everything end-to-end and push

Verify the entire repo is in a working state, then push to GitHub.

- [ ] **Step 1: Run the full test suite**

```bash
.agents/bin/python -m pytest -v
```

Expected: all green (smoke + types + llm + tools tests).

- [ ] **Step 2: Static check all notebook sources**

```bash
for f in notebooks/_src/*.py; do
  echo "checking $f"
  .agents/bin/python -c "import ast; ast.parse(open('$f').read())"
done
```

Expected: each prints "checking ..." with no exception.

- [ ] **Step 3: Execute all three notebooks (requires API key)**

```bash
for f in notebooks/01_bare_agent_loop.ipynb notebooks/02_tool_use.ipynb notebooks/03_structured_outputs.ipynb; do
  echo "executing $f"
  .agents/bin/jupyter execute "$f"
done
```

Expected: each completes without error. (NB 03's "compare with prompt-only JSON" cell may print failures — that's the lesson, not an error.)

- [ ] **Step 4: Re-sync notebook sources to capture any execution outputs**

```bash
for f in notebooks/_src/*.py; do
  .agents/bin/jupytext --sync "$f"
done
```

This makes sure the committed `.ipynb` files contain executed cell
outputs so a learner cloning the repo sees expected results without
running them first.

- [ ] **Step 5: Commit any output diffs**

```bash
git add notebooks/
if git diff --cached --quiet; then
  echo "no output diffs to commit"
else
  git commit -m "chore: update notebook outputs after end-to-end run

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
fi
```

- [ ] **Step 6: Push to GitHub**

```bash
git push origin main
```

Expected: pushes successfully. CI runs on GitHub.

- [ ] **Step 7: Verify CI passes**

```bash
gh run watch
```

Or check the Actions tab on GitHub. Expected: green build.

---

## Self-review (already performed by the plan author)

- **Spec coverage:** Module A (NB 01-03) is fully covered by Tasks 7-9.
  The `agentlab` package skeleton + `llm` + `tools` + `types`
  (Tasks 2, 4, 5, 6) cover the conventions section. README, LICENSE,
  .env.example, and CI cover the repo-layout and conventions sections.
  Modules B/C/D are explicitly deferred to later plans (stated in the
  plan header).
- **Placeholders:** none. Every step shows actual code or an exact
  command.
- **Type consistency:** `Answer`, `Citation`, `AgentLoopResult`,
  `ToolRegistry`, and `tool` are defined once and used consistently
  across tasks. Function signatures match between definition and use.
- **Out-of-scope guards:** the plan does not pull in chromadb, MCP,
  computer-use, or local-model deps — those are deferred to module-b/c/d
  optional groups and the corresponding future plans.
