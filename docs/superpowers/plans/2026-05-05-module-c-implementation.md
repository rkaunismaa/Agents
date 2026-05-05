# Module C — Multi-Agent + MCP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver Module C of the curriculum (NB 09-12) — four executable JupyterLab notebooks that teach building an MCP server, consuming MCP from an agent, orchestrator-worker patterns, and parallel/durable spine integration — plus the supporting `src/agentlab/mcp_helpers.py` and `src/agentlab/spine.py` library, the canonical FastMCP `mcp_servers/notes_server.py`, and a 5-task extension to `data/eval_tasks.jsonl`.

**Architecture:** Same shape as Modules A & B. Reusable code lands in `src/agentlab/` with TDD-driven tests. Notebooks are authored as `.py` files in `notebooks/_src/` and synced to `.ipynb` via jupytext. NB 09 builds the notes server (low-level `Server` → `FastMCP`); the FastMCP version becomes the canonical `mcp_servers/notes_server.py`. NB 10 wires an MCP client + tool routing into the existing agent loop, landing helpers in `mcp_helpers.py`. NB 11 builds an orchestrator-worker pattern inline two ways (scratch + `claude-agent-sdk`). NB 12 extracts the orchestrator to `spine.py`, evolves it with async fan-out + cancellation + JSONL checkpointing + MCP-backed memory, then runs the extended eval suite.

**Tech Stack additions:** None new. The `[module-c]` extra (`mcp>=1.0.0`, `claude-agent-sdk>=0.1.0`) is already declared in `pyproject.toml` and just needs to be sync'd into `.agents`.

**Spec:** [`../specs/2026-05-05-module-c-multi-agent-mcp-design.md`](../specs/2026-05-05-module-c-multi-agent-mcp-design.md).

**Subsequent plans:** Module D (NB 13-15 — autonomous + local + capstone) gets its own plan after Module C ships.

---

## Files created/modified in this plan

```
# Created
src/agentlab/mcp_helpers.py                        # mcp_tools_to_anthropic + MCPToolRouter
src/agentlab/spine.py                              # Subagent + Orchestrator (sync/async/checkpoint)
tests/test_mcp_helpers.py                          # TDD tests for mcp_helpers
tests/test_spine.py                                # TDD tests for spine
mcp_servers/__init__.py                            # empty marker
mcp_servers/notes_server.py                        # canonical FastMCP notes server
notebooks/_src/09_building_an_mcp_server.py
notebooks/_src/10_consuming_mcp_in_an_agent.py
notebooks/_src/11_orchestrator_and_subagents.py
notebooks/_src/12_parallel_and_durable.py
notebooks/09_building_an_mcp_server.ipynb          # jupytext-generated
notebooks/10_consuming_mcp_in_an_agent.ipynb
notebooks/11_orchestrator_and_subagents.ipynb
notebooks/12_parallel_and_durable.ipynb
data/checkpoints/.gitkeep                          # placeholder so dir exists

# Modified
.gitignore                                         # add data/notes.json + data/checkpoints/
data/eval_tasks.jsonl                              # +5 parallel-friendly rows (t11-t15)
README.md                                          # flip Module C row to ✅, add NB descriptions
pyproject.toml                                     # pin claude-agent-sdk minor after first NB 11 run
```

---

## Task C-1: Branch + module-c dependency sync

**Files:**
- No file edits. Branch creation + env sync only.

- [ ] **Step 1: Confirm branch + clean working tree**

```bash
git status
git log --oneline -1
```

Expected: branch `feature/module-c`, HEAD at `dba24e8 docs(spec): Module C — Multi-Agent + MCP design spec`. Working tree may show modified `.gitignore` and untracked `.venv` (pre-existing, leave alone).

If not on `feature/module-c`, run `git checkout feature/module-c`.

- [ ] **Step 2: Sync the env with module-c extras**

```bash
UV_PROJECT_ENVIRONMENT=.agents uv sync --extra dev --extra module-b --extra module-c
```

Expected: solver completes, installs `mcp` and `claude-agent-sdk` (+ their transitive deps). May download a few packages on first run. No errors. Module B's deps (chromadb, sentence-transformers, opentelemetry) remain installed.

- [ ] **Step 3: Smoke-test the imports**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/python -c "import mcp; from mcp.server.fastmcp import FastMCP; from mcp.server import Server; from mcp.client.stdio import stdio_client; from mcp import ClientSession; import claude_agent_sdk; print('OK')"
```

Expected: prints `OK`. If `claude_agent_sdk` import path differs in the installed version, run `UV_PROJECT_ENVIRONMENT=.agents .agents/bin/python -c "import claude_agent_sdk; print(claude_agent_sdk.__file__)"` to discover the actual surface and adjust later imports accordingly.

- [ ] **Step 4: Verify MCP CLI is available**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/mcp --help
```

Expected: prints usage including `dev` subcommand. This binary will host the MCP Inspector in NB 09.

- [ ] **Step 5: No commit yet** — `uv.lock` may have updated but the dep additions were already in `pyproject.toml`. Ship the `uv.lock` change with the next functional commit (Task C-2's `mcp_helpers.py` work).

---

## Task C-2: `src/agentlab/mcp_helpers.py` (TDD)

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_mcp_helpers.py`
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/mcp_helpers.py`

**Why TDD now:** NB 10 imports `mcp_tools_to_anthropic` and `MCPToolRouter`. Land them with green tests before the NB depends on them.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mcp_helpers.py`:

```python
"""Tests for agentlab.mcp_helpers: schema conversion + tool router.

We do not spawn a real MCP server here — these are unit tests against
fakes that mimic the SDK's `ClientSession` surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentlab.mcp_helpers import MCPToolRouter, mcp_tools_to_anthropic


# ── Fakes that mimic mcp.types ─────────────────────────────────────


@dataclass
class _FakeMCPTool:
    name: str
    description: str
    inputSchema: dict


@dataclass
class _FakeListToolsResult:
    tools: list[_FakeMCPTool]


@dataclass
class _FakeContent:
    type: str
    text: str


@dataclass
class _FakeCallResult:
    content: list[_FakeContent]
    isError: bool = False


class _FakeSession:
    """Minimal stand-in for mcp.ClientSession."""

    def __init__(self, tools: list[_FakeMCPTool], call_results: dict[str, _FakeCallResult]):
        self._tools = tools
        self._call_results = call_results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(tools=list(self._tools))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeCallResult:
        self.calls.append((name, arguments))
        return self._call_results[name]


# ── mcp_tools_to_anthropic ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_tools_to_anthropic_basic():
    """Each MCP Tool maps to an Anthropic tool dict with the same schema."""
    session = _FakeSession(
        tools=[
            _FakeMCPTool(
                name="add_note",
                description="Add a note.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["key", "content"],
                },
            ),
            _FakeMCPTool(
                name="list_notes",
                description="List all note keys.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ],
        call_results={},
    )

    result = await mcp_tools_to_anthropic(session)

    assert isinstance(result, list) and len(result) == 2
    assert result[0] == {
        "name": "add_note",
        "description": "Add a note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["key", "content"],
        },
    }
    assert result[1]["name"] == "list_notes"
    assert result[1]["input_schema"] == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_mcp_tools_to_anthropic_handles_missing_description():
    """A None or missing description becomes the empty string (Anthropic requires str)."""
    session = _FakeSession(
        tools=[_FakeMCPTool(name="foo", description=None, inputSchema={"type": "object"})],  # type: ignore[arg-type]
        call_results={},
    )

    result = await mcp_tools_to_anthropic(session)

    assert result[0]["description"] == ""


# ── MCPToolRouter ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_call_returns_text_content():
    """Successful call returns the joined text content blocks."""
    session = _FakeSession(
        tools=[],
        call_results={
            "add_note": _FakeCallResult(content=[_FakeContent(type="text", text="ok: stored 'hello'")]),
        },
    )
    router = MCPToolRouter(session)

    result = await router.call("add_note", {"key": "h", "content": "hello"})

    assert result == "ok: stored 'hello'"
    assert session.calls == [("add_note", {"key": "h", "content": "hello"})]


@pytest.mark.asyncio
async def test_router_call_concatenates_multiple_text_blocks():
    """If the server returns multiple text blocks, the router joins them with newlines."""
    session = _FakeSession(
        tools=[],
        call_results={
            "list_notes": _FakeCallResult(
                content=[
                    _FakeContent(type="text", text="key1"),
                    _FakeContent(type="text", text="key2"),
                ]
            ),
        },
    )
    router = MCPToolRouter(session)

    result = await router.call("list_notes", {})

    assert result == "key1\nkey2"


@pytest.mark.asyncio
async def test_router_call_error_result_raises():
    """isError=True surfaces as a RuntimeError carrying the error text."""
    session = _FakeSession(
        tools=[],
        call_results={
            "get_note": _FakeCallResult(
                content=[_FakeContent(type="text", text="key not found: missing")],
                isError=True,
            ),
        },
    )
    router = MCPToolRouter(session)

    with pytest.raises(RuntimeError, match="key not found"):
        await router.call("get_note", {"key": "missing"})


@pytest.mark.asyncio
async def test_router_knows_tool():
    """`knows(name)` is True iff the session listed that tool."""
    session = _FakeSession(
        tools=[_FakeMCPTool(name="add_note", description="d", inputSchema={"type": "object"})],
        call_results={},
    )
    router = MCPToolRouter(session)
    await router.refresh()

    assert router.knows("add_note") is True
    assert router.knows("delete_note") is False
```

- [ ] **Step 2: Run tests to verify they fail with import errors**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_mcp_helpers.py -v
```

Expected: `ERRORS` collecting tests — `ModuleNotFoundError: No module named 'agentlab.mcp_helpers'`. This is the red phase.

- [ ] **Step 3: Implement `src/agentlab/mcp_helpers.py`**

Create `src/agentlab/mcp_helpers.py`:

```python
"""Helpers for consuming MCP servers from an Anthropic agent loop.

`mcp_tools_to_anthropic` converts MCP tool listings into the dict shape
the Anthropic SDK expects under `tools=[...]`.

`MCPToolRouter` wraps an MCP `ClientSession` and dispatches tool calls,
joining text-content blocks from the server response and surfacing
`isError=True` results as exceptions.
"""
from __future__ import annotations

from typing import Any


async def mcp_tools_to_anthropic(session: Any) -> list[dict]:
    """Discover tools on `session` and return Anthropic-shaped tool dicts.

    Each MCP `Tool` already carries a JSON Schema in `inputSchema`, which
    is what the Anthropic SDK wants under `input_schema`.
    """
    listed = await session.list_tools()
    out: list[dict] = []
    for tool in listed.tools:
        out.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            }
        )
    return out


class MCPToolRouter:
    """Routes Anthropic tool calls back through an MCP `ClientSession`.

    Usage:
        router = MCPToolRouter(session)
        await router.refresh()                     # populate known tool names
        if router.knows(name):
            result = await router.call(name, args) # str

    The router only handles tool routing; resource and prompt access stay
    on `session` directly (those are app-driven, not agent-driven — see
    NB 10 reflect).
    """

    def __init__(self, session: Any):
        self._session = session
        self._known: set[str] = set()

    async def refresh(self) -> None:
        listed = await self._session.list_tools()
        self._known = {t.name for t in listed.tools}

    def knows(self, name: str) -> bool:
        return name in self._known

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        text = "\n".join(
            getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text"
        )
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool '{name}' returned error: {text}")
        return text
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_mcp_helpers.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Verify the rest of the suite still passes**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -q
```

Expected: all existing tests pass + the 6 new ones. `-m 'not eval'` is the default in `pyproject.toml` so eval tests stay skipped.

- [ ] **Step 6: Commit**

```bash
git add src/agentlab/mcp_helpers.py tests/test_mcp_helpers.py uv.lock
git commit -m "$(cat <<'EOF'
feat(mcp_helpers): MCP tool discovery + router for Anthropic agent loop

mcp_tools_to_anthropic converts MCP tool listings to Anthropic's
tool-dict schema; MCPToolRouter dispatches tool calls back through
an MCP ClientSession and joins text-content responses. NB 10 will
import these to wire the notes server into run_agent_loop.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

(Bundling `uv.lock` with this commit is fine — module-c deps were already declared, this just records them in the lockfile alongside the first code that uses them.)

---

## Task C-3: NB 09 — Building an MCP server from scratch

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/mcp_servers/__init__.py` (empty)
- Create: `/home/rob/PythonEnvironments/Agents/mcp_servers/notes_server.py` (canonical FastMCP)
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/09_building_an_mcp_server.py`
- Modify: `/home/rob/PythonEnvironments/Agents/.gitignore` (add `data/notes.json`)
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/09_building_an_mcp_server.ipynb`

- [ ] **Step 1: Create the `mcp_servers/` package marker**

Create `mcp_servers/__init__.py` as an empty file:

```bash
mkdir -p mcp_servers && touch mcp_servers/__init__.py
```

- [ ] **Step 2: Write the canonical FastMCP notes server**

Create `mcp_servers/notes_server.py`:

```python
"""Canonical notes server (FastMCP).

The low-level `Server`-based version of this same surface lives inline
in NB 09 as a teaching artifact. This file is what NB 10 spawns as a
subprocess and what NB 12's spine connects to via stdio.

Persistence: a single JSON file at `data/notes.json` (relative to repo
root). The server validates the file on read and refuses to start if
the schema doesn't match — it never silently overwrites learner data.

Run standalone:
    python mcp_servers/notes_server.py

Inspect interactively:
    mcp dev mcp_servers/notes_server.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTES_PATH = REPO_ROOT / "data" / "notes.json"

mcp = FastMCP("notes-server")


def _load_notes() -> dict[str, dict]:
    """Return {key: {content, created_at}} or {} if file is missing.

    On schema mismatch, prints to stderr and exits non-zero so the
    caller (notebook subprocess or `mcp dev`) sees the error.
    """
    if not NOTES_PATH.exists():
        return {}
    try:
        raw = json.loads(NOTES_PATH.read_text())
    except json.JSONDecodeError as exc:
        print(
            f"notes-server: {NOTES_PATH} is not valid JSON ({exc}); "
            f"refusing to start. Inspect or delete the file manually.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not isinstance(raw, dict):
        print(
            f"notes-server: {NOTES_PATH} top level must be an object "
            f"({{key: {{content, created_at}}}}); refusing to start.",
            file=sys.stderr,
        )
        sys.exit(2)
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "content" not in entry or "created_at" not in entry:
            print(
                f"notes-server: malformed entry for key '{key}' in "
                f"{NOTES_PATH}; expected {{content, created_at}}. "
                f"Refusing to start.",
                file=sys.stderr,
            )
            sys.exit(2)
    return raw


def _save_notes(notes: dict[str, dict]) -> None:
    NOTES_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTES_PATH.write_text(json.dumps(notes, indent=2, sort_keys=True))


@mcp.tool()
def add_note(key: str, content: str) -> str:
    """Add or overwrite a note with the given key."""
    notes = _load_notes()
    notes[key] = {
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_notes(notes)
    return f"stored '{key}'"


@mcp.tool()
def get_note(key: str) -> str:
    """Return the content of the note with the given key, or an error message."""
    notes = _load_notes()
    if key not in notes:
        return f"key not found: {key}"
    return notes[key]["content"]


@mcp.tool()
def list_notes() -> str:
    """Return all note keys, one per line, sorted alphabetically."""
    notes = _load_notes()
    if not notes:
        return "(no notes)"
    return "\n".join(sorted(notes.keys()))


@mcp.tool()
def delete_note(key: str) -> str:
    """Delete the note with the given key. Returns whether it existed."""
    notes = _load_notes()
    if key not in notes:
        return f"key not found: {key}"
    del notes[key]
    _save_notes(notes)
    return f"deleted '{key}'"


@mcp.resource("notes://{key}")
def note_resource(key: str) -> str:
    """Read a single note as a resource."""
    notes = _load_notes()
    if key not in notes:
        raise ValueError(f"unknown note: {key}")
    return notes[key]["content"]


@mcp.prompt()
def summarize_notes() -> str:
    """Prompt template that asks the agent to digest all current notes."""
    notes = _load_notes()
    if not notes:
        return "There are no notes to summarize."
    body = "\n\n".join(
        f"### {key}\n{entry['content']}" for key, entry in sorted(notes.items())
    )
    return (
        "Below are the user's saved notes. Produce a concise digest "
        "grouping related items and surfacing recurring themes. Use "
        "neutral, direct language.\n\n"
        f"{body}"
    )


if __name__ == "__main__":
    mcp.run()
```

- [ ] **Step 3: Update `.gitignore` for runtime files**

The repo's `.gitignore` already has uncommitted local changes (pre-existing). Don't stage those; only add the new lines for Module C runtime files. Append at the end of `.gitignore`:

```
# Module C runtime artifacts
/data/notes.json
/data/checkpoints/
```

(Use a literal `data/notes.json` ignore pattern — both `data/notes.json` and `/data/notes.json` work; the leading `/` anchors to repo root and avoids accidentally ignoring nested `data/notes.json` files.)

- [ ] **Step 4: Smoke-test the FastMCP server starts**

```bash
UV_PROJECT_ENVIRONMENT=.agents timeout 3 .agents/bin/python mcp_servers/notes_server.py < /dev/null ; echo "exit=$?"
```

Expected: exit code 124 (timeout killed it after 3s; the server was happily idling on stdio waiting for input). Any other exit code (1, 2, etc.) means a startup error — read stderr.

- [ ] **Step 5: Write the NB 09 source**

Create `notebooks/_src/09_building_an_mcp_server.py`:

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
# # NB 09 — Building an MCP server from scratch
#
# **Goal:** show how MCP works at the protocol level, then graduate to
# the idiomatic API.
#
# We build the **same notes server** twice:
# 1. **Low-level `Server`** — handlers wired by hand to the stdio
#    transport. ~80–100 lines. You see the protocol shape.
# 2. **`FastMCP`** — decorator-based, signature → schema (like our
#    `ToolRegistry` from NB 02). ~30 lines. Same surface.
#
# Then we add a **Prompt** (the third MCP primitive) and an appendix on
# Streamable HTTP transport.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="09 — Building an MCP server from scratch",
    estimate="$0.01",
    model="claude-sonnet-4-6 (one demo call only)",
)

# %% [markdown]
# ## Step 1 — Design the notes server (no code yet)
#
# **On-disk shape** (single JSON file at `data/notes.json`):
#
# ```json
# {
#   "fav_test_lib": {
#     "content": "pytest",
#     "created_at": "2026-05-05T12:34:56+00:00"
#   }
# }
# ```
#
# **Tools the server exposes:**
#
# | Name | Inputs | Returns |
# |---|---|---|
# | `add_note` | `key: str, content: str` | confirmation string |
# | `get_note` | `key: str` | content or "key not found" |
# | `list_notes` | — | newline-separated keys |
# | `delete_note` | `key: str` | confirmation or "key not found" |
#
# **Resources:** each note as `notes://{key}`. Resources are pulled
# explicitly by a host app — the agent doesn't auto-discover them.
#
# **Prompt:** `summarize_notes` — a templated prompt the host app can
# offer the user (e.g. as a slash command) that injects the current
# notes corpus and asks for a digest.

# %% [markdown]
# ## Step 2 — Low-level `Server` (the protocol-shaped version)
#
# We import the SDK's primitive layer and wire each handler explicitly.
# This is what `FastMCP` is built on top of — see how much glue it's
# hiding from us.

# %%
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server


def _make_low_level_server(notes_path: Path) -> Server:
    """Return a low-level `Server` exposing the notes API on stdio.

    We construct the server inside a function so the notebook can
    reuse it without leaking module-level state.
    """
    server = Server("notes-server-lowlevel")

    def _load() -> dict[str, dict]:
        return json.loads(notes_path.read_text()) if notes_path.exists() else {}

    def _save(notes: dict[str, dict]) -> None:
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(json.dumps(notes, indent=2, sort_keys=True))

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="add_note",
                description="Add or overwrite a note.",
                inputSchema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["key", "content"],
                },
            ),
            types.Tool(
                name="get_note",
                description="Return note content for a given key.",
                inputSchema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
            types.Tool(
                name="list_notes",
                description="List all note keys.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="delete_note",
                description="Delete a note by key.",
                inputSchema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                },
            ),
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict | None) -> list[types.TextContent]:
        args = arguments or {}
        notes = _load()
        if name == "add_note":
            notes[args["key"]] = {
                "content": args["content"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _save(notes)
            return [types.TextContent(type="text", text=f"stored '{args['key']}'")]
        if name == "get_note":
            entry = notes.get(args["key"])
            return [types.TextContent(type="text", text=entry["content"] if entry else f"key not found: {args['key']}")]
        if name == "list_notes":
            text = "\n".join(sorted(notes)) if notes else "(no notes)"
            return [types.TextContent(type="text", text=text)]
        if name == "delete_note":
            if args["key"] in notes:
                del notes[args["key"]]
                _save(notes)
                return [types.TextContent(type="text", text=f"deleted '{args['key']}'")]
            return [types.TextContent(type="text", text=f"key not found: {args['key']}")]
        raise ValueError(f"unknown tool: {name}")

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        notes = _load()
        return [
            types.Resource(
                uri=f"notes://{key}",
                name=key,
                description=f"Note '{key}'",
                mimeType="text/plain",
            )
            for key in sorted(notes)
        ]

    @server.read_resource()
    async def _read_resource(uri: types.AnyUrl) -> str:
        # uri looks like notes://key
        prefix = "notes://"
        s = str(uri)
        if not s.startswith(prefix):
            raise ValueError(f"unsupported scheme: {uri}")
        key = s[len(prefix):]
        notes = _load()
        if key not in notes:
            raise ValueError(f"unknown note: {key}")
        return notes[key]["content"]

    return server


# This is the entrypoint you'd run as `python this_file.py` to host the
# low-level server on stdio. We don't *run* it from the notebook (it
# would block waiting on stdin); we just construct it to show the
# shape.
LOW_LEVEL_SERVER = _make_low_level_server(Path("data/notes_lowlevel.json"))
print(f"Low-level server: {LOW_LEVEL_SERVER.name}")
print("Handlers wired:", "list_tools, call_tool, list_resources, read_resource")

# %% [markdown]
# Notes on Step 2:
# - **Five handlers** were registered by hand. Each one is an `async`
#   function bound to a specific MCP request type via decorator.
# - **`types.TextContent(type="text", ...)`** is the raw shape MCP uses
#   for tool results. FastMCP will hide this — but you should see it
#   once.
# - **No schema generation from signatures.** We hand-rolled JSON
#   Schema for each tool. FastMCP introspects Python signatures
#   (exactly like NB 02's `ToolRegistry`).

# %% [markdown]
# ## Step 3 — Hit it with the MCP Inspector
#
# The MCP CLI ships an inspector that lets you talk to a server
# interactively. From a separate terminal, run:
#
# ```bash
# UV_PROJECT_ENVIRONMENT=.agents .agents/bin/mcp dev mcp_servers/notes_server.py
# ```
#
# (We're pointing it at the canonical FastMCP version — same surface,
# just shorter source. If you point it at a low-level version saved to
# disk, the result is identical.)
#
# It opens a browser UI. Try:
# - **Tools tab:** call `add_note` with `key="hello", content="world"`,
#   then `list_notes`, then `get_note` with `key="hello"`.
# - **Resources tab:** browse to `notes://hello`.
# - **Prompts tab:** invoke `summarize_notes`.
#
# This is the first time we've talked to an MCP server **without an
# agent in the loop**. It clarifies that MCP is a protocol, not a
# Claude-specific thing.

# %% [markdown]
# ## Step 4 — Refactor to FastMCP (the canonical version)
#
# `mcp_servers/notes_server.py` (already on disk) is the FastMCP
# rewrite of the low-level server above. Open it side-by-side and
# compare:
#
# - Each `@mcp.tool()` function's signature **becomes the input
#   schema** automatically. No manual JSON Schema.
# - Resources use a parameterized URI template (`notes://{key}`) and
#   the function's argument matches `{key}`.
# - One file, ~30 lines of meaningful code (vs ~100 for the low-level).

# %%
from pathlib import Path as _Path

# Show the FastMCP source for comparison.
canonical_src = _Path("mcp_servers/notes_server.py").read_text()
print(f"FastMCP version: {len(canonical_src.splitlines())} lines\n")
# Print only the decorated function block to keep output short
import re
funcs = re.findall(r"@mcp\.\w+\([^)]*\)\n[\s\S]*?(?=\n@mcp\.|\nif __name__|\Z)", canonical_src)
for f in funcs:
    print(f.split('\n')[0], '...')

# %% [markdown]
# ## Step 5 — Add a Prompt (third MCP primitive)
#
# `summarize_notes` is already registered in the canonical server with
# `@mcp.prompt()`. It returns a prompt template that includes the
# current notes corpus and asks for a digest.
#
# Why prompts matter: they're **app-driven**, not agent-driven. The
# host app (e.g. a chat UI) might surface them as slash commands. The
# agent doesn't auto-call them; the user (or app) picks them.
#
# Below: invoke the prompt manually + run it through Claude to show the
# end-to-end shape. (This is the only Claude API call in this
# notebook.)

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from mcp_servers.notes_server import summarize_notes

# Seed two notes so the prompt has something to summarize.
client = get_client()
NOTES_FILE = Path("data/notes.json")

if not NOTES_FILE.exists():
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps({
        "fav_test_lib": {"content": "pytest", "created_at": "2026-05-05T12:00:00+00:00"},
        "fav_http_lib": {"content": "httpx for async, requests for sync", "created_at": "2026-05-05T12:01:00+00:00"},
    }, indent=2))

prompt_text = summarize_notes()
print("--- prompt template (first 500 chars) ---")
print(prompt_text[:500])

response = client.messages.create(
    model=DEFAULT_MODEL,
    max_tokens=512,
    messages=[{"role": "user", "content": prompt_text}],
)
print("\n--- model digest ---")
for block in response.content:
    if hasattr(block, "text"):
        print(block.text)

# %% [markdown]
# ## Appendix — Streamable HTTP transport
#
# stdio is great for local development. For multi-client or networked
# scenarios, you'd swap the transport. With FastMCP that's one line:
#
# ```python
# # at the bottom of mcp_servers/notes_server.py:
# if __name__ == "__main__":
#     mcp.run(transport="streamable-http")
# ```
#
# Then `mcp dev` connects over HTTP. The handlers don't change. The
# client side (NB 10) uses a different connect helper:
#
# ```python
# from mcp.client.streamable_http import streamablehttp_client
# async with streamablehttp_client("http://localhost:8000/mcp") as (read, write, _):
#     async with ClientSession(read, write) as session:
#         ...
# ```
#
# We stay on stdio for the rest of Module C — fewer moving parts.

# %% [markdown]
# ## Reflect
#
# - **The protocol is small.** Five handler types (list/call tools,
#   list/read resources, list/get prompts) cover the surface. FastMCP
#   is a quality-of-life wrapper, not a different protocol.
# - **Tools, resources, prompts have different drivers.** Tools are
#   agent-driven (the model picks them). Resources are app-driven (the
#   host fetches them). Prompts are user/app-driven (slash commands,
#   menus). Get this mental model right and the rest of MCP is
#   straightforward.
# - **Next:** NB 10 wires this server into an agent loop and shows the
#   moment when the spine stops needing in-process memory.
```

- [ ] **Step 6: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/09_building_an_mcp_server.py
```

Expected: writes `notebooks/09_building_an_mcp_server.ipynb`.

- [ ] **Step 7: Pause for user to run the notebook**

The user runs the notebook in VS Code (kernel: Python (.agents)). Verify all cells execute without error. The `mcp dev` step (Step 3) is a separate terminal demo — the user can verify the inspector opens or skip it; the in-notebook code does not depend on the inspector running.

Confirm: Step 5's Claude call returns a coherent digest; `data/notes.json` exists at end of run.

- [ ] **Step 8: After confirmation, commit**

```bash
git add mcp_servers/__init__.py mcp_servers/notes_server.py notebooks/_src/09_building_an_mcp_server.py notebooks/09_building_an_mcp_server.ipynb .gitignore
git commit -m "$(cat <<'EOF'
feat(nb09): MCP server two ways — low-level Server then FastMCP

NB 09 builds the notes server twice (low-level handlers, then
@mcp.tool / @mcp.resource / @mcp.prompt decorators), inspects it
via mcp dev, and demos the summarize_notes prompt via Claude.
mcp_servers/notes_server.py is the FastMCP version — canonical
runnable file consumed by NB 10 and NB 12.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task C-4: NB 10 — Consuming MCP in an agent

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/10_consuming_mcp_in_an_agent.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/10_consuming_mcp_in_an_agent.ipynb`

- [ ] **Step 1: Write the NB 10 source**

Create `notebooks/_src/10_consuming_mcp_in_an_agent.py`:

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
# # NB 10 — Consuming MCP in an agent
#
# **Goal:** wire an MCP client into the agent loop. Auto-discover
# tools. Walk through the tool/resource/prompt distinction from the
# consumer side.
#
# We connect to the canonical notes server from NB 09, route Claude's
# tool calls through MCP, and demonstrate the MCP-earning-its-keep
# moment: notes that **outlive the process**.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="10 — Consuming MCP in an agent",
    estimate="$0.02–0.03",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Connect to the notes server
#
# We spawn `mcp_servers/notes_server.py` as a subprocess via the SDK's
# stdio client and open a `ClientSession`. The handshake returns the
# server's capabilities.

# %%
import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


SERVER_PARAMS = StdioServerParameters(
    command="python",
    args=["mcp_servers/notes_server.py"],
)


async def show_handshake():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            init_result = await session.initialize()
            print(f"Server name: {init_result.serverInfo.name}")
            print(f"Server version: {init_result.serverInfo.version}")
            print(f"Protocol version: {init_result.protocolVersion}")
            caps = init_result.capabilities
            print(f"Capabilities: tools={caps.tools is not None}, "
                  f"resources={caps.resources is not None}, "
                  f"prompts={caps.prompts is not None}")


await show_handshake()

# %% [markdown]
# ## Step 2 — Auto-discover tools → Anthropic schemas
#
# `mcp_tools_to_anthropic` (from `agentlab.mcp_helpers`, landed in C-2)
# converts the server's tool listing into the dict shape the Anthropic
# SDK wants under `tools=[...]`.

# %%
from agentlab.mcp_helpers import mcp_tools_to_anthropic


async def show_discovered_tools():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await mcp_tools_to_anthropic(session)
            for t in tools:
                req = t["input_schema"].get("required", [])
                print(f"  {t['name']}({', '.join(req) or '—'}): {t['description']}")


await show_discovered_tools()

# %% [markdown]
# ## Step 3 — Wire MCP into the agent loop
#
# We adapt NB 02's `run_agent_loop` shape: when the model emits a
# `tool_use` block whose name is one of the MCP-discovered tools,
# route it through `MCPToolRouter.call(...)`. Local tools (e.g.
# `web_search`) work alongside MCP tools — same registry shape, just
# routed differently.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.mcp_helpers import MCPToolRouter

client = get_client()

SYSTEM_PROMPT = """You are a helpful assistant with persistent notes.
You can store key/value notes via add_note and retrieve them via
get_note or list_notes. Use them when the user asks you to remember
or recall information."""


async def chat_with_notes(question: str, max_turns: int = 6) -> str:
    """Run a single-question agent loop against the notes server."""
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await mcp_tools_to_anthropic(session)
            router = MCPToolRouter(session)
            await router.refresh()

            messages = [{"role": "user", "content": question}]
            for _ in range(max_turns):
                response = client.messages.create(
                    model=DEFAULT_MODEL,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    tools=tools,
                    messages=messages,
                )
                if response.stop_reason == "end_turn":
                    text = "".join(getattr(b, "text", "") for b in response.content)
                    return text.strip()
                # Append assistant turn so tool_use blocks are preserved for the next call.
                messages.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        if router.knows(block.name):
                            try:
                                result_text = await router.call(block.name, dict(block.input))
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_text,
                                })
                            except RuntimeError as exc:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(exc),
                                    "is_error": True,
                                })
                        else:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"unknown tool: {block.name}",
                                "is_error": True,
                            })
                if not tool_results:
                    raise RuntimeError("Stopped without end_turn and no tool calls.")
                messages.append({"role": "user", "content": tool_results})
            raise RuntimeError("Out of turns.")

# %% [markdown]
# ## Step 4 — Demo conversation
#
# Three turns. Each one is its own session (the MCP server is
# subprocess-spawned per call, not long-lived) — but the **notes file**
# is shared, so state persists across calls. This is the key insight:
# MCP-backed memory outlives any single agent process.

# %%
# Wipe any stale notes from earlier runs to make this demo deterministic.
import os
notes_file = "data/notes.json"
if os.path.exists(notes_file):
    os.remove(notes_file)

print("--- Turn 1 ---")
turn1 = await chat_with_notes(
    "Remember that my favorite Python testing library is pytest. Store it under key 'fav_test_lib'."
)
print(turn1)

print("\n--- Turn 2 (same session, different agent invocation) ---")
turn2 = await chat_with_notes("What was my favorite testing library?")
print(turn2)

print("\n--- Turn 3 (cold start — new agent invocation, same notes file) ---")
turn3 = await chat_with_notes("List my notes, then tell me the value of fav_test_lib.")
print(turn3)

# %% [markdown]
# **What just happened.** Each `chat_with_notes` call spawns a fresh
# subprocess of the MCP server, opens a fresh client session, and runs
# its own agent loop — yet the agent recalls the previous turn's note
# because the *file* on disk persists. **This is MCP earning its
# keep**: state lives outside the agent process.
#
# Compare with NB 06's in-process `KeyValueMemory` — the moment that
# Python kernel restarts, you lose everything. Notes server: survive
# kernel restart, survive notebook restart, survive next week.

# %% [markdown]
# ## Step 5 — Resources walkthrough
#
# Resources are the second MCP primitive. **They're app-driven, not
# agent-driven**: your harness pulls them by URI, the agent doesn't
# auto-discover them.

# %%
async def show_resources():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_resources()
            print(f"Resources advertised: {len(listed.resources)}")
            for r in listed.resources:
                print(f"  {r.uri} — {r.description}")
            if listed.resources:
                first_uri = listed.resources[0].uri
                content = await session.read_resource(first_uri)
                print(f"\nRead {first_uri}:")
                print(f"  {content.contents[0].text!r}")


await show_resources()

# %% [markdown]
# **Why resources, not tools, for "browseable content"?** Because tools
# are decided by the model on each turn — so every browseable note
# would clutter the model's tool list. Resources let your *host*
# decide what to show in a UI (a sidebar, a /notes slash command, etc.)
# without bloating the agent's context.

# %% [markdown]
# ## Step 6 — Prompts walkthrough
#
# Prompts are the third primitive. They're **template fixtures** the
# host app can offer — typically as slash commands or menu items. The
# agent doesn't call prompts; the user (or the harness) does.

# %%
async def show_prompts():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_prompts()
            print(f"Prompts advertised: {len(listed.prompts)}")
            for p in listed.prompts:
                print(f"  {p.name}: {p.description}")
            if listed.prompts:
                got = await session.get_prompt(listed.prompts[0].name)
                msg = got.messages[0]
                preview = getattr(msg.content, "text", str(msg.content))[:200]
                print(f"\nGet '{listed.prompts[0].name}' → {len(got.messages)} message(s)")
                print(f"  {preview}...")


await show_prompts()

# %% [markdown]
# ## Reflect
#
# - **Tool / resource / prompt = three different drivers.** Tool: agent
#   decides. Resource: app fetches. Prompt: user/app picks. Match the
#   primitive to the decision-maker.
# - **MCP earns its keep when state outlives the process.** A purely
#   in-process `KeyValueMemory` would be just as good for a single
#   notebook session. Once you want notes that survive kernel
#   restarts — or shared across processes — MCP pays back the protocol
#   tax.
# - **The agent loop didn't change much.** Just a different routing
#   path for tool calls. NB 02's `run_agent_loop` was almost there;
#   `MCPToolRouter` is the small adapter.
# - **Next:** NB 11 introduces the orchestrator-worker pattern. We
#   leave MCP behind for one notebook to focus on subagent dispatch,
#   then bring everything together in NB 12.
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/10_consuming_mcp_in_an_agent.py
```

Expected: writes `notebooks/10_consuming_mcp_in_an_agent.ipynb`.

- [ ] **Step 3: Pause for user to run the notebook**

The user runs the notebook in VS Code (kernel: Python (.agents)). Verify:
- Step 1's handshake prints a server name + protocol version.
- Step 4's three turns succeed; Turn 2 and Turn 3 correctly recall "pytest".
- Steps 5–6 list the resource + prompt without errors.

If the import block at Step 5 (`from mcp_servers.notes_server import summarize_notes`) fails because `mcp_servers` isn't on `sys.path`, the issue is the working directory; `chdir_to_repo_root()` should have fixed it but if not, the user can add `import sys; sys.path.insert(0, '.')` before the import. **Adjust the source if needed** and re-sync. (Document the fix in the commit message if so.)

- [ ] **Step 4: After confirmation, commit**

```bash
git add notebooks/_src/10_consuming_mcp_in_an_agent.py notebooks/10_consuming_mcp_in_an_agent.ipynb
git commit -m "$(cat <<'EOF'
feat(nb10): consume MCP from agent loop — auto-discover + cross-process memory

NB 10 spawns the notes server subprocess, auto-discovers its tools
into Anthropic-shaped schemas, routes tool_use blocks through
MCPToolRouter, and demos cross-session note persistence (notes
outlive the agent process — the MCP-earning-its-keep moment).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task C-5: NB 11 — Orchestrator + sequential subagents

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/11_orchestrator_and_subagents.py`
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/11_orchestrator_and_subagents.ipynb`
- Modify (after first run): `/home/rob/PythonEnvironments/Agents/pyproject.toml` (pin `claude-agent-sdk` minor)

- [ ] **Step 1: Write the NB 11 source**

Create `notebooks/_src/11_orchestrator_and_subagents.py`:

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
# # NB 11 — Orchestrator + sequential subagents
#
# **Goal:** introduce orchestrator-worker. Build it from scratch first,
# then rebuild the same shape with `claude-agent-sdk` so you see what
# the framework is hiding.
#
# **Sequential dispatch only** — parallelism, cancellation, and
# checkpointing are NB 12's job.
#
# **Demo task across both stages:** *"Compare httpx, aiohttp, and
# requests. For each, summarize what it's good at and call out one
# gotcha. Then rank them for an async-first project."*

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="11 — Orchestrator + sequential subagents",
    estimate="$0.04–0.06 (two full multi-agent runs)",
    model="claude-sonnet-4-6 (orchestrator + workers)",
)

QUESTION = (
    "Compare httpx, aiohttp, and requests. For each, summarize what "
    "it's good at and call out one gotcha. Then rank them for an "
    "async-first project."
)

# %% [markdown]
# ## Step 1 — Why subagents at all?
#
# Three reasons:
#
# 1. **Token isolation.** Each worker has its own context window — no
#    pollution from earlier subtasks. The orchestrator sees only
#    summaries, not full traces.
# 2. **Role specialization.** Different system prompts steer different
#    workers (researcher digs, summarizer compresses, ranker compares).
# 3. **Failure boundaries.** One worker fails, the others continue.
#    The orchestrator decides whether the partial result is
#    synthesizable or whether to retry.
#
# In NB 12 we'll add a fourth: **parallelism + cancellation.** Today
# we'll see the lesson at sequential dispatch, where the structure is
# easiest to read.

# %% [markdown]
# ## Step 2 — Stage A: scratch nested-Claude orchestrator
#
# The outer agent has two tools: `dispatch_subagent(role, task)` and
# `submit_final(answer)`. Each `dispatch_subagent` call runs a fresh
# `messages.create` loop with a role-specific system prompt and
# returns the worker's final text. **Sequential** — one subagent at a
# time.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.types import Answer

client = get_client()

RESEARCHER_SYSTEM = """You are a focused technical researcher. Use
web_search to find authoritative facts about a single library. Return
3-5 bullet points: what it's good at, current state, one gotcha. No
prose, no preamble. End with one citation URL per fact in
parentheses."""

SUMMARIZER_SYSTEM = """You are a precise summarizer. Take the
researcher's bullet points and rewrite them as one tight paragraph
(~80 words) plus a one-line gotcha. No new claims; preserve the
researcher's wording for technical specifics."""

RANKER_SYSTEM = """You are a comparative analyst. Given short
summaries of N libraries for a stated use case, rank them 1..N with
one sentence of justification per ranking. End with a single
recommendation."""

DISPATCH_TOOL = {
    "name": "dispatch_subagent",
    "description": "Run a subagent with a specific role on a specific task.",
    "input_schema": {
        "type": "object",
        "properties": {
            "role": {
                "type": "string",
                "enum": ["researcher", "summarizer", "ranker"],
                "description": "Which subagent to invoke.",
            },
            "task": {
                "type": "string",
                "description": "The task description for the subagent.",
            },
        },
        "required": ["role", "task"],
    },
}

SUBMIT_FINAL_TOOL = {
    "name": "submit_final",
    "description": "Submit the final ranked answer with citations.",
    "input_schema": Answer.model_json_schema(),
}

ROLE_SYSTEMS = {
    "researcher": RESEARCHER_SYSTEM,
    "summarizer": SUMMARIZER_SYSTEM,
    "ranker": RANKER_SYSTEM,
}


def run_subagent(role: str, task: str, max_turns: int = 6) -> str:
    """Run one subagent loop. Returns the subagent's final text."""
    system = ROLE_SYSTEMS[role]
    messages = [{"role": "user", "content": task}]
    tools = [{"type": "web_search_20250305", "name": "web_search"}] if role == "researcher" else []
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=1024,
            system=system,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason == "end_turn":
            text = "".join(getattr(b, "text", "") for b in response.content)
            return text.strip()
        messages.append({"role": "assistant", "content": response.content})
        # We only allow web_search; managed tools loop themselves, so we just
        # continue if there's any non-end_turn stop reason.
    raise RuntimeError(f"{role} subagent ran out of turns.")


ORCHESTRATOR_SYSTEM = """You are the orchestrator of a research team.

You have three role types of subagents:
- researcher: gathers facts via web_search
- summarizer: compresses researcher output
- ranker: compares summaries and produces a final ranking

Workflow:
1. Dispatch one researcher per item.
2. Dispatch one summarizer for each researcher's output.
3. Dispatch one ranker over all summaries.
4. Submit the final answer with citations via submit_final.

Use ONLY the dispatch_subagent and submit_final tools. Never produce
free-form prose."""


def run_orchestrator(question: str, max_turns: int = 12) -> Answer:
    messages = [{"role": "user", "content": question}]
    for turn in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL,
            max_tokens=2048,
            system=ORCHESTRATOR_SYSTEM,
            tools=[DISPATCH_TOOL, SUBMIT_FINAL_TOOL],
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_final":
                return Answer.model_validate(block.input)
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use" and block.name == "dispatch_subagent":
                role = block.input["role"]
                task = block.input["task"]
                print(f"[turn {turn + 1}] orchestrator → {role}: {task[:80]}...")
                worker_text = run_subagent(role, task)
                print(f"[turn {turn + 1}] {role} → orchestrator ({len(worker_text)} chars)")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": worker_text,
                })
        if not tool_results:
            if response.stop_reason == "end_turn":
                raise RuntimeError("orchestrator ended without submit_final.")
            raise RuntimeError(f"orchestrator stuck (stop_reason={response.stop_reason}).")
        messages.append({"role": "user", "content": tool_results})
    raise RuntimeError("orchestrator ran out of turns.")


print("=== Stage A: scratch orchestrator ===\n")
answer_a = run_orchestrator(QUESTION)
print(f"\nFinal answer summary ({len(answer_a.summary)} chars):")
print(answer_a.summary)
print(f"\nCitations: {len(answer_a.citations)}")

# %% [markdown]
# ### What you just saw
#
# - The orchestrator emitted a **sequence of `dispatch_subagent` tool
#   calls**, each one running an inner `messages.create` loop with a
#   different system prompt.
# - Each subagent **had its own context window** — none of them saw
#   the other subagents' messages, only their assigned task.
# - The orchestrator's final move was a `submit_final` call carrying a
#   validated `Answer` object.
# - This is **plain Python on top of the Anthropic SDK**. No
#   framework. You can step through `run_orchestrator` in a debugger.

# %% [markdown]
# ## Step 3 — Stage B: rebuild with `claude-agent-sdk`
#
# Same task. Same `Answer` shape. We use the SDK's subagent definition
# mechanism to express the same orchestrator-worker pattern with less
# code.
#
# **Note:** the SDK API in `claude-agent-sdk>=0.1.0` is recent and may
# evolve. The cell below uses the surface as of this notebook's
# writing. If imports fail, run
# `python -c "import claude_agent_sdk; help(claude_agent_sdk)"` to
# check the current export list and adjust the imports/method names
# below.

# %%
import json

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition

# Define the three workers as named subagents the orchestrator can dispatch.
SUBAGENTS: dict[str, AgentDefinition] = {
    "researcher": AgentDefinition(
        description="Gathers facts via web_search; returns 3-5 bullets with citations.",
        prompt=RESEARCHER_SYSTEM,
        tools=["WebSearch"],
        model="sonnet",
    ),
    "summarizer": AgentDefinition(
        description="Compresses researcher output to one tight paragraph + gotcha.",
        prompt=SUMMARIZER_SYSTEM,
        tools=[],
        model="sonnet",
    ),
    "ranker": AgentDefinition(
        description="Compares summaries; ranks and recommends.",
        prompt=RANKER_SYSTEM,
        tools=[],
        model="sonnet",
    ),
}

SDK_OPTIONS = ClaudeAgentOptions(
    system_prompt=ORCHESTRATOR_SYSTEM + (
        "\n\nWhen you call subagents, address them as the named agents "
        "'researcher', 'summarizer', or 'ranker'. The SDK will dispatch."
    ),
    agents=SUBAGENTS,
    allowed_tools=["WebSearch"],  # The orchestrator itself doesn't search; workers do.
    max_turns=12,
)


async def run_orchestrator_sdk(question: str) -> str:
    """SDK-orchestrated version. Returns the final assistant text."""
    final_text_parts: list[str] = []
    async with ClaudeSDKClient(options=SDK_OPTIONS) as sdk:
        await sdk.query(question)
        async for message in sdk.receive_response():
            # The SDK streams structured messages. We accumulate text from
            # the final assistant message; subagent activity is internal.
            kind = type(message).__name__
            if kind == "AssistantMessage":
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        final_text_parts.append(text)
    return "\n".join(final_text_parts).strip()


print("=== Stage B: claude-agent-sdk ===\n")
sdk_text = await run_orchestrator_sdk(QUESTION)
print(f"\nSDK final text ({len(sdk_text)} chars):")
print(sdk_text[:1200])

# %% [markdown]
# ### Side-by-side
#
# | Aspect | Stage A (scratch) | Stage B (SDK) |
# |---|---|---|
# | Lines of code | ~120 | ~30 |
# | Visible in notebook | every dispatch + every worker turn | only final assistant message |
# | Debuggable | step into `run_subagent` | breakpoint via SDK hooks |
# | Dispatch loop | hand-rolled tool routing | SDK manages |
# | Output | `Answer` (Pydantic) | free-form text* |
#
# \* Forcing structured output with the SDK requires a structured-output
# wrapper that's beyond this notebook's scope; the *pattern* is the
# same, the surface is just abstracted.

# %% [markdown]
# ## Step 4 — Reflect
#
# - **The SDK earns its keep when:** subagents need their own tools,
#   you want hooks (e.g. permission gates), the orchestration logic
#   would otherwise re-implement what the SDK already provides
#   (worker dispatch, retries, token accounting).
# - **The scratch loop wins when:** you need explicit control over
#   worker task handles (parallelism, cancellation, checkpointing),
#   you want to step through the orchestrator in a debugger, or you
#   don't want another dependency.
# - **NB 12 will pick scratch.** Why: parallel `asyncio.gather`,
#   timeout cancellation, and per-worker JSONL checkpointing are all
#   easier to thread through a hand-rolled orchestrator than to coax
#   out of the SDK's lifecycle. Constraint-fit, not framework-rejection.
#
# **Next:** NB 12 extracts Stage A's orchestrator to
# `src/agentlab/spine.py`, then upgrades it.
```

- [ ] **Step 2: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/11_orchestrator_and_subagents.py
```

Expected: writes `notebooks/11_orchestrator_and_subagents.ipynb`.

- [ ] **Step 3: Pause for user to run the notebook**

The user runs the notebook in VS Code (kernel: Python (.agents)). Verify:
- Stage A prints orchestrator → role lines for each dispatch, eventually returns an `Answer` with non-empty summary and ≥1 citation.
- Stage B runs without import errors. If imports fail, the SDK's exported names may differ from the cell's expectations — adjust `from claude_agent_sdk import ...` to match the installed version (run `dir(claude_agent_sdk)` first), then re-sync the source. The SDK is at version 0.1.x and may rename surface between minors.

If Stage B's API surface differs significantly from `ClaudeSDKClient` / `ClaudeAgentOptions` / `AgentDefinition`:
- The shape of the lesson stays the same: define agents, hand them to a client, await responses.
- Adjust imports + the `SUBAGENTS` / `SDK_OPTIONS` construction to match the actual SDK while preserving the lesson.
- Keep the side-by-side comparison cell (Step 4 inline table) accurate — update line counts if material.

- [ ] **Step 4: Pin the SDK minor in `pyproject.toml`**

Once NB 11 runs end-to-end, lock `claude-agent-sdk` to the working minor to insulate the curriculum from drift.

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/python -c "import claude_agent_sdk; print(claude_agent_sdk.__version__ if hasattr(claude_agent_sdk, '__version__') else 'unknown')"
```

If a version is reported (e.g. `0.1.7`), edit `pyproject.toml`'s `[module-c]` extra:

```toml
module-c = [
    "mcp>=1.0.0",
    "claude-agent-sdk>=0.1.7,<0.2.0",
]
```

(Substitute the actual version. Use `>=X.Y.Z,<X.(Y+1).0` to allow patch bumps but block API drift.)

If `__version__` is unavailable, leave the spec at `>=0.1.0` and document the working version in NB 11's cost banner instead.

- [ ] **Step 5: After confirmation, commit**

```bash
git add notebooks/_src/11_orchestrator_and_subagents.py notebooks/11_orchestrator_and_subagents.ipynb pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
feat(nb11): orchestrator + sequential subagents — scratch then claude-agent-sdk

NB 11 builds the orchestrator-worker pattern twice: hand-rolled with
nested messages.create + dispatch_subagent + submit_final tools, then
rebuilt with claude-agent-sdk's AgentDefinition. Same task, same Answer
shape, side-by-side comparison. Pins claude-agent-sdk minor for
curriculum stability.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task C-6: `src/agentlab/spine.py` (TDD, comprehensive)

**Why this is one task, not three:** the spec splits the spine evolution across NB 12 steps, but the *library work* is one cohesive piece — Subagent + Orchestrator with sync dispatch, async fan-out, cancellation, and JSONL checkpointing. We TDD all of it here, in a single library task, and NB 12 then becomes a pure consumer (import + wire + demo).

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/tests/test_spine.py`
- Create: `/home/rob/PythonEnvironments/Agents/src/agentlab/spine.py`
- Create: `/home/rob/PythonEnvironments/Agents/data/checkpoints/.gitkeep`

- [ ] **Step 1: Create the checkpoints directory marker**

```bash
mkdir -p data/checkpoints && touch data/checkpoints/.gitkeep
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_spine.py`:

```python
"""Tests for agentlab.spine: Subagent dataclass + Orchestrator.

We do not call any LLM here. The Orchestrator under test is dispatched
via an injected `dispatch` callable so workers are deterministic
fakes.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agentlab.spine import Orchestrator, Subagent, WorkerResult


# ── Fixtures ───────────────────────────────────────────────────────


def _make_workers() -> list[Subagent]:
    return [
        Subagent(role="researcher", system_prompt="be a researcher"),
        Subagent(role="summarizer", system_prompt="be a summarizer"),
        Subagent(role="ranker", system_prompt="be a ranker"),
    ]


# ── Sync run: dispatch is called once per worker, results collected ─


def test_orchestrator_run_dispatches_each_worker_once():
    """In sync mode, every worker runs once and produces a WorkerResult."""
    seen: list[tuple[str, str]] = []

    def fake_dispatch(worker: Subagent, task: str) -> str:
        seen.append((worker.role, task))
        return f"{worker.role}: {task[:20]}"

    orch = Orchestrator(workers=_make_workers(), dispatch=fake_dispatch)
    results = orch.run_sync("compare httpx, aiohttp, requests")

    assert len(results) == 3
    assert {r.role for r in results} == {"researcher", "summarizer", "ranker"}
    assert all(isinstance(r, WorkerResult) for r in results)
    assert all(r.error is None for r in results)
    assert {role for role, _ in seen} == {"researcher", "summarizer", "ranker"}


def test_orchestrator_run_records_dispatch_errors():
    """If dispatch raises, the WorkerResult carries the error string and result is None."""

    def flaky_dispatch(worker: Subagent, task: str) -> str:
        if worker.role == "summarizer":
            raise RuntimeError("boom")
        return f"{worker.role}: ok"

    orch = Orchestrator(workers=_make_workers(), dispatch=flaky_dispatch)
    results = orch.run_sync("test")

    bad = next(r for r in results if r.role == "summarizer")
    good = [r for r in results if r.role != "summarizer"]
    assert bad.result is None
    assert "boom" in bad.error
    assert all(g.error is None and g.result for g in good)


# ── Async run: workers fan out via asyncio.gather ──────────────────


@pytest.mark.asyncio
async def test_orchestrator_run_async_runs_workers_concurrently():
    """Three workers each sleep 0.1s; total wall-clock should be <0.2s."""
    import time

    async def slow_dispatch(worker: Subagent, task: str) -> str:
        await asyncio.sleep(0.1)
        return f"{worker.role}-done"

    orch = Orchestrator(workers=_make_workers(), dispatch_async=slow_dispatch)
    start = time.perf_counter()
    results = await orch.run_async("test")
    elapsed = time.perf_counter() - start

    assert len(results) == 3
    assert elapsed < 0.2, f"async run took {elapsed:.3f}s, expected <0.2s"


@pytest.mark.asyncio
async def test_orchestrator_run_async_timeout_cancels_slow_worker():
    """One worker exceeds its timeout; result has error, others succeed."""

    async def mixed_dispatch(worker: Subagent, task: str) -> str:
        if worker.role == "researcher":
            await asyncio.sleep(0.5)  # will exceed timeout
        else:
            await asyncio.sleep(0.05)
        return f"{worker.role}-done"

    orch = Orchestrator(workers=_make_workers(), dispatch_async=mixed_dispatch)
    results = await orch.run_async("test", worker_timeout=0.1)

    by_role = {r.role: r for r in results}
    assert by_role["researcher"].result is None
    assert "timeout" in by_role["researcher"].error.lower()
    assert by_role["summarizer"].result == "summarizer-done"
    assert by_role["ranker"].result == "ranker-done"


# ── Checkpointing ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_orchestrator_writes_checkpoint_per_worker(tmp_path):
    """Each successful worker appends a JSONL line under checkpoint_dir/run_id.jsonl."""
    async def quick(worker: Subagent, task: str) -> str:
        return f"{worker.role}-result"

    orch = Orchestrator(
        workers=_make_workers(), dispatch_async=quick, checkpoint_dir=tmp_path
    )
    await orch.run_async("test", run_id="abc123")

    checkpoint = tmp_path / "abc123.jsonl"
    assert checkpoint.exists()
    lines = [json.loads(line) for line in checkpoint.read_text().splitlines() if line.strip()]
    assert len(lines) == 3
    assert {entry["role"] for entry in lines} == {"researcher", "summarizer", "ranker"}
    assert all(entry["run_id"] == "abc123" for entry in lines)


@pytest.mark.asyncio
async def test_orchestrator_resume_skips_completed_workers(tmp_path):
    """Resume reads cached results from JSONL; only re-runs missing workers."""
    # Pre-populate the checkpoint with one completed worker.
    cp_path = tmp_path / "abc123.jsonl"
    cp_path.write_text(json.dumps({
        "run_id": "abc123",
        "role": "researcher",
        "result": "cached-researcher",
        "error": None,
    }) + "\n")

    seen_roles: list[str] = []

    async def dispatch(worker: Subagent, task: str) -> str:
        seen_roles.append(worker.role)
        return f"{worker.role}-fresh"

    orch = Orchestrator(
        workers=_make_workers(), dispatch_async=dispatch, checkpoint_dir=tmp_path
    )
    results = await orch.run_async("test", run_id="abc123")

    by_role = {r.role: r for r in results}
    assert by_role["researcher"].result == "cached-researcher"
    assert seen_roles == ["summarizer", "ranker"]  # researcher skipped


# ── Subagent dataclass ─────────────────────────────────────────────


def test_subagent_is_a_dataclass_with_role_and_system_prompt():
    s = Subagent(role="x", system_prompt="y")
    assert s.role == "x" and s.system_prompt == "y"
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_spine.py -v
```

Expected: collection errors — `ModuleNotFoundError: No module named 'agentlab.spine'`.

- [ ] **Step 4: Implement `src/agentlab/spine.py`**

Create `src/agentlab/spine.py`:

```python
"""Multi-agent orchestrator + worker primitives.

Extracted from NB 11 Stage A and extended for NB 12: parallel async
fan-out via asyncio.gather, per-worker timeout cancellation, and
JSONL checkpointing for resumability.

Construction is dispatch-injected so the same Orchestrator can be
exercised with a fake (in tests) or with a real Claude-backed worker
loop (in NB 12).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

DispatchSync = Callable[["Subagent", str], str]
DispatchAsync = Callable[["Subagent", str], Awaitable[str]]


@dataclass(frozen=True)
class Subagent:
    """A worker definition: a role name + the system prompt that role uses."""
    role: str
    system_prompt: str


@dataclass
class WorkerResult:
    role: str
    result: Optional[str]
    error: Optional[str]


class Orchestrator:
    """Dispatch a fixed set of subagents over a single question.

    Two execution modes:
    - `run_sync(question)` — sequential, requires `dispatch` (sync callable)
    - `run_async(question, run_id=..., worker_timeout=...)` — parallel,
      requires `dispatch_async` (awaitable callable)

    Optional checkpointing: pass `checkpoint_dir` and a `run_id`. Each
    worker's WorkerResult is appended to `checkpoint_dir/<run_id>.jsonl`
    on completion; calling `run_async` again with the same `run_id`
    skips workers already in the checkpoint.
    """

    def __init__(
        self,
        workers: list[Subagent],
        dispatch: DispatchSync | None = None,
        dispatch_async: DispatchAsync | None = None,
        checkpoint_dir: Path | None = None,
    ):
        if dispatch is None and dispatch_async is None:
            raise ValueError("Provide at least one of dispatch / dispatch_async")
        self.workers = list(workers)
        self.dispatch = dispatch
        self.dispatch_async = dispatch_async
        self.checkpoint_dir = checkpoint_dir

    # ── sync ───────────────────────────────────────────────────────

    def run_sync(self, question: str) -> list[WorkerResult]:
        if self.dispatch is None:
            raise RuntimeError("run_sync requires `dispatch` (sync callable)")
        results: list[WorkerResult] = []
        for worker in self.workers:
            try:
                text = self.dispatch(worker, question)
                results.append(WorkerResult(role=worker.role, result=text, error=None))
            except Exception as exc:
                results.append(WorkerResult(role=worker.role, result=None, error=repr(exc)))
        return results

    # ── async + checkpointing ──────────────────────────────────────

    async def run_async(
        self,
        question: str,
        run_id: str | None = None,
        worker_timeout: float = 60.0,
    ) -> list[WorkerResult]:
        if self.dispatch_async is None:
            raise RuntimeError("run_async requires `dispatch_async` (awaitable callable)")
        run_id = run_id or uuid.uuid4().hex[:8]
        cached = self._load_checkpoint(run_id)

        async def _one(worker: Subagent) -> WorkerResult:
            if worker.role in cached:
                return cached[worker.role]
            try:
                text = await asyncio.wait_for(
                    self.dispatch_async(worker, question), timeout=worker_timeout
                )
                wr = WorkerResult(role=worker.role, result=text, error=None)
            except asyncio.TimeoutError:
                wr = WorkerResult(
                    role=worker.role,
                    result=None,
                    error=f"timeout after {worker_timeout}s",
                )
            except asyncio.CancelledError:
                # Cooperative cancellation: re-raise so gather sees it.
                raise
            except Exception as exc:
                wr = WorkerResult(role=worker.role, result=None, error=repr(exc))
            self._append_checkpoint(run_id, wr)
            return wr

        return await asyncio.gather(*(_one(w) for w in self.workers))

    def resume(self, run_id: str) -> list[WorkerResult]:
        """Return cached results from the checkpoint without re-running.

        For partial resumes (some workers missing), prefer `run_async`
        with the same `run_id` — it skips cached workers and runs the
        rest.
        """
        cached = self._load_checkpoint(run_id)
        if not cached:
            raise FileNotFoundError(f"No checkpoint for run_id={run_id}")
        return [cached[w.role] for w in self.workers if w.role in cached]

    # ── checkpoint internals ───────────────────────────────────────

    def _checkpoint_path(self, run_id: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return self.checkpoint_dir / f"{run_id}.jsonl"

    def _load_checkpoint(self, run_id: str) -> dict[str, WorkerResult]:
        path = self._checkpoint_path(run_id)
        if path is None or not path.exists():
            return {}
        out: dict[str, WorkerResult] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            out[entry["role"]] = WorkerResult(
                role=entry["role"], result=entry.get("result"), error=entry.get("error")
            )
        return out

    def _append_checkpoint(self, run_id: str, wr: WorkerResult) -> None:
        path = self._checkpoint_path(run_id)
        if path is None:
            return
        entry = {
            "run_id": run_id,
            "role": wr.role,
            "result": wr.result,
            "error": wr.error,
        }
        with path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/test_spine.py -v
```

Expected: 7 passed. The async timeout test sleeps 0.5s under the 0.1s timeout, so it should complete in well under 0.2s.

- [ ] **Step 6: Verify the rest of the suite still passes**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -q
```

Expected: all tests pass; eval tests stay deselected by default.

- [ ] **Step 7: Commit**

```bash
git add src/agentlab/spine.py tests/test_spine.py data/checkpoints/.gitkeep
git commit -m "$(cat <<'EOF'
feat(spine): Orchestrator + Subagent with sync, async, timeout, checkpointing

src/agentlab/spine.py exposes Subagent (frozen dataclass) and
Orchestrator with run_sync, run_async (asyncio.gather + per-worker
timeout via asyncio.wait_for), and JSONL checkpointing under
checkpoint_dir/<run_id>.jsonl. resume(run_id) replays without
re-running; run_async(run_id=...) does partial resume, skipping
cached workers. Dispatch is injected (sync or async callable) so the
same orchestrator works with fakes in tests and Claude-backed
workers in NB 12.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task C-7: NB 12 — ✦ Parallel + durable spine integration

**Files:**
- Create: `/home/rob/PythonEnvironments/Agents/notebooks/_src/12_parallel_and_durable.py`
- Modify: `/home/rob/PythonEnvironments/Agents/data/eval_tasks.jsonl` (append 5 rows: t11–t15)
- Generated by jupytext: `/home/rob/PythonEnvironments/Agents/notebooks/12_parallel_and_durable.ipynb`

- [ ] **Step 1: Append 5 parallel-friendly tasks to `data/eval_tasks.jsonl`**

Open `data/eval_tasks.jsonl`. After the existing `t10` row, append:

```jsonl
{"id":"t11","question":"Compare httpx, aiohttp, and requests for an async-first Python project. For each, give a one-line gotcha. Then rank them.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Mentions all three libraries; includes a one-line gotcha per library; ends with a ranking and recommendation."}
{"id":"t12","question":"Compare FastAPI, Flask, and Django for building a JSON API. Cover ergonomics, async support, and deployment. Then rank them.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Covers all three frameworks across the three axes (ergonomics/async/deployment); produces a ranking with justification."}
{"id":"t13","question":"For each of pytest, unittest, and nose2, summarize current status and one strength. Rank by current adoption.","expected_tool_calls":["web_search"],"reference_answer":"pytest","judge_rubric":"Covers all three test frameworks; identifies pytest as #1 by adoption; mentions unittest's stdlib status."}
{"id":"t14","question":"Compare Pydantic v2, attrs, and msgspec for data modeling. Note one tradeoff each. Then rank for an HTTP API.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Mentions all three libraries; calls out a meaningful tradeoff per library; ranks with rationale tied to the HTTP-API use case."}
{"id":"t15","question":"Summarize the latest stable releases of Python 3.12, 3.13, and 3.14 (or whichever three are current). For each, give one notable feature. Rank by readiness for production today.","expected_tool_calls":["web_search"],"reference_answer":null,"judge_rubric":"Three Python versions covered; one feature per version; production-readiness ranking with rationale."}
```

Expected total: 15 tasks. Verify:

```bash
wc -l data/eval_tasks.jsonl
```

- [ ] **Step 2: Write the NB 12 source**

Create `notebooks/_src/12_parallel_and_durable.py`:

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
# # NB 12 — ✦ Parallel + durable spine
#
# **Goal:** combine MCP-backed memory + parallel subagent fan-out +
# cancellation + checkpointing into the extracted spine library. Close
# the ✦ Module-C spine progression.
#
# Everything in this notebook **imports** from the library
# (`agentlab.spine`, `agentlab.mcp_helpers`). The notebook is the
# integration layer: wire the parts, run the demos, observe traces,
# extend the eval set.
#
# **Architectural call:** we extend NB 11 Stage A (manual orchestrator),
# not Stage B (Agent SDK). Reason: explicit task handles + cancellation
# + checkpointing are easier to thread through a hand-rolled
# orchestrator than to coax out of the SDK lifecycle. Constraint-fit,
# not framework-rejection.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="12 — ✦ Parallel + durable spine",
    estimate="$0.10–0.15 (3-4 spine runs + eval extension)",
    model="claude-sonnet-4-6 (workers + orchestrator)",
)

QUESTION = (
    "For each of these three open-source projects (httpx, FastAPI, "
    "Pydantic), summarize the README, fetch the latest GitHub release, "
    "and rank them by recent activity."
)

# %% [markdown]
# ## Step 0 — Spine extracted (recap)
#
# The orchestrator + Subagent definitions from NB 11 Stage A now live
# at `src/agentlab/spine.py`, with extensions for async fan-out,
# per-worker timeout, and JSONL checkpointing. NB 12 imports them.

# %%
from agentlab.spine import Orchestrator, Subagent, WorkerResult
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.types import Answer

client = get_client()

WORKERS = [
    Subagent(role="researcher", system_prompt=(
        "You are a focused technical researcher. Use web_search to "
        "summarize the README of the given project and fetch the "
        "latest release. Return: 'README: <60-word summary>. Latest "
        "release: <tag and date>.' End with one citation URL per fact."
    )),
    Subagent(role="summarizer", system_prompt=(
        "You are a precise summarizer. Compress the researcher's "
        "output to one tight paragraph (~80 words) with no new claims."
    )),
    Subagent(role="ranker", system_prompt=(
        "You are a comparative analyst. Given short summaries, rank "
        "the projects by recent activity (latest release recency + "
        "README signal). Return a ranked list with one-sentence "
        "justifications and a single recommendation."
    )),
]

# %% [markdown]
# ## Step 1 — Async workers + asyncio.gather
#
# `Orchestrator.run_async` fans out workers concurrently. We define a
# Claude-backed `dispatch_async` that runs a subagent loop with the
# worker's system prompt.

# %%
import asyncio
import time


async def claude_dispatch(worker: Subagent, task: str) -> str:
    """Run a worker as a single Claude call (with web_search for researcher)."""
    tools = [{"type": "web_search_20250305", "name": "web_search"}] if worker.role == "researcher" else []
    messages = [{"role": "user", "content": task}]
    # asyncio-friendly thin wrapper around the sync SDK call.
    response = await asyncio.to_thread(
        client.messages.create,
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=worker.system_prompt,
        tools=tools,
        messages=messages,
    )
    return "".join(getattr(b, "text", "") for b in response.content).strip()


orch = Orchestrator(workers=WORKERS, dispatch_async=claude_dispatch)

print("--- Parallel run (no MCP, no checkpoints) ---")
start = time.perf_counter()
parallel_results = await orch.run_async(QUESTION, worker_timeout=120)
parallel_elapsed = time.perf_counter() - start
print(f"Wall-clock: {parallel_elapsed:.1f}s\n")
for r in parallel_results:
    snippet = (r.result or "")[:120]
    status = "✓" if r.error is None else f"✗ {r.error}"
    print(f"  [{r.role}] {status}  {snippet}")

# %% [markdown]
# ## Step 2 — Cancellation
#
# We introduce a tighter `worker_timeout` and a deliberately slow
# wrapper around the researcher to show graceful degradation. The
# orchestrator returns whatever workers finished in time; the slow one
# carries an error string.

# %%
async def slow_dispatch(worker: Subagent, task: str) -> str:
    if worker.role == "researcher":
        await asyncio.sleep(5)  # exceeds the tight timeout below
    return await claude_dispatch(worker, task)


slow_orch = Orchestrator(workers=WORKERS, dispatch_async=slow_dispatch)

print("--- Cancellation demo (researcher times out) ---")
canceled_results = await slow_orch.run_async(QUESTION, worker_timeout=2)
for r in canceled_results:
    status = "✓" if r.error is None else f"✗ {r.error}"
    print(f"  [{r.role}] {status}")

# %% [markdown]
# ## Step 3 — Checkpointing
#
# Same `Orchestrator`, this time with `checkpoint_dir` set. Each
# worker's `WorkerResult` is appended to
# `data/checkpoints/<run_id>.jsonl`. We then re-run with the same
# `run_id` and observe that completed workers are skipped.

# %%
from pathlib import Path

CHECKPOINT_DIR = Path("data/checkpoints")
ckpt_orch = Orchestrator(
    workers=WORKERS, dispatch_async=claude_dispatch, checkpoint_dir=CHECKPOINT_DIR
)

run_id = "demo_run"
# Wipe any prior checkpoint for this run_id so the demo is deterministic.
checkpoint_file = CHECKPOINT_DIR / f"{run_id}.jsonl"
checkpoint_file.unlink(missing_ok=True)

print("--- First run (writes checkpoint) ---")
first_results = await ckpt_orch.run_async(QUESTION, run_id=run_id, worker_timeout=120)
print(f"Wrote {checkpoint_file.stat().st_size} bytes to {checkpoint_file}")

print("\n--- Second run with same run_id (should skip all workers) ---")
import time as _time
start = _time.perf_counter()
second_results = await ckpt_orch.run_async(QUESTION, run_id=run_id, worker_timeout=120)
elapsed = _time.perf_counter() - start
print(f"Wall-clock: {elapsed:.2f}s (expect ~0s — all workers cached)")
for r in second_results:
    snippet = (r.result or "")[:80]
    print(f"  [{r.role}] {snippet}...")

# %% [markdown]
# ## Step 4 — MCP-backed memory
#
# We now connect to the notes server (NB 09) as a subprocess. The
# orchestrator passes an `MCPToolRouter` to its workers via a
# closure-captured wrapper, so workers can write intermediate findings
# as notes (`add_note`) and the synthesizer can read them back.
#
# **The MCP-earning-its-keep moment for the spine:** after the
# orchestrator exits, the notes survive as a file. A new orchestrator
# session — even after a kernel restart — can recall them.

# %%
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentlab.mcp_helpers import MCPToolRouter

SERVER_PARAMS = StdioServerParameters(command="python", args=["mcp_servers/notes_server.py"])


def _make_mcp_dispatch(router: MCPToolRouter, run_id: str):
    """Build a dispatch_async that writes each worker's result as an MCP note."""

    async def mcp_dispatch(worker: Subagent, task: str) -> str:
        text = await claude_dispatch(worker, task)
        # Persist the worker's output as a note keyed by run + role.
        try:
            await router.call("add_note", {"key": f"{run_id}:{worker.role}", "content": text})
        except RuntimeError as exc:
            print(f"  [warn] could not persist {worker.role} note: {exc}")
        return text

    return mcp_dispatch


async def run_with_mcp(question: str, run_id: str) -> list[WorkerResult]:
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            router = MCPToolRouter(session)
            await router.refresh()
            mcp_orch = Orchestrator(
                workers=WORKERS,
                dispatch_async=_make_mcp_dispatch(router, run_id),
                checkpoint_dir=CHECKPOINT_DIR,
            )
            return await mcp_orch.run_async(question, run_id=run_id, worker_timeout=120)


print("--- MCP-backed run ---")
mcp_run_id = "mcp_demo"
(CHECKPOINT_DIR / f"{mcp_run_id}.jsonl").unlink(missing_ok=True)
mcp_results = await run_with_mcp(QUESTION, run_id=mcp_run_id)
for r in mcp_results:
    print(f"  [{r.role}] persisted ✓" if r.error is None else f"  [{r.role}] ✗ {r.error}")

# Verify the notes survived: open the notes server independently and list keys.
async def list_notes_after():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            router = MCPToolRouter(session)
            keys_text = await router.call("list_notes", {})
            return keys_text


print("\n--- Notes after the run (cold-started server) ---")
print(await list_notes_after())

# %% [markdown]
# ## Step 5 — OTel traces
#
# Same instrumentation idea as NB 08, but spans now tree as:
#
# ```
# agent.run
# ├── subagent.researcher  (parallel sibling)
# │   └── llm.complete
# ├── subagent.summarizer  (parallel sibling)
# │   └── llm.complete
# └── subagent.ranker      (parallel sibling)
#     └── llm.complete
# ```
#
# We use `ConsoleSpanExporter` to print spans inline (Jaeger swap is
# the same one-cell appendix as NB 08).

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agentlab.spine")


async def traced_dispatch(worker: Subagent, task: str) -> str:
    with tracer.start_as_current_span(f"subagent.{worker.role}") as span:
        span.set_attribute("worker.role", worker.role)
        span.set_attribute("task.length", len(task))
        with tracer.start_as_current_span("llm.complete"):
            text = await claude_dispatch(worker, task)
        span.set_attribute("result.length", len(text))
        return text


traced_orch = Orchestrator(workers=WORKERS, dispatch_async=traced_dispatch)

print("--- Traced parallel run ---")
with tracer.start_as_current_span("agent.run") as span:
    span.set_attribute("question.length", len(QUESTION))
    span.set_attribute("parallel_dispatch", True)
    traced_results = await traced_orch.run_async(QUESTION, worker_timeout=120)

# Force flush so spans print before the next cell.
_ = provider.force_flush()

# %% [markdown]
# ## Step 6 — Eval extension
#
# `data/eval_tasks.jsonl` was extended in this commit's task list with
# 5 parallel-friendly tasks (`t11`-`t15`) — multi-source comparisons.
# The existing pytest harness in
# `tests/eval/test_research_assistant.py` parametrizes over the JSONL,
# so the new rows are picked up automatically.
#
# **Run from a terminal** (not the notebook):
#
# ```bash
# UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -m eval tests/eval/ -v
# ```
#
# We do not run the full eval suite from the notebook (slow + costs
# real money). The cell below runs **one** parallel-friendly task
# through the spine to verify the architecture *is* fanning out.

# %%
import json

with open("data/eval_tasks.jsonl") as f:
    tasks = [json.loads(line) for line in f if line.strip()]

t11 = next(t for t in tasks if t["id"] == "t11")
print(f"Running {t11['id']}: {t11['question']}\n")

start = time.perf_counter()
results_t11 = await traced_orch.run_async(t11["question"], worker_timeout=120)
elapsed = time.perf_counter() - start
print(f"Wall-clock: {elapsed:.1f}s (parallel dispatch — should be << 3× single-worker time)")

# %% [markdown]
# ## Reflect
#
# - **The spine ✦ progression is complete.**
#   - Memory (NB 06) → retrieval (NB 07) → measurable (NB 08) →
#     multi-agent + durable + MCP-backed (this NB).
# - **What NB 15 will do.** No new architecture — wrap `Orchestrator`
#   as a CLI script with env-driven config, tracing on by default,
#   README, and a `Makefile`-or-shell entrypoint.
# - **Why we picked manual over SDK in this NB.** Explicit task
#   handles + cancellation + checkpointing are hard to thread through
#   the SDK's lifecycle. Constraint-fit, not framework-rejection.
# - **What we deliberately did not build.** Distributed checkpointing,
#   subagent retry/circuit-breaker patterns, multi-server MCP
#   composition. Real concerns for production; out of curriculum
#   scope. NB 15 will note them but not implement.
#
# **Module C is done.** Module D (NB 13–15) will introduce computer
# use, local models, and the capstone CLI extraction.
```

- [ ] **Step 3: Sync the notebook**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/12_parallel_and_durable.py
```

Expected: writes `notebooks/12_parallel_and_durable.ipynb`.

- [ ] **Step 4: Pause for user to run the notebook**

The user runs the notebook in VS Code (kernel: Python (.agents)). Verify:
- Step 1 prints wall-clock < 3× the slowest single-worker time (parallel actually fans out).
- Step 2 prints `✗ timeout` for `researcher`, `✓` for the other two.
- Step 3's second run prints elapsed ~0s (cache hit) and shows all three roles' cached snippets.
- Step 4 prints `persisted ✓` per worker, then a non-empty `list_notes` output containing both `mcp_demo:researcher` and the others.
- Step 5 prints span output (lots of JSON-ish lines from `ConsoleSpanExporter`).
- Step 6 (single t11 task) completes without errors.

If the SDK's `claude-agent-sdk` import in NB 11 changed any names, this notebook is unaffected — NB 12 uses the manual path only.

- [ ] **Step 5: Optionally run the eval extension from a terminal**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -m eval tests/eval/ -v
```

Expected: 45 test instances (15 tasks × 3 styles), with some LLM-judge variance. Deterministic and reference-based should be ≥80% pass; LLM-judge is informational. (This is the full $0.05–0.10 eval pass + the new 5 tasks; budget ~$0.10–0.15 in total.)

- [ ] **Step 6: After confirmation, commit**

```bash
git add notebooks/_src/12_parallel_and_durable.py notebooks/12_parallel_and_durable.ipynb data/eval_tasks.jsonl
git commit -m "$(cat <<'EOF'
feat(nb12): ✦ parallel + durable spine — async, cancellation, MCP, evals

NB 12 imports agentlab.spine.Orchestrator, demos parallel fan-out vs
sequential, graceful timeout cancellation, JSONL checkpointing with
resume, MCP-backed memory (notes outlive the orchestrator process),
and OTel tracing of the parallel sibling spans. Adds 5
parallel-friendly tasks (t11-t15) to data/eval_tasks.jsonl, picked up
automatically by tests/eval/test_research_assistant.py. Closes the ✦
Module-C spine progression.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task C-8: README + Module C retrospective

**Files:**
- Modify: `/home/rob/PythonEnvironments/Agents/README.md`

- [ ] **Step 1: Read the current README**

```bash
cat README.md
```

Refresh on the Module B section's structure — the same shape applies to Module C.

- [ ] **Step 2: Flip the Module C status row**

Edit the Status table. Replace:

```markdown
| **C · Multi-agent + MCP** | 09 build MCP server · 10 consume MCP · 11 orchestrator + subagents · 12 parallel + durable | ⏳ planned |
```

with:

```markdown
| **C · Multi-agent + MCP** | [09 build MCP server](notebooks/09_building_an_mcp_server.ipynb) · [10 consume MCP](notebooks/10_consuming_mcp_in_an_agent.ipynb) · [11 orchestrator + subagents](notebooks/11_orchestrator_and_subagents.ipynb) · [12 parallel + durable](notebooks/12_parallel_and_durable.ipynb) | ✅ shipped |
```

- [ ] **Step 3: Add a "What Module C teaches" section**

After the existing "What Module B teaches" section, add:

```markdown
## What Module C teaches

- **[NB 09 — Building an MCP server from scratch](notebooks/09_building_an_mcp_server.ipynb)**
  Build the same notes server twice: low-level `Server` with manual
  handlers, then `FastMCP` decorators. Inspect via `mcp dev`. Adds a
  `summarize_notes` Prompt to demonstrate the third MCP primitive.
  Stdio transport main flow; Streamable HTTP appendix.
- **[NB 10 — Consuming MCP in an agent](notebooks/10_consuming_mcp_in_an_agent.ipynb)**
  Spawn the notes server as a subprocess, auto-discover its tools,
  route Anthropic tool calls through `MCPToolRouter`, and demo
  cross-session note persistence — the moment MCP earns its keep over
  in-process memory. Lands `mcp_helpers.py` in `src/agentlab/`.
- **[NB 11 — Orchestrator + sequential subagents](notebooks/11_orchestrator_and_subagents.ipynb)**
  Orchestrator-worker pattern two ways: hand-rolled with nested
  `messages.create` + `dispatch_subagent` tool, then rebuilt with
  `claude-agent-sdk`'s `AgentDefinition`. Same task, side-by-side
  comparison.
- **[NB 12 — ✦ Parallel + durable spine](notebooks/12_parallel_and_durable.ipynb)**
  Imports the extracted `agentlab.spine.Orchestrator`. Demos parallel
  fan-out via `asyncio.gather`, graceful timeout cancellation, JSONL
  checkpointing with resume, MCP-backed memory, and OTel traces of
  the parallel sibling spans. Extends the spine eval set with 5
  parallel-friendly tasks.
```

- [ ] **Step 4: Update Quickstart to include `module-c`**

Replace:

```markdown
uv sync --extra dev --extra module-b
```

with:

```markdown
uv sync --extra dev --extra module-b --extra module-c
```

- [ ] **Step 5: Update Repo layout to mention new files**

In the existing repo-layout block, update the `src/agentlab/` subtree and add the `mcp_servers/` line if not already present:

```
src/agentlab/              # the small library imported by every notebook
  ├ llm.py                 # Anthropic client wrapper + run_agent_loop
  ├ tools.py               # ToolRegistry + schema generation
  ├ types.py               # Pydantic models (Answer, Citation)
  ├ memory.py              # ConversationBuffer + KeyValueMemory + SemanticMemory
  ├ mcp_helpers.py         # mcp_tools_to_anthropic + MCPToolRouter
  └ spine.py               # Subagent + Orchestrator (sync/async/checkpoint)
```

And replace `mcp_servers/` line if it says "added in Module C":

```
mcp_servers/               # MCP servers (notes_server.py — NB 09)
```

- [ ] **Step 6: Update Costs section**

Replace:

```markdown
Module A end-to-end is roughly **$0.05–0.10**; Module B end-to-end is
**$0.10–0.20** (the eval suite drives most of that — opt in with
`pytest -m eval`). Module C and the capstone will run higher; those
notebooks default to Claude Haiku where pedagogically acceptable.
```

with:

```markdown
Module A end-to-end is roughly **$0.05–0.10**; Module B end-to-end is
**$0.10–0.20**; Module C end-to-end is **$0.17–0.25**. The eval suite
(15 tasks × 3 styles after Module C) drives the largest single chunk
and is opt-in via `pytest -m eval`. Module D (capstone) will run
higher.
```

- [ ] **Step 7: Self-review the README diff**

```bash
git diff README.md
```

Expected diff: ~5 hunks — status row, new "What Module C teaches" section, quickstart line, repo layout block, costs paragraph. No accidental whitespace changes elsewhere.

- [ ] **Step 8: Commit**

```bash
git add README.md
git commit -m "$(cat <<'EOF'
docs: flip Module C to ✅ shipped + add NB 09-12 descriptions

Status row, "What Module C teaches" section with per-NB hooks,
quickstart now syncs module-c extra, repo layout shows mcp_helpers.py
+ spine.py + mcp_servers/, costs section accounts for Module C
($0.17–$0.25 end-to-end with the extended 15-task eval set).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 9: Push the branch and merge to main**

After all 8 tasks ship and the user confirms, the standard wrap-up:

```bash
git push -u origin feature/module-c
git checkout main
git merge --ff-only feature/module-c
git push origin main
```

(Branch stays per the established preference; do not delete `feature/module-c`.)

---

## Wrap-up checklist

After Task C-8 ships, verify:

- [ ] All 4 notebooks (NB 09-12) execute end-to-end without error
- [ ] `pytest tests/test_mcp_helpers.py tests/test_spine.py` — green
- [ ] `pytest -m eval tests/eval/ -v` — 45 test instances (15 × 3); deterministic + reference ≥80%
- [ ] `mcp dev mcp_servers/notes_server.py` opens the inspector
- [ ] README Module C row shows ✅ shipped; "What Module C teaches" present
- [ ] `feature/module-c` merged fast-forward to `main`; both pushed to origin
- [ ] Module C end-to-end cost stays within $0.17–$0.25
