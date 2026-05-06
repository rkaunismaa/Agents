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
import json
from datetime import datetime, timezone
from pathlib import Path

import mcp.types as types
from mcp.server import Server


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
# - **Four handlers** were registered by hand. Each one is an `async`
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
# - The handler bodies + decorators sum to ~55 lines (vs ~100 for the
#   hand-rolled low-level handlers above). The Step 4 cell prints the
#   total file size; most of the rest is the schema-validating
#   `_load_notes` helper plus docstrings.

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
