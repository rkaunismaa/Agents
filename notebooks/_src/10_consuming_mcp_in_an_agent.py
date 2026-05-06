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
# route it through `MCPToolRouter.call(...)`. The `else` branch is a
# safety net that returns an `unknown tool` error — mixing MCP tools
# with locally-defined tools is left to NB 12, which combines both
# registries in the durable spine.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.mcp_helpers import MCPToolRouter

client = get_client()

SYSTEM_PROMPT = """You are a helpful assistant with persistent notes.
You can store key/value notes via add_note and retrieve them via
get_note or list_notes.

Before answering any question about what the user told you, prefers,
or asked you to remember, ALWAYS call list_notes first to see what's
stored. Don't say "I don't know" without checking your notes."""


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
            print(f"Static resources advertised: {len(listed.resources)}")
            for r in listed.resources:
                print(f"  {r.uri} — {r.description}")

            # Our notes://{key} resource is a *template* (URI has a
            # parameter), so it surfaces via list_resource_templates,
            # not list_resources. Templates are read by constructing
            # a concrete URI and calling read_resource on it.
            templates = await session.list_resource_templates()
            print(f"\nResource templates advertised: {len(templates.resourceTemplates)}")
            for t in templates.resourceTemplates:
                print(f"  {t.uriTemplate} — {t.description}")

            if templates.resourceTemplates:
                concrete_uri = "notes://fav_test_lib"
                content = await session.read_resource(concrete_uri)
                print(f"\nRead {concrete_uri}:")
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
