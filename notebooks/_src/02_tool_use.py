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
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()  # so `data/seed.txt` in Step 4 resolves from repo root
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
