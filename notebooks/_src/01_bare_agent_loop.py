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
