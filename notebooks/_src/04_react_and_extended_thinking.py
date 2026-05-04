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
    max_tokens=8192,  # must exceed thinking.budget_tokens; leaves ~4K for the visible answer
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
