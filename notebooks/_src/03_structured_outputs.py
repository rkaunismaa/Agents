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
