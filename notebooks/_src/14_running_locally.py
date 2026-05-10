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
# # NB 14 — Running locally with Ollama / LM Studio
#
# **Goal:** swap the Anthropic client for a local model via an
# OpenAI-compatible endpoint. Understand what translates cleanly and
# where local models break.
#
# **Setup:** Ollama must be running (`ollama serve` in a terminal if not
# already). We use `llama3.2` as the primary model. LM Studio is shown
# as a one-cell endpoint swap.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="14 — Running locally with Ollama / LM Studio",
    estimate="$0.01–0.02 (comparison Claude calls only; local runs are free)",
    model="claude-sonnet-4-6 (comparison only)",
)

# %% [markdown]
# ## Setup — verify Ollama + pull model

# %%
import subprocess
import time as _time

LOCAL_MODEL = "llama3.2"
result = subprocess.run(["ollama", "pull", LOCAL_MODEL], capture_output=True, text=True)
print(result.stdout[-500:] or result.stderr[-500:])

# %% [markdown]
# ## Step 1 — Client swap

# %%
from openai import OpenAI

local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Verify: run the NB 01 bare question.
response = local_client.chat.completions.create(
    model=LOCAL_MODEL,
    messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
    max_tokens=16,
)
print(f"llama3.2 says: {response.choices[0].message.content}")

# %% [markdown]
# ## Step 2 — Tool use
#
# Can the local model follow the tool-use protocol?
# The Anthropic SDK uses `input_schema`; the OpenAI SDK uses `parameters`.
# Both describe JSON schema for the tool's arguments.

# %%
from agentlab.llm import get_client

anthropic_client = get_client()
TASK = "Use the add tool to compute 17 + 25."

# Claude (Anthropic SDK + Anthropic tool schema)
claude_resp = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    tools=[{
        "name": "add",
        "description": "Add two integers.",
        "input_schema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
    }],
    messages=[{"role": "user", "content": TASK}],
)
print("Claude stop_reason:", claude_resp.stop_reason)
print("Claude content types:", [b.type for b in claude_resp.content])

# Local model (OpenAI SDK + OpenAI tool schema)
local_resp = local_client.chat.completions.create(
    model=LOCAL_MODEL,
    messages=[{"role": "user", "content": TASK}],
    tools=[{
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        },
    }],
    max_tokens=256,
)
print("\nllama3.2 finish_reason:", local_resp.choices[0].finish_reason)
print("llama3.2 tool_calls:", local_resp.choices[0].message.tool_calls)

# %% [markdown]
# ## Step 3 — Structured outputs
#
# Does the local model return valid `Answer` JSON when asked via
# prompt-only schema injection?

# %%
import json
from agentlab.types import Answer

SCHEMA = json.dumps(Answer.model_json_schema(), indent=2)
PROMPT = (
    f"Answer the following question. Return your response as JSON matching "
    f"exactly this schema:\n{SCHEMA}\n\nQuestion: What is the capital of France?"
)

# Claude
claude_ans = anthropic_client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=256,
    messages=[{"role": "user", "content": PROMPT}],
)
print("Claude:", claude_ans.content[0].text[:300])

# Local model
local_ans = local_client.chat.completions.create(
    model=LOCAL_MODEL,
    messages=[{"role": "user", "content": PROMPT}],
    max_tokens=256,
)
print("\nllama3.2:", local_ans.choices[0].message.content[:300])

# %% [markdown]
# ## Step 4 — Quantitative eval
#
# Run three tasks from `data/eval_tasks.jsonl` against both models inline.
# Task format: `{id, question, judge_rubric, ...}`.
# We compare response text side by side and measure latency.

# %%
with open("data/eval_tasks.jsonl") as f:
    task_map = {t["id"]: t for line in f
                if line.strip()
                for t in [json.loads(line)]}

EVAL_IDS = ["t01", "t05", "t11"]


def run_claude_task(task: dict) -> tuple[str, float]:
    t0 = _time.perf_counter()
    r = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": task["question"]}],
    )
    return "".join(getattr(b, "text", "") for b in r.content).strip(), _time.perf_counter() - t0


def run_local_task(task: dict) -> tuple[str, float]:
    t0 = _time.perf_counter()
    r = local_client.chat.completions.create(
        model=LOCAL_MODEL,
        messages=[{"role": "user", "content": task["question"]}],
        max_tokens=512,
    )
    return r.choices[0].message.content or "", _time.perf_counter() - t0


for tid in EVAL_IDS:
    task = task_map[tid]
    cr, ct = run_claude_task(task)
    lr, lt = run_local_task(task)
    print(f"\n--- {tid}: {task['question'][:70]}")
    print(f"Rubric: {task['judge_rubric']}")
    print(f"Claude  ({ct:.1f}s): {cr[:200]}")
    print(f"llama3.2 ({lt:.1f}s): {lr[:200]}")

# %% [markdown]
# ## Step 5 — LM Studio swap
#
# LM Studio's OpenAI-compatible server defaults to port 1234.
# Uncomment the cell below after starting LM Studio and loading a model.
# The code is identical to Step 1; only `base_url` changes.

# %%
# lm_studio_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lmstudio")
# response = lm_studio_client.chat.completions.create(
#     model="local-model",  # Replace with the model name shown in LM Studio's UI
#     messages=[{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
#     max_tokens=16,
# )
# print(f"LM Studio says: {response.choices[0].message.content}")

# %% [markdown]
# ## Reflect
#
# - **What the OpenAI-compat layer buys:** `chat.completions.create` works
#   against Ollama, LM Studio, Groq, Together, and any OpenAI-compatible
#   provider. Swapping is a one-line `base_url` change — as Step 5 showed.
# - **Where local models break hardest:** tool-use protocol compliance
#   (missing or malformed `tool_calls` blocks), long-context recall
#   (retrieval quality drops sharply past 4k tokens), and structured output
#   reliability (JSON schema adherence is inconsistent without constrained
#   decoding or grammar sampling).
# - **When to reach for local:** privacy or compliance requirements that
#   preclude cloud APIs, cost at high throughput, offline use, or rapid
#   iteration where API latency and billing add friction.
