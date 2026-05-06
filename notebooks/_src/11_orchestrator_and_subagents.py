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
import time

from anthropic import RateLimitError as _RateLimitError

from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.types import Answer

client = get_client()


def _create(**kwargs):
    """Thin wrapper around client.messages.create with rate-limit backoff."""
    for delay in (20, 40, 60):
        try:
            return client.messages.create(**kwargs)
        except _RateLimitError:
            print(f"[rate limit] waiting {delay}s before retry...")
            time.sleep(delay)
    return client.messages.create(**kwargs)

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
        response = _create(
            model=DEFAULT_MODEL,
            max_tokens=4096,
            system=system,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason in ("end_turn", "max_tokens"):
            text = "".join(getattr(b, "text", "") for b in response.content)
            return text.strip()
        messages.append({"role": "assistant", "content": response.content})
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
        response = _create(
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
    last_text: str = ""
    async with ClaudeSDKClient(options=SDK_OPTIONS) as sdk:
        await sdk.query(question)
        async for message in sdk.receive_response():
            # Track the *last* AssistantMessage rather than accumulating all
            # of them — intermediate orchestrator turns (if the SDK surfaces
            # them) would otherwise corrupt the final answer.
            if type(message).__name__ == "AssistantMessage":
                parts = []
                for block in getattr(message, "content", []) or []:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
                if parts:
                    last_text = "\n".join(parts)
    return last_text.strip()


print("=== Stage B: claude-agent-sdk ===\n")
sdk_text = await run_orchestrator_sdk(QUESTION)
print(f"\nSDK final text ({len(sdk_text)} chars):")
print(sdk_text[:1200])

# %% [markdown]
# ### Side-by-side
#
# | Aspect | Stage A (scratch) | Stage B (SDK) |
# |---|---|---|
# | Lines of code | ~120 | ~50 |
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
