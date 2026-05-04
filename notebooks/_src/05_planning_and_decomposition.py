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
# # NB 05 — Planning & decomposition
#
# **Goal:** plan-then-execute. A *planner* produces a TODO list; an
# *executor* walks the list; on failure, a *re-planner* revises. Then a
# *synthesizer* turns the executed plan into a structured `Answer`.
#
# **Demo task:** *"Compare httpx, aiohttp, and requests for an async-first
# Python project; recommend one."*
#
# Why this task: three subgoals (one per library), a synthesis step,
# and a natural failure mode if a search returns nothing useful.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="05 — Planning & decomposition",
    estimate="$0.03",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — Define the plan type
#
# A `Plan` is a list of `Step`s. Each `Step` has a description, the
# tool to call, the args, and a status. We use Pydantic so the planner
# can't return malformed plans.

# %%
from typing import Literal

from pydantic import BaseModel, Field


class Step(BaseModel):
    description: str = Field(description="What this step accomplishes, in plain English.")
    tool: Literal["web_search", "synthesize"] = Field(
        description="Which tool runs this step. 'synthesize' is the final summarizer."
    )
    args: dict = Field(description="Arguments for the tool. For web_search: {'query': str}.")
    status: Literal["pending", "done", "failed"] = "pending"
    result: str | None = None


class Plan(BaseModel):
    goal: str
    steps: list[Step]


# %% [markdown]
# ## Step 2 — Planner: tool-use coercion forces a valid `Plan`
#
# We define a `submit_plan` tool whose `input_schema` is the Plan's
# JSON schema. The planner *must* call it.

# %%
from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()

submit_plan_tool = {
    "name": "submit_plan",
    "description": "Submit your plan. Must be called exactly once.",
    "input_schema": Plan.model_json_schema(),
}

PLANNER_SYSTEM = """You are a planner. Decompose the user's goal into 3-5
concrete steps. Each step uses one of: web_search (gather information),
synthesize (only as the final step, to produce the recommendation).
Call submit_plan exactly once."""


def make_plan(goal: str) -> Plan:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=PLANNER_SYSTEM,
        tools=[submit_plan_tool],
        messages=[{"role": "user", "content": goal}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_plan":
            return Plan.model_validate(block.input)
    raise RuntimeError("Planner did not call submit_plan.")


GOAL = (
    "Compare httpx, aiohttp, and requests for a new async-first Python "
    "project. Recommend one."
)

plan = make_plan(GOAL)
for i, step in enumerate(plan.steps, 1):
    print(f"{i}. [{step.tool}] {step.description}")

# %% [markdown]
# ## Step 3 — Executor
#
# Walk the plan. For `web_search` steps, run a single-turn agent with
# the managed web_search tool and capture the answer. For `synthesize`,
# defer to Step 5. Mark each step `done` or `failed`.

# %%
def execute_search_step(step: Step) -> str:
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system="You are a research assistant. Use web_search; reply in 2-3 sentences with citations.",
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": step.args.get("query", step.description)}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content)
    sources: list[str] = []
    for block in response.content:
        if getattr(block, "type", None) != "web_search_tool_result":
            continue
        content = getattr(block, "content", None)
        if not isinstance(content, list):
            continue
        for r in content:
            url = getattr(r, "url", None)
            title = getattr(r, "title", None) or url
            if url:
                sources.append(f"- {title}: {url}")
    if not text.strip() or "no results" in text.lower():
        raise RuntimeError("empty search result")
    if sources:
        text = f"{text.strip()}\n\nSources:\n" + "\n".join(sources)
    return text.strip()


def execute(plan: Plan) -> Plan:
    for step in plan.steps:
        if step.tool != "web_search":
            continue
        try:
            step.result = execute_search_step(step)
            step.status = "done"
            print(f"✓ {step.description}")
        except Exception as exc:
            step.status = "failed"
            step.result = str(exc)
            print(f"✗ {step.description}  ({exc})")
    return plan


plan = execute(plan)

# %% [markdown]
# ## Step 4 — Re-planning on failure
#
# If any step failed, send the partially-executed plan back to the
# planner and ask for a revision. One revision pass max.

# %%
def replan_if_needed(plan: Plan) -> Plan:
    failed = [s for s in plan.steps if s.status == "failed"]
    if not failed:
        return plan
    print(f"\n[re-plan triggered: {len(failed)} step(s) failed]")
    revised = make_plan(
        f"Original goal: {plan.goal}\n\n"
        f"Previous plan had failures: {[s.description for s in failed]}\n"
        f"Revise the plan to work around them."
    )
    return execute(revised)


plan = replan_if_needed(plan)

# %% [markdown]
# ## Step 5 — Synthesize the final answer
#
# Reuse NB 03's `Answer` model. The synthesizer reads all the executed
# steps and produces the recommendation.

# %%
from agentlab.types import Answer

submit_answer_tool = {
    "name": "submit_answer",
    "description": (
        "Submit your final answer. The summary must be at least one sentence "
        "and you MUST include at least one citation drawn from the Sources "
        "listed in the research below. Do not invent URLs."
    ),
    "input_schema": Answer.model_json_schema(),
}

SYNTH_SYSTEM = """Synthesize a recommendation from the research below.
You must call submit_answer exactly once, with citations drawn from the
'Sources:' lists in the research. Use at least one citation; ideally 2-3.
Do not invent URLs — only cite URLs that appear in the research."""


def synthesize(plan: Plan) -> Answer:
    body = f"Goal: {plan.goal}\n\nResearch:\n"
    for step in plan.steps:
        if step.tool == "web_search" and step.status == "done":
            body += f"\n## {step.description}\n{step.result}\n"
    response = client.messages.create(
        model=DEFAULT_MODEL,
        max_tokens=2048,
        system=SYNTH_SYSTEM,
        tools=[submit_answer_tool],
        messages=[{"role": "user", "content": body}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_answer":
            return Answer.model_validate(block.input)
    raise RuntimeError("Synthesizer did not call submit_answer.")


answer = synthesize(plan)
print(answer.summary)
print()
for c in answer.citations:
    print(f"  - {c.title} — {c.url}")

# %% [markdown]
# ## Reflect
#
# - **Plan-then-execute beats free-form looping** when subgoals are
#   independent. The agent can't "forget" to compare candidate B
#   because it's an explicit step.
# - **Re-planning on failure** is cheap insurance against tool flakiness.
# - **Tool-use coercion is doing a lot of work here** — both the
#   planner and synthesizer use it. NB 03's pattern compounds.
# - **When NOT to plan:** single-tool tasks. NB 02's `run_agent_loop`
#   handles those without ceremony.
# - **Next:** NB 06 introduces memory so the spine project can answer
#   follow-ups.
