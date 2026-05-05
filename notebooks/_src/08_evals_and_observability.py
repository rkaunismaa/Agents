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
#     display_name: .agents
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NB 08 — ✦ Evals + observability (spine integration)
#
# **Goal:** make "is the agent working?" measurable. We layer three
# eval styles + OpenTelemetry tracing on the research-assistant spine.
#
# Three eval styles, each with a different failure mode:
# 1. **Deterministic asserts** — fast, free, catches structural bugs;
#    misses content quality.
# 2. **LLM-as-judge** — flexible, subjective; flaky and costs money.
# 3. **Reference-based** — exact match for known-answer tasks; doesn't
#    fit open-ended questions.
#
# Plus OpenTelemetry traces so you can *see* what the agent did.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="08 — Evals + observability (✦ spine + measurement)",
    estimate="$0.05–0.10",
    model="claude-sonnet-4-6 (judges: claude-haiku-4-5)",
)

# %% [markdown]
# ## Step 1 — Load the eval tasks
#
# 10 tasks live in `data/eval_tasks.jsonl`. Mix of factual lookups,
# explainers, and known-answer questions.

# %%
import json
from pathlib import Path

tasks = [
    json.loads(line)
    for line in Path("data/eval_tasks.jsonl").read_text().splitlines()
    if line.strip()
]
print(f"Loaded {len(tasks)} eval tasks.")
for t in tasks[:3]:
    print(f"  [{t['id']}] {t['question']}")
print("  ...")

# %% [markdown]
# ## Step 2 — Trace the spine with OpenTelemetry
#
# We wrap `research()` with spans for `agent.run`, `tool.web_search`,
# and `llm.complete`. Use `ConsoleSpanExporter` so spans print to the
# notebook output as JSON.

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agentlab.spine")

# %%
from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.types import Answer

client = get_client()

# Inline copy of the spine `research()` (per the curriculum's "no
# library extraction until NB 15" rule). Same logic as NB 03 and as
# tests/eval/research_under_test.py; consolidated in NB 15.
SUBMIT_ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your final answer with citations.",
    "input_schema": Answer.model_json_schema(),
}
SPINE_SYSTEM = """You are a careful technical research assistant.
1. Use web_search for facts you aren't sure of.
2. Call submit_answer with summary + citations when ready.
3. Do not produce free-form text after submit_answer."""


def research(question: str, max_turns: int = 8):
    tool_calls = []
    messages = [{"role": "user", "content": question}]
    for _ in range(max_turns):
        response = client.messages.create(
            model=DEFAULT_MODEL, max_tokens=2048, system=SPINE_SYSTEM,
            tools=[
                {"type": "web_search_20250305", "name": "web_search"},
                SUBMIT_ANSWER_TOOL,
            ],
            messages=messages,
        )
        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append(block.name)
                if block.name == "submit_answer":
                    return Answer.model_validate(block.input), tool_calls
            if block.type == "server_tool_use":
                tool_calls.append(getattr(block, "name", "server_tool"))
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            raise RuntimeError("end_turn without submit_answer.")
    raise RuntimeError("Out of turns.")


def traced_research(question: str):
    with tracer.start_as_current_span("agent.run") as span:
        span.set_attribute("question", question)
        answer, tool_calls = research(question)
        span.set_attribute("tool_calls", ",".join(tool_calls))
        span.set_attribute("citation_count", len(answer.citations))
        return answer, tool_calls


answer, tool_calls = traced_research(tasks[1]["question"])  # MCP question
print(f"Tool calls: {tool_calls}")
print(f"Summary: {answer.summary[:300]}")

# Force the exporter to flush so spans appear before the next cell.
provider.force_flush()

# %% [markdown]
# ## Step 3 — Eval style 1: deterministic asserts
#
# For each task: did `research()` return a valid `Answer`? Did it call
# the expected tools? Are there ≥1 citations?

# %%
from agentlab.types import Answer

deterministic_results = []
for task in tasks[:3]:  # subset for cost; full suite via pytest below
    try:
        answer, tool_calls = research(task["question"])
        ok = isinstance(answer, Answer) and len(answer.citations) >= 1
        ok &= all(t in tool_calls for t in task["expected_tool_calls"])
        deterministic_results.append((task["id"], ok))
    except Exception as e:
        deterministic_results.append((task["id"], False))
for tid, ok in deterministic_results:
    print(f"{'✓' if ok else '✗'} {tid}")

# %% [markdown]
# ## Step 4 — Eval style 2: LLM-as-judge
#
# A separate Haiku call rates the summary 1-5 against each task's
# rubric. Anything ≤ 2 is a fail.

# %%
JUDGE_SYSTEM = """You are an evaluator. Score the answer 1-5 against the rubric:
1 = wildly wrong / off-topic
2 = partially relevant but missing key facts
3 = correct but minimal
4 = correct, well-cited, complete
5 = exemplary

Output ONLY the integer score on its own line."""


def judge(question: str, rubric: str, answer: Answer) -> int:
    prompt = (
        f"Question: {question}\n\nRubric: {rubric}\n\n"
        f"Answer summary: {answer.summary}\n\n"
        f"Citations: {[c.url for c in answer.citations]}"
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(getattr(b, "text", "") for b in response.content).strip()
    return int(text.split()[0])


judge_results = []
for task in tasks[:3]:
    answer, _ = research(task["question"])
    score = judge(task["question"], task["judge_rubric"], answer)
    judge_results.append((task["id"], score))
for tid, score in judge_results:
    flag = "✓" if score >= 3 else "✗"
    print(f"{flag} {tid}: {score}/5")

# %% [markdown]
# ## Step 5 — Eval style 3: reference-based
#
# For tasks with a `reference_answer`, substring match against the
# summary or citations.

# %%
ref_results = []
for task in tasks[:3]:
    if task["reference_answer"] is None:
        ref_results.append((task["id"], "skip"))
        continue
    answer, _ = research(task["question"])
    ref = task["reference_answer"].lower()
    summary = answer.summary.lower()
    cited = " ".join(c.title.lower() + " " + (c.quote or "").lower() for c in answer.citations)
    ok = ref in summary or ref in cited
    ref_results.append((task["id"], "✓" if ok else "✗"))
for tid, flag in ref_results:
    print(f"{flag} {tid}")

# %% [markdown]
# ## Step 6 — Run the full suite via pytest
#
# Eval tests live in `tests/eval/` and are gated behind
# `@pytest.mark.eval`. Run them with:
#
# ```bash
# pytest -m eval tests/eval/ -v
# ```
#
# That runs all 10 tasks × 3 styles = 30 test instances. Skip pytest
# from the notebook (it's a separate command); the snippet above covers
# the demo subset.

# %% [markdown]
# ## Step 7 — Where each eval style breaks down
#
# - **Deterministic asserts** are great for structural checks
#   (validation passed? expected tools called?) but blind to whether
#   the answer is *correct*. A confidently-wrong answer with one
#   citation passes.
# - **LLM-as-judge** is flexible but flaky. The same answer can score
#   3 or 4 across runs. Use cheap models (Haiku) and run twice for
#   stability.
# - **Reference-based** is rock-solid when you can write a reference
#   answer, but most agent tasks are open-ended. Half our tasks have
#   `reference_answer: null` for that reason.
#
# **Use them in combination.** Deterministic catches bugs you don't
# even mean to introduce; reference catches regressions on known
# answers; LLM-judge surfaces quality drift.

# %% [markdown]
# ## Appendix — Swap to a real OTel collector (Jaeger)
#
# To send traces to a real UI:
#
# ```bash
# docker run -d --name jaeger \
#   -p 16686:16686 -p 4317:4317 \
#   jaegertracing/all-in-one:latest
# ```
#
# Then swap the exporter:
#
# ```python
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
#
# provider = TracerProvider()
# provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)))
# trace.set_tracer_provider(provider)
# ```
#
# Open http://localhost:16686 and click into traces.

# %% [markdown]
# ## Reflect
#
# - **Evals turn vibes into numbers.** "Feels right" doesn't survive a
#   model upgrade; pinned eval scores do.
# - **Tracing turns black boxes into something you can ask questions of.**
#   When the spine answers slowly, traces tell you whether it was the
#   model, the tool, or your own glue code.
# - **Module B is done.** The research assistant has memory (NB 06),
#   knows when to retrieve (NB 07), and is measurable (this NB).
# - **Next (Module C):** MCP. We build a server (NB 09), consume one
#   (NB 10), and start orchestrating subagents (NB 11-12).
