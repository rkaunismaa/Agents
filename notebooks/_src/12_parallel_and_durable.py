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
# # NB 12 — ✦ Parallel + durable spine
#
# **Goal:** combine MCP-backed memory + parallel subagent fan-out +
# cancellation + checkpointing into the extracted spine library. Close
# the ✦ Module-C spine progression.
#
# Everything in this notebook **imports** from the library
# (`agentlab.spine`, `agentlab.mcp_helpers`). The notebook is the
# integration layer: wire the parts, run the demos, observe traces,
# extend the eval set.
#
# **Architectural call:** we extend NB 11 Stage A (manual orchestrator),
# not Stage B (Agent SDK). Reason: explicit task handles + cancellation
# + checkpointing are easier to thread through a hand-rolled
# orchestrator than to coax out of the SDK lifecycle. Constraint-fit,
# not framework-rejection.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="12 — ✦ Parallel + durable spine",
    estimate="$0.10–0.15 (3-4 spine runs + eval extension)",
    model="claude-sonnet-4-6 (workers + orchestrator)",
)

QUESTION = (
    "For each of these three open-source projects (httpx, FastAPI, "
    "Pydantic), summarize the README, fetch the latest GitHub release, "
    "and rank them by recent activity."
)

# %% [markdown]
# ## Step 0 — Spine extracted (recap)
#
# The orchestrator + Subagent definitions from NB 11 Stage A now live
# at `src/agentlab/spine.py`, with extensions for async fan-out,
# per-worker timeout, and JSONL checkpointing. NB 12 imports them.

# %%
from agentlab.spine import Orchestrator, Subagent, WorkerResult
from agentlab.llm import DEFAULT_MODEL, get_client

client = get_client()

WORKERS = [
    Subagent(role="researcher", system_prompt=(
        "You are a focused technical researcher. Use web_search to "
        "summarize the README of the given project and fetch the "
        "latest release. Return: 'README: <60-word summary>. Latest "
        "release: <tag and date>.' End with one citation URL per fact."
    )),
    Subagent(role="summarizer", system_prompt=(
        "You are a precise summarizer. From your training knowledge, "
        "write one tight paragraph (~80 words) covering all the "
        "projects in the question — key purpose, current state, and "
        "one standout fact per project. No preamble, no hedging."
    )),
    Subagent(role="ranker", system_prompt=(
        "You are a comparative analyst. From your training knowledge, "
        "rank the projects in the question by recent activity and "
        "ecosystem momentum. One-sentence justification per project, "
        "then a single recommendation. Be direct."
    )),
]

# %% [markdown]
# ## Step 1 — Async workers + asyncio.gather
#
# `Orchestrator.run_async` fans out workers concurrently. We define a
# Claude-backed `dispatch_async` that runs a subagent loop with the
# worker's system prompt.

# %%
import asyncio
import random
import time
from anthropic import RateLimitError as _RateLimitError


async def claude_dispatch(worker: Subagent, task: str) -> str:
    """Run a worker as a single Claude call (researcher uses web_search)."""
    tools = [{"type": "web_search_20250305", "name": "web_search"}] if worker.role == "researcher" else []
    messages = [{"role": "user", "content": task}]
    # Backoff with jitter: desynchronises parallel workers so they don't all
    # retry at the same instant and saturate the 30 k TPM limit together.
    for delay in (10, 20, 40, 90):
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=DEFAULT_MODEL,
                max_tokens=1024,
                system=worker.system_prompt,
                tools=tools,
                messages=messages,
            )
            return "".join(getattr(b, "text", "") for b in response.content).strip()
        except _RateLimitError:
            actual = delay + random.uniform(0, delay * 0.5)
            print(f"  [rate limit] {worker.role} waiting {actual:.0f}s...")
            await asyncio.sleep(actual)
    response = await asyncio.to_thread(
        client.messages.create,
        model=DEFAULT_MODEL,
        max_tokens=1024,
        system=worker.system_prompt,
        tools=tools,
        messages=messages,
    )
    return "".join(getattr(b, "text", "") for b in response.content).strip()


orch = Orchestrator(workers=WORKERS, dispatch_async=claude_dispatch)

print("--- Parallel run (no MCP, no checkpoints) ---")
start = time.perf_counter()
parallel_results = await orch.run_async(QUESTION, worker_timeout=240)
parallel_elapsed = time.perf_counter() - start
print(f"Wall-clock: {parallel_elapsed:.1f}s\n")
for r in parallel_results:
    snippet = (r.result or "")[:120]
    status = "✓" if r.error is None else f"✗ {r.error}"
    print(f"  [{r.role}] {status}  {snippet}")

# %% [markdown]
# ## Step 2 — Cancellation
#
# We introduce a tighter `worker_timeout` and a deliberately slow
# wrapper around the researcher to show graceful degradation. The
# orchestrator returns whatever workers finished in time; the slow one
# carries an error string.

# %%
async def slow_dispatch(worker: Subagent, task: str) -> str:
    if worker.role == "researcher":
        await asyncio.sleep(60)  # exceeds the tight timeout below
        return await claude_dispatch(worker, task)  # never reached within 30s
    # Non-researcher workers: return a stub — keeps the demo rate-limit-independent.
    await asyncio.sleep(0.1)
    return f"{worker.role}: stub result"


slow_orch = Orchestrator(workers=WORKERS, dispatch_async=slow_dispatch)

print("--- Cancellation demo (researcher times out) ---")
canceled_results = await slow_orch.run_async(QUESTION, worker_timeout=30)
for r in canceled_results:
    status = "✓" if r.error is None else f"✗ {r.error}"
    print(f"  [{r.role}] {status}")

# %% [markdown]
# ## Step 3 — Checkpointing
#
# Same `Orchestrator`, this time with `checkpoint_dir` set. Each
# successful worker's `WorkerResult` is appended to
# `data/checkpoints/<run_id>.jsonl`. We then re-run with the same
# `run_id` and observe that completed workers are skipped.
#
# We use a lightweight fake dispatch here: the learning goal is
# write-then-resume mechanics (not another live API call). Step 1
# already demonstrated parallelism; firing a second parallel fan-out
# immediately after it would saturate the 30k TPM rate-limit window.

# %%
from pathlib import Path

CHECKPOINT_DIR = Path("data/checkpoints")


async def _fake_dispatch(worker: Subagent, task: str) -> str:
    """Lightweight stand-in for the checkpoint demo — no API call."""
    await asyncio.sleep(0.05)
    return f"{worker.role}: stub (checkpoint demo)"


ckpt_orch = Orchestrator(
    workers=WORKERS, dispatch_async=_fake_dispatch, checkpoint_dir=CHECKPOINT_DIR
)

run_id = "demo_run"
# Wipe any prior checkpoint for this run_id so the demo is deterministic.
checkpoint_file = CHECKPOINT_DIR / f"{run_id}.jsonl"
checkpoint_file.unlink(missing_ok=True)

print("--- First run (writes checkpoint) ---")
first_results = await ckpt_orch.run_async(QUESTION, run_id=run_id, worker_timeout=240)
size = checkpoint_file.stat().st_size if checkpoint_file.exists() else 0
print(f"Wrote {size} bytes to {checkpoint_file}")

print("\n--- Second run with same run_id (should skip all workers) ---")
start = time.perf_counter()
second_results = await ckpt_orch.run_async(QUESTION, run_id=run_id, worker_timeout=240)
elapsed = time.perf_counter() - start
print(f"Wall-clock: {elapsed:.2f}s (expect ~0s — all workers cached)")
for r in second_results:
    snippet = (r.result or "")[:80]
    print(f"  [{r.role}] {snippet}...")

# %% [markdown]
# ## Step 4 — MCP-backed memory
#
# We connect to the notes server (NB 09) as a subprocess. Rather than
# firing another parallel fan-out (which would re-saturate the rate
# limit after Steps 1–3), we seed MCP from the `demo_run` checkpoint
# written in Step 3. The lesson is durability, not invocation: notes
# written via `add_note` survive a cold-started server.
#
# **The MCP-earning-its-keep moment for the spine:** after the writer
# exits, the notes are still there. A new session — even after a
# kernel restart — can recall them via `list_notes` / `get_note`.

# %%
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agentlab.mcp_helpers import MCPToolRouter

SERVER_PARAMS = StdioServerParameters(command="python", args=["mcp_servers/notes_server.py"])


async def write_checkpoint_to_mcp(source_run_id: str, dest_run_id: str) -> list[WorkerResult]:
    """Write cached worker results as MCP notes. No new API calls needed."""
    cached = ckpt_orch._load_checkpoint(source_run_id)
    results: list[WorkerResult] = []
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            router = MCPToolRouter(session)
            await router.refresh()
            for worker in WORKERS:
                wr = cached.get(worker.role)
                if wr and wr.result:
                    try:
                        await router.call(
                            "add_note",
                            {"key": f"{dest_run_id}:{worker.role}", "content": wr.result},
                        )
                        print(f"  [{worker.role}] persisted ✓")
                    except RuntimeError as exc:
                        print(f"  [{worker.role}] warn: {exc}")
                    results.append(wr)
    return results


async def list_notes_after():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            router = MCPToolRouter(session)
            keys_text = await router.call("list_notes", {})
            return keys_text


print("--- MCP-backed run (seeding from demo_run checkpoint) ---")
mcp_run_id = "mcp_demo"
mcp_results = await write_checkpoint_to_mcp("demo_run", mcp_run_id)

print("\n--- Notes after the run (cold-started server) ---")
print(await list_notes_after())

# %% [markdown]
# ## Step 5 — OTel traces
#
# Same instrumentation idea as NB 08, but spans now tree as:
#
# ```
# agent.run
# ├── subagent.researcher  (parallel sibling)
# │   └── llm.complete
# ├── subagent.summarizer  (parallel sibling)
# │   └── llm.complete
# └── subagent.ranker      (parallel sibling)
#     └── llm.complete
# ```
#
# We use `ConsoleSpanExporter` to print spans inline (Jaeger swap is
# the same one-cell appendix as NB 08).

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("agentlab.spine")


async def traced_dispatch(worker: Subagent, task: str) -> str:
    with tracer.start_as_current_span(f"subagent.{worker.role}") as span:
        span.set_attribute("worker.role", worker.role)
        span.set_attribute("task.length", len(task))
        with tracer.start_as_current_span("llm.complete"):
            text = await claude_dispatch(worker, task)
        span.set_attribute("result.length", len(text))
        return text


traced_orch = Orchestrator(workers=WORKERS, dispatch_async=traced_dispatch)

print("--- Traced parallel run ---")
with tracer.start_as_current_span("agent.run") as span:
    span.set_attribute("question.length", len(QUESTION))
    span.set_attribute("parallel_dispatch", True)
    traced_results = await traced_orch.run_async(QUESTION, worker_timeout=240)

# Force flush so spans print before the next cell.
_ = provider.force_flush()

# %% [markdown]
# ## Step 6 — Eval extension
#
# `data/eval_tasks.jsonl` was extended in this commit's task list with
# 5 parallel-friendly tasks (`t11`-`t15`) — multi-source comparisons.
# The existing pytest harness in
# `tests/eval/test_research_assistant.py` parametrizes over the JSONL,
# so the new rows are picked up automatically.
#
# **Run from a terminal** (not the notebook):
#
# ```bash
# UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest -m eval tests/eval/ -v
# ```
#
# We do not run the full eval suite from the notebook (slow + costs
# real money). The cell below runs **one** parallel-friendly task
# through the spine to verify the architecture *is* fanning out.

# %%
import json

with open("data/eval_tasks.jsonl") as f:
    tasks = [json.loads(line) for line in f if line.strip()]

t11 = next(t for t in tasks if t["id"] == "t11")
print(f"Running {t11['id']}: {t11['question']}\n")

start = time.perf_counter()
results_t11 = await traced_orch.run_async(t11["question"], worker_timeout=240)
elapsed = time.perf_counter() - start
print(f"Wall-clock: {elapsed:.1f}s (parallel dispatch — should be << 3× single-worker time)")

# %% [markdown]
# ## Reflect
#
# - **The spine ✦ progression is complete.**
#   - Memory (NB 06) → retrieval (NB 07) → measurable (NB 08) →
#     multi-agent + durable + MCP-backed (this NB).
# - **What NB 15 will do.** No new architecture — wrap `Orchestrator`
#   as a CLI script with env-driven config, tracing on by default,
#   README, and a `Makefile`-or-shell entrypoint.
# - **Why we picked manual over SDK in this NB.** Explicit task
#   handles + cancellation + checkpointing are hard to thread through
#   the SDK's lifecycle. Constraint-fit, not framework-rejection.
# - **What we deliberately did not build.** Distributed checkpointing,
#   subagent retry/circuit-breaker patterns, multi-server MCP
#   composition. Real concerns for production; out of curriculum
#   scope. NB 15 will note them but not implement.
#
# **Module C is done.** Module D (NB 13–15) will introduce computer
# use, local models, and the capstone CLI extraction.
