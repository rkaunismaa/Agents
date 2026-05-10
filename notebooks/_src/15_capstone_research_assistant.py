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
# # NB 15 — ✦ Capstone: research assistant as a deployable script
#
# **Goal:** graduate the spine from notebook to CLI artifact.
#
# By this point everything is already built:
# `agentlab.spine` has `Orchestrator` + `Subagent`; the dispatch
# function, rate-limit backoff, and OTel tracing are all in place.
# This notebook shows the finished `research_assistant.py`, runs it
# from the terminal, and verifies the eval suite is still green.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="15 — ✦ Capstone: research assistant",
    estimate="$0.10–0.20 (2–3 spine runs + eval suite)",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Step 1 — The script

# %%
import subprocess
from pathlib import Path

from rich.console import Console
from rich.syntax import Syntax

console = Console()
source = Path("research_assistant.py").read_text()
console.print(Syntax(source, "python", theme="monokai", line_numbers=True))

# %% [markdown]
# ## Step 2 — Run from the terminal (via subprocess)

# %%
QUESTION = "What is the Model Context Protocol (MCP) and who created it?"

result = subprocess.run(
    ["python", "research_assistant.py", QUESTION],
    capture_output=True,
    text=True,
)
print(result.stdout)
if result.returncode != 0:
    print("STDERR:", result.stderr[-500:])

# %% [markdown]
# ## Step 3 — Checkpoint resume
#
# Run twice with the same `--run-id`. The second run should complete
# in under a second — all workers are loaded from the JSONL checkpoint.

# %%
import time

RUN_ID = "nb15_demo"

# First run — writes checkpoint.
r1 = subprocess.run(
    ["python", "research_assistant.py", QUESTION, "--run-id", RUN_ID],
    capture_output=True, text=True,
)
print("First run output (truncated):")
print(r1.stdout[:600])

# Second run — all workers cached.
start = time.perf_counter()
r2 = subprocess.run(
    ["python", "research_assistant.py", QUESTION, "--run-id", RUN_ID],
    capture_output=True, text=True,
)
elapsed = time.perf_counter() - start
print(f"\nSecond run wall-clock: {elapsed:.2f}s (expect <1s — workers cached)")
print(r2.stdout[:300])

# %% [markdown]
# ## Step 4 — OTel tracing
#
# Run with `--trace`. The span tree `agent.run → subagent.{role} → llm.complete`
# appears inline via `ConsoleSpanExporter`. Workers are cached, so no new API
# calls are made — spans still emit from the local context manager code.

# %%
r_trace = subprocess.run(
    ["python", "research_assistant.py", QUESTION, "--run-id", RUN_ID, "--trace"],
    capture_output=True, text=True,
)
print("Output (truncated):")
print(r_trace.stdout[:1500])

# %% [markdown]
# ## Step 5 — Eval suite green
#
# Run `pytest tests/eval/ -m eval -v` as a subprocess.
# This is the "eval suite green" success criterion from the curriculum spec.
#
# **Note:** this makes real API calls (~$0.10–0.20). It confirms the
# research assistant spine passes all 15 eval tasks.

# %%
eval_result = subprocess.run(
    [".agents/bin/pytest", "tests/eval/", "-m", "eval", "-v"],
    capture_output=True, text=True,
)
print(eval_result.stdout[-3000:])
assert eval_result.returncode == 0, f"Eval suite failed:\n{eval_result.stdout[-1000:]}"
print("✓ Eval suite passed.")

# %% [markdown]
# ## Reflect
#
# - **The spine ✦ progression is complete.**
#   Bare loop (NB 01) → structured outputs (NB 03) → evals + OTel (NB 08)
#   → multi-agent durable (NB 12) → deployable artifact (NB 15).
# - **What changed between notebook and script:** nothing architectural.
#   Wrapping `Orchestrator` in a `typer` command added env-driven config,
#   CLI flags, and rich output around code that already worked.
# - **What Module D deliberately did not build:** autonomous multi-step
#   computer use tasks, fine-tuned local models, production Docker
#   deployment. Real concerns for production; out of curriculum scope.
# - **Where to go next:** `agentlab` is the seed of a production codebase.
#   Swap in a real vector store, add a database-backed checkpoint store,
#   wire it to a Slack bot, or serve it via FastAPI.
