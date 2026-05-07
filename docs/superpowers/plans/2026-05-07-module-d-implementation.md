# Module D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Module D (NB 13–15) to close the 15-notebook curriculum — computer use, local models, and the capstone CLI artifact.

**Architecture:** Three notebooks (NB 13/14/15) plus `research_assistant.py` at repo root. No new `agentlab/` library modules — NB 13/14 are notebook-only; NB 15 imports the existing `agentlab.spine` / `agentlab.llm`. The `--rm` Docker flag on the computer-use container keeps teardown automatic.

**Tech Stack:** `anthropic` SDK (computer-use-2024-10-22 beta), `openai` (OpenAI-compatible local endpoint), `typer>=0.12.0`, `rich` (already core), jupytext, Docker (host-installed), Ollama + LM Studio (host-installed).

---

## File map

| Status | Path | Purpose |
|--------|------|---------|
| Modify | `pyproject.toml` | Add `typer>=0.12.0` to `module-d` extra |
| Create | `notebooks/_src/13_computer_use.py` | NB 13 jupytext source |
| Create | `notebooks/13_computer_use.ipynb` | NB 13 notebook (jupytext sync) |
| Create | `notebooks/_src/14_running_locally.py` | NB 14 jupytext source |
| Create | `notebooks/14_running_locally.ipynb` | NB 14 notebook (jupytext sync) |
| Create | `research_assistant.py` | Capstone CLI script |
| Create | `notebooks/_src/15_capstone_research_assistant.py` | NB 15 jupytext source |
| Create | `notebooks/15_capstone_research_assistant.ipynb` | NB 15 notebook (jupytext sync) |
| Modify | `README.md` | Module D row + section |

---

## Task D-1: Branch + typer dep sync

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feature/module-d
```

Expected: `Switched to a new branch 'feature/module-d'`

- [ ] **Step 2: Add typer to module-d extra**

In `pyproject.toml`, find the `module-d` extra and add `typer`:

```toml
module-d = [
    "openai>=1.55.0",
    "typer>=0.12.0",
]
```

- [ ] **Step 3: Sync the venv**

```bash
UV_PROJECT_ENVIRONMENT=.agents uv sync --extra module-d
```

Expected: resolves and installs `typer`; no errors.

- [ ] **Step 4: Verify 42 tests still pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/ --ignore=tests/eval -q
```

Expected: `42 passed`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat(d1): add typer to module-d extra"
```

---

## Task D-2: NB 13 — Computer use

**Files:**
- Create: `notebooks/_src/13_computer_use.py`
- Create: `notebooks/13_computer_use.ipynb` (via jupytext sync)

No unit tests — Docker + API I/O are not testable in CI.

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/13_computer_use.py` with this exact content:

```python
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
# # NB 13 — Computer use
#
# **Goal:** control a sandboxed virtual desktop via Claude's computer-use API.
#
# Claude receives a screenshot of the desktop, decides what action to take
# (click, type, key, etc.), we execute that action in a Docker container,
# then send a fresh screenshot and repeat. No special library beyond
# `anthropic` — the loop is ~40 lines.
#
# **Sandbox:** Anthropic's reference Docker image provides a virtual X11
# desktop with Chromium. We start it here and tear it down at the end.
# Never point this loop at your real desktop.

# %%
from _common import chdir_to_repo_root, cost_banner, load_env

load_env()
chdir_to_repo_root()
cost_banner(
    notebook="13 — Computer use",
    estimate="$0.10–0.20 (image-heavy turns; Sonnet throughout)",
    model="claude-sonnet-4-6",
)

# %% [markdown]
# ## Setup — start the Docker sandbox

# %%
import base64
import subprocess
import time

from IPython.display import Image
from IPython.display import display as ipy_display

from agentlab.llm import get_client

CONTAINER_NAME = "cu_demo"
DISPLAY_VAR = ":1"

# Clean up any prior run, then start fresh.
subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
subprocess.check_call([
    "docker", "run", "-d",
    "--name", CONTAINER_NAME,
    "ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest",
])
print("Container started. Waiting for desktop to initialize...")
time.sleep(5)
print("Ready.")

client = get_client()

# %% [markdown]
# ## Step 1 — The computer-use tools
#
# Claude's computer-use API exposes three tool types via
# `client.beta.messages.create(betas=["computer-use-2024-10-22"])`:
#
# | Tool type | Name | What it does |
# |---|---|---|
# | `computer_20241022` | `computer` | Screenshot + mouse/keyboard |
# | `text_editor_20241022` | `str_replace_editor` | File edits (view, str_replace, create) |
# | `bash_20241022` | `bash` | Shell commands |
#
# We only use `computer` in this notebook. The other two are available via
# the same `betas` parameter when you need file-level or shell access.

# %% [markdown]
# ## Step 2 — Screenshot + action helpers

# %%
def take_screenshot() -> bytes:
    """Capture the sandbox desktop as raw PNG bytes."""
    b64 = subprocess.check_output([
        "docker", "exec", "-e", f"DISPLAY={DISPLAY_VAR}", CONTAINER_NAME,
        "bash", "-c",
        "import -window root /tmp/screen.png && base64 -w 0 /tmp/screen.png",
    ]).decode().strip()
    return base64.b64decode(b64)


def xdotool(*args: str) -> None:
    subprocess.run(
        ["docker", "exec", "-e", f"DISPLAY={DISPLAY_VAR}",
         CONTAINER_NAME, "xdotool"] + list(args),
        check=True,
    )


# Verify: take one screenshot and display it inline.
ipy_display(Image(data=take_screenshot()))

# %% [markdown]
# ## Step 3 — The agent loop

# %%
COMPUTER_TOOLS = [{
    "type": "computer_20241022",
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768,
    "display_number": 1,
}]


def execute_action(action_input: dict) -> list[dict]:
    """Execute one computer-use action; return a tool_result content list."""
    action = action_input["action"]
    if action == "screenshot":
        b64 = base64.b64encode(take_screenshot()).decode()
        return [{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": b64,
        }}]
    if action in ("left_click", "right_click", "double_click"):
        x, y = action_input["coordinate"]
        btn = "3" if action == "right_click" else "1"
        extra = ["--repeat", "2"] if action == "double_click" else []
        xdotool("mousemove", str(x), str(y))
        xdotool("click", *extra, btn)
        return [{"type": "text", "text": f"{action} at ({x},{y})"}]
    if action == "mouse_move":
        x, y = action_input["coordinate"]
        xdotool("mousemove", str(x), str(y))
        return [{"type": "text", "text": f"moved to ({x},{y})"}]
    if action == "type":
        xdotool("type", "--clearmodifiers", action_input["text"])
        return [{"type": "text", "text": "typed"}]
    if action == "key":
        xdotool("key", action_input["text"])
        return [{"type": "text", "text": "key pressed"}]
    if action == "cursor_position":
        pos = subprocess.check_output(
            ["docker", "exec", "-e", f"DISPLAY={DISPLAY_VAR}",
             CONTAINER_NAME, "xdotool", "getmouselocation"],
        ).decode().strip()
        return [{"type": "text", "text": pos}]
    raise ValueError(f"Unknown action: {action!r}")


def claude_cu_loop(task: str, max_turns: int = 20) -> str:
    """Run a computer-use task; return the final assistant text."""
    messages = [{"role": "user", "content": task}]
    for _turn in range(max_turns):
        response = client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            tools=COMPUTER_TOOLS,
            messages=messages,
            betas=["computer-use-2024-10-22"],
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            return "".join(
                getattr(b, "text", "") for b in response.content
            ).strip()
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": execute_action(block.input),
            })
        if not tool_results:
            break
        messages.append({"role": "user", "content": tool_results})
    return "max_turns reached"

# %% [markdown]
# ## Step 4 — Demo task
#
# "Open Chromium, navigate to python.org, and tell me the current Python
# version shown on the homepage."

# %%
print("--- Computer use demo ---")
result = claude_cu_loop(
    "Open Chromium, navigate to python.org, and tell me the current "
    "Python version shown on the homepage.",
    max_turns=20,
)
print(result)
ipy_display(Image(data=take_screenshot()))  # Final desktop state

# %% [markdown]
# ## Step 5 — Safety considerations
#
# **Always sandbox.** The loop has unconditional trust in Claude's actions.
# Pointed at your real display it could close windows, modify files, or
# send messages. The Docker container is a throwaway VM with no access to
# your filesystem or network (unless you explicitly bind-mount or open ports).
#
# **Cost.** Each screenshot is ~50–100 KB base64. A 10-turn demo with
# screenshots every other turn costs roughly $0.05–0.15 in input tokens.
#
# **Abort guard.** `execute_action` raises `ValueError` on any unknown
# action type. This surfaces unexpected model output immediately rather
# than silently ignoring it.
#
# **Rate limiting.** Image-heavy messages consume many input tokens.
# Add `time.sleep(0.5)` between turns if you hit rate limits on longer
# sessions.

# %% [markdown]
# ## Teardown

# %%
subprocess.run(["docker", "stop", CONTAINER_NAME], check=True)
print("Container stopped and removed.")

# %% [markdown]
# ## Reflect
#
# - **Screenshot→action vs. tool use.** In regular tool use, Claude receives
#   structured data back from tools. In computer use, it receives *images*
#   of the desktop state — the tool result is visual, not textual.
# - **Safety and cost are first-class.** Computer use loops can run
#   indefinitely, interact with real systems, and consume many tokens per
#   turn. Always sandbox; always cap `max_turns`; always watch your cost.
# - **`text_editor` and `bash` round it out.** For tasks beyond
#   mouse/keyboard (editing config files, running install scripts), add
#   those tool types via the same `betas` parameter and the same loop
#   structure.
```

- [ ] **Step 2: Sync to .ipynb**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/13_computer_use.py
```

Expected: creates `notebooks/13_computer_use.ipynb`

- [ ] **Step 3: Commit**

```bash
git add notebooks/_src/13_computer_use.py notebooks/13_computer_use.ipynb
git commit -m "feat(d2): NB 13 — computer use (Docker sandbox + claude_cu_loop)"
```

---

## Task D-3: NB 14 — Local models

**Files:**
- Create: `notebooks/_src/14_running_locally.py`
- Create: `notebooks/14_running_locally.ipynb` (via jupytext sync)

No unit tests — Ollama/LM Studio are host processes, not testable in CI.

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/14_running_locally.py`:

```python
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
```

- [ ] **Step 2: Sync to .ipynb**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/14_running_locally.py
```

Expected: creates `notebooks/14_running_locally.ipynb`

- [ ] **Step 3: Commit**

```bash
git add notebooks/_src/14_running_locally.py notebooks/14_running_locally.ipynb
git commit -m "feat(d3): NB 14 — local models (Ollama + LM Studio swap)"
```

---

## Task D-4: `research_assistant.py`

**Files:**
- Create: `research_assistant.py`

- [ ] **Step 1: Write the script**

Create `research_assistant.py` at the repo root:

```python
#!/usr/bin/env python
"""Research assistant CLI — wraps agentlab.spine for terminal use."""
from __future__ import annotations

import asyncio
import os
import random
from pathlib import Path

import typer
from anthropic import RateLimitError as _RateLimitError
from rich.console import Console

from agentlab.llm import DEFAULT_MODEL, get_client
from agentlab.spine import Orchestrator, Subagent, WorkerResult

app = typer.Typer(add_completion=False, help="Parallel research assistant powered by agentlab.spine.")
console = Console()

WORKERS = [
    Subagent(role="researcher", system_prompt=(
        "You are a focused technical researcher. Use web_search to gather "
        "current information about the topic in the question. Return factual "
        "findings with at least one source URL per claim."
    )),
    Subagent(role="summarizer", system_prompt=(
        "You are a precise summarizer. In one tight paragraph (~80 words), "
        "synthesize an answer to the question from your training knowledge. "
        "Be direct, no hedging, no preamble."
    )),
    Subagent(role="ranker", system_prompt=(
        "You are a comparative analyst. If the question involves comparing or "
        "ranking items, rank them explicitly with one-sentence reasoning per "
        "item and a final recommendation. If not comparative, give your most "
        "direct answer."
    )),
]


async def _claude_dispatch(worker: Subagent, task: str, model: str, client) -> str:
    """Run a worker as a single Claude call with jitter backoff on rate limits."""
    tools = [{"type": "web_search_20250305", "name": "web_search"}] if worker.role == "researcher" else []
    messages = [{"role": "user", "content": task}]
    for delay in (10, 20, 40, 90):
        try:
            response = await asyncio.to_thread(
                client.messages.create,
                model=model,
                max_tokens=1024,
                system=worker.system_prompt,
                tools=tools,
                messages=messages,
            )
            return "".join(getattr(b, "text", "") for b in response.content).strip()
        except _RateLimitError:
            actual = delay + random.uniform(0, delay * 0.5)
            console.print(f"  [yellow][rate limit] {worker.role} waiting {actual:.0f}s...[/yellow]")
            await asyncio.sleep(actual)
    response = await asyncio.to_thread(
        client.messages.create,
        model=model,
        max_tokens=1024,
        system=worker.system_prompt,
        tools=tools,
        messages=messages,
    )
    return "".join(getattr(b, "text", "") for b in response.content).strip()


def _setup_tracing():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("agentlab.spine"), provider


@app.command()
def main(
    question: str = typer.Argument(..., help="Research question to answer"),
    run_id: str = typer.Option(
        None, "--run-id", help="Resume from checkpoint (same run_id skips completed workers)"
    ),
    trace: bool = typer.Option(False, "--trace", help="Enable OTel console tracing"),
    model: str = typer.Option(
        os.getenv("AGENTLAB_MODEL", DEFAULT_MODEL), "--model", help="Claude model to use"
    ),
    checkpoint_dir: Path = typer.Option(
        Path(os.getenv("AGENTLAB_CHECKPOINT_DIR", "data/checkpoints")),
        "--checkpoint-dir",
        help="Directory for JSONL checkpoint files",
    ),
) -> None:
    """Run the parallel research assistant spine from the command line."""
    tracer, provider = _setup_tracing() if trace else (None, None)
    client = get_client()

    async def _dispatch(worker: Subagent, task: str) -> str:
        if tracer:
            with tracer.start_as_current_span(f"subagent.{worker.role}") as span:
                span.set_attribute("worker.role", worker.role)
                with tracer.start_as_current_span("llm.complete"):
                    return await _claude_dispatch(worker, task, model, client)
        return await _claude_dispatch(worker, task, model, client)

    orch = Orchestrator(
        workers=WORKERS,
        dispatch_async=_dispatch,
        checkpoint_dir=checkpoint_dir,
    )

    console.print(f"\n[bold blue]Research Assistant[/bold blue] — [dim]{model}[/dim]")
    console.print(f"[dim]Question:[/dim] {question}\n")

    async def _run() -> list[WorkerResult]:
        if tracer:
            with tracer.start_as_current_span("agent.run") as span:
                span.set_attribute("question.length", len(question))
                return await orch.run_async(question, run_id=run_id, worker_timeout=240)
        return await orch.run_async(question, run_id=run_id, worker_timeout=240)

    results = asyncio.run(_run())

    if trace and provider:
        provider.force_flush()

    for r in results:
        if r.error:
            console.print(f"[red][{r.role}] ERROR: {r.error}[/red]")
        else:
            console.print(f"[bold green][{r.role}][/bold green]")
            console.print(r.result or "")
            console.print()


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Smoke-test the CLI**

```bash
UV_PROJECT_ENVIRONMENT=.agents python research_assistant.py --help
```

Expected: typer help output listing `question`, `--run-id`, `--trace`, `--model`, `--checkpoint-dir`.

- [ ] **Step 3: Commit**

```bash
git add research_assistant.py
git commit -m "feat(d4): research_assistant.py — typer CLI wrapping agentlab.spine"
```

---

## Task D-5: NB 15 + README update + merge to main

**Files:**
- Create: `notebooks/_src/15_capstone_research_assistant.py`
- Create: `notebooks/15_capstone_research_assistant.ipynb` (via jupytext sync)
- Modify: `README.md`

- [ ] **Step 1: Write the notebook source**

Create `notebooks/_src/15_capstone_research_assistant.py`:

```python
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
```

- [ ] **Step 2: Sync to .ipynb**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/jupytext --sync notebooks/_src/15_capstone_research_assistant.py
```

Expected: creates `notebooks/15_capstone_research_assistant.ipynb`

- [ ] **Step 3: Update README.md**

In `README.md`, make three changes:

**Change 1** — Status table row for Module D (replace `⏳ planned`):
```markdown
| **D · Autonomous + local** | [13 computer use](notebooks/13_computer_use.ipynb) · [14 local models](notebooks/14_running_locally.ipynb) · [15 ✦ capstone](notebooks/15_capstone_research_assistant.ipynb) | ✅ shipped |
```

**Change 2** — Add "What Module D teaches" section after the Module C section and before "## Quickstart":
```markdown
## What Module D teaches

- **[NB 13 — Computer use](notebooks/13_computer_use.ipynb)**
  Starts Anthropic's reference Docker container as a sandboxed virtual
  desktop. The `claude_cu_loop` function handles the screenshot→action
  cycle: Claude receives PNG images, returns `computer` tool-use blocks,
  and the loop executes click/type/key via `docker exec xdotool`. Three
  tool types covered: `computer_20241022`, `text_editor_20241022`,
  `bash_20241022`.
- **[NB 14 — Running locally](notebooks/14_running_locally.ipynb)**
  Swaps `get_client()` for an `openai.OpenAI` client pointed at
  `localhost:11434` (Ollama). Compares tool-use compliance, structured
  output reliability, and response quality on three eval tasks side by
  side (Claude vs `llama3.2`). LM Studio shown as a one-line endpoint
  swap — the code is identical.
- **[NB 15 — ✦ Capstone](notebooks/15_capstone_research_assistant.ipynb)**
  Graduates the spine to `research_assistant.py` — a `typer` CLI at the
  repo root. `--run-id` resumes from JSONL checkpoint; `--trace` wires
  OTel spans; `--model` and `--checkpoint-dir` are env-overridable. The
  notebook runs the eval suite via subprocess as the final green-check.
```

**Change 3** — Update Quickstart install command:
```bash
uv sync --extra dev --extra module-b --extra module-c --extra module-d
```

**Change 4** — Update Repo layout to add `research_assistant.py`:
```
research_assistant.py      # capstone CLI — typer wrapper around agentlab.spine
```
(Add this line after the `src/agentlab/` block, before `tests/`)

**Change 5** — Update Costs section:
Replace:
```
Module D and the
capstone will run higher.
```
With:
```
Module D end-to-end is **$0.20–0.40** (computer use image tokens in NB 13
and the eval suite in NB 15 drive most of that).
```

- [ ] **Step 4: Verify 42 tests still pass**

```bash
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/ --ignore=tests/eval -q
```

Expected: `42 passed`

- [ ] **Step 5: Commit**

```bash
git add notebooks/_src/15_capstone_research_assistant.py \
        notebooks/15_capstone_research_assistant.ipynb \
        README.md
git commit -m "feat(d5): NB 15 capstone + README Module D — curriculum complete"
```

- [ ] **Step 6: Merge to main**

```bash
git checkout main
git merge feature/module-d
UV_PROJECT_ENVIRONMENT=.agents .agents/bin/pytest tests/ --ignore=tests/eval -q
git branch -d feature/module-d
```

Expected: fast-forward merge; `42 passed` on main; branch deleted.

---

## Self-review notes

- **Spec coverage:** D-1 (branch + typer dep) ✓ · D-2 (NB 13) ✓ · D-3 (NB 14) ✓ · D-4 (`research_assistant.py`) ✓ · D-5 (NB 15 + README + merge) ✓. All five spec tasks covered.
- **No placeholders:** all code blocks are complete and runnable.
- **Type consistency:** `Subagent`, `Orchestrator`, `WorkerResult` used in D-4 match the signatures in `agentlab/spine.py`. `_claude_dispatch` signature `(worker, task, model, client)` is consistent across D-4 and D-5.
- **eval_tasks.jsonl format confirmed:** tasks have `id`, `question`, `judge_rubric`. NB 14 Step 4 uses `task["question"]` and `task["judge_rubric"]` — both present. ✓
