# Module D — Autonomous & Local Design Spec

**Date:** 2026-05-07
**Owner:** rkaunismaa
**Branch:** `feature/module-d`

## Goal

Close the 15-notebook curriculum. Three notebooks:

- **NB 13** — Claude's computer-use API controlling a sandboxed virtual desktop
- **NB 14** — Swapping the Anthropic client for a local model via OpenAI-compatible endpoint
- **NB 15 ✦** — Graduating the spine from notebook to deployable CLI artifact (`research_assistant.py`)

## Decisions (from brainstorming, 2026-05-07)

| # | Dimension | Decision |
|---|---|---|
| 1 | NB 13 sandbox | Anthropic reference Docker container (`ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest`); started/stopped from notebook via subprocess |
| 2 | NB 13 tool scope | `computer_20241022` only in the demo; `text_editor` and `bash` tools explained but not exercised |
| 3 | NB 13 demo task | "Open Chromium, navigate to python.org, and tell me the current Python version shown on the homepage" — deterministic success condition |
| 4 | NB 13 model | `claude-sonnet-4-6` (computer use requires a capable model; Haiku not supported) |
| 5 | NB 14 primary engine | Ollama (`llama3.2`); LM Studio shown as a one-cell endpoint swap |
| 6 | NB 14 comparison | Quantitative eval on t01, t05, t11 from `data/eval_tasks.jsonl`; side-by-side Claude vs local table |
| 7 | NB 15 CLI library | `typer` (user preference); `rich` for progress output |
| 8 | NB 15 artifact | `research_assistant.py` at repo root — imports `agentlab.spine`, `agentlab.llm`, `agentlab.mcp_helpers` |
| 9 | NB 15 spine wiring | `Orchestrator` + `Subagent` from `agentlab.spine`; `claude_dispatch` inline in script |
| 10 | New library modules | None — NB 13/14 are notebook-only; NB 15 artifact is the script itself |
| 11 | Branch | `feature/module-d` off `main` at current HEAD |
| 12 | `typer` dep | Added to `module-d` extra in `pyproject.toml` |
| 13 | Tests | No new library tests (42 still passes); NB 15 runs `pytest tests/eval/` via subprocess as the green-eval check |

## Notebooks

### NB 13 — Computer use

**Goal:** control a sandboxed virtual desktop via Claude's computer-use API.

**File:** `notebooks/_src/13_computer_use.py`

**Cost target:** $0.10–0.20 (image-heavy turns; Claude Sonnet 4.6 throughout)

**Structure:**

**Setup cell** — pull and start the Docker container:

```python
import subprocess
container_id = subprocess.check_output([
    "docker", "run", "-d", "--rm",
    "--name", "cu_demo",
    "ghcr.io/anthropics/anthropic-quickstarts:computer-use-demo-latest",
]).decode().strip()
```

A teardown cell at the bottom runs `docker stop cu_demo`.

**Step 1 — The tools.** Prose cell: explain `computer_20241022` (screenshot +
mouse/keyboard), `text_editor_20241022` (file edits), `bash_20241022` (shell).
Only `computer` is exercised in this notebook; the others are available via the
same `betas=["computer-use-2024-10-22"]` parameter.

**Step 2 — First screenshot.** Call `claude_cu_loop("Take a screenshot and
describe what you see")` with `max_turns=3`. Verifies the sandbox is live and
Claude can receive images. Display the screenshot inline using `IPython.display`.

**Step 3 — The agent loop.** Define `claude_cu_loop(task, max_turns=20)`:

- Uses `client.beta.messages.create` with `betas=["computer-use-2024-10-22"]`
- On each turn: if Claude returns a `computer` tool-use block with
  `action="screenshot"`, exec `docker exec cu_demo bash -c "import -window root
  /tmp/s.png && base64 /tmp/s.png"` and return the image as a `tool_result`
  content block
- For `left_click`, `right_click`, `double_click`, `type`, `key`: exec
  `docker exec cu_demo xdotool <action>` and return an empty `tool_result`
- Loop ends on `stop_reason == "end_turn"` or `max_turns` reached
- Unknown action types raise `ValueError` (abort-on-unknown guard)

**Step 4 — Demo task.** Run:

```python
result = claude_cu_loop(
    "Open Chromium, navigate to python.org, and tell me the current "
    "Python version shown on the homepage.",
    max_turns=20,
)
```

Print the final assistant text.

**Step 5 — Safety cell.** Prose + code:

- Why you always sandbox: never point `claude_cu_loop` at your host display
- Cost note: each screenshot is ~50–100 KB base64; 10-turn demo ≈ $0.05–0.15
- Abort guard already in the loop (`ValueError` on unknown action)
- Rate limiting: `asyncio.sleep(0.5)` between turns if needed

**Teardown cell:** `subprocess.run(["docker", "stop", "cu_demo"])`

**Reflect (3 bullets):**
- How the screenshot→action loop differs from tool use (images as conversation content, not just tool results)
- Safety and cost are first-class concerns, not afterthoughts
- What `text_editor` and `bash` add (file-level edits, shell — useful for longer autonomous tasks)

---

### NB 14 — Running locally with Ollama / LM Studio

**Goal:** swap the Anthropic client for a local model; understand what translates and what breaks.

**File:** `notebooks/_src/14_running_locally.py`

**Cost target:** $0.01–0.02 (comparison Claude calls only; local runs are free)

**Structure:**

**Setup cell** — verify Ollama is running, pull model if needed:

```python
import subprocess
subprocess.run(["ollama", "pull", "llama3.2"], check=True)
```

**Step 1 — Client swap.** Replace `get_client()`:

```python
from openai import OpenAI
local_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
```

Run the NB 01 bare question ("What is 2 + 2?") against `llama3.2`. Show it works.

**Step 2 — Tool use.** Run a one-tool task (web_search or the NB 01 `add` tool)
against the local model. Show a side-by-side result: Claude response vs llama3.2
response. Document common failure modes: missing `tool_use` blocks, wrong JSON
shape, hallucinated tool names.

**Step 3 — Structured outputs.** Run the `Answer` schema coercion from NB 03
against the local model. Does it comply? Show the raw response. Document what
breaks: missing fields, extra fields, wrong types.

**Step 4 — Quantitative eval.** Run eval tasks t01 (Module A), t05 (Module B),
t11 (Module C) from `data/eval_tasks.jsonl` against both Claude and llama3.2
inline in the notebook (the pytest harness is hardwired to the Anthropic client
and is not used here). Measure correctness and latency per call. Build a
comparison table:

| Task | Model | Correct | Latency |
|------|-------|---------|---------|
| t01  | claude-sonnet-4-6 | ✓ | ~2s |
| t01  | llama3.2 | ? | ~0.5s |
| ...  | ...   | ...     | ...     |

**Step 5 — LM Studio swap.** One cell — change `base_url` to
`http://localhost:1234/v1`. Show the same Step 1 call succeeds. The code is
identical; only the endpoint changes.

**Reflect (3 bullets):**
- What the OpenAI-compat layer buys: same client code works for any provider
- Where local models break hardest: tool-use protocol compliance, long-context recall, structured output reliability
- When to reach for local: privacy/cost at scale, offline use, experimentation without API budget

---

### NB 15 — ✦ Capstone: research assistant as a deployable script

**Goal:** graduate the spine from notebook to CLI artifact.

**File:** `notebooks/_src/15_capstone_research_assistant.py`

**Cost target:** $0.10–0.20 (2–3 spine runs + eval suite)

#### `research_assistant.py` (repo root)

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
from agentlab.spine import Orchestrator, Subagent

app = typer.Typer(help="Parallel research assistant powered by agentlab.spine.")
console = Console()

WORKERS = [
    Subagent(role="researcher",  system_prompt="..."),
    Subagent(role="summarizer",  system_prompt="..."),
    Subagent(role="ranker",      system_prompt="..."),
]


@app.command()
def main(
    question: str = typer.Argument(..., help="Research question to answer"),
    run_id: str   = typer.Option(None,  "--run-id",  help="Resume from checkpoint"),
    trace: bool   = typer.Option(False, "--trace",   help="Enable OTel console tracing"),
    model: str    = typer.Option(
        os.getenv("AGENTLAB_MODEL", DEFAULT_MODEL), "--model", help="Claude model"
    ),
    checkpoint_dir: Path = typer.Option(
        Path(os.getenv("AGENTLAB_CHECKPOINT_DIR", "data/checkpoints")),
        "--checkpoint-dir",
    ),
) -> None:
    ...

if __name__ == "__main__":
    app()
```

Env-driven defaults (`AGENTLAB_MODEL`, `AGENTLAB_CHECKPOINT_DIR`); CLI flags override.

**Structure of the notebook:**

**Step 1 — Show the script.** Display `research_assistant.py` source using
`rich.syntax.Syntax` so the learner sees the full artifact without leaving the
notebook.

**Step 2 — Run from terminal (via subprocess).** Capture stdout and print inline:

```python
result = subprocess.run(
    ["python", "research_assistant.py", QUESTION],
    capture_output=True, text=True,
)
print(result.stdout)
```

Demonstrates it works as a real script, not just importable code.

**Step 3 — Checkpoint resume.** Run twice with the same `--run-id`. Show the
second run completes near-instantly (workers loaded from checkpoint file).

**Step 4 — Tracing.** Run with `--trace`, capture OTel console output, show the
span tree: `agent.run → subagent.{role} → llm.complete`.

**Step 5 — Eval from CLI.** Run the eval suite as a subprocess:

```python
result = subprocess.run(
    [".agents/bin/pytest", "tests/eval/", "-m", "eval", "-v"],
    capture_output=True, text=True,
)
print(result.stdout)
assert result.returncode == 0, "eval suite failed"
```

This is the "eval suite green" success criterion.

**Reflect (4 bullets):**
- The spine ✦ progression is complete: bare loop (NB 01) → structured (NB 03) → evals + OTel (NB 08) → multi-agent durable (NB 12) → deployable artifact (NB 15)
- What changed between notebook and script: nothing architectural; just wiring env config and CLI args around what already worked
- What Module D deliberately did not build: autonomous multi-step computer use tasks, fine-tuned local models, production Docker deployment
- Where to go next: the agentlab library is the seed of a production codebase — swap in a real vector store, add a database-backed checkpoint store, wire to a Slack bot

---

## Library additions

None. All Module D code lives in notebooks or in `research_assistant.py`. The existing `agentlab.spine`, `agentlab.llm`, and `agentlab.mcp_helpers` are imported without modification.

## Dependencies

`pyproject.toml` change — add `typer>=0.12.0` to the `module-d` extra:

```toml
module-d = [
    "openai>=1.55.0",
    "typer>=0.12.0",
]
```

`rich` is already a core dependency.

## Tests

No new unit tests. The 42 existing tests continue to pass unchanged. NB 15 runs
`pytest tests/eval/ -m eval` as a subprocess assertion — that is the integration
test for the capstone.

## Task breakdown

| Task | Deliverable |
|------|-------------|
| D-1  | Branch `feature/module-d` + `typer` dep sync |
| D-2  | NB 13 — computer use (`notebooks/_src/13_computer_use.py` + `.ipynb`) |
| D-3  | NB 14 — local models (`notebooks/_src/14_running_locally.py` + `.ipynb`) |
| D-4  | `research_assistant.py` (typer CLI + spine wiring) |
| D-5  | NB 15 — ✦ capstone notebook + README update + merge to main |

D-4 before D-5 because the notebook displays and runs the script.

## Cost summary

| Notebook | Estimate |
|----------|----------|
| NB 13    | $0.10–0.20 (image-heavy computer use turns) |
| NB 14    | $0.01–0.02 (comparison Claude calls only) |
| NB 15    | $0.10–0.20 (spine runs + eval suite) |
| **Total**| **$0.21–0.42** |

## Success criteria

1. NB 13 completes the demo task (Chromium → python.org → Python version) without error.
2. NB 14 comparison table is populated; LM Studio swap cell runs without error.
3. `research_assistant.py` runs from the terminal, emits structured output, and supports `--run-id` resume and `--trace`.
4. NB 15 `pytest tests/eval/ -m eval` subprocess assertion passes.
5. All 42 existing unit tests still pass on `main` after merge.
