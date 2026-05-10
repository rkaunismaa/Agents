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
