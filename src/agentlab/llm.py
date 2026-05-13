"""Anthropic client wrapper and a reusable agent-loop helper.

NB 01 deliberately re-implements the loop by hand; from NB 02 onwards
notebooks can use ``run_agent_loop`` for cleaner code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from anthropic import Anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def get_client(api_key: str | None = None) -> Anthropic:
    """Build an Anthropic client.

    Reads ``ANTHROPIC_API_KEY`` from the environment if ``api_key`` isn't
    given. Raises ``RuntimeError`` with a friendly message if no key is
    available — better than the SDK's late KeyError.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env or pass api_key= explicitly."
        )
    return Anthropic(api_key=key)


@dataclass
class AgentLoopResult:
    """What ``run_agent_loop`` returns when the model finally says ``end_turn``."""

    final_text: str
    turns: int
    transcript: list[dict[str, Any]] = field(default_factory=list)


# Protocol instead of `Anthropic` so tests can pass a lightweight FakeClient
# (a simple dataclass with a `.messages.create(...)` stub) without importing
# or instantiating the real SDK. This keeps unit tests fast and offline.
class _ClientLike(Protocol):
    """Structural type accepted by run_agent_loop in place of a real Anthropic client.

    Any object with a `.messages` attribute that has a `.create(...)` method
    satisfies this protocol — including the FakeClient used in tests.
    """
    messages: Any  # duck-typed for the FakeClient used in tests


def _block_to_dict(block: Any) -> dict[str, Any]:
    """Convert an Anthropic SDK content block (or test fake) to a plain dict.

    The SDK returns Pydantic models (e.g. TextBlock, ToolUseBlock); tests use
    plain dataclasses. Both have the same attribute names but neither is a
    plain dict. Normalizing here means the rest of the loop never needs to
    know which kind it received.
    """
    out: dict[str, Any] = {"type": block.type}
    for attr in ("text", "name", "input", "id", "tool_use_id"):
        val = getattr(block, attr, None)
        if val is not None:
            out[attr] = val
    return out


def run_agent_loop(
    *,
    client: _ClientLike,
    system: str,
    user_message: str,
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, Callable[..., Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    max_turns: int = 10,
) -> AgentLoopResult:
    """Drive a Claude messages.create loop until end_turn.

    Each turn: call messages.create with the current transcript. If the
    response has tool_use blocks, run the matching handlers and append a
    user message containing tool_result blocks. Stop when the model
    returns end_turn (or another non-tool stop reason).
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

    for turn in range(1, max_turns + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        assistant_blocks = [_block_to_dict(b) for b in response.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        # Check for != "tool_use" rather than == "end_turn" so that
        # "max_tokens" and "stop_sequence" also terminate the loop cleanly.
        # Checking for end_turn would silently swallow truncated responses.
        if response.stop_reason != "tool_use":
            final_text = "".join(
                b.get("text", "") for b in assistant_blocks if b["type"] == "text"
            )
            return AgentLoopResult(
                final_text=final_text, turns=turn, transcript=messages
            )

        tool_results: list[dict[str, Any]] = []
        for block in assistant_blocks:
            if block["type"] != "tool_use":
                continue
            handler = tool_handlers.get(block["name"])
            if handler is None:
                # Return an error result rather than crashing. Claude sees the
                # error message and can explain the problem or try a different
                # tool — much better than an unhandled exception in the loop.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Error: no handler for tool {block['name']!r}",
                    "is_error": True,
                })
                continue
            try:
                result = handler(**(block.get("input") or {}))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": str(result),
                })
            except Exception as exc:
                # Surface tool exceptions back to the model as is_error results.
                # This lets Claude recover (retry with different args, explain
                # the failure) rather than propagating a Python exception that
                # would abort the entire loop and lose the transcript.
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": f"Error: {exc}",
                    "is_error": True,
                })

        messages.append({"role": "user", "content": tool_results})

    # Reaching here means Claude called tools on every turn without ever
    # returning end_turn. This is a runaway loop — raise rather than
    # returning a partial result that looks like a normal completion.
    raise RuntimeError(f"Agent loop did not terminate within max_turns={max_turns}.")
