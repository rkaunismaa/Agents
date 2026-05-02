"""Tests for agentlab.llm — client wrapper and agent loop helper."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agentlab.llm import (
    DEFAULT_MODEL,
    HAIKU_MODEL,
    AgentLoopResult,
    get_client,
    run_agent_loop,
)


def test_default_model_is_sonnet():
    assert "sonnet" in DEFAULT_MODEL.lower()


def test_haiku_model_is_haiku():
    assert "haiku" in HAIKU_MODEL.lower()


def test_get_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        get_client()


def test_get_client_uses_explicit_key():
    client = get_client(api_key="sk-test-explicit")
    # The Anthropic SDK exposes the key on the client instance.
    assert client.api_key == "sk-test-explicit"


# --- Fake client used to drive run_agent_loop without hitting the API ---

@dataclass
class FakeBlock:
    type: str
    text: str | None = None
    name: str | None = None
    input: dict | None = None
    id: str | None = None
    tool_use_id: str | None = None


@dataclass
class FakeResponse:
    content: list[FakeBlock]
    stop_reason: str = "end_turn"


@dataclass
class FakeMessages:
    scripted: list[FakeResponse]
    calls: list[dict] = field(default_factory=list)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.scripted.pop(0)


@dataclass
class FakeClient:
    messages: FakeMessages


def test_run_agent_loop_terminates_on_end_turn():
    client = FakeClient(messages=FakeMessages(scripted=[
        FakeResponse(content=[FakeBlock(type="text", text="Hello!")]),
    ]))

    result = run_agent_loop(
        client=client,
        system="You are a helpful agent.",
        user_message="Hi",
        tools=[],
        tool_handlers={},
    )

    assert isinstance(result, AgentLoopResult)
    assert result.final_text == "Hello!"
    assert result.turns == 1
    assert len(client.messages.calls) == 1


def test_run_agent_loop_executes_tool_then_continues():
    client = FakeClient(messages=FakeMessages(scripted=[
        FakeResponse(
            content=[FakeBlock(
                type="tool_use", id="t1", name="add", input={"a": 2, "b": 3}
            )],
            stop_reason="tool_use",
        ),
        FakeResponse(content=[FakeBlock(type="text", text="The answer is 5.")]),
    ]))

    def add(a: int, b: int) -> int:
        return a + b

    result = run_agent_loop(
        client=client,
        system="You are a calculator.",
        user_message="What is 2+3?",
        tools=[{"name": "add", "description": "add two ints",
                "input_schema": {"type": "object",
                                 "properties": {"a": {"type": "integer"},
                                                "b": {"type": "integer"}},
                                 "required": ["a", "b"]}}],
        tool_handlers={"add": add},
    )

    assert result.final_text == "The answer is 5."
    assert result.turns == 2
    # Second call should include the tool_result.
    second_call_messages = client.messages.calls[1]["messages"]
    assert any(
        block.get("type") == "tool_result"
        for msg in second_call_messages
        for block in (msg["content"] if isinstance(msg["content"], list) else [])
    )


def test_run_agent_loop_respects_max_turns():
    # Always returns tool_use, never end_turn — should bail out.
    def loop_response():
        return FakeResponse(
            content=[FakeBlock(type="tool_use", id="t1", name="noop", input={})],
            stop_reason="tool_use",
        )

    client = FakeClient(messages=FakeMessages(scripted=[loop_response() for _ in range(20)]))

    with pytest.raises(RuntimeError, match="max_turns"):
        run_agent_loop(
            client=client,
            system="x",
            user_message="x",
            tools=[{"name": "noop", "description": "x",
                    "input_schema": {"type": "object", "properties": {}}}],
            tool_handlers={"noop": lambda: None},
            max_turns=5,
        )
