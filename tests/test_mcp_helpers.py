"""Tests for agentlab.mcp_helpers: schema conversion + tool router.

We do not spawn a real MCP server here — these are unit tests against
fakes that mimic the SDK's `ClientSession` surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agentlab.mcp_helpers import MCPToolRouter, mcp_tools_to_anthropic


# ── Fakes that mimic mcp.types ─────────────────────────────────────


@dataclass
class _FakeMCPTool:
    name: str
    description: str
    inputSchema: dict


@dataclass
class _FakeListToolsResult:
    tools: list[_FakeMCPTool]


@dataclass
class _FakeContent:
    type: str
    text: str


@dataclass
class _FakeCallResult:
    content: list[_FakeContent]
    isError: bool = False


class _FakeSession:
    """Minimal stand-in for mcp.ClientSession."""

    def __init__(self, tools: list[_FakeMCPTool], call_results: dict[str, _FakeCallResult]):
        self._tools = tools
        self._call_results = call_results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> _FakeListToolsResult:
        return _FakeListToolsResult(tools=list(self._tools))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> _FakeCallResult:
        self.calls.append((name, arguments))
        return self._call_results[name]


# ── mcp_tools_to_anthropic ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_mcp_tools_to_anthropic_basic():
    """Each MCP Tool maps to an Anthropic tool dict with the same schema."""
    session = _FakeSession(
        tools=[
            _FakeMCPTool(
                name="add_note",
                description="Add a note.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["key", "content"],
                },
            ),
            _FakeMCPTool(
                name="list_notes",
                description="List all note keys.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ],
        call_results={},
    )

    result = await mcp_tools_to_anthropic(session)

    assert isinstance(result, list) and len(result) == 2
    assert result[0] == {
        "name": "add_note",
        "description": "Add a note.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["key", "content"],
        },
    }
    assert result[1]["name"] == "list_notes"
    assert result[1]["input_schema"] == {"type": "object", "properties": {}}


@pytest.mark.asyncio
async def test_mcp_tools_to_anthropic_handles_missing_description():
    """A None or missing description becomes the empty string (Anthropic requires str)."""
    session = _FakeSession(
        tools=[_FakeMCPTool(name="foo", description=None, inputSchema={"type": "object"})],  # type: ignore[arg-type]
        call_results={},
    )

    result = await mcp_tools_to_anthropic(session)

    assert result[0]["description"] == ""


# ── MCPToolRouter ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_router_call_returns_text_content():
    """Successful call returns the joined text content blocks."""
    session = _FakeSession(
        tools=[],
        call_results={
            "add_note": _FakeCallResult(content=[_FakeContent(type="text", text="ok: stored 'hello'")]),
        },
    )
    router = MCPToolRouter(session)

    result = await router.call("add_note", {"key": "h", "content": "hello"})

    assert result == "ok: stored 'hello'"
    assert session.calls == [("add_note", {"key": "h", "content": "hello"})]


@pytest.mark.asyncio
async def test_router_call_concatenates_multiple_text_blocks():
    """If the server returns multiple text blocks, the router joins them with newlines."""
    session = _FakeSession(
        tools=[],
        call_results={
            "list_notes": _FakeCallResult(
                content=[
                    _FakeContent(type="text", text="key1"),
                    _FakeContent(type="text", text="key2"),
                ]
            ),
        },
    )
    router = MCPToolRouter(session)

    result = await router.call("list_notes", {})

    assert result == "key1\nkey2"


@pytest.mark.asyncio
async def test_router_call_error_result_raises():
    """isError=True surfaces as a RuntimeError carrying the error text."""
    session = _FakeSession(
        tools=[],
        call_results={
            "get_note": _FakeCallResult(
                content=[_FakeContent(type="text", text="key not found: missing")],
                isError=True,
            ),
        },
    )
    router = MCPToolRouter(session)

    with pytest.raises(RuntimeError, match="key not found"):
        await router.call("get_note", {"key": "missing"})


@pytest.mark.asyncio
async def test_router_knows_tool():
    """`knows(name)` is True iff the session listed that tool."""
    session = _FakeSession(
        tools=[_FakeMCPTool(name="add_note", description="d", inputSchema={"type": "object"})],
        call_results={},
    )
    router = MCPToolRouter(session)
    await router.refresh()

    assert router.knows("add_note") is True
    assert router.knows("delete_note") is False
