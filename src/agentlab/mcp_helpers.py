"""Helpers for consuming MCP servers from an Anthropic agent loop.

`mcp_tools_to_anthropic` converts MCP tool listings into the dict shape
the Anthropic SDK expects under `tools=[...]`.

`MCPToolRouter` wraps an MCP `ClientSession` and dispatches tool calls,
joining text-content blocks from the server response and surfacing
`isError=True` results as exceptions.
"""
from __future__ import annotations

from typing import Any


async def mcp_tools_to_anthropic(session: Any) -> list[dict]:
    """Discover tools on `session` and return Anthropic-shaped tool dicts.

    Each MCP `Tool` already carries a JSON Schema in `inputSchema`, which
    is what the Anthropic SDK wants under `input_schema`.

    This is a standalone function rather than a method on MCPToolRouter because
    it is stateless: call it once at setup to get the tool list to pass to
    `client.messages.create(tools=...)`. The router handles per-call routing;
    the discovery step is separate.
    """
    listed = await session.list_tools()
    out: list[dict] = []
    for tool in listed.tools:
        out.append(
            {
                "name": tool.name,
                "description": tool.description or "",
                # MCP's inputSchema is already standard JSON Schema, which is
                # exactly what Anthropic's API expects under input_schema.
                # No conversion needed — just rename the key.
                "input_schema": tool.inputSchema,
            }
        )
    return out


class MCPToolRouter:
    """Routes Anthropic tool calls back through an MCP `ClientSession`.

    Usage:
        router = MCPToolRouter(session)
        await router.refresh()                     # populate known tool names
        if router.knows(name):
            result = await router.call(name, args) # str

    The router only handles tool routing; resource and prompt access stay
    on `session` directly (those are app-driven, not agent-driven — see
    NB 10 reflect).
    """

    def __init__(self, session: Any):
        """Wrap an MCP ClientSession; call refresh() before routing any tool calls."""
        self._session = session
        # `refresh()` is separate from `__init__` because `__init__` is
        # synchronous but `session.list_tools()` is async. Calling an async
        # method from `__init__` would require `asyncio.run()`, which fails
        # if an event loop is already running (as it is inside a notebook or
        # async test). Instead, callers explicitly `await router.refresh()`
        # after construction.
        self._known: set[str] = set()

    async def refresh(self) -> None:
        """Fetch the current tool list from the MCP server and update the known-tools set.

        Call this once after construction and again if the server's tool list
        may have changed (e.g. after a server reload).
        """
        listed = await self._session.list_tools()
        self._known = {t.name for t in listed.tools}

    def knows(self, name: str) -> bool:
        """Return True if name is a tool this router can handle.

        Use this to gate calls to `router.call()` when the agent loop manages
        tools from multiple sources (local handlers, MCP tools, managed tools).
        """
        # The agent loop may have tools from multiple sources (local tools,
        # MCP tools, Anthropic-managed tools like web_search). `knows()` lets
        # the loop decide which router to forward a call to without the router
        # raising an error on unknown names.
        return name in self._known

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call on the MCP server and return the text result.

        Raises RuntimeError if the server signals an error (isError=True), so
        the agent loop can catch it and return an is_error tool_result to Claude.
        """
        result = await self._session.call_tool(name, arguments)
        # MCP tool results can contain multiple content blocks (text, image,
        # resource). We join only text blocks because image/resource content
        # cannot be serialized to the plain string the agent loop expects as
        # a tool result. For richer content, the caller would need to handle
        # the raw `result.content` blocks directly.
        text = "\n".join(
            getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text"
        )
        # Surface MCP errors as Python exceptions so the agent loop's existing
        # error-handling path (catch → return is_error tool_result) handles
        # them uniformly, without special-casing MCP in the loop itself.
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool '{name}' returned error: {text}")
        return text
