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
    """
    listed = await session.list_tools()
    out: list[dict] = []
    for tool in listed.tools:
        out.append(
            {
                "name": tool.name,
                "description": tool.description or "",
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
        self._session = session
        self._known: set[str] = set()

    async def refresh(self) -> None:
        listed = await self._session.list_tools()
        self._known = {t.name for t in listed.tools}

    def knows(self, name: str) -> bool:
        return name in self._known

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        result = await self._session.call_tool(name, arguments)
        text = "\n".join(
            getattr(block, "text", "") for block in result.content if getattr(block, "type", "") == "text"
        )
        if getattr(result, "isError", False):
            raise RuntimeError(f"MCP tool '{name}' returned error: {text}")
        return text
