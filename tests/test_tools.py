"""Tests for agentlab.tools — the tool registry."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, Field

from agentlab.tools import ToolRegistry, tool


def test_register_simple_tool():
    registry = ToolRegistry()

    @registry.tool(description="Add two integers.")
    def add(a: int, b: int) -> int:
        return a + b

    schemas = registry.schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "add"
    assert schema["description"] == "Add two integers."
    assert schema["input_schema"]["properties"]["a"]["type"] == "integer"
    assert schema["input_schema"]["properties"]["b"]["type"] == "integer"
    assert set(schema["input_schema"]["required"]) == {"a", "b"}


def test_register_tool_with_pydantic_model():
    class SearchInput(BaseModel):
        query: str = Field(..., description="The search query.")
        max_results: int = Field(5, ge=1, le=20)

    registry = ToolRegistry()

    @registry.tool(description="Search the web.", input_model=SearchInput)
    def search(query: str, max_results: int = 5) -> list[str]:
        return [f"{query}#{i}" for i in range(max_results)]

    schema = registry.schemas()[0]
    assert schema["name"] == "search"
    props = schema["input_schema"]["properties"]
    assert props["query"]["description"] == "The search query."
    assert props["max_results"]["minimum"] == 1


def test_handlers_dispatch_correctly():
    registry = ToolRegistry()

    @registry.tool(description="x")
    def add(a: int, b: int) -> int:
        return a + b

    handlers = registry.handlers()
    assert handlers["add"](a=2, b=3) == 5


def test_tool_decorator_module_level():
    """The bare ``@tool`` decorator builds a global registry-free entry."""
    @tool(description="Negate an integer.")
    def neg(x: int) -> int:
        return -x

    assert neg.tool_name == "neg"
    assert neg.tool_schema["description"] == "Negate an integer."
    assert neg(5) == -5  # original callable still works


def test_optional_parameters_are_not_required():
    registry = ToolRegistry()

    @registry.tool(description="Greet someone.")
    def greet(name: str, formal: bool = False) -> str:
        return f"{'Greetings' if formal else 'Hi'}, {name}"

    schema = registry.schemas()[0]
    required = set(schema["input_schema"].get("required", []))
    assert required == {"name"}


def test_tool_name_must_be_unique():
    registry = ToolRegistry()

    @registry.tool(description="x")
    def dup() -> int:
        return 1

    with pytest.raises(ValueError, match="already registered"):
        @registry.tool(description="y")
        def dup() -> int:  # noqa: F811
            return 2
