"""Tool registry and schema generation for agentlab notebooks.

The registry pattern: a notebook defines a Python function and decorates
it with ``@registry.tool(...)``. The registry inspects the signature
(or an explicit Pydantic model) to produce the JSON schema Anthropic's
API expects, and exposes both ``.schemas()`` and ``.handlers()`` for
use with ``agentlab.llm.run_agent_loop``.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, create_model

_PY_TO_JSON = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}


def _python_type_to_jsonschema(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a fragment of JSON schema.

    Handles the primitive types we use in Module A. Lists, dicts, and
    Optional are supported; richer cases use ``input_model=`` with a
    Pydantic model directly.
    """
    origin = typing.get_origin(annotation)
    if origin is list:
        (item_type,) = typing.get_args(annotation)
        return {"type": "array", "items": _python_type_to_jsonschema(item_type)}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Union:
        non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _python_type_to_jsonschema(non_none[0])
    if annotation in _PY_TO_JSON:
        return {"type": _PY_TO_JSON[annotation]}
    return {"type": "string"}  # safe default


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    # Use get_type_hints to resolve string annotations (PEP 563 / from __future__ import annotations)
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(param_name, param.annotation)
        properties[param_name] = _python_type_to_jsonschema(annotation)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _schema_from_pydantic_model(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    # Anthropic accepts standard JSON schema; trim Pydantic's $defs unless used.
    schema.pop("title", None)
    return schema


class ToolRegistry:
    """Holds a set of tools and produces schemas + handlers for the API."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def tool(
        self,
        *,
        description: str,
        name: str | None = None,
        input_model: type[BaseModel] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: register ``fn`` as a tool.

        If ``input_model`` is given, its JSON schema is used; otherwise the
        function signature drives schema generation.
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            tool_name = name or fn.__name__
            if tool_name in self._entries:
                raise ValueError(f"Tool {tool_name!r} is already registered.")

            input_schema = (
                _schema_from_pydantic_model(input_model)
                if input_model is not None
                else _schema_from_signature(fn)
            )
            self._entries[tool_name] = {
                "schema": {
                    "name": tool_name,
                    "description": description,
                    "input_schema": input_schema,
                },
                "handler": fn,
            }
            return fn

        return decorator

    def schemas(self) -> list[dict[str, Any]]:
        return [entry["schema"] for entry in self._entries.values()]

    def handlers(self) -> dict[str, Callable[..., Any]]:
        return {name: entry["handler"] for name, entry in self._entries.items()}


def tool(
    *,
    description: str,
    name: str | None = None,
    input_model: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Module-level decorator: attach schema metadata to a function without a registry.

    Useful for one-off tools defined inline in a notebook.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        input_schema = (
            _schema_from_pydantic_model(input_model)
            if input_model is not None
            else _schema_from_signature(fn)
        )
        fn.tool_name = tool_name  # type: ignore[attr-defined]
        fn.tool_schema = {  # type: ignore[attr-defined]
            "name": tool_name,
            "description": description,
            "input_schema": input_schema,
        }
        return fn

    return decorator
