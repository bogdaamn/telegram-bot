from __future__ import annotations

import pytest

from tools import Tool, ToolRegistry


async def _echo(arguments: dict) -> str:
    return f"echo:{arguments.get('value')}"


async def _boom(arguments: dict) -> str:
    raise RuntimeError("kaboom")


def _make_registry() -> ToolRegistry:
    echo_tool = Tool(
        name="echo",
        description="Echo back a value.",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
        handler=_echo,
    )
    boom_tool = Tool(
        name="boom",
        description="Always raises.",
        parameters={"type": "object", "properties": {}},
        handler=_boom,
    )
    return ToolRegistry([echo_tool, boom_tool])


def test_schemas_shape():
    registry = _make_registry()
    schemas = registry.schemas()
    names = {s["function"]["name"] for s in schemas}
    assert names == {"echo", "boom"}
    assert all(s["type"] == "function" for s in schemas)


@pytest.mark.asyncio
async def test_invoke_dispatches_to_handler():
    registry = _make_registry()
    call = {"function": {"name": "echo", "arguments": {"value": "hi"}}}

    result = await registry.invoke(call)

    assert result == "echo:hi"


@pytest.mark.asyncio
async def test_invoke_unknown_tool():
    registry = _make_registry()
    call = {"function": {"name": "nope", "arguments": {}}}

    result = await registry.invoke(call)

    assert result == "ERROR: no such tool 'nope'"


@pytest.mark.asyncio
async def test_invoke_catches_handler_exception():
    registry = _make_registry()
    call = {"function": {"name": "boom", "arguments": {}}}

    result = await registry.invoke(call)

    assert result.startswith("ERROR:")
