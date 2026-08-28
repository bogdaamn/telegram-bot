"""Tool implementations for the agent: exec, skills, and session lookup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ToolFunction = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolFunction

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def invoke(self, call: dict[str, Any]) -> str:
        function = call.get("function", {})
        name = function.get("name", "")
        arguments = function.get("arguments") or {}

        tool = self._tools.get(name)
        if tool is None:
            return f"ERROR: no such tool '{name}'"

        try:
            return await tool.handler(arguments)
        except Exception as exc:  # a single bad tool call must not crash the loop
            logger.exception("Tool '%s' raised an exception", name)
            return f"ERROR: {exc}"
