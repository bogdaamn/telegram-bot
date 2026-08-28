"""LLM provider interface and Ollama implementation.

The Telegram layer talks only to `LLMClient`, so the underlying provider
(Ollama today) can be swapped out without touching bot/handler code.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 120.0


class LLMError(Exception):
    """Raised when the LLM provider is unavailable or returns an invalid response."""


class LLMClient(ABC):
    """Abstract interface for a tool-calling chat-style LLM backend."""

    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return the raw assistant message dict for the next turn."""
        raise NotImplementedError


def _normalize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    normalized = []
    for call in tool_calls:
        function = dict(call.get("function", {}))
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            function["arguments"] = json.loads(arguments) if arguments else {}
        elif arguments is None:
            function["arguments"] = {}
        normalized.append({**call, "function": function})
    return normalized


class OllamaClient(LLMClient):
    """LLM client backed by a local Ollama server's chat API."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        url = f"{self._base_url}/api/chat"
        payload: dict = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            logger.error("Could not connect to Ollama at %s: %s", self._base_url, exc)
            raise LLMError("Could not connect to the LLM service.") from exc
        except httpx.TimeoutException as exc:
            logger.error("Timed out waiting for Ollama response: %s", exc)
            raise LLMError("The LLM service took too long to respond.") from exc
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Ollama returned an error status %s: %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise LLMError("The LLM service returned an error.") from exc
        except httpx.HTTPError as exc:
            logger.error("HTTP error while calling Ollama: %s", exc)
            raise LLMError("Could not reach the LLM service.") from exc
        except ValueError as exc:
            logger.error("Ollama returned a response that was not valid JSON: %s", exc)
            raise LLMError("The LLM service returned an invalid response.") from exc

        try:
            message = data["message"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama response shape: %s", data)
            raise LLMError("The LLM service returned an invalid response.") from exc

        tool_calls = message.get("tool_calls") or []
        return {
            "role": message.get("role", "assistant"),
            "content": message.get("content") or "",
            "tool_calls": _normalize_tool_calls(tool_calls),
        }
