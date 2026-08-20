"""LLM provider interface and Ollama implementation.

The Telegram layer talks only to `LLMClient`, so the underlying provider
(Ollama today) can be swapped out without touching bot/handler code.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 120.0


class LLMError(Exception):
    """Raised when the LLM provider is unavailable or returns an invalid response."""


class LLMClient(ABC):
    """Abstract interface for a chat-style LLM backend."""

    @abstractmethod
    async def generate_reply(self, user_message: str) -> str:
        """Return the assistant's reply to a single, standalone user message."""
        raise NotImplementedError


class OllamaClient(LLMClient):
    """LLM client backed by a local Ollama server's chat API."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def generate_reply(self, user_message: str) -> str:
        url = f"{self._base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": user_message}],
            "stream": False,
        }

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
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            logger.error("Unexpected Ollama response shape: %s", data)
            raise LLMError("The LLM service returned an invalid response.") from exc

        if not isinstance(content, str) or not content.strip():
            logger.error("Ollama returned an empty reply: %s", data)
            raise LLMError("The LLM service returned an empty response.")

        return content.strip()
