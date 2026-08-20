"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the process environment.
load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    ollama_base_url: str
    ollama_model: str


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def load_config() -> Config:
    """Read and validate configuration from environment variables."""
    telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not telegram_bot_token:
        raise ConfigError(
            "TELEGRAM_BOT_TOKEN is required. Set it in your .env file or environment."
        )

    ollama_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/")
    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    return Config(
        telegram_bot_token=telegram_bot_token,
        ollama_base_url=ollama_base_url,
        ollama_model=ollama_model,
    )
