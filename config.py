"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_AGENT_MAX_STEPS = 8
MIN_AGENT_MAX_STEPS = 5
MAX_AGENT_MAX_STEPS = 10
DEFAULT_SESSIONS_DIR = "sessions"
DEFAULT_SKILLS_DIR = "skills"
DEFAULT_ALLOWED_HTTP_HOSTS = "wttr.in"


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    ollama_base_url: str
    ollama_model: str
    allowed_user_ids: frozenset[int]
    agent_max_steps: int
    sessions_dir: Path
    skills_dir: Path
    allowed_http_hosts: frozenset[str]


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


def _parse_allowed_user_ids(raw: str | None) -> frozenset[int]:
    if not raw or not raw.strip():
        raise ConfigError(
            "ALLOWED_TELEGRAM_USER_IDS is required and must not be empty. "
            "Set it to a comma-separated list of Telegram numeric user ids."
        )
    ids: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            ids.append(int(part))
        except ValueError as exc:
            raise ConfigError(
                f"ALLOWED_TELEGRAM_USER_IDS contains a non-integer value: '{part}'"
            ) from exc
    if not ids:
        raise ConfigError("ALLOWED_TELEGRAM_USER_IDS is required and must not be empty.")
    return frozenset(ids)


def _parse_agent_max_steps(raw: str | None) -> int:
    if raw is None or not raw.strip():
        return DEFAULT_AGENT_MAX_STEPS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"AGENT_MAX_STEPS must be an integer, got '{raw}'") from exc
    if not (MIN_AGENT_MAX_STEPS <= value <= MAX_AGENT_MAX_STEPS):
        raise ConfigError(
            f"AGENT_MAX_STEPS must be between {MIN_AGENT_MAX_STEPS} and "
            f"{MAX_AGENT_MAX_STEPS}, got {value}"
        )
    return value


def _parse_allowed_http_hosts(raw: str | None) -> frozenset[str]:
    raw = raw or DEFAULT_ALLOWED_HTTP_HOSTS
    hosts = frozenset(h.strip().lower() for h in raw.split(",") if h.strip())
    if not hosts:
        raise ConfigError("ALLOWED_HTTP_HOSTS must not be empty if set.")
    return hosts


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
        allowed_user_ids=_parse_allowed_user_ids(os.getenv("ALLOWED_TELEGRAM_USER_IDS")),
        agent_max_steps=_parse_agent_max_steps(os.getenv("AGENT_MAX_STEPS")),
        sessions_dir=Path(os.getenv("SESSIONS_DIR", DEFAULT_SESSIONS_DIR)),
        skills_dir=Path(os.getenv("SKILLS_DIR", DEFAULT_SKILLS_DIR)),
        allowed_http_hosts=_parse_allowed_http_hosts(os.getenv("ALLOWED_HTTP_HOSTS")),
    )
