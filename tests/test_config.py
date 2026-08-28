from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, load_config

ENV_KEYS = [
    "TELEGRAM_BOT_TOKEN", "OLLAMA_BASE_URL", "OLLAMA_MODEL",
    "ALLOWED_TELEGRAM_USER_IDS", "AGENT_MAX_STEPS",
    "SESSIONS_DIR", "SKILLS_DIR", "ALLOWED_HTTP_HOSTS",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "386668609")


def test_load_config_applies_defaults():
    cfg = load_config()
    assert cfg.ollama_model == "qwen2.5:7b"
    assert cfg.agent_max_steps == 8
    assert cfg.sessions_dir == Path("sessions")
    assert cfg.skills_dir == Path("skills")
    assert cfg.allowed_http_hosts == frozenset({"wttr.in"})


def test_missing_allowed_user_ids_raises(monkeypatch):
    monkeypatch.delenv("ALLOWED_TELEGRAM_USER_IDS", raising=False)
    with pytest.raises(ConfigError, match="ALLOWED_TELEGRAM_USER_IDS"):
        load_config()


def test_empty_allowed_user_ids_raises(monkeypatch):
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "   ")
    with pytest.raises(ConfigError, match="ALLOWED_TELEGRAM_USER_IDS"):
        load_config()


def test_allowed_user_ids_parses_multiple(monkeypatch):
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "1, 2,3")
    cfg = load_config()
    assert cfg.allowed_user_ids == frozenset({1, 2, 3})


def test_allowed_user_ids_rejects_non_integer(monkeypatch):
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "abc")
    with pytest.raises(ConfigError, match="non-integer"):
        load_config()


def test_agent_max_steps_out_of_range_raises(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "20")
    with pytest.raises(ConfigError, match="AGENT_MAX_STEPS"):
        load_config()


def test_agent_max_steps_accepts_boundaries(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "5")
    assert load_config().agent_max_steps == 5
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    assert load_config().agent_max_steps == 10


def test_allowed_http_hosts_custom(monkeypatch):
    monkeypatch.setenv("ALLOWED_HTTP_HOSTS", "wttr.in, example.com")
    cfg = load_config()
    assert cfg.allowed_http_hosts == frozenset({"wttr.in", "example.com"})
