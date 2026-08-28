# Minimal AI Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing stateless Telegram → Ollama forwarder into a minimal autonomous agent: an agentic loop, a restricted `exec` tool, on-demand Skill files, and per-chat session persistence with `/new`.

**Architecture:** A small `agent.run()` loop calls `OllamaClient.chat()` with tool schemas from a `ToolRegistry`, executing tool calls (allowlisted `exec`, `read_skill`, `list_sessions`, `read_session`) until the model returns plain content or a step budget is exhausted. `main.py` wires Telegram (aiogram) to this loop; `session.py` persists every turn to an append-only JSONL file per chat so `/new` can export a transcript and wipe the chat.

**Tech Stack:** Python 3.11+, aiogram 3.30 (already in `.venv`), httpx, python-dotenv, pytest + pytest-asyncio + respx for tests. Ollama's native tools API, default model `qwen2.5:7b` (already pulled locally, confirmed via `ollama show qwen2.5:7b` to support `tools`).

**Spec:** `docs/2026-08-28-ai-agent-design.md` — read it alongside this plan; section numbers below (e.g. "Spec §7.3") refer to it.

## Global Constraints

- Python 3.11+, `aiogram>=3.4,<4`, `httpx>=0.27,<1`, `python-dotenv>=1.0,<2` (existing, unchanged).
- New test-only deps: `pytest>=8.0,<9`, `pytest-asyncio>=0.24,<1`, `respx>=0.21,<1`.
- `subprocess`/`asyncio.create_subprocess_exec` calls must always use `shell=False` (i.e. never build a shell string) — no exceptions, anywhere.
- `agent.run()` (the loop function body) must stay small enough to read in one sitting — **hard cap: 80 lines**, excluding imports/type defs/comments (Spec §6).
- `AGENT_MAX_STEPS` default `8`, valid range `[5, 10]` inclusive; out-of-range is a `ConfigError`, not a silent clamp (Spec §11).
- Single allowlisted Telegram user id for this deployment: `386668609`.
- `sessions/` directory is gitignored in full (both `.jsonl` and `.txt` files) — session logs never reach git.
- TDD throughout: for every unit below, the test file is written and run to confirm failure *before* the implementation file is written.
- Every async test is marked `@pytest.mark.asyncio` (strict mode — see Task 1's `pytest.ini`).
- No live Telegram or live Ollama calls in the automated suite — mock/fake/respx only.

## Execution Notes (read before dispatching)

- **Task 1 is a hard prerequisite for everything else.** It creates `requirements.txt`, `pytest.ini`, `.gitignore`, `.env.example`, and extends `config.py`. Every later task's tests depend on the test deps it installs; running two tasks against a not-yet-updated `requirements.txt` concurrently would race. Run Task 1 alone, fully, first.
- **Tasks 2–7 have no file overlap with each other and no import dependency on each other's implementations** (each is tested with fakes/tmp_path, not real collaborators). Once Task 1 is done, dispatch Tasks 2, 3, 4, 5, 6, 7 to separate subagents in parallel.
- **Tasks 8 and 9 are the integration phase** and must run strictly after all of Tasks 2–7 are complete, in order (8 then 9) — `agent.py`'s real-world usage needs a real `ToolRegistry` (Task 7) and `OllamaClient` (Task 6), and `main.py` needs literally everything.

---

### Task 1: Repo setup and config extension

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Modify: `.gitignore`
- Create: `.env.example`
- Modify: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Config` dataclass with fields `telegram_bot_token: str`, `ollama_base_url: str`, `ollama_model: str`, `allowed_user_ids: frozenset[int]`, `agent_max_steps: int`, `sessions_dir: Path`, `skills_dir: Path`, `allowed_http_hosts: frozenset[str]`; `load_config() -> Config`; `ConfigError` (unchanged, existing exception class). All later tasks that touch `main.py` rely on exactly these field names and types.

- [ ] **Step 1: Update `requirements.txt`**

```
aiogram>=3.4,<4
httpx>=0.27,<1
python-dotenv>=1.0,<2
pytest>=8.0,<9
pytest-asyncio>=0.24,<1
respx>=0.21,<1
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = strict
```

- [ ] **Step 3: Install test deps and create `tests/` package**

Run: `.venv/bin/pip install -r requirements.txt`
Then create an empty `tests/__init__.py` (so `tools/` imports resolve the same whether run from repo root or via `pytest`).

- [ ] **Step 4: Add `sessions/` to `.gitignore`**

Append to the existing `.gitignore`:

```
# Session transcripts and active session state (never pushed)
sessions/
```

- [ ] **Step 5: Restore and extend `.env.example`**

```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
ALLOWED_TELEGRAM_USER_IDS=386668609
AGENT_MAX_STEPS=8
SESSIONS_DIR=sessions
SKILLS_DIR=skills
ALLOWED_HTTP_HOSTS=wttr.in
```

- [ ] **Step 6: Write the failing config tests — `tests/test_config.py`**

```python
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
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL — `Config` has no field `allowed_user_ids` (or similar `TypeError`/`AttributeError`), since `config.py` hasn't been extended yet.

- [ ] **Step 8: Implement `config.py`**

Replace the full contents of `config.py` with:

```python
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
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add requirements.txt pytest.ini .gitignore .env.example config.py tests/test_config.py tests/__init__.py
git commit -m "feat: extend config for agent (allowlist, max steps, sessions/skills dirs) and add test infra"
```

---

## Phase 1 — Parallel tasks (dispatch Tasks 2–7 concurrently after Task 1 is committed)

### Task 2: Command allowlist and restricted `exec` tool

**Files:**
- Create: `tools/__init__.py` (empty placeholder — Task 7 fills this in; create it now only so `tools.allowlist` and `tools.exec_tool` are importable as a package; do not add any code to it in this task)
- Create: `tools/allowlist.py`
- Create: `tools/exec_tool.py`
- Test: `tests/test_allowlist.py`
- Test: `tests/test_exec_tool.py`

**Interfaces:**
- Consumes: nothing beyond stdlib.
- Produces: `AllowlistEntry` dataclass (`validate: Callable[[list[str]], tuple[bool, list[str] | None, str | None]]`, `timeout_seconds: float`); `build_registry(allowed_http_hosts: frozenset[str]) -> dict[str, AllowlistEntry]`; `validate_command(command: str, registry: dict[str, AllowlistEntry]) -> tuple[bool, list[str] | None, str | None]`; `async def run_exec(command: str, registry: dict[str, AllowlistEntry]) -> str`. Task 9 (`main.py`) calls `build_registry(config.allowed_http_hosts)` and wraps `run_exec` as a `Tool` handler.

- [ ] **Step 1: Create the empty package file**

Create `tools/__init__.py` with just a docstring:

```python
"""Tool implementations for the agent: exec, skills, and session lookup."""
```

- [ ] **Step 2: Write the failing allowlist tests — `tests/test_allowlist.py`**

```python
from __future__ import annotations

import pytest

from tools.allowlist import build_registry, validate_command

REGISTRY = build_registry(frozenset({"wttr.in"}))


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "smctemp; whoami",
        "smctemp && curl http://evil.com",
        "curl http://wttr.in/Minsk",
        "curl https://evil.com",
        "curl -L https://wttr.in/Minsk",
        "curl https://wttr.in/Minsk -o /tmp/x",
        "curl file:///etc/passwd",
        "/bin/sh -c id",
        "top -l 1 -n 5",
        "smctemp `whoami`",
        "",
        "   ",
    ],
)
def test_denies_unsafe_or_disallowed_commands(command):
    ok, resolved, reason = validate_command(command, REGISTRY)
    assert ok is False
    assert resolved is None
    assert reason


@pytest.mark.parametrize(
    "command",
    [
        "smctemp -c",
        "smctemp -c -i25 -n180 -f",
        "pmset -g therm",
        "top -l 1 -n 0 -s 0",
        "vm_stat",
        "curl -s https://wttr.in/Minsk?0",
        "curl https://wttr.in/Minsk?0 --max-time 5",
    ],
)
def test_allows_permitted_commands(command):
    ok, resolved, reason = validate_command(command, REGISTRY)
    assert ok is True
    assert resolved is not None
    assert reason is None


def test_curl_resolved_argv_always_has_max_time():
    ok, resolved, _ = validate_command("curl -s https://wttr.in/Minsk?0", REGISTRY)
    assert ok is True
    assert "--max-time" in resolved


def test_unknown_host_denied():
    ok, _, reason = validate_command("curl https://example.com", REGISTRY)
    assert ok is False
    assert "example.com" in reason
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_allowlist.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.allowlist'`.

- [ ] **Step 4: Implement `tools/allowlist.py`**

```python
"""Pure validation logic for the `exec` tool's command allowlist.

No I/O here — this module only decides whether a command line is allowed
and, if so, what argv to actually run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable
from urllib.parse import urlsplit
import shlex

ValidatorResult = tuple[bool, list[str] | None, str | None]

SHELL_METACHARACTERS = set(";|&><`$()\n")
CURL_ALLOWED_FLAGS = {"-s", "-sS"}


@dataclass(frozen=True)
class AllowlistEntry:
    validate: Callable[[list[str]], ValidatorResult]
    timeout_seconds: float


def _fixed_argv_validator(*allowed_argvs: list[str]) -> Callable[[list[str]], ValidatorResult]:
    allowed = [list(a) for a in allowed_argvs]

    def validate(argv: list[str]) -> ValidatorResult:
        if list(argv) in allowed:
            return True, list(argv), None
        return False, None, f"'{' '.join(argv)}' does not match an allowed argument form"

    return validate


def _make_curl_validator(allowed_hosts: frozenset[str]) -> Callable[[list[str]], ValidatorResult]:
    def validate(argv: list[str]) -> ValidatorResult:
        flags: list[str] = []
        max_time: str | None = None
        urls: list[str] = []
        i = 1
        while i < len(argv):
            token = argv[i]
            if token == "--max-time":
                i += 1
                if i >= len(argv):
                    return False, None, "--max-time requires a value"
                value = argv[i]
                if not value.isdigit() or not (1 <= int(value) <= 15):
                    return False, None, "--max-time must be an integer between 1 and 15"
                max_time = value
            elif token in CURL_ALLOWED_FLAGS:
                flags.append(token)
            elif token.startswith("-"):
                return False, None, f"curl flag '{token}' is not permitted"
            else:
                urls.append(token)
            i += 1

        if len(urls) != 1:
            return False, None, "curl requires exactly one URL argument"

        parts = urlsplit(urls[0])
        if parts.scheme != "https":
            return False, None, "curl URL must use https://"
        if parts.hostname not in allowed_hosts:
            return False, None, f"host '{parts.hostname}' is not in the allowed host list"

        resolved = ["curl", *flags, urls[0], "--max-time", max_time or "10"]
        return True, resolved, None

    return validate


def build_registry(allowed_http_hosts: frozenset[str]) -> dict[str, AllowlistEntry]:
    return {
        "smctemp": AllowlistEntry(
            validate=_fixed_argv_validator(
                ["smctemp", "-c"],
                ["smctemp", "-c", "-i25", "-n180", "-f"],
            ),
            timeout_seconds=5.0,
        ),
        "pmset": AllowlistEntry(
            validate=_fixed_argv_validator(["pmset", "-g", "therm"]),
            timeout_seconds=5.0,
        ),
        "top": AllowlistEntry(
            validate=_fixed_argv_validator(["top", "-l", "1", "-n", "0", "-s", "0"]),
            timeout_seconds=5.0,
        ),
        "vm_stat": AllowlistEntry(
            validate=_fixed_argv_validator(["vm_stat"]),
            timeout_seconds=5.0,
        ),
        "curl": AllowlistEntry(
            validate=_make_curl_validator(allowed_http_hosts),
            timeout_seconds=10.0,
        ),
    }


def validate_command(command: str, registry: dict[str, AllowlistEntry]) -> ValidatorResult:
    if any(ch in SHELL_METACHARACTERS for ch in command):
        return False, None, "shell metacharacters are not permitted"

    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return False, None, f"could not parse command: {exc}"

    if not argv:
        return False, None, "empty command"

    name = PurePosixPath(argv[0]).name
    if argv[0] != name:
        return False, None, "command must not include a path"

    entry = registry.get(name)
    if entry is None:
        return False, None, f"'{name}' is not an allowed command"

    return entry.validate(argv)
```

- [ ] **Step 5: Run allowlist tests to verify they pass**

Run: `.venv/bin/pytest tests/test_allowlist.py -v`
Expected: all PASS.

- [ ] **Step 6: Write the failing exec tool tests — `tests/test_exec_tool.py`**

```python
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.allowlist import build_registry
from tools.exec_tool import run_exec

REGISTRY = build_registry(frozenset({"wttr.in"}))


@pytest.mark.asyncio
async def test_denied_command_never_calls_subprocess(monkeypatch):
    mock_create = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create)

    result = await run_exec("rm -rf /", REGISTRY)

    assert result.startswith("DENIED:")
    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_allowed_command_returns_stdout(monkeypatch):
    process = MagicMock()
    process.communicate = AsyncMock(return_value=(b"64.2\n", b""))
    mock_create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create)

    result = await run_exec("smctemp -c", REGISTRY)

    assert result == "64.2"
    mock_create.assert_awaited_once_with(
        "smctemp", "-c",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )


@pytest.mark.asyncio
async def test_timeout_returns_error_and_kills_process(monkeypatch):
    process = MagicMock()

    async def fake_communicate():
        await asyncio.sleep(0)
        return b"", b""

    process.communicate = fake_communicate
    process.kill = MagicMock()
    process.wait = AsyncMock()
    mock_create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create)

    async def fake_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    result = await run_exec("smctemp -c", REGISTRY)

    assert result == "ERROR: command timed out"
    process.kill.assert_called_once()


@pytest.mark.asyncio
async def test_output_is_truncated(monkeypatch):
    huge = b"x" * 5000
    process = MagicMock()
    process.communicate = AsyncMock(return_value=(huge, b""))
    mock_create = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create)

    result = await run_exec("vm_stat", REGISTRY)

    assert result.endswith("[truncated]")
    assert len(result) < 5000


@pytest.mark.asyncio
async def test_missing_binary_returns_error(monkeypatch):
    mock_create = AsyncMock(side_effect=FileNotFoundError())
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mock_create)

    result = await run_exec("smctemp -c", REGISTRY)

    assert result == "ERROR: 'smctemp' is not installed on this system"
```

- [ ] **Step 7: Run to verify failure**

Run: `.venv/bin/pytest tests/test_exec_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.exec_tool'`.

- [ ] **Step 8: Implement `tools/exec_tool.py`**

```python
"""The restricted `exec` tool: validates against the allowlist, then runs."""

from __future__ import annotations

import asyncio

from tools.allowlist import AllowlistEntry, validate_command

OUTPUT_CAP_CHARS = 4000
TRUNCATION_MARKER = "\n... [truncated]"


async def run_exec(command: str, registry: dict[str, AllowlistEntry]) -> str:
    ok, resolved_argv, reason = validate_command(command, registry)
    if not ok:
        return f"DENIED: {reason}"

    name = resolved_argv[0]
    entry = registry[name]

    try:
        process = await asyncio.create_subprocess_exec(
            *resolved_argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError:
        return f"ERROR: '{name}' is not installed on this system"

    try:
        stdout, _ = await asyncio.wait_for(
            process.communicate(), timeout=entry.timeout_seconds
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return "ERROR: command timed out"

    output = stdout.decode("utf-8", errors="replace")
    if len(output) > OUTPUT_CAP_CHARS:
        output = output[:OUTPUT_CAP_CHARS] + TRUNCATION_MARKER
    return output.strip()
```

- [ ] **Step 9: Run exec tool tests to verify they pass**

Run: `.venv/bin/pytest tests/test_exec_tool.py -v`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add tools/__init__.py tools/allowlist.py tools/exec_tool.py tests/test_allowlist.py tests/test_exec_tool.py
git commit -m "feat: add restricted exec tool with declarative command allowlist"
```

---

### Task 3: Session persistence (`session.py`)

**Files:**
- Create: `session.py`
- Test: `tests/test_session.py`

**Interfaces:**
- Consumes: nothing beyond stdlib.
- Produces: `class SessionStore` with `__init__(self, path: Path)`, `append_message(message: dict) -> None`, `append_telegram_message_id(message_id: int) -> None`, `messages() -> list[dict]`, `telegram_message_ids() -> list[int]`, `render_transcript() -> str`, `clear() -> None`. Task 8 (`agent.py`) calls `append_message`/`messages`. Task 9 (`main.py`) calls all six methods.

- [ ] **Step 1: Write the failing tests — `tests/test_session.py`**

```python
from __future__ import annotations

from session import SessionStore


def test_append_and_read_back_messages(tmp_path):
    path = tmp_path / "active-1.jsonl"
    store = SessionStore(path)

    store.append_message({"role": "user", "content": "hi"})
    store.append_telegram_message_id(101)
    store.append_message({"role": "assistant", "content": "hello"})
    store.append_telegram_message_id(102)

    assert store.messages() == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert store.telegram_message_ids() == [101, 102]


def test_reload_from_disk_preserves_order(tmp_path):
    path = tmp_path / "active-1.jsonl"
    store = SessionStore(path)
    store.append_message({"role": "user", "content": "hi"})
    store.append_telegram_message_id(101)
    store.append_message({"role": "assistant", "content": "hello"})

    reloaded = SessionStore(path)

    assert reloaded.messages() == store.messages()
    assert reloaded.telegram_message_ids() == [101]


def test_render_transcript_format(tmp_path):
    store = SessionStore(tmp_path / "active-1.jsonl")
    store.append_message({"role": "user", "content": "hi"})
    store.append_message({"role": "assistant", "content": "hello"})

    assert store.render_transcript() == "[user] hi\n[assistant] hello"


def test_clear_removes_file_and_state(tmp_path):
    path = tmp_path / "active-1.jsonl"
    store = SessionStore(path)
    store.append_message({"role": "user", "content": "hi"})

    store.clear()

    assert not path.exists()
    assert store.messages() == []
    assert store.telegram_message_ids() == []


def test_creates_parent_directory_if_missing(tmp_path):
    path = tmp_path / "nested" / "active-1.jsonl"
    store = SessionStore(path)

    store.append_message({"role": "user", "content": "hi"})

    assert path.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'session'`.

- [ ] **Step 3: Implement `session.py`**

```python
"""Per-chat session persistence: append-only JSONL, replayable on reload."""

from __future__ import annotations

import json
from pathlib import Path


class SessionStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._records: list[dict] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._records.append(json.loads(line))

    def append_message(self, message: dict) -> None:
        self._append({"kind": "msg", "message": message})

    def append_telegram_message_id(self, message_id: int) -> None:
        self._append({"kind": "tg", "message_id": message_id})

    def _append(self, record: dict) -> None:
        self._records.append(record)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()

    def messages(self) -> list[dict]:
        return [r["message"] for r in self._records if r["kind"] == "msg"]

    def telegram_message_ids(self) -> list[int]:
        return [r["message_id"] for r in self._records if r["kind"] == "tg"]

    def render_transcript(self) -> str:
        lines = []
        for msg in self.messages():
            role = msg.get("role", "?")
            content = msg.get("content", "")
            lines.append(f"[{role}] {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._records = []
        if self._path.exists():
            self._path.unlink()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_session.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add session.py tests/test_session.py
git commit -m "feat: add append-only per-chat session persistence"
```

---

### Task 4: Skills index and skill content files

**Files:**
- Create: `tools/skills_tool.py`
- Create: `skills/sysmon.md`
- Create: `skills/weather-api.md`
- Create: `skills/morning-routine.md`
- Test: `tests/test_skills_tool.py`

**Interfaces:**
- Consumes: nothing beyond stdlib.
- Produces: `class Skill` (`name: str`, `description: str`, `command: str | None`, `body: str`); `class SkillsIndex` with `__init__(self, skills_dir: Path)`, `index_text() -> str`, `slash_commands() -> dict[str, str]` (maps command name → skill name), `read_skill(name: str) -> str`. Task 9 (`main.py`) uses `index_text()` in the system prompt, `slash_commands()` to register `/sysmon` and `/morning`, and wraps `read_skill` as a `Tool` handler.

- [ ] **Step 1: Write the failing tests — `tests/test_skills_tool.py`**

```python
from __future__ import annotations

from pathlib import Path

from tools.skills_tool import SkillsIndex


def _write_skill(
    dir_: Path,
    filename: str,
    name: str,
    description: str,
    body: str,
    command: str | None = None,
) -> None:
    frontmatter_lines = ["---", f"name: {name}", f"description: {description}"]
    if command:
        frontmatter_lines.append(f"command: {command}")
    frontmatter_lines.append("---")
    text = "\n".join(frontmatter_lines) + "\n\n" + body + "\n"
    (dir_ / filename).write_text(text, encoding="utf-8")


def test_index_text_lists_name_and_description_only(tmp_path):
    _write_skill(
        tmp_path, "a.md", "sysmon", "Report CPU and memory.",
        "Full body with secret instructions.",
    )
    index = SkillsIndex(tmp_path)

    text = index.index_text()

    assert "sysmon" in text
    assert "Report CPU and memory." in text
    assert "secret instructions" not in text


def test_read_skill_returns_body(tmp_path):
    _write_skill(tmp_path, "a.md", "sysmon", "desc", "Do the thing.")
    index = SkillsIndex(tmp_path)

    assert index.read_skill("sysmon") == "Do the thing."


def test_read_skill_unknown_name(tmp_path):
    index = SkillsIndex(tmp_path)

    assert index.read_skill("nope") == "ERROR: no such skill 'nope'"


def test_slash_commands_only_for_skills_with_command(tmp_path):
    _write_skill(tmp_path, "a.md", "sysmon", "desc", "body", command="sysmon")
    _write_skill(tmp_path, "b.md", "weather-api", "desc2", "body2")
    index = SkillsIndex(tmp_path)

    assert index.slash_commands() == {"sysmon": "sysmon"}


def test_index_on_missing_directory_is_empty(tmp_path):
    index = SkillsIndex(tmp_path / "does-not-exist")

    assert index.index_text() == ""
    assert index.slash_commands() == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_skills_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.skills_tool'`.

- [ ] **Step 3: Implement `tools/skills_tool.py`**

```python
"""Skill file loading: a frontmatter index plus on-demand body reads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    command: str | None
    body: str


def _parse_skill_file(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path}: missing frontmatter opening '---'")

    frontmatter: dict[str, str] = {}
    i = 1
    while i < len(lines) and lines[i].strip() != "---":
        line = lines[i]
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()
        i += 1
    if i >= len(lines):
        raise ValueError(f"{path}: missing frontmatter closing '---'")

    body = "\n".join(lines[i + 1 :]).strip()

    if "name" not in frontmatter or "description" not in frontmatter:
        raise ValueError(f"{path}: frontmatter must include 'name' and 'description'")

    return Skill(
        name=frontmatter["name"],
        description=frontmatter["description"],
        command=frontmatter.get("command") or None,
        body=body,
    )


class SkillsIndex:
    def __init__(self, skills_dir: Path) -> None:
        self._skills: dict[str, Skill] = {}
        if skills_dir.exists():
            for path in sorted(skills_dir.glob("*.md")):
                skill = _parse_skill_file(path)
                self._skills[skill.name] = skill

    def index_text(self) -> str:
        lines = [f"- {s.name}: {s.description}" for s in self._skills.values()]
        return "\n".join(lines)

    def slash_commands(self) -> dict[str, str]:
        return {s.command: s.name for s in self._skills.values() if s.command}

    def read_skill(self, name: str) -> str:
        skill = self._skills.get(name)
        if skill is None:
            return f"ERROR: no such skill '{name}'"
        return skill.body
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_skills_tool.py -v`
Expected: all PASS.

- [ ] **Step 5: Create `skills/sysmon.md`**

```markdown
---
name: sysmon
description: Report current CPU temperature and memory in use.
command: sysmon
---

Use the `exec` tool to check system health:

1. Run `smctemp -c` to get the CPU temperature in Celsius. If that command
   is denied or fails, run `pmset -g therm` instead and report the thermal
   pressure level (Nominal/Fair/Serious/Critical) rather than a number.
2. Run `top -l 1 -n 0 -s 0` and read the `PhysMem: ... used` line to report
   memory currently in use.
3. Reply with one short message summarizing both: CPU temperature (or
   thermal pressure) and memory in use.

If any command is denied, do not retry a different variant of the same
denied command — report the limitation to the user instead of spending
more steps on it.
```

- [ ] **Step 6: Create `skills/weather-api.md`**

```markdown
---
name: weather-api
description: Rules for looking up weather via the wttr.in command-line API.
---

To check the weather for a city, use the `exec` tool with `curl` against
`https://wttr.in/<City>?0` for a compact one-line forecast, or
`https://wttr.in/<City>?1` for a slightly fuller one. Replace `<City>` with
the requested city name (URL-encode spaces as `%20` if needed).

Only `curl` requests to `https://wttr.in/...` are permitted — no other
hosts, no `http://`, no extra flags like `-L`, `-o`, or `-d`. If the
request is denied, do not retry with different curl flags; report that
weather lookup is unavailable for that request.
```

- [ ] **Step 7: Create `skills/morning-routine.md`**

```markdown
---
name: morning-routine
description: Morning summary combining weather and system health.
command: morning
---

Prepare a morning summary in three steps:

1. Read the `weather-api` skill (via `read_skill`) and follow it to fetch
   today's forecast for Minsk (`https://wttr.in/Minsk?0`).
2. Read the `sysmon` skill (via `read_skill`) and follow it to check
   current CPU temperature and memory usage.
3. Combine both results into one short summary message for the user:
   weather first, then system health.

Do not repeat a denied command from either sub-skill — if a step fails,
note it briefly in the summary and continue with the rest.
```

- [ ] **Step 8: Commit**

```bash
git add tools/skills_tool.py tests/test_skills_tool.py skills/
git commit -m "feat: add skill index/read_skill tool and sysmon, weather-api, morning-routine skills"
```

---

### Task 5: Saved-session lookup tools (`sessions_tool.py`)

**Files:**
- Create: `tools/sessions_tool.py`
- Test: `tests/test_sessions_tool.py`

**Interfaces:**
- Consumes: nothing beyond stdlib.
- Produces: `SESSION_FILENAME_PATTERN: re.Pattern`; `make_session_filename(chat_id: int, timestamp: str) -> str` (raises `ValueError` on a malformed timestamp); `class SessionsTool` with `__init__(self, sessions_dir: Path)`, `list_sessions() -> str`, `read_session(filename: str) -> str`. Task 9 (`main.py`) calls `make_session_filename` when writing the `/new` transcript, and wraps `list_sessions`/`read_session` as `Tool` handlers.

- [ ] **Step 1: Write the failing tests — `tests/test_sessions_tool.py`**

```python
from __future__ import annotations

import pytest

from tools.sessions_tool import SessionsTool, make_session_filename


def test_make_session_filename_valid():
    assert make_session_filename(123, "20260828T123456Z") == "20260828T123456Z-chat123.txt"


def test_make_session_filename_rejects_bad_timestamp():
    with pytest.raises(ValueError):
        make_session_filename(123, "not-a-timestamp")


def test_list_sessions_empty(tmp_path):
    tool = SessionsTool(tmp_path)
    assert tool.list_sessions() == "No saved sessions yet."


def test_list_and_read_session_round_trip(tmp_path):
    filename = make_session_filename(1, "20260828T120000Z")
    (tmp_path / filename).write_text("[user] hi\n[assistant] hello", encoding="utf-8")

    tool = SessionsTool(tmp_path)

    assert filename in tool.list_sessions()
    assert tool.read_session(filename) == "[user] hi\n[assistant] hello"


@pytest.mark.parametrize(
    "malicious_name",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "20260828T120000Z-chat1.txt/../../etc/passwd",
        "not-even-close.txt",
    ],
)
def test_read_session_rejects_traversal(tmp_path, malicious_name):
    tool = SessionsTool(tmp_path)
    assert tool.read_session(malicious_name) == f"ERROR: no such session '{malicious_name}'"


def test_read_session_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("secret", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    filename = make_session_filename(1, "20260828T120000Z")
    (sessions_dir / filename).symlink_to(outside)

    tool = SessionsTool(sessions_dir)

    assert tool.read_session(filename) == f"ERROR: no such session '{filename}'"


def test_read_session_truncates_long_content(tmp_path):
    filename = make_session_filename(1, "20260828T120000Z")
    (tmp_path / filename).write_text("x" * 9000, encoding="utf-8")
    tool = SessionsTool(tmp_path)

    result = tool.read_session(filename)

    assert result.endswith("[truncated]")
    assert len(result) < 9000
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_sessions_tool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.sessions_tool'`.

- [ ] **Step 3: Implement `tools/sessions_tool.py`**

```python
"""Read-only lookup tools over exported session transcripts.

The filename pattern deliberately allows no `/` or `..` — path traversal
is impossible by construction, not just by the containment check below.
"""

from __future__ import annotations

import re
from pathlib import Path

SESSION_FILENAME_PATTERN = re.compile(r"^\d{8}T\d{6}Z-chat\d+\.txt$")
READ_CAP_CHARS = 8000
TRUNCATION_MARKER = "\n... [truncated]"


def make_session_filename(chat_id: int, timestamp: str) -> str:
    """`timestamp` must be compact UTC ISO with no colons, e.g. '20260828T123456Z'."""
    filename = f"{timestamp}-chat{chat_id}.txt"
    if not SESSION_FILENAME_PATTERN.match(filename):
        raise ValueError(f"invalid timestamp for session filename: {timestamp}")
    return filename


class SessionsTool:
    def __init__(self, sessions_dir: Path) -> None:
        self._dir = sessions_dir

    def list_sessions(self) -> str:
        if not self._dir.exists():
            return "No saved sessions yet."
        files = sorted(self._dir.glob("*.txt"), reverse=True)
        if not files:
            return "No saved sessions yet."
        return "\n".join(f.name for f in files)

    def read_session(self, filename: str) -> str:
        if not SESSION_FILENAME_PATTERN.match(filename):
            return f"ERROR: no such session '{filename}'"

        base = self._dir.resolve()
        target = (self._dir / filename).resolve()

        if target.parent != base or not target.is_file():
            return f"ERROR: no such session '{filename}'"

        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > READ_CAP_CHARS:
            text = text[:READ_CAP_CHARS] + TRUNCATION_MARKER
        return text
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_sessions_tool.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/sessions_tool.py tests/test_sessions_tool.py
git commit -m "feat: add list_sessions/read_session tools with traversal-proof filenames"
```

---

### Task 6: `OllamaClient.chat` (tool-calling)

**Files:**
- Modify: `llm_client.py` (replace `generate_reply` with `chat`)
- Test: `tests/test_ollama_chat.py`

**Interfaces:**
- Consumes: nothing beyond `httpx`.
- Produces: `class LLMClient(ABC)` with abstract `async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict`; `class OllamaClient(LLMClient)` implementing it, returning `{"role": str, "content": str, "tool_calls": list[dict]}` where each entry is `{"function": {"name": str, "arguments": dict}}` (arguments always a `dict`, never a JSON string). `LLMError` unchanged. Task 8 (`agent.py`) and Task 9 (`main.py`) depend on exactly this return shape.

- [ ] **Step 1: Write the failing tests — `tests/test_ollama_chat.py`**

```python
from __future__ import annotations

import json

import httpx
import pytest
import respx

from llm_client import LLMError, OllamaClient

BASE_URL = "http://localhost:11434"


@pytest.mark.asyncio
@respx.mock
async def test_chat_sends_tools_in_payload():
    route = respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "hi", "tool_calls": []}},
        )
    )
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    tools = [{"type": "function", "function": {"name": "exec", "parameters": {}}}]
    result = await client.chat([{"role": "user", "content": "hi"}], tools=tools)

    assert result["content"] == "hi"
    assert result["tool_calls"] == []
    sent_body = json.loads(route.calls[0].request.content)
    assert sent_body["tools"] == tools


@pytest.mark.asyncio
@respx.mock
async def test_chat_parses_string_encoded_arguments():
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "exec",
                                "arguments": '{"command": "smctemp -c"}',
                            }
                        }
                    ],
                }
            },
        )
    )
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    result = await client.chat([{"role": "user", "content": "temp?"}])

    assert result["tool_calls"][0]["function"]["arguments"] == {"command": "smctemp -c"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_passes_through_dict_arguments():
    respx.post(f"{BASE_URL}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "exec", "arguments": {"command": "vm_stat"}}}
                    ],
                }
            },
        )
    )
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    result = await client.chat([{"role": "user", "content": "mem?"}])

    assert result["tool_calls"][0]["function"]["arguments"] == {"command": "vm_stat"}


@pytest.mark.asyncio
@respx.mock
async def test_chat_raises_llm_error_on_connection_failure():
    respx.post(f"{BASE_URL}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
@respx.mock
async def test_chat_raises_llm_error_on_bad_status():
    respx.post(f"{BASE_URL}/api/chat").mock(return_value=httpx.Response(500, text="boom"))
    client = OllamaClient(base_url=BASE_URL, model="qwen2.5:7b")

    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_ollama_chat.py -v`
Expected: FAIL — `OllamaClient` has no attribute `chat`.

- [ ] **Step 3: Implement `llm_client.py`**

Replace the full contents of `llm_client.py` with:

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_ollama_chat.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_client.py tests/test_ollama_chat.py
git commit -m "feat: replace generate_reply with tool-calling OllamaClient.chat"
```

---

### Task 7: Generic `ToolRegistry`

**Files:**
- Modify: `tools/__init__.py` (created empty in Task 2 — fill it in now)
- Test: `tests/test_tool_registry.py`

**Interfaces:**
- Consumes: nothing beyond stdlib (tested entirely with fake handlers — no dependency on Tasks 2/4/5's real implementations).
- Produces: `class Tool` (`name: str`, `description: str`, `parameters: dict`, `handler: Callable[[dict], Awaitable[str]]`) with a `schema() -> dict` method returning Ollama's function-tool JSON shape; `class ToolRegistry` with `__init__(self, tools: list[Tool])`, `schemas() -> list[dict]`, `async def invoke(self, call: dict) -> str` (never raises — catches handler exceptions and returns an `ERROR:`-prefixed string). Task 8 (`agent.py`) calls `schemas()`/`invoke()`. Task 9 (`main.py`) constructs the real `Tool` instances wrapping Task 2/4/5's functions and passes them to `ToolRegistry`.

- [ ] **Step 1: Write the failing tests — `tests/test_tool_registry.py`**

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_tool_registry.py -v`
Expected: FAIL — `ImportError: cannot import name 'Tool' from 'tools'`.

- [ ] **Step 3: Implement `tools/__init__.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest tests/test_tool_registry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/__init__.py tests/test_tool_registry.py
git commit -m "feat: add generic ToolRegistry with exception-safe invoke"
```

---

## Phase 2 — Integration (sequential, only after ALL of Tasks 2–7 are committed)

### Task 8: The agentic loop (`agent.py`)

**Files:**
- Create: `agent.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: any object satisfying `chat(messages, tools=None) -> dict` (Task 6's `OllamaClient` in production; a scripted fake in tests), any object satisfying `schemas() -> list[dict]` / `async invoke(call) -> str` (Task 7's `ToolRegistry` in production; a fake in tests), any object satisfying `append_message(message) -> None` / `messages() -> list[dict]` (Task 3's `SessionStore` in production; a fake in tests).
- Produces: `STEP_BUDGET_EXHAUSTED_NOTICE: str`; `async def run(llm, registry, session, user_text: str, on_step: Callable[[dict], Awaitable[None]], max_steps: int) -> str`. Task 9 (`main.py`) calls this directly with real `OllamaClient`, `ToolRegistry`, and `SessionStore` instances.

**Constraint:** the `run()` function body must be ≤ 80 lines (Global Constraints).

- [ ] **Step 1: Write the failing tests — `tests/test_agent_loop.py`**

```python
from __future__ import annotations

import pytest

from agent import STEP_BUDGET_EXHAUSTED_NOTICE, run


class FakeSession:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    def append_message(self, message: dict) -> None:
        self._messages.append(message)

    def messages(self) -> list[dict]:
        return list(self._messages)


class FakeRegistry:
    def __init__(self, invoke_result: str = "ok") -> None:
        self.invoke_result = invoke_result
        self.invoked_calls: list[dict] = []

    def schemas(self) -> list[dict]:
        return []

    async def invoke(self, call: dict) -> str:
        self.invoked_calls.append(call)
        return self.invoke_result


class ScriptedLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def chat(self, messages: list[dict], tools=None) -> dict:
        self.call_count += 1
        return self._responses.pop(0)


def _no_tool_call_response(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": []}


def _tool_call_response(name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": {}}}],
    }


@pytest.mark.asyncio
async def test_returns_immediately_with_no_tool_calls():
    llm = ScriptedLLM([_no_tool_call_response("hello")])
    registry = FakeRegistry()
    session = FakeSession()
    steps: list[dict] = []

    async def on_step(call):
        steps.append(call)

    result = await run(llm, registry, session, "hi", on_step, max_steps=8)

    assert result == "hello"
    assert llm.call_count == 1
    assert steps == []


@pytest.mark.asyncio
async def test_calls_tool_then_stops():
    llm = ScriptedLLM([_tool_call_response("echo"), _no_tool_call_response("done")])
    registry = FakeRegistry(invoke_result="echo result")
    session = FakeSession()
    steps: list[dict] = []

    async def on_step(call):
        steps.append(call)

    result = await run(llm, registry, session, "hi", on_step, max_steps=8)

    assert result == "done"
    assert llm.call_count == 2
    assert len(steps) == 1
    assert len(registry.invoked_calls) == 1


@pytest.mark.asyncio
async def test_stops_at_max_steps_and_does_not_persist_notice():
    responses = [_tool_call_response("echo") for _ in range(5)]
    llm = ScriptedLLM(responses)
    registry = FakeRegistry(invoke_result="echo result")
    session = FakeSession()

    async def on_step(call):
        pass

    result = await run(llm, registry, session, "hi", on_step, max_steps=5)

    assert result == STEP_BUDGET_EXHAUSTED_NOTICE
    assert llm.call_count == 5
    assert all(
        m.get("content") != STEP_BUDGET_EXHAUSTED_NOTICE for m in session.messages()
    )


@pytest.mark.asyncio
async def test_tool_error_surfaces_as_tool_message_not_exception():
    llm = ScriptedLLM([_tool_call_response("boom"), _no_tool_call_response("recovered")])
    registry = FakeRegistry(invoke_result="ERROR: kaboom")
    session = FakeSession()

    async def on_step(call):
        pass

    result = await run(llm, registry, session, "hi", on_step, max_steps=8)

    assert result == "recovered"
    tool_messages = [m for m in session.messages() if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "ERROR: kaboom"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_agent_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent'`.

- [ ] **Step 3: Implement `agent.py`**

```python
"""The agentic loop.

Kept intentionally small: this function is the entire harness (<= 80
lines by design). Telegram, config, and tool-specific logic all live
elsewhere (main.py, tools/) — this file must stay generic.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

STEP_BUDGET_EXHAUSTED_NOTICE = (
    "I've hit my step limit working on this without reaching a final answer. "
    "Try rephrasing or breaking the request into smaller steps."
)


class ChatClient(Protocol):
    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...


class Registry(Protocol):
    def schemas(self) -> list[dict]: ...
    async def invoke(self, call: dict) -> str: ...


class Session(Protocol):
    def append_message(self, message: dict) -> None: ...
    def messages(self) -> list[dict]: ...


OnStep = Callable[[dict], Awaitable[None]]


async def run(
    llm: ChatClient,
    registry: Registry,
    session: Session,
    user_text: str,
    on_step: OnStep,
    max_steps: int,
) -> str:
    session.append_message({"role": "user", "content": user_text})

    for _ in range(max_steps):
        assistant_msg = await llm.chat(session.messages(), tools=registry.schemas())
        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            session.append_message(assistant_msg)
            return assistant_msg.get("content") or ""

        session.append_message(assistant_msg)

        for call in tool_calls:
            await on_step(call)
            result_text = await registry.invoke(call)
            session.append_message(
                {
                    "role": "tool",
                    "tool_name": call.get("function", {}).get("name", ""),
                    "content": result_text,
                }
            )

    return STEP_BUDGET_EXHAUSTED_NOTICE
```

- [ ] **Step 4: Verify the line-count constraint**

Run: `awk '/^async def run/,/^    return STEP_BUDGET_EXHAUSTED_NOTICE$/' agent.py | wc -l`
Expected: well under 80 (the function body above is ~30 lines).

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/pytest tests/test_agent_loop.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add agent.py tests/test_agent_loop.py
git commit -m "feat: add the minimal agentic loop with step-budget protection"
```

---

### Task 9: Telegram integration (`main.py`) and docs

**Files:**
- Modify: `main.py` (full rewrite)
- Modify: `README.md`

**Interfaces:**
- Consumes: `Config`/`load_config`/`ConfigError` (Task 1), `AllowlistEntry`/`build_registry` (Task 2), `run_exec` (Task 2), `SessionStore` (Task 3), `Skill`/`SkillsIndex` (Task 4), `SESSION_FILENAME_PATTERN`/`make_session_filename`/`SessionsTool` (Task 5), `LLMClient`/`LLMError`/`OllamaClient` (Task 6), `Tool`/`ToolRegistry` (Task 7), `agent.run`/`STEP_BUDGET_EXHAUSTED_NOTICE` (Task 8).
- Produces: a runnable bot; no new importable interfaces (this is the leaf of the dependency graph).

**No automated tests for this task** — the spec (§12) is explicit that Telegram/LLM integration is verified manually, not with mocks that would just re-assert the same contracts already covered by Tasks 1–8. This task's own verification is Step 5 (import smoke test) plus the manual QA checklist added to the README.

- [ ] **Step 1: Implement `main.py`**

Replace the full contents of `main.py` with:

```python
"""Application startup, middleware, and Telegram handlers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

import agent
from config import Config, ConfigError, load_config
from llm_client import LLMClient, LLMError, OllamaClient
from session import SessionStore
from tools import Tool, ToolRegistry
from tools.allowlist import build_registry
from tools.exec_tool import run_exec
from tools.sessions_tool import SessionsTool, make_session_filename
from tools.skills_tool import SkillsIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

USER_FRIENDLY_ERROR = (
    "Sorry, I couldn't get a response from the language model right now. "
    "Please try again in a moment."
)
UNSUPPORTED_MESSAGE_NOTICE = "Sorry, I can only understand text messages right now."
UNAUTHORIZED_NOTICE = "Sorry, this bot is not available to you."
BUSY_NOTICE = "Still working on your previous message — one at a time, please."
DELETE_BATCH_SIZE = 100

dispatcher = Dispatcher()

_sessions: dict[int, SessionStore] = {}
_locks: dict[int, asyncio.Lock] = {}


def _session_path(config: Config, chat_id: int) -> Path:
    return config.sessions_dir / f"active-{chat_id}.jsonl"


def _get_session(config: Config, chat_id: int) -> SessionStore:
    if chat_id not in _sessions:
        _sessions[chat_id] = SessionStore(_session_path(config, chat_id))
    return _sessions[chat_id]


def _get_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _locks:
        _locks[chat_id] = asyncio.Lock()
    return _locks[chat_id]


@dispatcher.message.outer_middleware()
async def auth_middleware(handler, message: Message, data: dict):
    config: Config = data["config"]
    if message.from_user is None or message.from_user.id not in config.allowed_user_ids:
        logger.warning(
            "Rejected message from unauthorized user_id=%s chat_id=%s",
            message.from_user.id if message.from_user else None,
            message.chat.id,
        )
        await message.answer(UNAUTHORIZED_NOTICE)
        return None
    return await handler(message, data)


def _build_tool_registry(
    config: Config, skills_index: SkillsIndex, sessions_tool: SessionsTool
) -> ToolRegistry:
    exec_registry = build_registry(config.allowed_http_hosts)

    async def _exec_handler(arguments: dict) -> str:
        return await run_exec(arguments.get("command", ""), exec_registry)

    async def _read_skill_handler(arguments: dict) -> str:
        return skills_index.read_skill(arguments.get("name", ""))

    async def _list_sessions_handler(arguments: dict) -> str:
        return sessions_tool.list_sessions()

    async def _read_session_handler(arguments: dict) -> str:
        return sessions_tool.read_session(arguments.get("filename", ""))

    tools = [
        Tool(
            name="exec",
            description=(
                "Run one allowlisted system command line. Permitted commands: "
                "'smctemp -c' (CPU temperature), 'pmset -g therm' (thermal "
                "pressure fallback), 'top -l 1 -n 0 -s 0' and 'vm_stat' "
                "(memory usage), and 'curl' against https://wttr.in/... only. "
                "Any other command is denied. Do not retry a denied command "
                "with different flags."
            ),
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=_exec_handler,
        ),
        Tool(
            name="read_skill",
            description="Read the full instructions for one named skill.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_read_skill_handler,
        ),
        Tool(
            name="list_sessions",
            description="List saved past conversation transcript filenames.",
            parameters={"type": "object", "properties": {}},
            handler=_list_sessions_handler,
        ),
        Tool(
            name="read_session",
            description="Read the contents of one saved conversation transcript by filename.",
            parameters={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
            handler=_read_session_handler,
        ),
    ]
    return ToolRegistry(tools)


def _system_prompt(skills_index: SkillsIndex) -> str:
    return (
        "You are a helpful assistant with tool access. Only use the `exec` "
        "tool for the commands described in its schema — never attempt "
        "anything else with it. The following skills are available; call "
        "read_skill(name) to load one when relevant:\n"
        f"{skills_index.index_text()}"
    )


async def _run_turn(
    message: Message,
    config: Config,
    llm: LLMClient,
    registry: ToolRegistry,
    skills_index: SkillsIndex,
    user_text: str,
) -> None:
    chat_id = message.chat.id
    lock = _get_lock(chat_id)
    if lock.locked():
        await message.answer(BUSY_NOTICE)
        return

    async with lock:
        session = _get_session(config, chat_id)
        if not session.messages():
            session.append_message({"role": "system", "content": _system_prompt(skills_index)})
        session.append_telegram_message_id(message.message_id)

        async def on_step(call: dict) -> None:
            name = call.get("function", {}).get("name", "?")
            args = call.get("function", {}).get("arguments", {})
            sent = await message.answer(f"⚙️ {name}: {args}")
            session.append_telegram_message_id(sent.message_id)

        try:
            reply = await agent.run(
                llm, registry, session, user_text, on_step, config.agent_max_steps
            )
        except LLMError as exc:
            logger.error("LLM request failed for chat %s: %s", chat_id, exc)
            await message.answer(USER_FRIENDLY_ERROR)
            return

        sent = await message.answer(reply)
        session.append_telegram_message_id(sent.message_id)


def _register_handlers(
    config: Config, llm: LLMClient, registry: ToolRegistry, skills_index: SkillsIndex
) -> None:
    @dispatcher.message(Command("new"))
    async def handle_new(message: Message) -> None:
        chat_id = message.chat.id
        session = _get_session(config, chat_id)

        transcript = session.render_transcript()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = make_session_filename(chat_id, timestamp)
        config.sessions_dir.mkdir(parents=True, exist_ok=True)
        (config.sessions_dir / filename).write_text(transcript, encoding="utf-8")

        ids = session.telegram_message_ids() + [message.message_id]
        deleted = 0
        failed = 0
        for i in range(0, len(ids), DELETE_BATCH_SIZE):
            batch = ids[i : i + DELETE_BATCH_SIZE]
            try:
                await message.bot.delete_messages(chat_id=chat_id, message_ids=batch)
                deleted += len(batch)
            except TelegramBadRequest:
                for mid in batch:
                    try:
                        await message.bot.delete_message(chat_id=chat_id, message_id=mid)
                        deleted += 1
                    except TelegramBadRequest:
                        failed += 1

        session.clear()
        _sessions.pop(chat_id, None)

        note = f"Started a new conversation. Deleted {deleted} message(s)."
        if failed:
            note += f" {failed} message(s) were too old to delete (Telegram's 48-hour limit)."
        await message.answer(note)

    def _make_skill_handler(skill_name: str):
        async def handler(message: Message) -> None:
            await _run_turn(
                message, config, llm, registry, skills_index,
                f"Follow the '{skill_name}' skill.",
            )
        return handler

    for command, skill_name in skills_index.slash_commands().items():
        dispatcher.message.register(_make_skill_handler(skill_name), Command(command))

    @dispatcher.message(F.text)
    async def handle_text_message(message: Message) -> None:
        await _run_turn(message, config, llm, registry, skills_index, message.text)

    @dispatcher.message()
    async def handle_non_text_message(message: Message) -> None:
        await message.answer(UNSUPPORTED_MESSAGE_NOTICE)


async def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    llm: LLMClient = OllamaClient(base_url=config.ollama_base_url, model=config.ollama_model)
    skills_index = SkillsIndex(config.skills_dir)
    sessions_tool = SessionsTool(config.sessions_dir)
    registry = _build_tool_registry(config, skills_index, sessions_tool)

    _register_handlers(config, llm, registry, skills_index)

    bot = Bot(token=config.telegram_bot_token, default=DefaultBotProperties(parse_mode=None))

    logger.info(
        "Starting bot with Ollama model '%s' at %s", config.ollama_model, config.ollama_base_url
    )

    try:
        await dispatcher.start_polling(bot, config=config)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Import smoke test**

Run: `.venv/bin/python -c "import main"`
Expected: no output, exit code 0 (confirms all imports resolve and there are no syntax errors; `load_config()` is not called at import time, so a missing `.env` does not matter here).

- [ ] **Step 3: Run the full automated test suite**

Run: `.venv/bin/pytest -v`
Expected: every test from Tasks 1–8 passes (no regressions from the `main.py` rewrite, since nothing in the suite imports `main`).

- [ ] **Step 4: Update `README.md`**

Add a new `## Prerequisites` line for `smctemp`, update the architecture list, replace the old single-turn description, and add a manual QA checklist. Apply these edits to the existing `README.md`:

- In the "Prerequisites" section, add:

  ```markdown
  - [`smctemp`](https://github.com/narugit/smctemp) for CPU temperature (no
    sudo required to read temperatures on Apple Silicon):

    ```bash
    brew tap narugit/tap
    brew install narugit/tap/smctemp
    ```
  ```

- In the "Architecture" section, replace the existing bullet list with:

  ```markdown
  - `main.py` — application startup, Telegram (aiogram) handlers, auth
    middleware, and `/new` session reset
  - `config.py` — environment/configuration loading
  - `llm_client.py` — `LLMClient` interface and the Ollama implementation,
    using Ollama's native tool-calling API
  - `agent.py` — the minimal agentic loop (tool calls until a final answer
    or the step budget runs out)
  - `session.py` — append-only per-chat session persistence
  - `tools/` — the restricted `exec` allowlist, skill loading, and saved
    session lookup tools
  - `skills/` — Markdown skill files the agent can read on demand
  ```

- Replace the paragraph under "A minimal Telegram bot that forwards..." with:

  ```markdown
  A Telegram bot backed by a local LLM (via [Ollama](https://ollama.com))
  that can call a small set of restricted tools to check system health,
  look up the weather, and manage its own conversation history. Each chat
  is a continuous session persisted to disk; `/new` exports the
  conversation to a text file and starts fresh.
  ```

- Add a new section near the end, before "Troubleshooting":

  ```markdown
  ## Manual QA checklist

  Automated tests cover the allowlist, exec tool, session persistence,
  skill loading, agent loop, and Ollama client in isolation (`pytest`).
  These steps verify the pieces work together for real:

  - [ ] Send a plain text message; confirm a reply comes back.
  - [ ] Send `/sysmon`; confirm it reports a CPU temperature (or thermal
        pressure) and memory usage, with a `⚙️ exec: ...` trace
        message before the summary.
  - [ ] Send `/morning`; confirm it reports weather for Minsk and then a
        system health summary.
  - [ ] Send `/new`; confirm the chat's messages disappear and a
        `sessions/<timestamp>-chat<id>.txt` file appears.
  - [ ] Ask "what did we talk about before I ran /new?" and confirm the
        agent can call `list_sessions`/`read_session` to answer from the
        exported file.
  - [ ] If you have a second Telegram account, message the bot from it and
        confirm it's refused (not in `ALLOWED_TELEGRAM_USER_IDS`).
  ```

- [ ] **Step 5: Commit**

```bash
git add main.py README.md
git commit -m "feat: wire agent loop, tools, and /new into the Telegram bot"
```
