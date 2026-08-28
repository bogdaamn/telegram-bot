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
