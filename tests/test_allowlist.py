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
