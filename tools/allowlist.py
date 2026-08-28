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
