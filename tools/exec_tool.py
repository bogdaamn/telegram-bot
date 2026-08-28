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
