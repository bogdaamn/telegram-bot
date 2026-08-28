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
