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
