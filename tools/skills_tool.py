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
