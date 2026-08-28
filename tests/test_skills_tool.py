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
