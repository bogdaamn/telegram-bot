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
