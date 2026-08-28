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
