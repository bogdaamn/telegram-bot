from __future__ import annotations

import pytest

from agent import STEP_BUDGET_EXHAUSTED_NOTICE, run


class FakeSession:
    def __init__(self) -> None:
        self._messages: list[dict] = []

    def append_message(self, message: dict) -> None:
        self._messages.append(message)

    def messages(self) -> list[dict]:
        return list(self._messages)


class FakeRegistry:
    def __init__(self, invoke_result: str = "ok") -> None:
        self.invoke_result = invoke_result
        self.invoked_calls: list[dict] = []

    def schemas(self) -> list[dict]:
        return []

    async def invoke(self, call: dict) -> str:
        self.invoked_calls.append(call)
        return self.invoke_result


class ScriptedLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def chat(self, messages: list[dict], tools=None) -> dict:
        self.call_count += 1
        return self._responses.pop(0)


def _no_tool_call_response(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": []}


def _tool_call_response(name: str) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": name, "arguments": {}}}],
    }


@pytest.mark.asyncio
async def test_returns_immediately_with_no_tool_calls():
    llm = ScriptedLLM([_no_tool_call_response("hello")])
    registry = FakeRegistry()
    session = FakeSession()
    steps: list[dict] = []

    async def on_step(call):
        steps.append(call)

    result = await run(llm, registry, session, "hi", on_step, max_steps=8)

    assert result == "hello"
    assert llm.call_count == 1
    assert steps == []


@pytest.mark.asyncio
async def test_calls_tool_then_stops():
    llm = ScriptedLLM([_tool_call_response("echo"), _no_tool_call_response("done")])
    registry = FakeRegistry(invoke_result="echo result")
    session = FakeSession()
    steps: list[dict] = []

    async def on_step(call):
        steps.append(call)

    result = await run(llm, registry, session, "hi", on_step, max_steps=8)

    assert result == "done"
    assert llm.call_count == 2
    assert len(steps) == 1
    assert len(registry.invoked_calls) == 1


@pytest.mark.asyncio
async def test_stops_at_max_steps_and_does_not_persist_notice():
    responses = [_tool_call_response("echo") for _ in range(5)]
    llm = ScriptedLLM(responses)
    registry = FakeRegistry(invoke_result="echo result")
    session = FakeSession()

    async def on_step(call):
        pass

    result = await run(llm, registry, session, "hi", on_step, max_steps=5)

    assert result == STEP_BUDGET_EXHAUSTED_NOTICE
    assert llm.call_count == 5
    assert all(
        m.get("content") != STEP_BUDGET_EXHAUSTED_NOTICE for m in session.messages()
    )


@pytest.mark.asyncio
async def test_tool_error_surfaces_as_tool_message_not_exception():
    llm = ScriptedLLM([_tool_call_response("boom"), _no_tool_call_response("recovered")])
    registry = FakeRegistry(invoke_result="ERROR: kaboom")
    session = FakeSession()

    async def on_step(call):
        pass

    result = await run(llm, registry, session, "hi", on_step, max_steps=8)

    assert result == "recovered"
    tool_messages = [m for m in session.messages() if m.get("role") == "tool"]
    assert tool_messages[0]["content"] == "ERROR: kaboom"
