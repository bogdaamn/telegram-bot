"""The agentic loop.

Kept intentionally small: this function is the entire harness (<= 80
lines by design). Telegram, config, and tool-specific logic all live
elsewhere (main.py, tools/) — this file must stay generic.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Protocol

STEP_BUDGET_EXHAUSTED_NOTICE = (
    "I've hit my step limit working on this without reaching a final answer. "
    "Try rephrasing or breaking the request into smaller steps."
)


class ChatClient(Protocol):
    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...


class Registry(Protocol):
    def schemas(self) -> list[dict]: ...
    async def invoke(self, call: dict) -> str: ...


class Session(Protocol):
    def append_message(self, message: dict) -> None: ...
    def messages(self) -> list[dict]: ...


OnStep = Callable[[dict], Awaitable[None]]


async def run(
    llm: ChatClient,
    registry: Registry,
    session: Session,
    user_text: str,
    on_step: OnStep,
    max_steps: int,
) -> str:
    session.append_message({"role": "user", "content": user_text})

    for _ in range(max_steps):
        assistant_msg = await llm.chat(session.messages(), tools=registry.schemas())
        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            session.append_message(assistant_msg)
            return assistant_msg.get("content") or ""

        session.append_message(assistant_msg)

        for call in tool_calls:
            await on_step(call)
            result_text = await registry.invoke(call)
            session.append_message(
                {
                    "role": "tool",
                    "tool_name": call.get("function", {}).get("name", ""),
                    "content": result_text,
                }
            )

    return STEP_BUDGET_EXHAUSTED_NOTICE
