"""Application startup, middleware, and Telegram handlers."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message

import agent
from config import Config, ConfigError, load_config
from llm_client import LLMClient, LLMError, OllamaClient
from session import SessionStore
from tools import Tool, ToolRegistry
from tools.allowlist import build_registry
from tools.exec_tool import run_exec
from tools.sessions_tool import SessionsTool, make_session_filename
from tools.skills_tool import SkillsIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

USER_FRIENDLY_ERROR = (
    "Sorry, I couldn't get a response from the language model right now. "
    "Please try again in a moment."
)
UNSUPPORTED_MESSAGE_NOTICE = "Sorry, I can only understand text messages right now."
UNAUTHORIZED_NOTICE = "Sorry, this bot is not available to you."
BUSY_NOTICE = "Still working on your previous message — one at a time, please."
EMPTY_REPLY_NOTICE = (
    "(The model returned an empty response for that — try rephrasing.)"
)
DELETE_BATCH_SIZE = 100

dispatcher = Dispatcher()

_sessions: dict[int, SessionStore] = {}
_locks: dict[int, asyncio.Lock] = {}


def _session_path(config: Config, chat_id: int) -> Path:
    return config.sessions_dir / f"active-{chat_id}.jsonl"


def _get_session(config: Config, chat_id: int) -> SessionStore:
    if chat_id not in _sessions:
        _sessions[chat_id] = SessionStore(_session_path(config, chat_id))
    return _sessions[chat_id]


def _get_lock(chat_id: int) -> asyncio.Lock:
    if chat_id not in _locks:
        _locks[chat_id] = asyncio.Lock()
    return _locks[chat_id]


@dispatcher.message.outer_middleware()
async def auth_middleware(handler, message: Message, data: dict):
    config: Config = data["config"]
    if message.from_user is None or message.from_user.id not in config.allowed_user_ids:
        logger.warning(
            "Rejected message from unauthorized user_id=%s chat_id=%s",
            message.from_user.id if message.from_user else None,
            message.chat.id,
        )
        await message.answer(UNAUTHORIZED_NOTICE)
        return None
    return await handler(message, data)


def _build_tool_registry(
    config: Config, skills_index: SkillsIndex, sessions_tool: SessionsTool
) -> ToolRegistry:
    exec_registry = build_registry(config.allowed_http_hosts)

    async def _exec_handler(arguments: dict) -> str:
        return await run_exec(arguments.get("command", ""), exec_registry)

    async def _read_skill_handler(arguments: dict) -> str:
        return skills_index.read_skill(arguments.get("name", ""))

    async def _list_sessions_handler(arguments: dict) -> str:
        return sessions_tool.list_sessions()

    async def _read_session_handler(arguments: dict) -> str:
        return sessions_tool.read_session(arguments.get("filename", ""))

    tools = [
        Tool(
            name="exec",
            description=(
                "Run one allowlisted system command line. Permitted commands: "
                "'smctemp -c' (CPU temperature), 'pmset -g therm' (thermal "
                "pressure fallback), 'top -l 1 -n 0 -s 0' and 'vm_stat' "
                "(memory usage), and 'curl' against https://wttr.in/... only. "
                "Any other command is denied. Do not retry a denied command "
                "with different flags."
            ),
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            handler=_exec_handler,
        ),
        Tool(
            name="read_skill",
            description="Read the full instructions for one named skill.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            handler=_read_skill_handler,
        ),
        Tool(
            name="list_sessions",
            description="List saved past conversation transcript filenames.",
            parameters={"type": "object", "properties": {}},
            handler=_list_sessions_handler,
        ),
        Tool(
            name="read_session",
            description="Read the contents of one saved conversation transcript by filename.",
            parameters={
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
            handler=_read_session_handler,
        ),
    ]
    return ToolRegistry(tools)


def _system_prompt(skills_index: SkillsIndex) -> str:
    return (
        "You are a helpful assistant with tool access. Only use the `exec` "
        "tool for the commands described in its schema — never attempt "
        "anything else with it. The following skills are available:\n"
        f"{skills_index.index_text()}\n"
        "Whenever the user's request matches one of these skills, you MUST "
        "call read_skill(name) first and carry out its instructions with the "
        "`exec` tool before writing your reply. Do not describe what a skill "
        "does instead of running it. Never send an empty reply: if a tool "
        "call fails or is denied, say so in one short sentence."
    )


async def _run_turn(
    message: Message,
    config: Config,
    llm: LLMClient,
    registry: ToolRegistry,
    skills_index: SkillsIndex,
    user_text: str,
) -> None:
    chat_id = message.chat.id
    lock = _get_lock(chat_id)
    if lock.locked():
        await message.answer(BUSY_NOTICE)
        return

    async with lock:
        session = _get_session(config, chat_id)
        if not session.messages():
            session.append_message({"role": "system", "content": _system_prompt(skills_index)})
        session.append_telegram_message_id(message.message_id)

        async def on_step(call: dict) -> None:
            name = call.get("function", {}).get("name", "?")
            args = call.get("function", {}).get("arguments", {})
            sent = await message.answer(f"⚙️ {name}: {args}")
            session.append_telegram_message_id(sent.message_id)

        try:
            reply = await agent.run(
                llm, registry, session, user_text, on_step, config.agent_max_steps
            )
        except LLMError as exc:
            logger.error("LLM request failed for chat %s: %s", chat_id, exc)
            await message.answer(USER_FRIENDLY_ERROR)
            return

        if not reply.strip():
            logger.warning(
                "Agent returned an empty final reply for chat %s (model produced no "
                "tool calls and no content)", chat_id,
            )
            reply = EMPTY_REPLY_NOTICE

        sent = await message.answer(reply)
        session.append_telegram_message_id(sent.message_id)


def _register_handlers(
    config: Config, llm: LLMClient, registry: ToolRegistry, skills_index: SkillsIndex
) -> None:
    @dispatcher.message(Command("new"))
    async def handle_new(message: Message) -> None:
        chat_id = message.chat.id
        session = _get_session(config, chat_id)

        transcript = session.render_transcript()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = make_session_filename(chat_id, timestamp)
        config.sessions_dir.mkdir(parents=True, exist_ok=True)
        (config.sessions_dir / filename).write_text(transcript, encoding="utf-8")

        ids = session.telegram_message_ids() + [message.message_id]
        deleted = 0
        failed = 0
        for i in range(0, len(ids), DELETE_BATCH_SIZE):
            batch = ids[i : i + DELETE_BATCH_SIZE]
            try:
                await message.bot.delete_messages(chat_id=chat_id, message_ids=batch)
                deleted += len(batch)
            except TelegramBadRequest:
                for mid in batch:
                    try:
                        await message.bot.delete_message(chat_id=chat_id, message_id=mid)
                        deleted += 1
                    except TelegramBadRequest:
                        failed += 1

        session.clear()
        _sessions.pop(chat_id, None)

        note = f"Started a new conversation. Deleted {deleted} message(s)."
        if failed:
            note += f" {failed} message(s) were too old to delete (Telegram's 48-hour limit)."
        await message.answer(note)

    def _make_skill_handler(skill_name: str):
        async def handler(message: Message) -> None:
            await _run_turn(
                message, config, llm, registry, skills_index,
                f"Call read_skill(name=\"{skill_name}\") now, then carry out "
                f"its instructions using the exec tool as needed.",
            )
        return handler

    for command, skill_name in skills_index.slash_commands().items():
        dispatcher.message.register(_make_skill_handler(skill_name), Command(command))

    @dispatcher.message(F.text)
    async def handle_text_message(message: Message) -> None:
        await _run_turn(message, config, llm, registry, skills_index, message.text)

    @dispatcher.message()
    async def handle_non_text_message(message: Message) -> None:
        await message.answer(UNSUPPORTED_MESSAGE_NOTICE)


async def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    llm: LLMClient = OllamaClient(base_url=config.ollama_base_url, model=config.ollama_model)
    skills_index = SkillsIndex(config.skills_dir)
    sessions_tool = SessionsTool(config.sessions_dir)
    registry = _build_tool_registry(config, skills_index, sessions_tool)

    _register_handlers(config, llm, registry, skills_index)

    bot = Bot(token=config.telegram_bot_token, default=DefaultBotProperties(parse_mode=None))

    logger.info(
        "Starting bot with Ollama model '%s' at %s", config.ollama_model, config.ollama_base_url
    )

    try:
        await dispatcher.start_polling(bot, config=config)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
