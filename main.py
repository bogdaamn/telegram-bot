"""Application startup and Telegram handlers.

Each incoming text message is forwarded to the configured LLM client and the
reply is sent straight back — no conversation history is kept or persisted.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message

from config import ConfigError, load_config
from llm_client import LLMClient, LLMError, OllamaClient

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

dispatcher = Dispatcher()


@dispatcher.message(F.text)
async def handle_text_message(message: Message, llm_client: LLMClient) -> None:
    """Forward a single text message to the LLM and reply with its answer."""
    try:
        reply = await llm_client.generate_reply(message.text)
    except LLMError as exc:
        logger.error("LLM request failed for chat %s: %s", message.chat.id, exc)
        await message.answer(USER_FRIENDLY_ERROR)
        return

    await message.answer(reply)


@dispatcher.message()
async def handle_non_text_message(message: Message) -> None:
    """Gracefully acknowledge any non-text message instead of failing silently."""
    await message.answer(UNSUPPORTED_MESSAGE_NOTICE)


async def main() -> None:
    try:
        config = load_config()
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    llm_client: LLMClient = OllamaClient(
        base_url=config.ollama_base_url,
        model=config.ollama_model,
    )

    bot = Bot(
        token=config.telegram_bot_token,
        default=DefaultBotProperties(parse_mode=None),
    )

    logger.info("Starting bot with Ollama model '%s' at %s", config.ollama_model, config.ollama_base_url)

    try:
        await dispatcher.start_polling(bot, llm_client=llm_client)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
