# Telegram → Ollama Bot

A minimal Telegram bot that forwards each user text message to a local LLM
(via [Ollama](https://ollama.com)) and replies with the generated response.
Every message is handled independently — no conversation history is kept or
persisted, so restarting the bot never restores previous context.

## Architecture

- `main.py` — application startup and Telegram (aiogram) handlers
- `config.py` — environment/configuration loading
- `llm_client.py` — LLM-provider interface (`LLMClient`) and the Ollama
  implementation (`OllamaClient`)

The Telegram layer only calls `LLMClient.generate_reply(...)`; it has no
Ollama-specific HTTP logic, so a different provider can be plugged in later
by implementing `LLMClient`.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and runnable locally
- A Telegram bot token from [@BotFather](https://t.me/BotFather)

## Install and run Ollama

```bash
# Install Ollama (see https://ollama.com/download for other platforms)
brew install ollama

# Start the Ollama server (leave this running in its own terminal)
ollama serve

# Pull the default model used by this bot
ollama pull qwen3:1.7b
```

## Setup

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env` file from the example:

   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and set your bot token (and optionally override the Ollama
   settings):

   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen3:1.7b
   ```

   - `TELEGRAM_BOT_TOKEN` is required.
   - `OLLAMA_BASE_URL` defaults to `http://localhost:11434` if unset.
   - `OLLAMA_MODEL` defaults to `qwen3:1.7b` if unset.

   Never commit `.env` — it's already excluded via `.gitignore`.

## Run the bot

With `ollama serve` running and the model pulled:

```bash
python main.py
```

The bot uses long polling, so no public URL or webhook setup is needed. Send
it any text message on Telegram and it will reply with the LLM's response.
Non-text messages (photos, stickers, voice notes, etc.) get a friendly notice
instead of being silently ignored.

## Troubleshooting

- **"TELEGRAM_BOT_TOKEN is required" on startup** — you haven't set the token
  in `.env`, or `.env` isn't in the directory you're running `python main.py`
  from.
- **Bot replies with "Sorry, I couldn't get a response from the language
  model right now."**
  - Check that `ollama serve` is running.
  - Check that `OLLAMA_BASE_URL` matches where Ollama is listening (default
    `http://localhost:11434`).
  - Check that the model in `OLLAMA_MODEL` has been pulled: `ollama list`.
  - Look at the bot's terminal output — errors are logged there (with no
    secrets included) with the underlying cause (connection refused,
    timeout, bad response, etc.).
- **Bot doesn't respond at all** — confirm `TELEGRAM_BOT_TOKEN` is correct and
  that the process is still running; check the terminal for a Telegram API
  authentication error.
- **Slow first replies** — the first request to a newly pulled model can be
  slow while Ollama loads it into memory; subsequent replies should be
  faster.
