# Telegram → Ollama Bot

A Telegram bot backed by a local LLM (via [Ollama](https://ollama.com))
that can call a small set of restricted tools to check system health,
look up the weather, and manage its own conversation history. Each chat
is a continuous session persisted to disk; `/new` exports the
conversation to a text file and starts fresh.

## Architecture

- `main.py` — application startup, Telegram (aiogram) handlers, auth
  middleware, and `/new` session reset
- `config.py` — environment/configuration loading
- `llm_client.py` — `LLMClient` interface and the Ollama implementation,
  using Ollama's native tool-calling API
- `agent.py` — the minimal agentic loop (tool calls until a final answer
  or the step budget runs out)
- `session.py` — append-only per-chat session persistence
- `tools/` — the restricted `exec` allowlist, skill loading, and saved
  session lookup tools
- `skills/` — Markdown skill files the agent can read on demand

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and runnable locally
- A Telegram bot token from [@BotFather](https://t.me/BotFather)
- [`smctemp`](https://github.com/narugit/smctemp) for CPU temperature (no
  sudo required to read temperatures on Apple Silicon):

  ```bash
  brew tap narugit/tap
  brew install narugit/tap/smctemp
  ```

## Install and run Ollama

```bash
# Install Ollama (see https://ollama.com/download for other platforms)
brew install ollama

# Start the Ollama server (leave this running in its own terminal)
ollama serve

# Pull the default model used by this bot
ollama pull qwen2.5:7b
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

4. Edit `.env` and set your bot token and allowed user id (and optionally
   override the other settings):

   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen2.5:7b
   ALLOWED_TELEGRAM_USER_IDS=386668609
   AGENT_MAX_STEPS=8
   SESSIONS_DIR=sessions
   SKILLS_DIR=skills
   ALLOWED_HTTP_HOSTS=wttr.in
   ```

   - `TELEGRAM_BOT_TOKEN` is required.
   - `ALLOWED_TELEGRAM_USER_IDS` is required — a comma-separated list of
     Telegram numeric user ids allowed to talk to the bot. Anyone else is
     refused. Find your own id by messaging [@userinfobot](https://t.me/userinfobot).
   - `OLLAMA_BASE_URL` defaults to `http://localhost:11434` if unset.
   - `OLLAMA_MODEL` defaults to `qwen2.5:7b` if unset.
   - `AGENT_MAX_STEPS` defaults to `8` (must be between 5 and 10) — the
     agent's tool-call step budget per message.
   - `SESSIONS_DIR` / `SKILLS_DIR` default to `sessions` / `skills`.
   - `ALLOWED_HTTP_HOSTS` defaults to `wttr.in` — hosts the `exec` tool's
     `curl` command may reach.

   Never commit `.env` — it's already excluded via `.gitignore`.

## Run the bot

With `ollama serve` running and the model pulled:

```bash
python main.py
```

The bot uses long polling, so no public URL or webhook setup is needed. Only
the Telegram user id(s) in `ALLOWED_TELEGRAM_USER_IDS` can talk to it. Each
chat keeps a continuous conversation (persisted to `sessions/`, restored on
restart); send `/new` to export the conversation to a `.txt` file and start
fresh. `/sysmon` and `/morning` run the corresponding skills directly.
Non-text messages (photos, stickers, voice notes, etc.) get a friendly notice
instead of being silently ignored.

## Manual QA checklist

Automated tests cover the allowlist, exec tool, session persistence,
skill loading, agent loop, and Ollama client in isolation (`pytest`).
These steps verify the pieces work together for real:

- [ ] Send a plain text message; confirm a reply comes back.
- [ ] Send `/sysmon`; confirm it reports a CPU temperature (or thermal
      pressure) and memory usage, with a `⚙️ exec: ...` trace
      message before the summary.
- [ ] Send `/morning`; confirm it reports weather for Minsk and then a
      system health summary.
- [ ] Send `/new`; confirm the chat's messages disappear and a
      `sessions/<timestamp>-chat<id>.txt` file appears.
- [ ] Ask "what did we talk about before I ran /new?" and confirm the
      agent can call `list_sessions`/`read_session` to answer from the
      exported file.
- [ ] If you have a second Telegram account, message the bot from it and
      confirm it's refused (not in `ALLOWED_TELEGRAM_USER_IDS`).

## Troubleshooting

- **"TELEGRAM_BOT_TOKEN is required" / "ALLOWED_TELEGRAM_USER_IDS is
  required" on startup** — you haven't set that variable in `.env`, or
  `.env` isn't in the directory you're running `python main.py` from.
- **"Sorry, this bot is not available to you."** — your Telegram user id
  isn't in `ALLOWED_TELEGRAM_USER_IDS`. Add it and restart the bot.
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
