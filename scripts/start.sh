#!/usr/bin/env bash
# Start the larchanka-bot Telegram bot as a background process.
# Intended to be invoked via the `start larchanka-bot` shell function
# (see the block appended to ~/.zshrc), but can also be run directly:
#   ./scripts/start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/.bot.pid"
LOG_FILE="$PROJECT_DIR/bot.log"

cd "$PROJECT_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "larchanka-bot is already running (PID $(cat "$PID_FILE"))."
  exit 0
fi

if [[ ! -f main.py ]]; then
  echo "Error: main.py not found in $PROJECT_DIR." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Error: .env not found. Run: cp .env.example .env, then set TELEGRAM_BOT_TOKEN." >&2
  exit 1
fi

if [[ -x .venv/bin/python ]]; then
  PYTHON=.venv/bin/python
else
  echo "Warning: .venv not found, falling back to 'python3' on PATH." >&2
  PYTHON=python3
fi

OLLAMA_BASE_URL="$(grep -E '^OLLAMA_BASE_URL=' .env 2>/dev/null | cut -d= -f2- || true)"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
if ! curl -s -o /dev/null -m 2 "$OLLAMA_BASE_URL"; then
  echo "Warning: could not reach Ollama at $OLLAMA_BASE_URL — the bot will start, but replies will fail until Ollama is running." >&2
fi

nohup "$PYTHON" main.py >> "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
disown

echo "larchanka-bot started (PID $(cat "$PID_FILE")). Logs: $LOG_FILE"
