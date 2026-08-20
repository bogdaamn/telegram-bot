#!/usr/bin/env bash
# Stop the larchanka-bot Telegram bot background process.
# Intended to be invoked via the `stop larchanka-bot` shell function
# (see the block appended to ~/.zshrc), but can also be run directly:
#   ./scripts/stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/.bot.pid"

cd "$PROJECT_DIR"

if [[ ! -f "$PID_FILE" ]]; then
  echo "larchanka-bot is not running (no PID file)."
  exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "larchanka-bot is not running (stale PID file removed)."
  rm -f "$PID_FILE"
  exit 0
fi

kill "$PID"

for _ in 1 2 3 4 5 6 7 8 9 10; do
  if ! kill -0 "$PID" 2>/dev/null; then
    break
  fi
  sleep 0.5
done

if kill -0 "$PID" 2>/dev/null; then
  echo "larchanka-bot did not stop gracefully, forcing kill (PID $PID)."
  kill -9 "$PID" 2>/dev/null || true
fi

rm -f "$PID_FILE"
echo "larchanka-bot stopped."
