# Design: Minimal AI Agent for the Telegram Bot

- Status: Approved
- Date: 2026-08-28
- Repo: `telegram-bot` (larchanka-training)

## 1. Purpose

Turn the existing stateless Telegram → Ollama forwarder into a minimal
autonomous agent: a small agentic loop, a restricted `exec` tool, on-demand
Skill files, and per-chat session persistence with `/new`. This is homework —
the implementation must stay small and legible, not production-grade.

## 2. Non-goals (explicit YAGNI)

- No streaming responses.
- No `/verbose` toggle — the live step trace is always on.
- No retry/backoff around Ollama or Telegram calls beyond what already exists.
- No sqlite/database — flat files only.
- No dashboard, metrics, or multi-user session isolation beyond the single
  allowlisted user.
- No real calendar/email integration (`gog cli`) — not installed, out of
  scope; the morning-routine skill uses only weather + local sysmon.
- No support for group chats or multiple concurrent allowlisted users beyond
  what a per-chat lock naturally provides.

## 3. Current state (baseline)

- `main.py` — aiogram 3 dispatcher, one `F.text` handler that calls
  `LLMClient.generate_reply(text)` and replies. No history.
- `config.py` — `Config` dataclass from env vars via `python-dotenv`.
- `llm_client.py` — `LLMClient` ABC, `OllamaClient` implementation using
  `POST /api/chat` with `stream: false`, single-turn `messages`.
- No tests, no tool-calling, no persistence.
- `.env.example` was deleted in the working tree but `scripts/start.sh` and
  `README.md` both still reference it. This is restored as part of this
  work (Section 11).

## 4. Target architecture

```
main.py             aiogram wiring: middleware, handlers, slash commands, DI
config.py           env config (extended)
llm_client.py       LLMClient ABC + OllamaClient — gains tool-aware chat()
agent.py            the harness / agentic loop (hard cap: <= 80 lines)
session.py          SessionStore: append-only jsonl, reload, .txt export
tools/
  __init__.py       ToolRegistry: name -> Tool, exports Ollama JSON schemas
  allowlist.py      declarative command registry + arg validators (pure)
  exec_tool.py      exec tool: validate -> subprocess (shell=False)
  skills_tool.py    read_skill tool
  sessions_tool.py  list_sessions, read_session tools
skills/
  sysmon.md
  weather-api.md
  morning-routine.md
sessions/            gitignored: active-<chat_id>.jsonl + <ts>-chat<id>.txt
tests/
  test_allowlist.py
  test_exec_tool.py
  test_agent_loop.py
  test_session.py
  test_skills_tool.py
  test_sessions_tool.py
  test_ollama_chat.py
```

The Telegram layer still only depends on `LLMClient` + `agent.run(...)`; it
has no Ollama-specific or tool-specific logic, preserving the existing
separation of concerns.

## 5. Tool-call protocol

Ollama's native tools API (`/api/chat` with a `tools` array), matching the
OpenAI-style tool-calling shape it supports. No text-based protocol, no
regex parsing of model output.

Contract with `llm_client.py`:

```python
class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Return the raw assistant message dict (content and/or tool_calls)."""
```

`generate_reply` is removed; `main.py`'s single-turn handler goes with it
(subsumed by the agent loop, which handles the single-turn case as a
zero-tool-call loop).

Two Ollama-specific quirks to handle in `OllamaClient.chat`:

1. `message.tool_calls[i].function.arguments` may already be a `dict`
   (Ollama's own quirk vs. OpenAI's JSON-string). Accept both: if it's a
   `str`, `json.loads` it; if it's already a `dict`, use it directly.
2. Tool results are appended to the message list as
   `{"role": "tool", "content": <string>, "tool_name": <name>}` before the
   next `chat()` call.

`OLLAMA_MODEL` default changes from `qwen3:1.7b` to `qwen2.5:7b` (already
pulled locally; verified via `ollama show qwen2.5:7b` to have the `tools`
capability and a 32k context window). Still overridable via env var.

## 6. The agentic loop (`agent.py`)

Hard constraint stated in the module docstring: **the loop function itself
must be small enough to read in one sitting — no more than 80 lines,
excluding imports, type definitions, and comments.** This is a homework
constraint on the implementer, not just documentation.

```python
MAX_STEPS_DEFAULT = 8  # env-tunable via AGENT_MAX_STEPS, clamped to [5, 10]

STEP_BUDGET_EXHAUSTED_NOTICE = (
    "I've hit my step limit working on this without reaching a final answer. "
    "Try rephrasing or breaking the request into smaller steps."
)

async def run(
    llm: LLMClient,
    registry: ToolRegistry,
    session: SessionStore,
    user_text: str,
    on_step: Callable[[dict], Awaitable[None]],
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
            session.append_message({
                "role": "tool",
                "tool_name": call["function"]["name"],
                "content": result_text,
            })
    return STEP_BUDGET_EXHAUSTED_NOTICE
```

Loop protection: `max_steps` is a required parameter (no infinite default);
`main.py` passes `config.agent_max_steps`. When the budget is exhausted, the
notice is returned as plain text and is **not** appended as an assistant
message (avoids polluting history with a message claiming a false "final
answer").

`on_step(call)` is how `main.py` renders the compact live trace (Section 9);
`agent.py` has zero Telegram knowledge — it takes a callback.

Errors from `registry.invoke` (a tool raising) are caught inside
`ToolRegistry.invoke` and turned into a `role: "tool"` content string
prefixed `ERROR:`, never raised up through `agent.run` — a single failing
tool call must not crash the loop or the Telegram handler.

## 7. `exec` tool and allowlist

### 7.1 Tool signature

```python
async def exec_tool(command: str) -> str:
    """Run one allowlisted command line and return combined stdout (or a
    DENIED:/ERROR: string). `command` is a single line, no shell operators."""
```

Exposed to the model as `exec` with a JSON schema: one required string
parameter `command`, plus a description enumerating exactly which
commands/purposes are permitted (so the model doesn't have to guess and
waste steps on denied attempts).

### 7.2 Validation pipeline (`tools/allowlist.py`, pure functions, no I/O)

1. Reject if `command` contains any of `; | & > < \` $ ( ) \n` (shell
   metacharacter guard) — return `DENIED: shell metacharacters are not
   permitted`.
2. `shlex.split(command)` (POSIX mode). Empty result -> `DENIED: empty
   command`.
3. Look up `Path(argv[0]).name` (basename only — rejects any path
   component, e.g. `/bin/rm`) against the registry keys. Unknown ->
   `DENIED: '<name>' is not an allowed command`.
4. Run the matched entry's validator against the full `argv`. Validator
   returns `(ok: bool, resolved_argv: list[str] | None, reason: str |
   None)`. On failure -> `DENIED: <reason>`.
5. On success, `exec_tool` runs `resolved_argv` via
   `asyncio.create_subprocess_exec(*resolved_argv, stdout=PIPE,
   stderr=STDOUT)` (never `shell=True`), with the entry's timeout
   (`asyncio.wait_for`) and an output cap (read up to N bytes, e.g. 4000,
   then truncate with a `... [truncated]` marker).

### 7.3 Allowlist registry (declarative table)

| Binary | Validator behavior | Timeout | Purpose |
|---|---|---|---|
| `smctemp` | argv must equal `["smctemp", "-c"]` OR `["smctemp", "-c", "-i25", "-n180", "-f"]` exactly | 5s | CPU temperature (°C) |
| `pmset` | argv must equal `["pmset", "-g", "therm"]` exactly | 5s | Thermal-pressure fallback if `smctemp` unavailable |
| `top` | argv must equal `["top", "-l", "1", "-n", "0", "-s", "0"]` exactly | 5s | Memory summary (`PhysMem: ...` line) |
| `vm_stat` | argv must equal `["vm_stat"]` exactly | 5s | Memory fallback (raw page counts) |
| `curl` | see 7.4 | 10s | Weather lookup (`wttr.in` only) |

Fixed-argv entries compare the full list with `==` — no partial matches, no
extra flags smuggled in.

### 7.4 `curl` validator (the one non-trivial entry)

- Allowed flags: `-s`, `-sS`, `--max-time` followed by a small integer
  (1–15). Any other flag -> denied.
- Exactly one non-flag argument: the URL.
- URL must parse via `urllib.parse.urlsplit` with `scheme == "https"` and
  `hostname` in the configured allowlist (`ALLOWED_HTTP_HOSTS`, default
  `wttr.in`).
- Explicitly denied even though `shlex`/scheme checks alone wouldn't catch
  them: `-L`/`--location` (would let a redirect leave the pinned host),
  `-o`/`--output`, `-d`/`--data*`, `-X`, `-H`, `-F`, `-T`/`--upload-file`
  (all denied by simply not being in the allowed-flags set).
- Resolved argv always appends `--max-time 10` itself if the model didn't
  provide one, so a hung request can't outlive the tool's own timeout.

### 7.5 Security tests (`tests/test_allowlist.py`) — written first (TDD)

Must-reject cases, each asserted to return a `DENIED:`-prefixed string
without touching `subprocess`:
`rm -rf /`, `smctemp; whoami`, `smctemp && curl http://evil.com`,
`curl http://wttr.in/Minsk` (http, not https), `curl https://evil.com`,
`curl -L https://wttr.in/Minsk`, `curl https://wttr.in/Minsk -o /tmp/x`,
`curl file:///etc/passwd`, `/bin/sh -c id`, `top -l 1 -n 5` (wrong argv),
`` `smctemp` `` with backticks embedded in a longer string, an empty
string, and a string containing only whitespace.

Must-accept cases: each exact allowed argv from the table above, and the
two `smctemp` variants.

## 8. Skills

### 8.1 File format

Each `skills/*.md` starts with a minimal frontmatter block (hand-parsed,
~15 lines, no PyYAML dependency — split on the first two `---` lines, then
one `key: value` per line):

```markdown
---
name: sysmon
description: Report current CPU temperature and memory in use.
command: sysmon
---

<body: full instructions for the model>
```

`command` is optional; when present, `main.py` registers `/​<command>` as a
slash command at startup.

### 8.2 Skill index and `read_skill` tool

At startup, `tools/skills_tool.py` scans `skills/*.md`, parses frontmatter
only (not the body), and builds an index: `{name: description}`. This index
— name + one-line description per skill, nothing else — is embedded in the
system prompt so the model knows what's available without spending context
on bodies it may not need this turn.

`read_skill(name: str) -> str` returns the full Markdown body for one named
skill, or `ERROR: no such skill '<name>'`. This is the only way skill bodies
reach the model.

### 8.3 The three skills (content requirements, not final prose)

**`sysmon.md`** (Type B action sequence, slash command `/sysmon`):
call `exec` with the CPU-temperature command (`smctemp -c`, falling back to
`pmset -g therm` if the model has learned `smctemp` is unavailable) and the
memory command (`top -l 1 -n 0 -s 0`), then summarize both in one short
message. Must explicitly state in the skill body: "if a command is denied,
do not retry a different variant of the same denied command — report the
limitation instead" (prevents step-budget burn on repeated denials).

**`weather-api.md`** (Type A — CLI/API usage rules): documents that
`https://wttr.in/<City>?0` gives a one-line forecast and `?1` a slightly
fuller one; only `curl` may be used; only `https://wttr.in/...` URLs are
reachable; the tool will deny anything else, so don't retry with different
flags on denial.

**`morning-routine.md`** (Type B, slash command `/morning`): 1) fetch
weather for a configured default city via the weather-api rules, 2) run
the sysmon check via the sysmon rules, 3) compose one combined summary
message for the user. References the other two skills by name rather than
duplicating their instructions — the model is expected to call
`read_skill` on each as needed.

## 9. Telegram integration (`main.py`)

- **Auth middleware**: outer middleware checks
  `message.from_user.id in config.allowed_user_ids`; on failure, sends a
  polite refusal, logs `chat_id` + `user_id`, and does not forward to the
  dispatcher pipeline. `ALLOWED_TELEGRAM_USER_IDS` is required and
  non-empty at config-load time — empty or missing is a `ConfigError`,
  never "allow everyone."
- **Per-chat lock**: `dict[int, asyncio.Lock]` in `main.py`. A message
  arriving while the chat's lock is held gets an immediate
  "still working on your previous message" reply and is dropped (not
  queued).
- **Live step trace**: `on_step` callback sends one short Telegram message
  per tool call before invoking it, e.g. `⚙️ exec: smctemp -c` or
  `⚙️ read_skill: sysmon`. Always on — no toggle.
- **`/new` command**: see Section 10.
- Non-text messages: unchanged friendly notice.

## 10. Session persistence and `/new`

### 10.1 File format (`sessions/active-<chat_id>.jsonl`)

Append-only, one JSON object per line, two kinds:

```json
{"kind": "msg", "message": {"role": "user", "content": "..."}}
{"kind": "tg", "message_id": 4821}
```

`kind: "msg"` records are replayed in order to reconstruct the Ollama
`messages` list on reload (`SessionStore.messages()` filters to these and
strips the `kind` wrapper). `kind: "tg"` records are appended for **every**
Telegram message tied to the conversation — the user's own messages, the
bot's replies, and every live-trace step message — so `/new` has a complete
deletion list. Writes are `open(path, "a")` + immediate `flush()`; no
buffering, so a crash loses at most nothing (append is synchronous per
call, loop is single-chat-at-a-time thanks to the per-chat lock).

### 10.2 Startup reload

On bot startup, for each `active-*.jsonl` found, `SessionStore` lazily
loads it on first message for that `chat_id` (not eagerly for all chats) —
reconstructing `messages()` from the `kind: "msg"` records. This makes a
bot restart mid-conversation resume with full context.

### 10.3 `/new` command sequence

1. Render a plain-text transcript from every `kind: "msg"` record —
   `[role] content` per line, in order — and write it to
   `sessions/<UTC-ISO-timestamp>-chat<chat_id>.txt`.
2. Collect every `kind: "tg"` message_id.
3. Call `bot.delete_messages(chat_id, ids)` in batches of at most 100
   (Telegram's bulk-delete limit). If a batch raises because it contains
   messages older than 48 hours (Telegram forbids deleting those), fall
   back to per-message `delete_message` calls within that batch, counting
   failures.
4. Delete `active-<chat_id>.jsonl` and drop the chat's in-memory
   `SessionStore` so the next message starts a fresh file.
5. Reply with a short confirmation that states how many messages were
   deleted and how many (if any) could not be deleted because they were
   older than 48 hours.

### 10.4 Reading back transcripts (`tools/sessions_tool.py`)

- `list_sessions() -> str`: lists filenames under `sessions/*.txt`
  (not the active `.jsonl`, which isn't a finished transcript), newest
  first.
- `read_session(filename: str) -> str`: validates `filename` matches
  `^[0-9T:\-]+Z-chat-?\d+\.txt$` (exact timestamp/chat-id shape produced by
  10.3, no path separators), then resolves
  `(SESSIONS_DIR / filename).resolve()` and checks it is still a child of
  `SESSIONS_DIR.resolve()` before reading — blocks both `../` traversal and
  symlink escapes. Returns file contents (truncated to a byte cap, e.g.
  8000 bytes with a truncation marker) or `ERROR: no such session
  '<filename>'`.
- Both tools are always in the registry (not gated behind a skill) per the
  requirement that "the agent must be able to find and read these saved
  session files if the user asks."

### 10.5 `.gitignore`

Add `sessions/` (the whole directory — both the active `.jsonl` and
exported `.txt` transcripts are session logs and must never be pushed).

## 11. Config (`config.py`)

New/changed fields on `Config`:

| Env var | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | changed default (was `qwen3:1.7b`) |
| `ALLOWED_TELEGRAM_USER_IDS` | *(required, no default)* | comma-separated ints; `ConfigError` if unset/empty |
| `AGENT_MAX_STEPS` | `8` | clamped to `[5, 10]`; out-of-range value is a `ConfigError`, not a silent clamp |
| `SESSIONS_DIR` | `sessions` | relative to project root |
| `SKILLS_DIR` | `skills` | relative to project root |
| `ALLOWED_HTTP_HOSTS` | `wttr.in` | comma-separated hostnames for the `curl` validator |

`.env.example` is restored (it was deleted from the working tree, but
`scripts/start.sh` and `README.md` both still reference it as the
onboarding step) and extended with the new keys above, `TELEGRAM_BOT_TOKEN`
and `OLLAMA_BASE_URL` unchanged from today.

## 12. Testing strategy (TDD)

Development order is red -> green -> refactor, test file before
implementation file, in this sequence (later steps depend on earlier ones
existing, so parallel implementation agents should follow the grouping in
the implementation plan rather than this exact order):

1. `tests/test_allowlist.py` before `tools/allowlist.py` (Section 7.5).
2. `tests/test_exec_tool.py` before `tools/exec_tool.py` — subprocess is
   mocked (`monkeypatch` on `asyncio.create_subprocess_exec`); covers
   timeout -> `ERROR: timed out`, output truncation, and a denied command
   never reaching the subprocess call (assert the mock was not called).
3. `tests/test_session.py` before `session.py` — append/reload round-trip,
   `/new`'s transcript rendering, and that a `kind: "tg"` record survives
   interleaved with `kind: "msg"` records in original order.
4. `tests/test_sessions_tool.py` before `tools/sessions_tool.py` — path
   traversal (`../../etc/passwd`, absolute paths, symlink-to-outside)
   all rejected; a valid filename round-trips.
5. `tests/test_skills_tool.py` before `tools/skills_tool.py` — frontmatter
   parsing, unknown skill name, and that the index used in the system
   prompt contains descriptions only (not bodies).
6. `tests/test_agent_loop.py` before finalizing `agent.py` — using a fake
   `LLMClient` (no real Ollama, no `respx` needed here since it's not
   HTTP-level) that is scripted to: (a) return content with no tool calls
   on turn 1 (loop returns immediately); (b) call a fake tool once then
   stop (loop makes exactly 2 `chat()` calls); (c) always return tool
   calls (loop stops exactly at `max_steps` and returns the budget-
   exhausted notice, and that notice is never appended to session
   history); (d) a tool that raises is surfaced as an `ERROR:`-prefixed
   tool message, not an exception out of `run()`.
7. `tests/test_ollama_chat.py` before extending `OllamaClient.chat` — uses
   `respx` to stub `POST /api/chat` and assert: `tools` is present in the
   request payload when passed; a response with `tool_calls[].function.
   arguments` as a JSON string is parsed into a dict; a response where
   `arguments` is already a dict passes through unchanged.

No live Telegram or live LLM calls in the automated suite. `README.md`
gains a short manual QA checklist (send a message, run `/sysmon`, run
`/morning`, run `/new` and confirm messages disappear and a `.txt` file
appears in `sessions/`, ask the bot to read that file back, try a message
from a non-allowlisted user id if a second Telegram account is available).

Adds to `requirements.txt`: `pytest`, `pytest-asyncio`, `respx`.

## 13. Prerequisites / setup changes

- `brew tap narugit/tap && brew install narugit/tap/smctemp` — confirmed
  via the project README that reading temperature needs no sudo on Apple
  Silicon (`sudo` is only used for the from-source install path, which
  this project does not use). If `smctemp` is absent at runtime, the
  `sysmon` skill body instructs the model to use the `pmset -g therm`
  fallback instead of retrying `smctemp`.
- No new Ollama model pull needed — `qwen2.5:7b` is already present
  locally.
- `scripts/start.sh` is otherwise unchanged; it already fails loudly if
  `.env` is missing, which continues to be the right behavior once
  `ALLOWED_TELEGRAM_USER_IDS` is a required var.

## 14. Open decisions made during design (recorded, not re-litigated)

- Tool protocol: native Ollama tools API, no text-based fallback parsing.
- Default model: `qwen2.5:7b`.
- Skills reach the model via an index + `read_skill`, not preloaded in full.
- CPU temperature via `smctemp` (brew), not sudo `powermetrics`.
- Three skills ship (`sysmon`, `weather-api`, `morning-routine`); no `gog`/
  calendar integration.
- `exec` allowlist is a declarative binary+validator registry, not a
  binary-name-only list or a fixed no-argument command catalog.
- Single allowlisted Telegram user id: `386668609`. No fail-open path.
- `/new` tracks and bulk-deletes every message id, reporting any
  older-than-48h messages it could not delete.
- Sessions persist to disk on every turn (not in-memory only), so a
  restart resumes context.
- Session recall uses dedicated `list_sessions`/`read_session` tools, not
  allowlisted `ls`/`cat`.
- Live per-tool-call trace is always on in Telegram; no `/verbose` toggle.
- Test coverage follows TDD: test file precedes implementation file for
  every unit listed in Section 12.
