---
name: weather-api
description: Rules for looking up weather via the wttr.in command-line API.
---

To check the weather for a city, use the `exec` tool with `curl` against
`https://wttr.in/<City>?0` for a compact one-line forecast, or
`https://wttr.in/<City>?1` for a slightly fuller one. Replace `<City>` with
the requested city name (URL-encode spaces as `%20` if needed).

Only `curl` requests to `https://wttr.in/...` are permitted — no other
hosts, no `http://`, no extra flags like `-L`, `-o`, or `-d`. If the
request is denied, do not retry with different curl flags; report that
weather lookup is unavailable for that request.
