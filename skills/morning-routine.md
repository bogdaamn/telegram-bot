---
name: morning-routine
description: Morning summary combining weather and system health.
command: morning
---

Prepare a morning summary in three steps:

1. Read the `weather-api` skill (via `read_skill`) and follow it to fetch
   today's forecast for Minsk (`https://wttr.in/Minsk?0`).
2. Read the `sysmon` skill (via `read_skill`) and follow it to check
   current CPU temperature and memory usage.
3. Combine both results into one short summary message for the user:
   weather first, then system health.

Do not repeat a denied command from either sub-skill — if a step fails,
note it briefly in the summary and continue with the rest.
