---
name: sysmon
description: Report current CPU temperature and memory in use.
command: sysmon
---

Use the `exec` tool to check system health:

1. Run `smctemp -c` to get the CPU temperature in Celsius. If that command
   is denied or fails, run `pmset -g therm` instead and report the thermal
   pressure level (Nominal/Fair/Serious/Critical) rather than a number.
2. Run `top -l 1 -n 0 -s 0` and read the `PhysMem: ... used` line to report
   memory currently in use.
3. Reply with one short message summarizing both: CPU temperature (or
   thermal pressure) and memory in use.

If any command is denied, do not retry a different variant of the same
denied command — report the limitation to the user instead of spending
more steps on it.
