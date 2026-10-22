# Schedules and rules

Automations are shared by server members. Anyone who can view and send in an automation's channel can inspect and manage it. Its channel stays fixed. Use `/schedule` or `/rule` to create, list, view, edit, delete, pause, or resume entries, or ask the bot in chat. `/schedule create` asks for the type (once, repeat every, daily or weekly) and opens a form with just that type's fields; `/schedule edit` does the same with the form filled in. `/rule create` and `/rule edit` open a form directly. The other commands autocomplete the entry as you type. Management replies are visible only to you.

Schedules can run once, at elapsed hour/day/week intervals, daily at a local clock time, or weekly on a weekday. The default timezone is Europe/London; specify another IANA timezone when creating a schedule. Daily and weekly entries use local clock time. A time skipped by daylight saving is skipped; a repeated time runs once at its first occurrence. After downtime a recurring schedule catches up once and resumes its timetable. The minimum gap between runs is measured from when a run actually starts, but a run that starts up to 15 minutes late (a busy queue) still counts as on time, so a schedule repeating exactly every minimum gap never skips a slot. Started runs are not replayed after a restart because tools may have already sent content. A one-off that finishes or fails releases its quota slot and remains viewable for seven days.

## Follow-ups

A follow-up is a one-off schedule the bot sets for itself when you ask it to come back to something later, for example "when the poll ends, make an event at the winning time". Ask in chat ("check back in 24 hours and tell me who won"); the bot writes down what it needs to do, because the run later sees no chat history. When it runs, the bot mentions the person who asked.

- Each person's follow-ups in a server must be at least `SCHEDULE_MIN_INTERVAL_HOURS` (default 4) apart. This counts ones that have already run in the last week, even if they were deleted, so follow-ups cannot be used as a faster repeating schedule. The first one can be soon, and one person's follow-up never blocks someone else's.
- They have their own quota (`FOLLOWUP_MAX_PER_GUILD`, default 3 waiting at once), so they never use up the server's schedule slot.
- A follow-up's run cannot set another follow-up.
- They appear in `/schedule list` marked "follow-up" and can be paused, resumed or deleted there. Their time cannot be edited; delete one and ask for a new one.

## Rules

Rules match message text in exactly their creation channel, including a thread only if created in that thread. `word` mode matches whole words or phrases; `substring` mode matches literal text inside other words. Matching ignores case and repeated whitespace. Bot messages, webhooks, edits, empty text, and `!reset_history` do not trigger rules. Each rule has a 60 second default cooldown. A rule job replaces an ordinary reply on that message; if no rule is admitted, ordinary chat still works.

Defaults are one schedule and four rules per server, a four hour minimum between recurring schedule runs, and a 60 second rule cooldown. Administrators can configure these using `AUTOMATIONS_*`, `SCHEDULE_*`, and `RULE_*` environment variables or Helm values. Zero quotas stop new entries of that kind. Disabling automations keeps stored entries but stops execution and hides management tools. Automatic runs may search the web, generate images, use the sandbox, and change personality; they cannot read or change personal memories or manage automations. An entry is suspended if its last editor loses access to its channel. A full queue skips rule matches; schedules stay due until capacity returns.
