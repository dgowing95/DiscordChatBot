<!--
What's New for the NEXT release. Shown once per server, as an embed, the
first time someone triggers the bot after that release is deployed.

- Features only: no bug fixes, security or performance notes.
- REPLACE the whole file in a feature PR; do not append. Every merge to
  main is its own release, and the release ships only what changed here.
- No user-facing features? Leave this file untouched - the release blanks
  it automatically, so the last release's notes are never shown again.
- One "## Feature name" heading per feature, then what it does and a
  "How to use:" line. Max 25 features, 1024 characters per section
  (checked by core/tests/whats_new_tests.py).
-->

## Read the conversation around a message
After finding an old message, I can now show what was said just before and after it, and keep going further back or forward. Paste a message link and I can read around that too.
How to use: ask "find where we talked about the trip and tell me what was decided", or paste a message link and ask "what was going on around this?"

## Code sandbox can work through thousands of messages
The code sandbox can now save every message that matches a search (bot embeds included) to a file and work through it with code, so it can count, add up or chart a bot's posts over months.
How to use: ask e.g. "use the code sandbox to add up each player's net value from all the Shared Chest embeds in #game and chart it".
