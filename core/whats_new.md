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

## Scheduled actions
Ask me to do something once or on a recurring interval, day, or weekday. I run it in the channel where you create it and use the current time when it runs.
How to use: ask "Every day at 9am, find today's Wordle answer" or use `/schedule create`.

## Message rules
Set a word, phrase, or substring that triggers an action when someone posts it in this channel. Matching rules answer together in one response.
How to use: ask "When someone says wordle, find today's answer" or use `/rule create`.
