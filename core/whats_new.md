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

## Code sandbox can look things up
While working on a code task, the sandbox can now search the web, read links, search this server's messages and look up events. It can also make or move a server event.
How to use: ask for code that needs facts, e.g. "make a chart of the last five Python release dates", or "build a quiz from what we said about the trip".
