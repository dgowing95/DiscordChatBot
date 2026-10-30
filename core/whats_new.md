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

## Better link reading
I read links much better now: Like YouTube videos (from their captions) and Reddit threads, and sites that only show their content once their scripts run. I also cache the content to speed up replies.
How to use: paste a link and @mention me, e.g. "what is said in this video?" or "what are people saying in this thread?"

## Find old messages
I can search the server's past messages to find something said long ago, and link you straight to it. I search the whole server, or just one channel if you say which.
How to use: @mention me, e.g. "find the message where Sam posted the trip link" or "what did Alex say about the patch in #general?"
