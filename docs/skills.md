# Skills

The bot learns how to do jobs that come up again, and reuses what it learned. A skill is a short
recipe the bot saves for your server: what to do, in order, and any script that did the work.

## How it works

- When a job takes several steps worth repeating (say, "turn the tracker bot's embeds into a weekly
  chart"), the bot saves how it did it. A **📘 Skill saved** message appears in the channel.
- Next time you ask something like it, the bot opens the skill first (**📖 Using a skill**) and
  follows it, so it is faster and more steady.
- After a long job the bot may also look back over it and decide whether it is worth keeping. That
  happens after its reply, and shows as the same 📘 / 📗 message with "after reviewing the last job".
- Skills belong to the whole server. Anyone can see and change them. They are kept even if the
  server's workspace is reset.

You can also just ask: "save that as a skill", "use the weekly chart skill".

## Managing skills: `/skills`

| Command | What it does |
|---|---|
| `/skills list` | The skills saved here, best used first |
| `/skills show <name> [version]` | Read one (private) |
| `/skills history <name>` | Who changed it, when, and why (private) |
| `/skills edit <name>` | Change its description and steps in a form |
| `/skills delete <name>` | Remove it. It can be brought back |
| `/skills restore <name>` | Bring back a deleted skill |
| `/skills revert <name> <version>` | Go back to an earlier version (adds a new version, history is kept) |
| `/skills export` | Download every skill and its history as a zip (private) |

Each change posts a message in the channel (📗 updated, 🗑️ deleted, ♻️ restored, ⏪ reverted), so
everyone can see what changed.

## Limits

Each server keeps up to 100 skills. A skill's steps can be up to 8,000 characters, with up to 10
small text files, 64 KB in all. The last 50 versions of each are kept.

## For the person running the bot

Settings are `SKILLS_*` (see `docs/upgrading-to-v3.md`, chart values `skills.*`).
`SKILLS_ENABLED=0` turns it all off (saved skills are kept). Skills live in Redis, not in the
workspace container: the copy in the container (`skills/<name>/`, read-only) is rewritten from Redis
every time a skill is used, so changing it there never changes the skill.
