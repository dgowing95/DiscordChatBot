# How the bot gets things done

Chatting with the bot works as it always did: mention it, or reply to it. When a message needs more
than an answer (look something up, read a link, make a chart, run some code, set a reminder) the bot
does the work itself and then answers in its own voice. This page shows what that looks like in a
server.

The bot has three sizes of job:

| | What it is | Where it happens |
|---|---|---|
| **A reply** | Plain chat: it answers from what it knows | In the channel |
| **A task** | It uses its tools (search, links, files, code, charts, images, polls, events, memories) and then answers | In the channel, while you wait |
| **A workflow** | A big job with many parts, planned first and run in steps | In a thread of its own ([workflows.md](workflows.md)) |

You never pick one: ask in plain words and the bot decides. `/task` starts a workflow by hand.

## What a task looks like

- **Each tool says what it did.** A search, a page it read, a poll or an event each post a small
  card in the channel, so you can see where an answer came from.
- **"🧠 Working on it"** appears when the bot runs code or works with files. One line on it says what
  the bot is doing right now, in plain words ("⏳ Installing python libraries", "⏳ Creating the
  script"), updated about every 30 seconds. With `/workspace progress on` it shows the commands and
  their output live instead.
- **Files arrive in the channel** before the answer: a chart, a CSV, a script.
- **Then the answer**, in the bot's own voice, with the facts it found.

Chat does not stop while the bot works: other people (and you) can keep talking, and the bot answers
them in the meantime.

### When it needs to ask you

If a guess would waste the work ("bar chart or line chart?"), the bot asks you first. The question
replies to your message and has buttons:

- the choices it suggests,
- **Answer…** to type your own,
- **You decide**, to let it choose.

Only the person who asked can answer. Using Discord's **Reply** on the question also answers it (the
bot reacts ✅); any other message is treated as a new request. With no answer in 2 minutes it carries
on with its best guess and says what it assumed. A reply the bot was not asked for never asks.

## The server's workspace

Each server has one **workspace**: a private computer where the bot runs code and keeps files. It
needs Docker on the machine that runs the bot.

- Every task gets **its own folder**, kept for 3 days, so "make the chart blue" finds the chart it just
  made.
- **`shared/`** holds what you asked the bot to keep ("keep this list for next week"). Every task in
  the server can see it.
- **Everyone in the server shares one workspace.** Files someone else left there are only data to the
  bot, never instructions.
- It stops after 15 minutes without a task, and is saved first, so nothing is lost. It is also saved
  every 30 minutes while it changes, and when the bot restarts.

### `/workspace`

| Command | What it does | Who can use it |
|---|---|---|
| `/workspace status` | Whether it is running, how many task folders it has, and when it was last saved (and how big that save was) | Manage Server |
| `/workspace reset` | Deletes every file in the server's workspace, after a confirm button (skills are kept) | Manage Server |
| `/workspace progress on\|off` | Show each task's commands and output live | Anyone |

## Memories

Tell the bot something worth remembering ("I prefer metric units", "my timezone is Europe/Berlin") and
it saves it (it reacts 💾). Memories belong to you, in this server only.

- **It uses them.** Your newest 20 memories go with every message you send it, so "what units do I
  like?" is answered at once.
- **To remove one**, ask: "forget that I prefer metric units".
- **To remove them all**, ask: "forget everything about me" (it reacts 🧹). This cannot be undone.

Automatic runs (schedules and rules) never see anyone's memories.

## Skills

When a job takes several steps worth repeating, the bot saves how it did it as a **skill** for the
server and uses it next time. See [skills.md](skills.md).

## In a voice call

In a call, the bot does the same work while it talks. It says a short "hold on" line while a task
runs, then gives the answer out loud; pictures and files go to a thread off the channel the call was
started from. Workflows are not started from a call. See [voice.md](voice.md).

## When something goes wrong

- **The bot says it stopped.** A task that repeats itself or runs too long is stopped, and the bot
  says what it got done.
- **"The safety guard blocked this request"**: the request did not pass the moderation check, so the
  workspace was not used.
- **It chats but cannot use its tools**: the workspace is not available (for example Docker is down
  on the bot's machine). Tell whoever runs the bot.
