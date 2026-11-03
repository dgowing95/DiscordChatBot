# Workflows (big jobs)

Some jobs are too big to finish inside one reply: a report built from many sources, or a small
program with several files. The bot runs those as a **workflow** in a thread of its own. It plans the
job, you approve the plan, and it works through the steps while the channel carries on as normal.

## Starting one

- **Just ask.** When a request has many parts, the bot hands it to a workflow by itself and tells you
  where the thread is.
- **`/task <request>`** starts one straight away. Put every detail in the request.

One workflow runs per server at a time. If one is already running, the bot says where it is.
While the bot is in a voice call in the server, `/task` says so and starts nothing; a workflow already
running carries on in its thread.

## The plan

The bot posts its plan in the thread: the goal, what you will get, and the steps. Only the person
who asked can answer it:

| | What it does |
|---|---|
| **Approve** | Starts the work |
| **Edit**, or just write in the thread | Say what to change. The bot writes a new plan (up to 3 times) |
| **Cancel** | Stops it before anything runs |

With no answer in 15 minutes it starts anyway, and says so. A workflow started by a schedule or a
rule starts at once.

## While it works

- Each step is done by a helper, one at a time. The plan shows each step's state (⏳ waiting,
  ▶️ running, ✅ done). Files a step makes are posted in the thread.
- Write in the thread to steer it. Your message gets 📨 and the bot answers it before the next step,
  changing the remaining steps if you ask.
- If a helper needs to know something, it asks you in the thread. Your next message there is the
  answer.
- Say just **stop** in the thread, or press **Cancel** on the plan, to stop it. Files made so far
  are kept.

## When it ends

The thread gets a closing message (done, partly done, failed or cancelled), the files are posted in
the channel you asked in, and the bot replies to you there.

Ask about the result in the thread afterwards ("make the chart bigger") and the bot works on the
same files.

A restart of the bot stops a running workflow; its thread says so. Ask again to start over.

## For server owners

| Setting | Default | What it does |
|---|---|---|
| `WORKFLOW_ENABLED` | on | Off removes `/task`, and the bot never starts a workflow |
| `WORKFLOW_FROM_AUTOMATIONS` | on | Schedules, rules and follow-ups may start one |
| `WORKFLOW_PLAN_APPROVAL_SECONDS` | 900 | How long the plan waits for an answer |
| `WORKFLOW_TIMEOUT` / `WORKFLOW_MAX_TURNS` | 4 h / 500 | Stops a workflow that never ends |
| `WORKFLOW_MAX_PER_GUILD` | 1 | Workflows running at once in one server |
| `SUBAGENT_MODEL` / `SUBAGENT_LLM_HOST` / `SUBAGENT_LLM_API_KEY` | the brain's | The helpers' model |

Workflows are the heaviest load the bot has. On a server with one LLM slot, chat still comes first.
