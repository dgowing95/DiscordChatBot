# Upgrading to v3

v3 is a major release. Many settings were renamed so their names say what they configure, and some
were removed. There are **no aliases** for the old names: an old name makes the bot refuse to start
(it logs each old name and its replacement), and an old values key makes `helm template` fail the same
way. That is on purpose, so a stale `.env` or values file never runs quietly on defaults.

Start with the steps below. The sections after them list every change: what it was, the new default
where it changed, and what to do.

## Upgrade steps

1. **Back up** your v2 values file and `.env`. You need them unchanged to roll back (see the end of this
   page).
2. **Convert them** with the tables below: [environment variables](#environment-variables),
   [chart values](#chart-values) and, if you bring your own Secret,
   [Secret key names](#secret-key-names). Both `helm template` and the bot's startup name any old key
   still there, with its replacement.
3. **Make Docker available to core**: the node's Docker socket is now required (compose already mounts
   it; the chart mounts `brain.workspace.dockerSocketPath`). Core will not start without it.
4. **Check Redis has room**: each active server keeps a saved workspace of up to
   `WORKSPACE_SNAPSHOT_MAX_BYTES` (50 MB) for `WORKSPACE_SNAPSHOT_TTL_SECONDS` (7 days), plus its skills.
5. **Deploy**, then run `/help` in a server to see the new commands.

## What users will notice

- **No more sandbox threads.** The bot does research, files, code and charts itself and answers in the
  channel where it was asked (details under [The brain](#the-brain-one-tool-for-the-persona-every-other-tool-behind-it)).
- **Big jobs run as workflows** in a thread of their own, with a plan to approve first
  ([docs/workflows.md](workflows.md)). `/task` starts one by hand.
- **It remembers what you told it**: saved memories are used now.
- **It learns skills**, per server, managed with `/skills` ([docs/skills.md](skills.md)).
- **`/workspace` replaces `/sandbox_progress_updates`** (status, reset, progress).
- It can ask you a question with buttons when a guess would waste the work.
- It no longer answers direct messages (it never worked there).

See [docs/brain.md](brain.md) for how it all looks in a server.

## Environment variables

Rename each old variable in your `.env` (docker compose) to the new name. Values keep their meaning
unless the note says otherwise.

| Old | New | Note |
|---|---|---|
| `MODEL` | `LLM_MODEL` | **Required now**: the bot will not start without it. The old code default (`qwen3:4b`) matched nothing compose or the chart serves. Compose's `llamacpp` service reads the same name |
| `LLM_PASS` | `LLM_API_KEY` | Same value. For Strata it is also the key compose gives the `strata` service |
| `REASONING_EFFORT` | `PERSONA_REASONING_EFFORT` | Default unchanged (medium) |
| `LLM_MAX_TURNS` | `PERSONA_MAX_TURNS` | **Default 20 → 4**: the persona has one tool now (see "The brain" below) |
| `LLM_CALL_TIMEOUT_SECONDS` | `PERSONA_CALL_TIMEOUT_SECONDS` | Default unchanged (600) |
| `LLM_CONTEXT_LENGTH` | removed | The context-window gauge (`discord_bot_llm_context_window_tokens`) is now read from the LLM server's `/props` (per-slot context × slots). Delete the line |
| `SANDBOX_MODEL` | `BRAIN_MODEL` | Empty still means the bot's own model |
| `SANDBOX_LLM_HOST` | `BRAIN_LLM_HOST` | Empty still means `LLM_HOST` |
| `SANDBOX_LLM_API_KEY` | `BRAIN_LLM_API_KEY` | Empty means `LLM_API_KEY` **only when the brain uses the bot's own server**. See "Keys stay with their server" below |
| `SANDBOX_REQUEST_TIMEOUT_SECONDS` | `BRAIN_CALL_TIMEOUT_SECONDS` | **One wall-clock bound per model call, default 600** (was a 180 s per-read timeout times 1 + retries, so 540 s by default). Also the per-read timeout |
| `SANDBOX_MAX_RETRIES` | removed | The brain's client always retries twice; `BRAIN_CALL_TIMEOUT_SECONDS` bounds the whole call. Delete the line |
| `WORKER_COUNT` | `QUEUE_WORKERS` | Default unchanged (2) |
| `OPENAI_API_KEY` | `CONTENT_GUARD_API_KEY` | Only ever used for the moderation check |
| `OPENAI_BASE_URL` | `CONTENT_GUARD_BASE_URL` | |
| `MODERATION_MODEL` | `CONTENT_GUARD_MODEL` | |
| `MODERATION_CACHE_TTL` | `CONTENT_GUARD_CACHE_TTL_SECONDS` | Default unchanged (600) |
| `SANDBOX_ENABLED` | removed | **Docker is required now** (see "The server workspace" below): the code sandbox is always on. Delete the line |
| `SANDBOX_IMAGE` | `WORKSPACE_IMAGE` | Default unchanged (`python:3.14-slim`) |
| `SANDBOX_MEMORY_LIMIT` | `WORKSPACE_MEMORY_LIMIT` | Now caps each server's long-lived workspace container |
| `SANDBOX_SNAPSHOT_MAX_BYTES` | `WORKSPACE_SNAPSHOT_MAX_BYTES` | Default unchanged (50 MB), now per server instead of per thread |
| `SANDBOX_SNAPSHOT_TTL_SECONDS` | `WORKSPACE_SNAPSHOT_TTL_SECONDS` | Default unchanged (7 days) |
| `SANDBOX_PERSIST_TIMEOUT_SECONDS` | `WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS` | **Default 180 → 120**: the Memory capability that needed the extra time is gone |
| `SANDBOX_INPUT_MAX_BYTES` | `WORKSPACE_INPUT_MAX_BYTES` | Default unchanged (25 MB) |
| `SANDBOX_MAX_TURNS` | `BRAIN_MAX_TURNS` | **Default 10 → 100**, and only a backstop now: the loop guard stops a stuck brain call. Choose again rather than copy the old value |
| `SANDBOX_TIMEOUT` | `BRAIN_TIMEOUT` | **Default 600 → 1800** seconds, a backstop like `BRAIN_MAX_TURNS`; lane waits count towards it |
| `SANDBOX_ASK_USER_TIMEOUT` | `WORKFLOW_ASK_TIMEOUT` | Default unchanged (300); used by workflows only |
| `LLAMA_ARG_CACHE_TYPE_K` | `LLAMA_CACHE_TYPE_K` | docker compose only; compose passes it on as llama.cpp's own `LLAMA_ARG_CACHE_TYPE_K` |
| `LLAMA_ARG_CACHE_TYPE_V` | `LLAMA_CACHE_TYPE_V` | docker compose only, as above |

Default changes without a rename:

| Variable | Old default | New default |
|---|---|---|
| `CONTENT_GUARD_DEBUG` | on | **off**: set it to `1` to log the guard's per-request detail |
| `BRAIN_REASONING_EFFORT` (new) | none sent | **medium**, like the persona: the brain now sends `reasoning_effort` |
| `PERSONA_MAX_TURNS` | 20 | **4** (it was `LLM_MAX_TURNS`) |
| `WORKSPACE_PREWARM` | off | **on**: every brain call needs the workspace, so it starts when the bot is about to answer |

New variables, all optional:

| Variable | What it does |
|---|---|
| `PERSONA_MODEL` / `PERSONA_LLM_HOST` / `PERSONA_LLM_API_KEY` | The persona's own LLM. Empty = `LLM_*` |
| `BRAIN_REASONING_EFFORT` | The brain's reasoning effort (default medium) |
| `LLM_SLOTS` | Slots of a self-hosted server whose `/props` does not report them (default 1). llama.cpp and Strata report them, so it is rarely needed |
| `LLM_CHAT_RESERVED` | Slots only chat replies may use (default 1) |
| `WORKSPACE_IDLE_SECONDS` | Seconds a server's workspace stays running after its last task (900) |
| `WORKSPACE_SNAPSHOT_INTERVAL_SECONDS` | Seconds between saves of a running workspace that changed (1800; 0 = only when it stops) |
| `WORKSPACE_MAX_WARM` | Workspaces running at once (4) |
| `WORKSPACE_JOB_RETENTION_HOURS` | Task folders older than this are deleted (72) |
| `WORKSPACE_CPU_LIMIT` / `WORKSPACE_PIDS_LIMIT` | CPUs (none) and processes (512) per workspace container |
| `WORKSPACE_INSTANCE` | Label that keeps two bots on one Docker daemon apart (`dcb-dev`; the chart uses `<namespace>-<release name>`) |
| `WORKSPACE_PREWARM` | Start a server's workspace when the bot is about to answer there (on) |
| `BRAIN_REPEAT_LIMIT` | The loop guard: the same tool call this many times in a row (or eight calls that bring nothing new) earns a warning, then a stop (3) |
| `BRAIN_TRIM_KEEP_RECENT` | Tool outputs kept whole in a long brain call's prompt; older ones are cut to their first and last 400 characters (6) |
| `BRAIN_UNPROMPTED_MAX_TURNS` / `BRAIN_UNPROMPTED_TIMEOUT` | Backstops for a brain call in a reply the bot was not asked for (the random reply chance): 40 turns, 180 seconds |
| `BRAIN_ASK_TIMEOUT` | Seconds the brain waits for the requester to answer its question (120) |
| `SKILLS_ENABLED` | New: `0` turns skills off (on by default). Saved skills are kept |
| `SKILLS_MAX_PER_GUILD` / `SKILLS_MAX_VERSIONS` / `SKILLS_MAX_BYTES` / `SKILLS_MAX_BODY_CHARS` / `SKILLS_MAX_FILES` | New: limits on skills: 100 per server, 50 versions each, 64 KB, 8,000 characters of steps, 10 files |
| `SKILLS_INDEX_MAX` | New: skills named in the brain's task (20); the rest are found by search |
| `SKILLS_REFLECTION_ENABLED` / `SKILLS_REFLECTION_MIN_TOOL_CALLS` | New: after a long brain job, a background call decides whether to save a skill (on, 6 tool calls) |
| `SKILLS_CURATOR_ENABLED` / `SKILLS_CURATOR_INTERVAL_HOURS` / `SKILLS_CURATOR_MIN_SKILLS` / `SKILLS_RETIRE_DAYS` | New: the daily skills tidy-up (on, every 24 hours, servers with 5+ skills, retires skills unused for 60 days) |
| `BRAIN_MAX_CONCURRENT` / `BRAIN_QUEUE_WAIT` | Brain calls at once (3), and seconds a call waits for a free one before the bot says it is busy (120) |
| `LLAMA_CTX_SIZE` / `LLAMA_PARALLEL` / `LLAMA_CACHE_RAM_MIB` | docker compose only: llama.cpp's total context (80000), slots (1) and prompt cache in MiB (8192). These were fixed in `docker-compose.yaml` before, at the same values |
| `CONTENT_GUARD_MAX_RETRIES` / `CONTENT_GUARD_RETRY_BACKOFF` | Retries of the moderation call on 429/5xx (2) and its base backoff in seconds (2). The chart had these keys before; now `.env` can set them too |
| `WORKFLOW_ENABLED` / `WORKFLOW_FROM_AUTOMATIONS` | New: workflows (`/task`, the brain's `start_workflow`; on), and whether schedules, rules and follow-ups may start one, approved at once (on) |
| `WORKFLOW_PLAN_APPROVAL_SECONDS` / `WORKFLOW_PLAN_REVISIONS` / `WORKFLOW_MAX_STEPS` | New: seconds a plan waits for Approve before it starts anyway (900), changes a plan may get (3), steps in one plan (8) |
| `WORKFLOW_TIMEOUT` / `WORKFLOW_MAX_TURNS` / `WORKFLOW_MAX_TOKENS` | New: caps on one workflow: seconds (3600), model turns (300) and tokens in and out (4,000,000; 0 = off), every agent together |
| `WORKFLOW_MAX_PER_GUILD` | New: workflows running at once per server (1) |
| `SUBAGENT_MODEL` / `SUBAGENT_LLM_HOST` / `SUBAGENT_LLM_API_KEY` | New: a workflow helper's own LLM. Empty = the brain's, then `LLM_*` |
| `SUBAGENT_REASONING_EFFORT` / `SUBAGENT_CALL_TIMEOUT_SECONDS` | New: a helper's reasoning effort and per-call timeout. Empty = the brain's |
| `SUBAGENT_MAX_TURNS` / `SUBAGENT_MAX_CONCURRENT` | New: turns per helper (100), researchers running at once (2, cut to the slots a self-hosted server has beside chat's) |
| `CORE_ENV_FILE` | docker compose only: the env file handed to core (`.env`), for running a second env file |
| `WORKFLOW_DEV_INJECT` | Local testing and E2E only: `!workflow <request>` starts a workflow from a webhook. Never set it in production |

### Keys stay with their server

A key now only falls back to a key for the **same server**. In v2, a brain (`SANDBOX_LLM_HOST`) or a
side job (`IMAGE_PROMPT_LLM_HOST`, `IMAGE_REVIEW_LLM_HOST`, `RESEARCH_LLM_HOST`) pointed at another
server with no key of its own was sent the bot's `LLM_PASS`, which would hand a local Strata key to a
third party. Now it sends no key, and the bot logs a warning at startup. If such a server needs a key,
set its own: `BRAIN_LLM_API_KEY`, `IMAGE_PROMPT_LLM_API_KEY`, and so on. The same rule already applied
to `LLM_EXTRA_HEADERS`.

### Side jobs follow the brain

The research brief, the image prompt rewrite and the image check (`RESEARCH_*`, `IMAGE_PROMPT_*`,
`IMAGE_REVIEW_*`) now fall back to the **brain's** settings (`BRAIN_*`), and only then to `LLM_*`. With
`BRAIN_*` unset nothing changes. **If you set `BRAIN_LLM_HOST`** (say, to OpenRouter), those three jobs
move there too unless their own `*_LLM_HOST` is set. The image check needs a vision model, so point
`IMAGE_REVIEW_LLM_HOST` back at the bot's own server if the brain's model cannot see pictures. A job
on the bot's own server asks for `LLM_MODEL` with `LLM_API_KEY`, never for the brain's model, so the host
alone is enough.

### The server workspace (Docker is required)

The code sandbox used to start a throwaway container for every task and, in a thread, save that
thread's files when the task ended (up to 3 minutes after the reply). Now each Discord server has
**one long-lived container**, its workspace, on the same Docker daemon:

- **Docker is required.** The bot checks the daemon at startup (for up to 60 seconds) and refuses to
  start with a clear error if it cannot reach it. Compose already mounts the socket; the chart now
  always mounts it (`brain.workspace.dockerSocketPath`). The image is pulled at startup if missing.
- Every task works in its own folder (`jobs/<id>/`). A follow-up in the same thread finds its files
  there for `WORKSPACE_JOB_RETENTION_HOURS` (72). Files the server asks to keep go in `shared/`,
  which every task in that server sees. **Everyone in a server shares one workspace.**
- The workspace is saved to Redis (`dcb:workspace:v1:*`) every 30 minutes when it changed, before it
  stops after 15 minutes unused, and when the bot shuts down. A restart re-attaches to the running
  container, so nothing is lost.
- Threads' saved workspaces from v2 are **not** carried over: v3 does not read them, and they expire
  within 7 days (a rollback to v2 within that week still finds them).
- `/sandbox_progress_updates` is now `/workspace progress on|off`; a server's old setting is carried
  over the first time it is read. New: `/workspace status` and `/workspace reset` (both need Manage
  Server; reset deletes every file in the server's workspace).
- Each container is labelled `dcb.role=workspace`, `dcb.guild` and `dcb.instance`. To remove them by
  hand: `docker rm -f $(docker ps -aq --filter label=dcb.role=workspace)`.
- E2E now runs the sandbox too, on the GitHub runner's own Docker.

### The brain: one tool for the persona, every other tool behind it

The personality that answers in chat (the persona) now has **one tool, `ask_brain`**. Everything it
could do itself before (web search, reading links, memory, images, polls, events, message search,
schedules, voice) and the code sandbox are the **brain's** tools now. The persona answers plain chat
itself and hands anything else to the brain, which does the work and reports facts back for the
persona to put in its own words. What users notice:

- **No more sandbox threads.** Code and file work happens inline, in the channel it was asked in:
  files arrive there, and a "🧠 Working on it" card appears only while the brain uses the
  workspace (a search or a poll shows just its own card, as before). Messages in an old sandbox
  thread are answered like any other.
- **The bot remembers what it saved about you.** Saved memories never reached the model in v2 (since
  August 2025); now the asker's newest 20 are shown to the persona with every message, so "what
  units do I like?" is answered at once.
- **The bot no longer answers direct messages**, and its slash commands are offered in servers only.
  It never worked in DMs.
- **The bot can ask you a question** when a guess would waste the work. It replies to your request
  with buttons: its suggested choices, "Answer…" to type your own, and "You decide". Using Discord's
  Reply on the question works too (it reacts ✅); any other message is a new request. In a voice call
  it says the question aloud and your next words are the answer. No answer within 2 minutes and it
  carries on with its best guess and says what it assumed. Replies the bot was not asked for never
  ask, and get much shorter limits.
- **Chat keeps working while the brain does.** Each running brain call gets a spare queue worker, so
  `QUEUE_WORKERS` stay free for new messages.
- **The bot learns skills.** When a job takes several steps worth repeating (say, turning a tracker
  bot's embeds into a weekly chart), the brain saves how it did it as a skill, per server, and uses
  it next time (📘 / 📖 cards say so). Anyone can see and manage them with `/skills` (list, show,
  history, edit, delete, restore, revert, export); every change is a new version and a delete can
  be undone. Skills live in Redis, so a `/workspace reset` or a restart never loses one. Once a
  day the bot tidies them: it merges two that do the same job, retires ones unused for 60 days and
  sharpens unclear descriptions (🗑️ / 📗 cards marked "daily tidy-up"; each can be undone).
- `/help` describes the brain's tools; the "Run code" entry is now "Code, files and charts".
- **Big jobs run as workflows** in a thread of their own (`docs/workflows.md`). The brain hands
  one over when a job has many parts (a report from many sources, a small program), or anyone can
  start one with `/task`. A plan is posted first, with Approve / Edit / Cancel for the person who
  asked; no answer in 15 minutes and it starts anyway. Before the plan it may look at how the job
  is laid out and ask what changes the plan. The steps run in turn, research side by side, and files
  arrive in the thread and in the channel, and the bot replies in the channel when it ends. Saying just
  "stop" in the thread, or Cancel, stops it. One runs per server at a time; a restart stops it
  (the thread says so). Thread state is kept under `dcb:workflows:v1:*` (the old
  `dcb:sandbox_conversation:*` keys simply expire).
- A brain call runs in the server's workspace, so **every tool request now needs Docker**, not just
  code tasks. With Docker down the bot still chats but cannot search, remember or draw.

### Lanes

On a self-hosted server (llama.cpp, Strata), chat replies now go ahead of brain work and side jobs
when every slot is busy. With one slot, the brain waits until no chat reply is running or waiting. The
time a call waits for a slot does not count toward its `*_CALL_TIMEOUT_SECONDS`.

## Chart values

| Old key | New key | Note |
|---|---|---|
| `llmEngine` | `llm.engine` | |
| `model` | `llm.model` | |
| `llm_host` | `llm.host` | |
| `llm_pass` | `llm.apiKey` | Now rendered into the Secret (as `LLM_API_KEY`), not the ConfigMap. Leave it empty for llama.cpp (core uses a placeholder); with `llm.engine: strata` core gets the Strata key as before |
| `reasoningEffort` | `persona.reasoningEffort` | |
| `llmMaxTurns` | `persona.maxTurns` | |
| `llmCallTimeout` | `persona.callTimeout` | |
| `llamacppContextLength` | `llamacpp.contextPerSlot` | **Per slot now**: the server's context is `contextPerSlot × llamacpp.parallel`. With the default `parallel: 1` the number is the same (40000) |
| `message_history` | `history.limit` | |
| `worker_count` | `queue.workers` | |
| `queue_max_size` | `queue.maxSize` | |
| `openaiApiKey` | `contentGuard.apiKey` | |
| `openaiBaseUrl` | `contentGuard.baseUrl` | |
| `moderationModel` | `contentGuard.model` | |
| `guardMaxRetries` | `contentGuard.maxRetries` | |
| `guardRetryBackoff` | `contentGuard.retryBackoff` | |
| `moderationCacheTtl` | `contentGuard.cacheTtlSeconds` | |
| `sandbox.model` | `brain.model` | |
| `sandbox.llmHost` | `brain.llmHost` | |
| `sandbox.apiKey` | `brain.apiKey` | |
| `sandbox.requestTimeout` | `brain.callTimeout` | Default 600; see `BRAIN_CALL_TIMEOUT_SECONDS` above |
| `sandbox.maxRetries` | removed | |
| `sandbox.enabled` | removed | Docker is required: the socket is always mounted |
| `sandbox.image` | `brain.workspace.image` | |
| `sandbox.dockerSocketPath` | `brain.workspace.dockerSocketPath` | |
| `sandbox.memoryLimit` | `brain.workspace.memoryLimit` | |
| `sandbox.snapshotMaxBytes` | `brain.workspace.snapshotMaxBytes` | |
| `sandbox.snapshotTtlSeconds` | `brain.workspace.snapshotTtlSeconds` | |
| `sandbox.persistTimeout` | `brain.workspace.snapshotTimeout` | Default 180 → 120 |
| `sandbox.inputMaxBytes` | `brain.workspace.inputMaxBytes` | |
| `sandbox.maxTurns` | `brain.maxTurns` | **Default 10 → 100**, see `BRAIN_MAX_TURNS` |
| `sandbox.timeout` | `brain.timeout` | **Default 600 → 1800**, see `BRAIN_TIMEOUT` |
| `sandbox.askUserTimeout` | `workflows.askTimeout` | The last `sandbox.` key: the `sandbox:` block is gone |
| `imagePullSecrets`, `podAnnotations`, `podLabels`, `resources` | removed | Chart scaffolding no template ever used; setting them never had an effect. Per-pod resources are `coreResources`, `llamacppResources`, … as before |

New keys: `contentGuard.enabled` (default true) and `contentGuard.debug` (default false) now reach core
from the chart. Also new:

| Key | Default | What it does |
|---|---|---|
| `llm.slots` | empty (= `llamacpp.parallel` or `strata.parallel`) | `LLM_SLOTS` |
| `llm.chatReserved` | 1 | `LLM_CHAT_RESERVED` |
| `llamacpp.cacheRamMiB` | empty (llama.cpp's 8192) | llama.cpp's `--cache-ram` |
| `persona.model`, `persona.llmHost`, `persona.apiKey` | empty (= `llm.*`) | The persona's own LLM; the key goes in the Secret as `PERSONA_LLM_API_KEY` |
| `brain.reasoningEffort` | medium | `BRAIN_REASONING_EFFORT` |
| `brain.repeatLimit`, `brain.trimKeepRecent` | 3, 6 | `BRAIN_REPEAT_LIMIT`, `BRAIN_TRIM_KEEP_RECENT` |
| `brain.unprompted.maxTurns`, `brain.unprompted.timeout`, `brain.askTimeout`, `brain.maxConcurrent`, `brain.queueWait` | 40, 180, 120, 3, 120 | `BRAIN_UNPROMPTED_*`, `BRAIN_ASK_TIMEOUT`, `BRAIN_MAX_CONCURRENT`, `BRAIN_QUEUE_WAIT` |
| `skills.enabled`, `skills.maxPerGuild`, `skills.maxVersions`, `skills.maxBytes`, `skills.maxBodyChars`, `skills.maxFiles`, `skills.indexMax`, `skills.reflection.enabled`, `skills.reflection.minToolCalls`, `skills.curator.enabled`, `skills.curator.intervalHours`, `skills.curator.minSkills`, `skills.curator.retireDays` | true, 100, 50, 65536, 8000, 10, 20, true, 6, true, 24, 5, 60 | The `SKILLS_*` settings above |
| `persona.maxTurns` | **4** (was 20 as `llmMaxTurns`) | `PERSONA_MAX_TURNS` |
| `brain.workspace.idleSeconds`, `snapshotIntervalSeconds`, `maxWarm`, `jobRetentionHours`, `cpuLimit`, `pidsLimit`, `prewarm` | 900, 1800, 4, 72, empty, 512, true | The `WORKSPACE_*` settings above. `WORKSPACE_INSTANCE` is `<namespace>-<release name>`. The core pod's `terminationGracePeriodSeconds` is `snapshotTimeout` + 30, for the shutdown save |
| `workflows.enabled`, `fromAutomations`, `askTimeout`, `planApprovalSeconds`, `planRevisions`, `maxSteps`, `timeout`, `maxTurns`, `maxTokens`, `maxPerGuild` | true, true, 300, 900, 3, 8, 3600, 300, 4000000, 1 | The `WORKFLOW_*` settings above |
| `subagents.model`, `llmHost`, `apiKey`, `reasoningEffort`, `callTimeout`, `maxTurns`, `maxConcurrent` | empty (= `brain.*`), 100, 2 | The `SUBAGENT_*` settings above; the key goes in the Secret as `SUBAGENT_LLM_API_KEY` |

With `llm.engine: strata`, the render now fails when `persona.apiKey`, `brain.apiKey`, `research.apiKey`,
`diffusion.promptRewrite.apiKey` or `diffusion.review.apiKey` is set without a host to go with it
(the job's own `llmHost`, or for the last three `brain.llmHost`). Such a key would replace Strata's own
key for that job, and Strata refuses any other key. Remove the key, or set the host it is for.

## Secret key names

If you bring your own Secret (`existingSecret`), rename its keys:

| Old key | New key |
|---|---|
| `OPENAI_API_KEY` | `CONTENT_GUARD_API_KEY` |
| `SANDBOX_LLM_API_KEY` | `BRAIN_LLM_API_KEY` |

New optional keys: `LLM_API_KEY`, for an LLM server outside the chart that checks a key,
`PERSONA_LLM_API_KEY`, for a persona on a server of its own, and `SUBAGENT_LLM_API_KEY`, for
workflow helpers on a server of their own.
`DISCORD_TOKEN`, `IMAGE_PROMPT_LLM_API_KEY`, `IMAGE_REVIEW_LLM_API_KEY` and `RESEARCH_LLM_API_KEY` are
unchanged.

## Metrics

| Metric | Change |
|---|---|
| `discord_bot_llm_context_window_tokens` | Read from each self-hosted LLM server's `/props` every 5 minutes instead of from `LLM_CONTEXT_LENGTH`, with a new `host` label (one series per server). It stays at its last value when a read fails, and is absent until the first read |
| `discord_bot_llm_slots{host,source}` | New: each self-hosted server's slots, and whether they came from `/props`, `LLM_SLOTS` or the default |
| `discord_bot_llm_lane_wait_seconds{lane}` | New: seconds a model call waited for a slot. Not part of `discord_bot_llm_call_seconds` |
| `discord_bot_llm_lane_in_use{host,lane}`, `discord_bot_llm_lane_waiting{host,lane}` | New: slots held and calls waiting, per server and lane |
| `discord_bot_workspace_starts_total{kind}`, `discord_bot_workspace_start_seconds{kind}` | New: each use of a server's workspace and how long it took to be ready (`cached`, `resumed`, `snapshot`, `fresh`) |
| `discord_bot_workspaces_warm` | New: workspaces running in this core process |
| `discord_bot_workspace_snapshot_seconds`, `discord_bot_workspace_snapshot_bytes`, `discord_bot_workspace_snapshot_failures_total{reason}` | New: each save's time and size, and saves that failed (`too_large`, `timeout`, `error`) |
| `discord_bot_workspace_reaped_total` | New: leftover workspace containers removed |
| `discord_bot_brain_calls_total{tier,status}`, `discord_bot_brain_call_seconds{tier,status}` | New: each brain call and how long it took (`done`, `partial`, `needs_input`, `failed`, `started_workflow`) |
| `discord_bot_brain_turns`, `discord_bot_brain_tool_calls` | New: model turns and tool calls per brain call |
| `discord_bot_brain_loop_stops_total{kind}` | New: brain calls the loop guard stopped (`repeat`, `stale`) |
| `discord_bot_brain_facts_dropped_total` | New: facts the brain marked must-keep that the persona's reply left out |
| `discord_bot_persona_escalations_total{result}` | New: persona replies that called the brain (`brain`) or answered alone (`answered`) |
| `discord_bot_brain_questions_total{outcome}` | New: the brain's questions to the requester (`answered`, `timeout`, `skipped`) |
| `discord_bot_brain_slot_wait_seconds` | New: seconds a brain call waited for one of `BRAIN_MAX_CONCURRENT` slots |
| `discord_bot_skills_saved_total{source,result}`, `discord_bot_skills_used_total`, `discord_bot_skill_success_total` | New: skills saved (by the brain, a review, a command, a revert; `created`, `updated`, `unchanged`), opened with `use_skill`, and brain calls that used one and ended done |
| `discord_bot_skill_reflection_total{action}` | New: reviews of finished jobs (`none`, `create`, `update`, `failed`) |
| `discord_bot_skill_curator_changes_total{kind}`, `discord_bot_skills_live` | New: the tidy-up's changes (`merge`, `retire`, `describe`; `refused` when a skill changed meanwhile or a check said no; `failed` passes), and live skills on every server together |
| `discord_bot_queue_workers` | New: queue workers running, spares for brain calls included |
| `discord_bot_llm_call_seconds{caller="brain"}` | New caller: the brain's model calls (the persona's stay `main`) |
| `discord_bot_sandbox_thread_messages_total{outcome}` | Renamed `discord_bot_workflow_thread_messages_total` (same label): thread messages now go to workflows. Update dashboards |
| `discord_bot_sandbox_input_presentation_seconds`, `discord_bot_sandbox_input_response_seconds`, `discord_bot_sandbox_stale_actions_blocked_total`, `discord_bot_sandbox_questions_total`, `discord_bot_sandbox_unresolved_input_total`, `discord_bot_sandbox_continuations_total` | Removed, with the sandbox thread they measured. Remove them from dashboards |
| `discord_bot_workflows_total{trigger,end_state}`, `discord_bot_workflow_seconds{end_state}`, `discord_bot_workflow_subagents`, `discord_bot_workflow_turns` | New: each workflow (`brain`, `slash`, `automation`; `done`, `partial`, `failed`, `cancelled`), its time, steps and model turns |
| `discord_bot_workflow_plan_approval_total{outcome}`, `discord_bot_workflows_running` | New: how plans were approved (`approved`, `edited`, `cancelled`, `timeout`, `auto`) and workflows running now |
| `discord_bot_llm_call_seconds{caller="workflow_manager"\|"workflow_subagent"}` | New callers: a workflow's manager and helpers |
| `discord_bot_workflow_scouts_total{outcome}`, `discord_bot_workflow_plan_questions_total{outcome}` | New: how scouts ended (`done`, `reported_at_limit`, `stopped`) and questions before the plan (`answered`, `unanswered`) |
| `discord_bot_workflow_tokens` | New: tokens, in and out, one workflow used |
| `discord_bot_workflow_verdicts_total{outcome}` | New: checks of finished work (`passed`, `failed`, `not_checked`, `sent_back`), and of research before it is built on (the same with `research_` in front) |
| `discord_bot_llm_empty_retries_total{lane}` | New: model calls cut off before they said anything, and so asked once more |

## Rolling back to v2

1. Deploy the last v2.x release **with your v2 values file and `.env`** (the backups from step 1). A v3
   file does not work with v2's templates.
2. Remove the workspace containers v3 left on the node, which v2 does not know about:
   `docker rm -f $(docker ps -aq --filter label=dcb.role=workspace)`.

What carries over: memories, settings, schedules and rules, polls and the web cache use the same Redis
keys in both. v2 ignores v3's own keys (`dcb:workspace:v1:*`, `dcb:skills:v1:*`, `dcb:workflows:v1:*`),
and still finds its old thread workspaces if you roll back within their 7 days. v2 does not use skills,
workflows or v3's saved workspaces; they stay in Redis for a later upgrade (saved workspaces expire
after 7 days).
