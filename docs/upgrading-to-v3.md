# Upgrading to v3

v3 is a major release. Many settings were renamed so their names say what they configure, and some
were removed. There are **no aliases** for the old names: an old name makes the bot refuse to start
(it logs each old name and its replacement), and an old values key makes `helm template` fail the same
way. That is on purpose, so a stale `.env` or values file never runs quietly on defaults.

This page grows as v3 is built. Each section lists what changed, the new default where it changed, and
what to do.

## Environment variables

Rename each old variable in your `.env` (docker compose) to the new name. Values keep their meaning
unless the note says otherwise.

| Old | New | Note |
|---|---|---|
| `MODEL` | `LLM_MODEL` | **Required now**: the bot will not start without it. The old code default (`qwen3:4b`) matched nothing compose or the chart serves. Compose's `llamacpp` service reads the same name |
| `LLM_PASS` | `LLM_API_KEY` | Same value. For Strata it is also the key compose gives the `strata` service |
| `REASONING_EFFORT` | `PERSONA_REASONING_EFFORT` | Default unchanged (medium) |
| `LLM_MAX_TURNS` | `PERSONA_MAX_TURNS` | Default unchanged (20) |
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
| `LLAMA_ARG_CACHE_TYPE_K` | `LLAMA_CACHE_TYPE_K` | docker compose only; compose passes it on as llama.cpp's own `LLAMA_ARG_CACHE_TYPE_K` |
| `LLAMA_ARG_CACHE_TYPE_V` | `LLAMA_CACHE_TYPE_V` | docker compose only, as above |

Default changes without a rename:

| Variable | Old default | New default |
|---|---|---|
| `CONTENT_GUARD_DEBUG` | on | **off**: set it to `1` to log the guard's per-request detail |
| `BRAIN_REASONING_EFFORT` (new) | none sent | **medium**, like the persona: the brain now sends `reasoning_effort` |

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
| `WORKSPACE_INSTANCE` | Label that keeps two bots on one Docker daemon apart (`dcb-dev`; the chart uses the release name) |
| `WORKSPACE_PREWARM` | Start a server's workspace on its first message (off; not used yet) |
| `LLAMA_CTX_SIZE` / `LLAMA_PARALLEL` / `LLAMA_CACHE_RAM_MIB` | docker compose only: llama.cpp's total context (80000), slots (1) and prompt cache in MiB (8192). These were fixed in `docker-compose.yaml` before, at the same values |

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
| `brain.workspace.idleSeconds`, `snapshotIntervalSeconds`, `maxWarm`, `jobRetentionHours`, `cpuLimit`, `pidsLimit`, `prewarm` | 900, 1800, 4, 72, empty, 512, false | The `WORKSPACE_*` settings above. `WORKSPACE_INSTANCE` is the release name |

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

New optional keys: `LLM_API_KEY`, for an LLM server outside the chart that checks a key, and
`PERSONA_LLM_API_KEY`, for a persona on a server of its own.
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
