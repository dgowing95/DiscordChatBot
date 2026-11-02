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
| `SANDBOX_LLM_API_KEY` | `BRAIN_LLM_API_KEY` | Empty still means `LLM_API_KEY` |
| `SANDBOX_REQUEST_TIMEOUT_SECONDS` | `BRAIN_CALL_TIMEOUT_SECONDS` | **One wall-clock bound per model call, default 600** (was a 180 s per-read timeout times 1 + retries, so 540 s by default). Also the per-read timeout |
| `SANDBOX_MAX_RETRIES` | removed | The brain's client always retries twice; `BRAIN_CALL_TIMEOUT_SECONDS` bounds the whole call. Delete the line |
| `WORKER_COUNT` | `QUEUE_WORKERS` | Default unchanged (2) |
| `OPENAI_API_KEY` | `CONTENT_GUARD_API_KEY` | Only ever used for the moderation check |
| `OPENAI_BASE_URL` | `CONTENT_GUARD_BASE_URL` | |
| `MODERATION_MODEL` | `CONTENT_GUARD_MODEL` | |
| `MODERATION_CACHE_TTL` | `CONTENT_GUARD_CACHE_TTL_SECONDS` | Default unchanged (600) |
| `LLAMA_ARG_CACHE_TYPE_K` | `LLAMA_CACHE_TYPE_K` | docker compose only; compose passes it on as llama.cpp's own `LLAMA_ARG_CACHE_TYPE_K` |
| `LLAMA_ARG_CACHE_TYPE_V` | `LLAMA_CACHE_TYPE_V` | docker compose only, as above |

Default changes without a rename:

| Variable | Old default | New default |
|---|---|---|
| `CONTENT_GUARD_DEBUG` | on | **off**: set it to `1` to log the guard's per-request detail |

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
| `imagePullSecrets`, `podAnnotations`, `podLabels`, `resources` | removed | Chart scaffolding no template ever used; setting them never had an effect. Per-pod resources are `coreResources`, `llamacppResources`, … as before |

New keys: `contentGuard.enabled` (default true) and `contentGuard.debug` (default false) now reach core
from the chart.

## Secret key names

If you bring your own Secret (`existingSecret`), rename its keys:

| Old key | New key |
|---|---|
| `OPENAI_API_KEY` | `CONTENT_GUARD_API_KEY` |
| `SANDBOX_LLM_API_KEY` | `BRAIN_LLM_API_KEY` |

New optional key: `LLM_API_KEY`, for an LLM server outside the chart that checks a key.
`DISCORD_TOKEN`, `IMAGE_PROMPT_LLM_API_KEY`, `IMAGE_REVIEW_LLM_API_KEY` and `RESEARCH_LLM_API_KEY` are
unchanged.

## Metrics

| Metric | Change |
|---|---|
| `discord_bot_llm_context_window_tokens` | Read from the LLM server's `/props` every 5 minutes instead of from `LLM_CONTEXT_LENGTH`. It stays at its last value when a read fails, and is 0 until the first read |
