# AGENTS.md

Overview of this repository and how to work with it (for humans and AI coding agents).

## What this is

A Python Discord bot that answers messages using a local LLM (llama.cpp server serving a GGUF model, e.g. `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`).
It is primarily deployed on **Kubernetes** via the Helm chart in `charts/dis-ai-bot`
(releases are produced by semantic-release; a prebuilt chart is downloadable from GitHub releases).
Redis is used as the settings store and user-memory store.

## Repository layout

```
core/                  # the main bot (the app that runs in production)
  main.py              # entrypoint: discord.Client, message queue, slash commands
  classes/
    message_handler.py     # per-message orchestration: history build, send/chunking
    text_llm_handler.py    # builds an `agents` Agent against the LLM server's (llama.cpp) OpenAI-compat API
    response_filter.py     # PURE (stdlib-only) response cleaning / thinking-block stripping
    content_guard.py       # OpenAI Moderations-based safety guard for web_search/fetch_url
    metrics.py             # PURE (stdlib + prometheus_client) Prometheus metrics:
                           #   all metric definitions + /metrics HTTP server (METRICS_PORT)
    user_memory.py         # JSON lists in Redis per (guild, user)
    config_manager.py      # per-guild settings in Redis (system prompt, temperature, ...)
    tool_functions.py      # agent function tools: web_search, fetch_url, weather, memory tools, generate_image
    image_generation.py    # client for the diffusion service + IMAGE_GEN_ENABLED flag
    sandbox_agent.py       # nested SandboxAgent + run_sandbox_task (throwaway Docker sandbox)
    sandbox_progress.py    # streams sandbox commands/output to one edited Discord message
                           #   (single embed: one field per command, state-coloured)
    common.py              # shared helpers (Discord tool embeds)
  tests/               # pytest suite (see Testing below)
  Dockerfile           # python:3.13-slim image, runs main.py
  requirements.txt
diffusionservice/      # standalone image service (text->image AND image->image/editing;
                       #   FastAPI + diffusers, queued single-worker, sd-turbo by default,
                       #   CPU-offloaded for low VRAM; img2img shares the loaded components)
charts/dis-ai-bot/     # Helm chart
docker-compose.yaml    # local dev: redis + llamacpp (GPU, llama.cpp) + diffusion (GPU) + core (mounts ./core)
.env / .env.example    # environment configuration (never commit .env)
```

## Runtime architecture

1. `main.py:` every Discord message goes on an `asyncio.Queue`; a single worker loop
   pops messages, builds a `MessageHandler`, and (if `should_process_message()`
   passes: mentioned, or passes the random reply-chance check) handles it.
2. `MessageHandler.handle_message()` builds the prompt
   (channel history -- most recent `MSG_HISTORY_LIMIT` (default 5) messages; the
   user's stored Redis memories are exposed to the agent through its function
   tools) and calls `TextLLMHandler.generate()`.
3. `TextLLMHandler` uses the **OpenAI `agents` SDK** pointed at the llama.cpp
   server's OpenAI-compatible endpoint (`LLM_HOST/v1`) with function tools attached.
4. The returned text is cleaned by `MessageHandler.filter_response()` (delegate:
   `core/classes/response_filter.py`, a pure module) and sent in **2000-char chunks**
   (`textwrap.wrap`, one `asyncio.sleep(1)` between sends).
5. Per-guild settings live in Redis under the `dcb` namespace; per-user memories under
   `guild:<id>:user:<id>`.
6. Image generation: when enabled (`IMAGE_GEN_ENABLED`, set from the chart's
   `diffusion.enabled`), the agent gets two tools — `generate_image(prompt)` and
   `edit_image(prompt, image_ref?, strength?)` — plus `/generate_image <prompt>`
   and `/edit_image <attachment> <prompt> [strength]` slash commands (registered
   in `main.py`). All of them POST to the standalone diffusion service
   (`DIFFUSION_URL/generate`), which runs in its own pod/container, queues
   requests (one image at a time) and replies with a PNG that is sent to the
   Discord channel. `edit_image`/`/edit_image` send the source image as base64
   (img2img; the service derives its img2img pipeline from the SAME loaded
   components, so no extra model/RAM/VRAM). `build_messages()` lists attached
   images by a short label (`[1]`, `[2]`, …) — NOT the signed CDN URL, because
   the model corrupts the 64-char hex signature when copying it into a tool
   arg (the fetch then 404s). `edit_image` resolves `image_ref` (a label, or
   "latest") to the real URL via the `attachment_refs` run context. Generation
   settings (`IMAGE_MODEL`, `IMAGE_STEPS`, `IMAGE_WIDTH`/`HEIGHT`, `IMAGE_OFFLOAD`,
   `IMAGE_QUEUE_SIZE`, `IMAGE_EDIT_STRENGTH`) live in the same configmap/env the
   diffusion pod reads.
7. Code sandbox: when enabled (`SANDBOX_ENABLED`, set from the chart's
   `sandbox.enabled`), the agent gets a `run_code_sandbox(task)` tool
   (no slash command). It runs a nested `SandboxAgent` (same LLM as the main
   agent by default — `SANDBOX_MODEL`/`SANDBOX_LLM_HOST`/`SANDBOX_LLM_API_KEY`
   can point it at a different OpenAI-compatible API, e.g. OpenRouter;
   Shell capability only — the Filesystem capability's `apply_patch` is a
   grammar tool the ChatCompletions API does not support, and exec_command
   already gives full filesystem access, empty workspace; the SDK's default
   sandbox base prompt is suppressed with `base_instructions=""` because it
   tells the model to call apply_patch, which aborts the run with
   ModelBehaviorError when the tool is absent) inside a THROWAWAY
   Docker container via `agents.sandbox.DockerSandboxClient` and returns the
   sandbox agent's final report. A fresh container is created and deleted for
   every call (nothing persists); the task must be self-contained. Live
   progress is a per-guild opt-in: the `/sandbox_progress_updates true|false`
   slash command stores the setting in Redis (default OFF — then only the one
   static "Running in sandbox" embed is sent). When on, a `RunHooks` attached
   to the nested run observes each `exec_command`/`write_stdin` call and its
   output and mirrors them into ONE Discord message edited in place: a
   single embed styled like the static "Tool Usage" embed (title "🐳
   Sandbox", description "Running in sandbox: {task}"), with ONE FIELD PER
   COMMAND — one-liners as the bold field name (`$ cmd`), heredocs under
   `⌨ Command` with their text and the fenced output in the value — and the
   accent colour reflecting state (yellow running / red failure / cyan
   idle); events are queued and the whole queue is batched into every
   throttled edit (15s to stay under Discord's 5-edits/minute limit; oldest
   fields evicted as a unit under the 25-field/6000-char embed budget —
   see `sandbox_progress.py`). The run's final state (done/timeout/failed)
   is flushed via `finalize()` before the tool returns. The core
   container needs the Docker daemon socket mounted (compose: socket bind
   mount; chart: hostPath volume gated on `sandbox.enabled`) plus the
   `docker`/`websocket-client` Python packages. Tasks go through the content
   guard first; `SANDBOX_MAX_TURNS`/`SANDBOX_TIMEOUT` bound each run.

### Environment variables

| Var | Purpose |
|---|---|
| `DISCORD_TOKEN` | required; bot token |
| `REDIS_HOST` | required; Redis host |
| `LLM_HOST` | llama.cpp server base URL (OpenAI-compat; core appends `/v1`). Points at the `llamacpp` service on :8081 in docker-compose (dev) and :8080 in the helm chart; the in-code fallback is `http://llamacpp:8080` |
| `LLM_PASS` | placeholder key — llama.cpp does not authenticate, but the OpenAI client requires a non-empty key |
| `MODEL` | the model, single source of truth: the bot requests this name AND the compose `llamacpp` service serves it (as `LLAMA_ARG_HF_REPO`); in the Helm chart the one `model` value feeds both. In-code default `qwen3:4b` |
| `OPENAI_API_KEY` | API key for the free OpenAI Moderations endpoint (web-tool guard); fail-open if unset |
| `CONTENT_GUARD_ENABLED` | `0`/`false` disables the content guard on web tools (default: on) |
| `CONTENT_GUARD_DEBUG` | `0`/`false` silences content-guard debug logging (default: on) |
| `METRICS_PORT` | port to serve the Prometheus `/metrics` endpoint on (default 9464); empty/`0` disables. Chart: `metrics.enabled`/`metrics.port` also add the pod port + a kube-prometheus-stack ServiceMonitor |
| `MSG_HISTORY_LIMIT` | how many prior channel messages to include, default 5 |
| `LLAMA_ARG_CACHE_TYPE_K`, `LLAMA_ARG_CACHE_TYPE_V` | optional; compose `llamacpp` service only: KV cache quantization type (llama.cpp `-ctk`/`-ctv`), default `q4_0`; in the Helm chart set via `llamacpp.cacheTypeK`/`cacheTypeV` |
| `IMAGE_GEN_ENABLED` | `0`/`false` removes the `generate_image` tool from the LLM (default: on). Chart: `diffusion.enabled` also removes the diffusion pod/PVC |
| `DIFFUSION_URL` | base URL of the diffusion service (core appends `/generate`); compose `diffusion` service on :8000 in dev, in-cluster `*-diffusion-service` in the chart; in-code fallback `http://diffusion:8000` |
| `IMAGE_MODEL` | HF repo id for the diffusion service (default `stabilityai/sd-turbo` — smallest practical model); the service downloads it into its `HF_HOME` volume on first boot |
| `IMAGE_STEPS` / `IMAGE_WIDTH` / `IMAGE_HEIGHT` | generation settings for the diffusion service (defaults: 4 steps, 512x512) |
| `IMAGE_OFFLOAD` | `model` (default: one pipeline component on GPU at a time, text encoder in CPU RAM) / `sequential` (lowest VRAM, slowest) / `none` (all on GPU) |
| `IMAGE_QUEUE_SIZE` | max queued image requests in the diffusion service (default 16); over that it returns 503 |
| `IMAGE_EDIT_STRENGTH` | default img2img strength, 0-1 exclusive: higher = more changes, lower = closer to the original (default 0.5) |
| `IMAGE_GEN_TIMEOUT` | seconds core waits on the diffusion service (default 300) |
| `SANDBOX_MODEL` | model id for the nested sandbox agent; empty (default) = the main bot's `MODEL`. Chart: `sandbox.model` |
| `SANDBOX_LLM_HOST` | base URL of the sandbox agent's LLM (core appends `/v1`); empty (default) = the main `LLM_HOST`. E.g. `https://openrouter.ai/api` for OpenRouter. Chart: `sandbox.llmHost` |
| `SANDBOX_LLM_API_KEY` | API key for the sandbox agent's LLM; empty (default) = the main `LLM_PASS` placeholder. Chart: `sandbox.apiKey` |

## Testing

- **Framework:** `pytest` (in `core/requirements.txt`). No conftest/pytest.ini; plain
  test files in `core/tests/`.
- **Prereq:** a Python 3.13+ venv with `pip install -r core/requirements.txt`
  (tests that import `config_manager`/`user_memory` need `redis`; `response_filter`
  tests are pure-stdlib).
- **How to run** (from the repo root — the `PYTHONPATH` is required because
  modules use both `core.classes...` and `classes...` import styles):

  ```bash
  PYTHONPATH=$(pwd) pytest core/tests/user_memory_tests.py core/tests/response_filter_tests.py core/tests/metrics_tests.py
  ```

  (On Windows PowerShell use `$env:PYTHONPATH=$(Get-Location)` — or
  `PYTHONPATH=$PWD pytest ...` in bash.)

- Keeping a module **pure and importable without the discord/agents SDKs** (like
  `response_filter.py`) is the intended pattern for anything you want to unit test —
  `MessageHandler` itself drags in `discord`, `agents`, Redis, etc.
- **CI:** `.github/workflows/tests.yaml` runs on every push to non-main branches,
  installs `core/requirements.txt` on Python 3.12, and calls the same pytest command
  as above. **When you add a new test file, add it to that command.**
- Conventions seen in existing tests: `pytest` fixtures + `unittest.mock` to stub
  Redis; docstring-style comments at the top of test files documenting how to run them.

## Conventions & gotchas

- Import style is inconsistent, and it works-leave it as-is: within `core/` modules import
  siblings as `from classes.X import ...` (works because the app runs with cwd `/app`
  in the container), while tests import as `from core.classes.X import ...`. Both
  resolve as namespace packages; preserve the existing style when editing.
- LLM responses may contain internal "thinking" reasoning blocks (open/close think-tags
  with an optional tab after the bracket). All stripping/regex logic lives in
  `core/classes/response_filter.py` — keep it pure, and cover new behaviour in
  `core/tests/response_filter_tests.py`.
- The free OpenAI Moderations endpoint is aggressively rate-limited (HTTP 429):
  `content_guard.py` retries 429/5xx with backoff, caches verdicts per input, and
  fails open when it cannot get an answer. Tunables are documented at the top of
  that module and in `.env.example`.
- `wrap(..., break_long_words=False)` silently drops whitespace-less runs longer than
  the chunk size (2000 chars) in `handle_message_send`; be aware when changing chunking.
- llama.cpp has no pull API: the `llamacpp` container downloads the model itself on boot
  (`LLAMA_ARG_HF_REPO` into the `LLAMA_CACHE` volume). Changing the model therefore requires
  restarting the server (`docker compose up -d` after editing `MODEL` in `.env`; `helm upgrade`
  in k8s) — `compose restart` alone does not re-read `.env`. On startup `main.py` verifies
  readiness by GETting `{LLM_HOST}/v1/models` and checking the configured `MODEL` is listed
  (`TextLLMHandler.check_model_ready`, fail-soft — a first-boot model may still be downloading).
- The chart's LLM PersistentVolume/Claim are still named `*-pvc-ollama` / `ollama-pv-claim`
  (hostPath `…/ollama`) on purpose, so data survives the Ollama → llama.cpp switch and keeps
  matching on upgrade — do not rename; the llamacpp pod mounts it at `/models`, which is also
  its `LLAMA_CACHE`, so the GGUF model (downloaded once via `--hf-repo`) persists across redeploys.
- The diffusion pod, its PVC and the `generate_image` tool are all gated by one switch:
  `diffusion.enabled` in the chart (→ `IMAGE_GEN_ENABLED` in the configmap). The service
  downloads its model into the `diffusers` volume on first boot, so first start is slow
  (the readiness probe on `/health` allows ~15 min); changing `IMAGE_MODEL` needs a pod
  restart (same as llamacpp: `compose restart`/`helm upgrade` re-uses the cached model,
  a new one is downloaded into the volume).
- Never commit `.env`; copy `.env.example` and fill in locally.
