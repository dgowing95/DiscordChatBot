# AGENTS.md

Overview of this repository and how to work with it (for humans and AI coding agents).

## What this is

A Python Discord bot that answers messages using a local LLM (llama.cpp server serving a GGUF model, e.g. `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`).
It is primarily deployed on **Kubernetes** via the Helm chart in `charts/dis-ai-bot`
(releases are cut by `.github/workflows/auto-tag.yaml`, which bumps a `vMAJOR.MINOR` tag on every
push to `main`, and the tag push starts `release.yaml`; a prebuilt chart is downloadable from GitHub
releases -- see Releasing below).
Redis is used as the settings store and user-memory store.

## Repository layout

```
core/                  # the main bot (the app that runs in production)
  main.py              # entrypoint: discord.Client, message queue, slash commands
  classes/
    message_handler.py     # per-message orchestration: history build, send/chunking
    message_queue.py       # PURE (stdlib-only) queue sizing (WORKER_COUNT / QUEUE_MAX_SIZE),
                           #   bounded queue factory + per-channel locks (scoped to build+send)
                           #   + in-flight task registry (prompt hint for still-running slow tools)
    text_llm_handler.py    # builds an `agents` Agent against the LLM server's (llama.cpp) OpenAI-compat API
    response_filter.py     # PURE (stdlib-only) response cleaning / thinking-block stripping
    content_guard.py       # OpenAI Moderations-based safety guard for web_search / fetch_url /
                           #   run_code_sandbox
    llm_config.py          # PURE (stdlib-only) MODEL / LLM_HOST / LLM_PASS defaults and the
                           #   temperature parser, shared by the main and sandbox agents
    redis_client.py        # the two shared, lazily-built Redis clients (str + binary); the
                           #   stores below hold no client of their own
    metrics.py             # PURE (stdlib + prometheus_client) Prometheus metrics:
                           #   all metric definitions + /metrics HTTP server (METRICS_PORT)
    user_memory.py         # JSON lists in Redis per (guild, user)
    config_manager.py      # per-guild settings in Redis (system prompt, temperature, ...)
    tool_functions.py      # agent function tools: web_search, fetch_url, memory tools, generate_image, run_code_sandbox
    image_generation.py    # client for the diffusion service + IMAGE_GEN_ENABLED flag
    image_prompt.py        # LLM rewrite of an image request -> SDXL prompt + negative prompt
    sandbox_agent.py       # nested SandboxAgent + run_sandbox_task (throwaway Docker sandbox)
                           #   + the pure builders for what the outer model is told
                           #   (sandbox_tool_result and friends)
    sandbox_progress.py    # streams sandbox commands/output to one edited Discord message
                           #   (single embed: one field per command, state-coloured)
    sandbox_snapshot_store.py  # Redis-backed workspace snapshots, keyed by thread id
    sandbox_thread_inbox.py    # routes thread messages to a sandbox run in flight
    common.py              # shared helpers (Discord tool embeds)
  tests/               # pytest suite (see Testing below)
  Dockerfile           # python:3.13-slim image, runs main.py
  requirements.txt     # runtime only
  requirements-dev.txt # the above plus pytest et al (what CI installs)
diffusionservice/      # standalone image service (text->image; FastAPI + diffusers,
                       #   queued single-worker, sd-turbo by default,
                       #   CPU-offloaded for low VRAM)
                       #   generation_params.py is the stdlib-only half (guidance /
                       #   negative-prompt policy), so it is unit-testable without torch
charts/dis-ai-bot/     # Helm chart (credentials render into templates/secret.yaml,
                       #   everything else into templates/configmap.yaml)
pyproject.toml         # pytest configuration - why bare `pytest` works from the repo root
docker-compose.yaml    # local dev: redis + llamacpp (GPU, llama.cpp) + diffusion (GPU) + core (mounts ./core)
.env / .env.example    # environment configuration (never commit .env)
```

## Runtime architecture

1. `main.py:` every Discord message is first checked by `should_handle_message()`
   in `on_message` (has content/embeds/attachments, not from the bot, not
   `!reset_history`, and either the bot is mentioned or the per-guild random
   reply-chance roll hits); only messages that pass go on a BOUNDED
   `asyncio.Queue` (`QUEUE_MAX_SIZE`, default 10 — once full, new messages are
   dropped and a mention gets a short "busy" reply), and a POOL of worker
   tasks (`WORKER_COUNT`, default 2) pops messages, builds a `MessageHandler`,
   and handles them. A per-channel `asyncio.Lock` (keyed by channel id) is
   SCOPED to the two fast phases of a handle — prompt build and the chunked
   send (`MessageHandler.handle_message`) — so the slow LLM/tool phase runs
   UNLOCKED: different channels run concurrently, AND a free worker can
   answer a NEW message in the SAME channel while the first is stuck in a
   slow tool (no interleaved chunks, consistent `channel.history()`
   snapshots, no deadlock — the lock is never held across an LLM/tool
   await). While a slow tool (sandbox / image gen) is running,
   `ToolMetricsHooks` registers it in the per-channel IN-FLIGHT REGISTRY
   (`classes/message_queue.py`) and any newer same-channel message's prompt
   gets a one-line hint ("🐳 code sandbox running for 4m 12s — <task>").
   Sizing, the queue factory, the lock registry and the in-flight registry
   live in `core/classes/message_queue.py` (pure, unit-tested).
2. `MessageHandler.handle_message()` builds the prompt
   (channel history -- most recent `MSG_HISTORY_LIMIT` (default 5) messages; the
   user's stored Redis memories are exposed to the agent through its function
   tools) and calls `TextLLMHandler.generate()`.
3. `TextLLMHandler` uses the **OpenAI `agents` SDK** pointed at the llama.cpp
   server's OpenAI-compatible endpoint (`LLM_HOST/v1`) with function tools attached.
4. The returned text is cleaned by `MessageHandler.filter_response()` (delegate:
   `core/classes/response_filter.py`, a pure module) and sent in **2000-char chunks**
   (`textwrap.wrap`, one `asyncio.sleep(1)` between sends). The model's reasoning
   is kept out of that answer and, by default, dropped; with `SHOW_THINKING=1` it is
   sent as follow-up message(s) wrapped in a spoiler-hidden code block
   (`||```...```||`, closed by default — click to reveal), chunked the same way
   (`response_filter.format_thinking_for_discord`, capped at
   `MAX_THINKING_CHUNKS` messages — a tool-calling run reasons on every turn).
   **The reasoning does not travel in the answer string.** With llama.cpp's
   default `--reasoning-format auto` and a thinking-enabled template, the server
   returns it out of band in `reasoning_content`; the SDK turns that into its own
   `reasoning_item` in `RunResult.new_items`, so `final_output` is already clean
   and there are no `<think>` tags left to find. `TextLLMHandler.generate()`
   therefore collects it with `response_filter.extract_reasoning_items(new_items)`
   (all turns, in order) and exposes it as `self.reasoning` for `MessageHandler`
   to send — `generate()` still returns the plain answer string, so its `"Error"`
   sentinel is unchanged. `extract_thinking()` on the answer text remains as a
   fallback for a server running `--reasoning-format none`, which does inline the
   tags. **The reasoning also survives a failed run**: when `Runner.run` raises
   (`MaxTurnsExceeded` after a reply chains several tool calls is the common one)
   `generate()` returns the `"Error"` sentinel as before, but first recovers the
   reasoning from the completed turns the SDK hangs off the exception
   (`AgentsException.run_data.new_items`), and `MessageHandler` sends it after the
   ❌. Without that, a failed tool run left the tool's embeds and files in the
   channel — posted during the unlocked phase, before the failure — with no answer
   and no reasoning, which reads as the bot going quiet mid-task.
5. Per-guild settings live in Redis under the `dcb` namespace; per-user memories under
   `guild:<id>:user:<id>`.
6. Image generation: when enabled (`IMAGE_GEN_ENABLED`, set from the chart's
   `diffusion.enabled`), the agent gets a `generate_image(prompt)` tool plus a
   `/generate_image <prompt>` slash command (registered in `main.py`). Both POST
   to the standalone diffusion service (`DIFFUSION_URL/generate`), which runs in
   its own pod/container, queues requests (one image at a time) and replies with
   a PNG that is sent to the Discord channel. It is text-to-image only; there is
   no image-editing path. Generation settings (`IMAGE_MODEL`, `IMAGE_STEPS`,
   `IMAGE_WIDTH`/`HEIGHT`, `IMAGE_GUIDANCE`, `IMAGE_NEGATIVE_PROMPT`,
   `IMAGE_LONG_PROMPT`, `IMAGE_OFFLOAD`, `IMAGE_QUEUE_SIZE`) live in the same
   configmap/env the diffusion pod reads. Both paths first send the request
   through `image_prompt.build_image_prompt` — one LLM call that rewrites it
   into an SDXL-shaped prompt plus a negative prompt, and falls back to the
   request verbatim on any failure. That is why the SDXL prompt rules are NOT
   in the `generate_image` docstring: the slash command never reads it.
   In the service, prompts longer than CLIP's 77-token window are encoded in
   chunks by compel rather than truncated, and distilled models (sd-turbo and
   friends) are pinned to `guidance_scale=0.0` with the negative prompt
   dropped, since diffusers skips the unconditional branch below CFG 1.
7. Code sandbox: when enabled (`SANDBOX_ENABLED`, from the chart's
   `sandbox.enabled`), the agent gets a `run_code_sandbox(task)` tool (no slash
   command). It runs a nested `SandboxAgent` in a THROWAWAY Docker container via
   `agents.sandbox.DockerSandboxClient` and returns that agent's own report.
   The core container needs the Docker daemon socket mounted (compose: a bind
   mount; chart: a hostPath gated on `sandbox.enabled`) plus the `docker` and
   `websocket-client` packages. Tasks go through the content guard first, and
   `SANDBOX_MAX_TURNS`/`SANDBOX_TIMEOUT` bound each run.

   How the pieces fit:

   | Concern | Where it lives |
   |---|---|
   | Orchestration: thread, guard, progress, artifacts, inbox lifecycle | `tool_functions.run_code_sandbox` |
   | What the outer model is told about a finished run | `sandbox_agent.sandbox_tool_result` (pure) |
   | Nested agent, its prompt, session/container lifecycle, artifact selection | `sandbox_agent.py` |
   | Live progress streamed into one edited message | `sandbox_progress.py` |
   | Workspace snapshots in Redis, keyed by thread id | `sandbox_snapshot_store.py` |
   | Mid-run steering from thread messages | `sandbox_thread_inbox.py` |

   The behaviour worth knowing before you change anything:

   - **Every call runs in a Discord thread.** `ensure_sandbox_thread` creates one
     off the triggering message (or reuses the thread you are already in), and
     everything — embeds, files, previews, questions — goes there.
   - **The container is disposable; the workspace is not.** In a thread, the
     workspace is persisted to Redis before teardown and restored on the next
     call in that SAME thread. Resume is thread-local by construction: the
     snapshot id IS the thread id, so asking anywhere else starts fresh.
   - **A run can be steered while it happens.** Messages posted in an active
     thread are routed to the sandbox instead of the outer LLM (📨 acknowledges
     receipt), and reach the model on the back of its next shell result.
   - **The sandbox agent chooses what is delivered** (`attach_file`) and writes
     the closing message the user reads. The outer model adds at most a
     sentence.
   - **A stopped run is never told to retry.** Timeout, max-turns and
     model-error each become `SandboxResult(ok=False, error=...)`, artifacts are
     recovered from the still-live container, and the model is pointed at a
     follow-up in the thread instead — which resumes rather than starting over.

   Every one of these carries its full rationale — the production incident that
   produced it, and what breaks if it is undone — in the docstring of the
   function that implements it. Read those before editing; they are the primary
   source, and this list is only a map. Start with `sandbox_agent.py`'s module
   docstring, then `run_sandbox_task`, `_deliver` and `sandbox_tool_result`.

### Prompt surface

   Everything the models are told lives in code, in three places: the
   `@function_tool` docstrings in `tool_functions.py` (what the outer model
   sees), each tool's return strings (read at reply-writing time — these are
   the ones that actually change behaviour), and `SANDBOX_INSTRUCTIONS` +
   its bullets in `sandbox_agent.py` (what the nested sandbox model sees).
   The outer agent's *system* prompt is not a lever: it is entirely
   `f"Answer as if you are {redis['dcb:{guild}:system']}"`, user-owned via
   `/system` and the `change_personality` tool.

   **One home per concept.** Both prompts grew by accretion — every observed
   bug fixed by adding text, none ever removed — until `SANDBOX_INSTRUCTIONS`
   was ~5000 characters with `out/` explained in five bullets and two of them
   flatly contradicting each other. Duplicated guidance is worse than terse
   guidance on a small local model: copies drift, and a model handed two
   versions of a rule follows neither reliably. So when adding guidance, edit
   the existing home rather than restating it nearby — the comment block above
   `SANDBOX_INSTRUCTIONS` records which bullets were merged and why.
   Sizes worth re-measuring before adding: `SANDBOX_INSTRUCTIONS` ~3700 chars
   (pinned by a test), `run_code_sandbox` ~1750 including its JSON schema.

   **Prefer a positive example over a list of prohibitions.** The design-
   ownership rule once carried a 350-character negative list ("no dimensions,
   colours or RGB values, no frame counts, no library choices…"); the model
   then retried a failed run with an invented canvas size, frame count and
   library — three items straight off that list. It now shows the shape
   instead: `task="Generate a gif of a cow doing a backflip."`

### Environment variables

| Var | Purpose |
|---|---|
| `DISCORD_TOKEN` | required; bot token |
| `REDIS_HOST` | required; Redis host |
| `LLM_HOST` | llama.cpp server base URL (OpenAI-compat; core appends `/v1`). Points at the `llamacpp` service on :8081 in docker-compose (dev) and :8080 in the helm chart; the in-code fallback is `http://llamacpp:8080` |
| `LLM_PASS` | placeholder key — llama.cpp does not authenticate, but the OpenAI client requires a non-empty key |
| `MODEL` | the model, single source of truth: the bot requests this name AND the compose `llamacpp` service serves it (as `LLAMA_ARG_HF_REPO`); in the Helm chart the one `model` value feeds both. In-code default `qwen3:4b` |
| `OPENAI_API_KEY` | API key for the free OpenAI Moderations endpoint (web-tool guard); fail-open if unset |
| `OPENAI_BASE_URL` | moderation API base (default `https://api.openai.com/v1`); any OpenAI-compatible `/v1/moderations` endpoint works. Chart: `openaiBaseUrl` |
| `MODERATION_MODEL` | moderation model name, e.g. `text-moderation-latest`; unset omits the parameter and uses the server default. Chart: `moderationModel` |
| `CONTENT_GUARD_ENABLED` | `0`/`false` disables the content guard on web tools (default: on) |
| `CONTENT_GUARD_DEBUG` | `0`/`false` silences content-guard debug logging (default: on) |
| `METRICS_PORT` | port to serve the Prometheus `/metrics` endpoint on (default 9464); empty/`0` disables. Chart: `metrics.enabled`/`metrics.port` also add a ClusterIP Service, the pod port, and a kube-prometheus-stack ServiceMonitor (labelled `release: kube-prometheus-stack` — the operator only imports ServiceMonitors with that label) |
| `LOG_LEVEL` | root log level (default `INFO`); `DEBUG` also dumps the raw agent run result. Logging replaced bare `print()` calls, which had no level to tune |
| `MSG_HISTORY_LIMIT` | how many prior channel messages to include, default 5. Chart: `message_history` |
| `REASONING_EFFORT` | sent to the LLM as the OpenAI-compat `reasoning_effort` field (low/medium/high, default medium). Chart: `reasoningEffort` |
| `LLM_MAX_TURNS` | max model turns for ONE reply from the main agent (helm: `llmMaxTurns`), default 20. A turn is one model response, however many tool calls it carries. Passed explicitly to `Runner.run` because the SDK's own default of 10 is easily overrun by a reply that chains several sandbox/image calls — and overrunning raises `MaxTurnsExceeded`, which costs the whole answer |
| `SHOW_THINKING` | `1`/`true` sends the model's reasoning as spoiler-hidden follow-up message(s); default (off) drops it entirely. Chart: `showThinking` |
| `WORKER_COUNT` | queue worker tasks (default 2, min 1); each handles one message at a time, a per-channel lock keeps same-channel order. Chart: `worker_count` |
| `QUEUE_MAX_SIZE` | max messages waiting on the bounded queue (default 10, min 1); when full new messages are dropped (a mention gets a short "busy" reply). Chart: `queue_max_size` |
| `LLAMA_ARG_CACHE_TYPE_K`, `LLAMA_ARG_CACHE_TYPE_V` | optional; compose `llamacpp` service only: KV cache quantization type (llama.cpp `-ctk`/`-ctv`), default `q4_0`; in the Helm chart set via `llamacpp.cacheTypeK`/`cacheTypeV` |
| `IMAGE_GEN_ENABLED` | `0`/`false` removes the `generate_image` tool from the LLM (default: on). Chart: `diffusion.enabled` also removes the diffusion pod/PVC |
| `DIFFUSION_URL` | base URL of the diffusion service (core appends `/generate`); compose `diffusion` service on :8000 in dev, in-cluster `*-diffusion-service` in the chart; in-code fallback `http://diffusion:8000` |
| `IMAGE_MODEL` | HF repo id for the diffusion service (default `stabilityai/sd-turbo` — smallest practical model); the service downloads it into its `HF_HOME` volume on first boot |
| `IMAGE_STEPS` / `IMAGE_WIDTH` / `IMAGE_HEIGHT` | generation settings for the diffusion service (defaults: 4 steps, 512x512) |
| `IMAGE_GUIDANCE` | CFG scale; unset leaves the pipeline's own (7.5 SD1.5 / 5.0 SDXL). Forced to 0.0 on distilled models. Chart: `diffusion.guidance` |
| `IMAGE_NEGATIVE_PROMPT` | baseline negative prompt, merged BEHIND the per-request one the rewriter produces, and the only one left when that rewrite is off or falls soft. Compose and the chart ship the same non-empty default; dropped automatically on distilled models. Chart: `diffusion.negativePrompt` |
| `IMAGE_LONG_PROMPT` | `0`/`false` reverts to truncating prompts at 77 CLIP tokens instead of chunk-encoding them (default: on). Chart: `diffusion.longPrompt` |
| `IMAGE_OFFLOAD` | `model` (default: one pipeline component on GPU at a time, text encoder in CPU RAM) / `sequential` (lowest VRAM, slowest) / `none` (all on GPU) |
| `IMAGE_QUEUE_SIZE` | max queued image requests in the diffusion service (default 16); over that it returns 503 |
| `IMAGE_GEN_TIMEOUT` | seconds core waits on the diffusion service (default 300) |
| `IMAGE_PROMPT_REWRITE_ENABLED` | `0`/`false` sends image requests to the service verbatim instead of rewriting them first (default: on). Chart: `diffusion.promptRewrite.enabled` |
| `IMAGE_PROMPT_MODEL` / `IMAGE_PROMPT_LLM_HOST` / `IMAGE_PROMPT_LLM_API_KEY` / `IMAGE_PROMPT_TIMEOUT` | the prompt rewrite's own LLM connection; each falls back to the bot's `MODEL` / `LLM_HOST` / `LLM_PASS` (timeout default 60s), exactly like the `SANDBOX_*` equivalents. Chart: `diffusion.promptRewrite.*` (the key via secret.yaml) |
| `SANDBOX_ENABLED` | `0`/`false` removes the `run_code_sandbox` tool from the LLM (default: on). Chart: `sandbox.enabled` also removes the Docker-socket hostPath mount |
| `SANDBOX_IMAGE` | container image for the sandbox workspace, pulled once onto the daemon (default `python:3.14-slim`). Chart: `sandbox.image` |
| `SANDBOX_MAX_TURNS` | max model turns for one sandbox task (default 10). Chart: `sandbox.maxTurns` |
| `SANDBOX_MODEL` | model id for the nested sandbox agent; empty (default) = the main bot's `MODEL`. Chart: `sandbox.model` |
| `SANDBOX_LLM_HOST` | base URL of the sandbox agent's LLM (core appends `/v1`); empty (default) = the main `LLM_HOST`. E.g. `https://openrouter.ai/api` for OpenRouter. Chart: `sandbox.llmHost` |
| `SANDBOX_LLM_API_KEY` | API key for the sandbox agent's LLM; empty (default) = the main `LLM_PASS` placeholder. Chart: `sandbox.apiKey` |
| `SANDBOX_ASK_USER_TIMEOUT` | max seconds the sandbox's `ask_user` tool waits for a reply in its thread before telling the model to proceed on its own (default 300); also clamped to whatever of the run's own `SANDBOX_TIMEOUT` budget remains. Chart: `sandbox.askUserTimeout` |
| `SANDBOX_PERSIST_TIMEOUT_SECONDS` | seconds allowed to persist a thread's workspace snapshot to Redis on container teardown, after `SANDBOX_TIMEOUT` has already elapsed (default 180 — generous since the `Memory` capability's own extraction runs here too). Chart: `sandbox.persistTimeout` |
| `SANDBOX_REQUEST_TIMEOUT_SECONDS` | seconds of silence on one HTTP request to the sandbox's LLM before the client gives up (default 180, down from the OpenAI client's unstated 600). This is httpx's per-read timeout: it catches a hung connection or a server that sends nothing, but NOT one that dribbles keep-alive padding while it works (OpenRouter pads non-streaming responses), which stays bounded only by `SANDBOX_TIMEOUT`. Chart: `sandbox.requestTimeout` |
| `SANDBOX_MAX_RETRIES` | how many times that client retries a failed request (default 2; 0 disables). Worst-case latency for one model call is (1 + this) x `SANDBOX_REQUEST_TIMEOUT_SECONDS`. Chart: `sandbox.maxRetries` |
| `SANDBOX_SNAPSHOT_MAX_BYTES` | max size of one thread's stored workspace snapshot in Redis (default 50MB). Chart: `sandbox.snapshotMaxBytes` |
| `SANDBOX_SNAPSHOT_TTL_SECONDS` | how long an unused thread's workspace snapshot survives in Redis (default 604800 = 7 days). Chart: `sandbox.snapshotTtlSeconds` |

## Releasing

- Every push to `main` runs `auto-tag.yaml`, which bumps the highest `vMAJOR.MINOR`
  tag by one minor and pushes it. That tag push is what starts `release.yaml`
  (tests -> two images -> chart -> GitHub release). Nothing dispatches
  `release.yaml`; its `workflow_dispatch` trigger is a manual escape hatch for
  re-running a release against an existing tag.
- **`TAG_PUSH_TOKEN` (repo secret) is what pushes the tag**, and it expires.
  `GITHUB_TOKEN` cannot do the job for two independent reasons: the "Restrict
  Tagging" ruleset allows ref creation only for repo admins, and it acts as the
  GitHub Actions app rather than a user; and a ref pushed with it deliberately
  fires no `push` event, so `release.yaml` would never start.
- To mint a replacement: your **account** settings (not the repo's) → Developer
  settings → Personal access tokens → Fine-grained tokens → Generate new token,
  scoped to `dgowing95/DiscordChatBot` only, with **Repository permissions →
  Contents: Read and write** (nothing else; `workflow` is not needed, the job
  pushes a tag and never touches `.github/workflows`). Then store it in the
  **repo's** settings → Secrets and variables → Actions, as `TAG_PUSH_TOKEN`.
- How the Auto Tag job fails tells you which half is wrong. Secret unset: the
  checkout step itself fails, because `token:` is a required input and an unset
  secret makes it empty rather than falling back to `GITHUB_TOKEN`. PAT expired
  or revoked: an authentication failure fetching the repo. `remote: error:
  GH013 ... Cannot create ref due to creations being restricted` on the push:
  the token is valid but its owner is not a bypass actor on the ruleset.

## Testing

- **Framework:** `pytest`, configured in `pyproject.toml` at the repo root
  (`python_files = *_tests.py` — the suite predates the default `test_*.py`
  convention — plus `testpaths` and `pythonpath`).
- **Prereq:** a Python 3.13+ venv with `pip install -r core/requirements-dev.txt`
  (that pulls in `core/requirements.txt` and adds the test-only packages;
  `response_filter` tests are pure-stdlib).
- **How to run** — from the repo root, no arguments and no `PYTHONPATH`:

  ```bash
  pytest
  ```

  This is exactly what CI runs, so a new test file is picked up automatically;
  there is no list to keep in step. ~460 tests, roughly 15 seconds.

- Tests import production modules as `classes.X`, the same name the app uses
  (it runs with cwd `/app`), and `pyproject.toml` puts `core/` on the path to
  make that resolve. Do NOT import them as `core.classes.X`: that resolves as a
  separate namespace package, giving a SECOND module object with its own
  globals, so a patch applied to one copy leaves the other untouched and any
  import-time state (metric registration, the channel-lock and in-flight
  registries) exists twice.
  `grep -rn "core\.classes\|core\.tests" core/` should stay empty.
- One test module may borrow a helper from another (message_handler_tests
  reuses response_filter_tests' reasoning-item builders), but import it by its
  BARE name -- `from response_filter_tests import _reasoning_item`. pytest puts
  core/tests on sys.path; the repo root only lands there when something else
  puts the cwd there, so a `core.tests.X` import resolves under `python -m
  pytest` and IDE runners but raises ModuleNotFoundError under bare `pytest`,
  locally and in CI alike.
- Nothing in the suite may write to `os.environ` directly — use `monkeypatch`.
  A bare `os.environ.setdefault("REDIS_HOST", "localhost")` in a test helper
  leaked process-wide and made every later test that touched `configManager`
  block on a real Redis connect timeout, which is what made the full suite
  appear to hang.

### Manual/live testing via a Discord webhook

Beyond pytest, the bot can be driven end-to-end (tool calls, sandbox runs,
image generation) against a real, running deployment by POSTing a message
through a Discord webhook — no real Discord account/client needed:

```bash
curl -sS -X POST "$TEST_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "<@BOT_USER_ID> your test message here"}'
```

- The bot only reacts to messages that pass `should_handle_message()`
  (`core/main.py`), so the content needs a real `<@id>` mention of the
  bot's Discord user ID — Discord parses `mentions` from that numeric ID
  in the content itself, regardless of who/what posted the message, so a
  plain `@botname` text string does nothing. Get the ID from Discord
  (right-click the bot → Copy User ID).
- `TEST_WEBHOOK_URL` (`.env`, local dev only) holds a webhook for the dev
  server/channel this points at. It is **not read by the app** — it's a
  standing convenience for curling test messages so it doesn't need
  rediscovering each session.
- Watch it process: `docker compose logs core -f` (local dev) or
  `kubectl -n <namespace> logs deployment/<release>-core-deployment -f`
  (a cluster deploy). `core`'s source is bind-mounted in docker-compose
  (`./core:/app`), so `docker compose restart core` alone picks up local
  code edits — no rebuild needed.
- **Never** commit a webhook URL, and never point `TEST_WEBHOOK_URL` (or
  any webhook pasted into a session) at a production channel — a webhook
  post is indistinguishable from real user traffic once it lands.

- Keeping a module **pure and importable without the discord/agents SDKs** (like
  `response_filter.py`) is the intended pattern for anything you want to unit test —
  `MessageHandler` itself drags in `discord`, `agents`, Redis, etc.
- **CI:** `.github/workflows/tests.yaml` runs `pytest` on every push, and
  `release.yaml` runs it again as a gate the image/chart jobs depend on
  (releases are cut straight off a push to `main`, so this is the only place
  the shipped commit is tested). New test files need no CI change.
- Conventions seen in existing tests: `pytest` fixtures + `unittest.mock` to stub
  Redis; docstring-style comments at the top of test files documenting how to run them.

## Conventions & gotchas

- There is ONE import name: `from classes.X import ...`, in production code and in
  tests alike (the app runs with cwd `/app`; `pyproject.toml` puts `core/` on the
  test path). It used to be inconsistent — tests used `core.classes.X` — and because
  the two resolve to separate module objects with separate globals, `metrics.py` and
  `message_queue.py` each needed a `sys.modules` aliasing hack to keep their
  import-time state from being created twice, and the sandbox tests had to know which
  copy to patch. Don't reintroduce the second name.
- A reasoning model delivers its thinking in one of two shapes: out of band in
  `reasoning_content` (llama.cpp's default — becomes a `reasoning_item` on the run
  result), or inline as open/close think-tags with an optional tab after the
  bracket. Both are handled in `core/classes/response_filter.py` — keep it pure
  (stdlib only; `extract_reasoning_items` duck-types the SDK's run items rather
  than importing them) and cover new behaviour in
  `core/tests/response_filter_tests.py`.
- The queue worker pool, bounded-queue sizing (WORKER_COUNT / QUEUE_MAX_SIZE),
  the per-channel locks (SCOPED to build+send — the LLM/tool phase runs
  unlocked) and the in-flight task registry (register_task_run /
  in_flight_hint) live in `core/classes/message_queue.py` — keep it pure
  (stdlib only). Cover changes to the concurrency model in
  `core/tests/message_queue_tests.py` (which also tests the `on_message` /
  `process_messages` wiring in `main.py`, imported directly — `main.py` guards
  `client.run()` behind `if __name__ == "__main__"` and builds its Redis client
  lazily, so importing it starts nothing and needs no environment), the registry itself in
  `core/tests/task_registry_tests.py`, the scoped-lock behaviour of
  `MessageHandler.handle_message` (concurrent generations, serialized sends,
  prompt hint) in `core/tests/message_handler_tests.py`, and the slow-tool
  registration in `ToolMetricsHooks` in `core/tests/metrics_tests.py`.
- The free OpenAI Moderations endpoint is aggressively rate-limited (HTTP 429):
  `content_guard.py` retries 429/5xx with backoff, caches verdicts per input, and
  fails open when it cannot get an answer. Tunables are documented at the top of
  that module and in `.env.example`.
- `wrap(..., break_long_words=False)` does not drop a whitespace-less run longer than
  the chunk size - it returns it as one OVERSIZED chunk, which Discord then rejects
  with `HTTPException`, unwinding past `handle_message()` and costing the whole reply.
  Both send paths therefore go through `response_filter.chunk_for_discord`, which
  hard-splits anything still over the limit. Use it rather than calling `wrap` directly.
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
