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
    history_policy.py      # PURE (stdlib-only) which history goes in a prompt: sliding vs
                           #   anchored window, per-channel anchor store, token budget
    attachment_cache.py    # PURE (stdlib-only) encoded images kept between prompt builds:
                           #   LRU + TTL, shared in-flight loads, delete/edit invalidation
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
    image_generation.py    # create_image (prompt, generate, check, retry), diffusion
                           #   client, /health capabilities, which image to edit
    image_prompt.py        # LLM rewrite of an image request -> SDXL prompt + negative prompt
    image_review.py        # vision check of a generated image: what it shows, what's missing
    side_llm.py            # shared client + thinking-off latch for those two side calls
    sandbox_agent.py       # nested SandboxAgent + run_sandbox_task (throwaway Docker sandbox)
                           #   + the pure builders for what the outer model is told
                           #   (sandbox_tool_result and friends)
    sandbox_progress.py    # streams sandbox commands/output to one edited Discord message
                           #   (single embed: one field per command, state-coloured)
    sandbox_snapshot_store.py  # Redis-backed workspace snapshots, keyed by thread id
    sandbox_thread_inbox.py    # PURE (stdlib-only) ledger of thread messages to a run in
                               #   flight: staged events, atomic per-thread claim
    sandbox_conversation.py    # conversation coordinator: model-input filter, tool gate,
                               #   respond_to_updates / ask_user, continuation nudges
    sandbox_conversation_store.py  # per-thread record of loose ends (late messages,
                               #   open question) for the next run, in Redis
    common.py              # shared helpers (Discord tool embeds, embed_from_data)
    whats_new.py           # PURE (stdlib-only) What's New notes: parse core/whats_new.md,
                           #   APP_VERSION / WHATS_NEW_ENABLED, embed data + Discord limits
    help_catalog.py        # PURE (stdlib-only) the hand-written /help listing
    automation_policy.py   # PURE (stdlib-only) schedule timing, DST, minimum gap,
                           #   rule matching, AUTOMATIONS_* / SCHEDULE_* / RULE_* settings
    automation_store.py    # versioned Redis records, quotas, revisions, atomic claims
    automation_runner.py   # rule admission, 15s schedule poll, automatic run execution
    automation_commands.py # /schedule and /rule slash command groups
    automation_tools.py    # the matching chat tools (list/get/create/update/delete)
  whats_new.md         # the NEXT release's user-facing features (see "What's New
                       #   notes" below) - rewritten by every feature PR
  tests/               # pytest suite (see Testing below)
  Dockerfile           # python:3.13-slim image, runs main.py
  requirements.txt     # runtime only
  requirements-dev.txt # the above plus pytest et al (what CI installs)
diffusionservice/      # standalone image service (text->image; FastAPI + diffusers,
                       #   queued single-worker, sd-turbo by default,
                       #   CPU-offloaded for low VRAM)
                       #   generation_params.py is the stdlib-only half (guidance /
                       #   negative-prompt policy), so it is unit-testable without torch
docs/automations.md    # user-facing guide to schedules and rules
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
   (channel history -- the messages BEFORE the triggering one, up to
   `MSG_HISTORY_LIMIT` (default 5) including the trigger; the user's stored
   Redis memories are exposed to the agent through its function tools) and
   calls `TextLLMHandler.generate()`. History is read before the trigger's id,
   not as "newest N minus one", because a trigger that waited in the queue has
   newer messages above it. `MSG_HISTORY_MODE=anchored` keeps the start of the
   window fixed between refreshes so llama.cpp can reuse its cached prompt
   prefix (`classes/history_policy.py` has the rules and why); every message,
   the trigger included, is formatted by the one `_format_group`, so a trigger
   serializes exactly as it will as history on the next turn. Older history is
   dropped whole when the prompt would not fit the llama.cpp per-slot context
   (read from its `/props`) minus `MSG_HISTORY_RESERVE_TOKENS`.
   The build has two halves: `select_messages` (under the channel lock: reads
   Discord and picks the window, estimating a flat token cost per image) and
   `prepare_messages` (after the lock: downloads and encodes the images). Encoded
   images are kept in `classes/attachment_cache.py` (64 MiB, 15 minutes, keyed by
   channel/message/attachment id, not URL), at most 4 downloads run at once
   through one shared aiohttp session, and Discord's raw delete/edit events in
   `main.py` drop a message's images; a history message deleted while images
   download is left out of the prompt.
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
   `diffusion.enabled`), the agent gets a `generate_image(prompt, edit_previous)`
   tool plus a `/generate_image <prompt>` slash command (registered in
   `main.py`). Both go through `image_generation.create_image`, which POSTs to
   the standalone diffusion service (`DIFFUSION_URL/generate`); that runs in its
   own pod/container, queues requests (one image at a time) and replies with a
   PNG. Generation settings (`IMAGE_MODEL`, `IMAGE_STEPS`, `IMAGE_WIDTH`/`HEIGHT`,
   `IMAGE_GUIDANCE`, `IMAGE_NEGATIVE_PROMPT`, `IMAGE_LONG_PROMPT`,
   `IMAGE_OFFLOAD`, `IMAGE_QUANTIZE`, `IMAGE_QUEUE_SIZE`) live in the same
   configmap/env the diffusion pod reads.

   The service handles two model families (`generation_params.pipeline_family`,
   read from the repo's `model_index.json`), and reports which on `/health` as
   `prompt_style` and `edits`; core reads that through
   `image_generation.service_capabilities` (only a successful read is cached —
   `/health` is 503 for minutes while a model loads):

   | | SD 1.5 / SDXL (e.g. Juggernaut-XL) | FLUX.2 [klein] 4B (prod) |
   |---|---|---|
   | Prompt | `image_prompt.build_image_prompt` rewrites it into an SDXL-shaped prompt + negative prompt (falls back to the request verbatim on any failure) | sent as written: its Qwen3 text encoder reads sentences |
   | Long prompts | chunk-encoded by compel (`CompelForSDXL`) past CLIP's 77 tokens | n/a |
   | Editing | no | yes: a `reference_image`, the tool's `edit_previous` |
   | Distilled pin | `guidance_scale=0.0`, negative prompt dropped | `guidance_scale=1.0` |
   | 8 GB card | fp16 + model CPU offload | `IMAGE_QUANTIZE=nf4` + model CPU offload; text encoder computes fp32 where the card lacks bf16 |

   The rewrite lives in core, not the `generate_image` docstring, because the
   slash command never reads that docstring.

   **Every image is looked at before it is posted** (`image_review.py`): one
   vision call to the bot's own LLM (llama.cpp has the model's vision projector
   loaded) returns what the picture shows and what was asked for but is
   missing. A miss earns one retry with those items stressed
   (`IMAGE_REVIEW_RETRIES`), the better attempt is posted, and the tool's return
   string tells the outer model what the posted image actually shows. Before
   this the bot captioned images from its own prompt, and so claimed exactly
   what the image model had missed. Checking fails soft (posted unchecked).

   **Editing** works only on images the bot generated
   (`image_generation.pick_edit_source`: attachment `generated-image.png` from
   the bot — the replied-to one, else the newest in the last 20 messages).
   Uploaded photos are never edited, which keeps "make him fatter" off photos of
   real people.
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
   | Orchestration: thread, claim, guard, progress, artifacts, closing note | `tool_functions.run_code_sandbox` |
   | What the outer model is told about a finished run | `sandbox_agent.sandbox_tool_result` (pure) |
   | Nested agent, its prompt, session/container lifecycle, artifact selection | `sandbox_agent.py` |
   | Live progress streamed into one edited message | `sandbox_progress.py` |
   | Workspace snapshots in Redis, keyed by thread id | `sandbox_snapshot_store.py` |
   | Thread messages: ledger, stages, per-thread claim (pure) | `sandbox_thread_inbox.py` |
   | Steering enforcement: input filter, gate, acks, questions, finishing | `sandbox_conversation.py` |
   | Loose ends carried to the next run in a thread | `sandbox_conversation_store.py` |

   The behaviour worth knowing before you change anything:

   - **Every call runs in a Discord thread.** `ensure_sandbox_thread` creates one
     off the triggering message (or reuses the thread you are already in), and
     everything — embeds, files, previews, questions — goes there.
   - **The container is disposable; the workspace is not.** In a thread, the
     workspace is persisted to Redis before teardown and restored on the next
     call in that SAME thread. Resume is thread-local by construction: the
     snapshot id IS the thread id, so asking anywhere else starts fresh.
   - **A run is a conversation, and steering is enforced, not hoped for.**
     Messages posted in a claimed thread go to the run's ledger instead of the
     outer LLM (📨 = the run WILL see it, never "applied"). They reach the
     model as their own user-role items at the next model call
     (`RunConfig.call_model_input_filter` — the one delivery path; the SDK
     does not keep filter-added items, so the ledger re-inserts each one at
     its anchor every call). Until the model answers with
     `respond_to_updates`, new shell commands, `attach_file` and previews are
     refused; a final answer with input unanswered gets a bounded
     continuation in the same container. Each message ends with a recorded
     outcome (done / accepted-but-unconfirmed / declined / never answered /
     arrived too late), and that — not "the sandbox adapted" — is what the
     outer model and the closing note report. A message that lands during a
     model call cannot interrupt it (mid-inference cancellation was judged
     unsafe); its stale actions are refused instead.
   - **One run per thread, claimed atomically.** `claim_run` happens right
     after the thread is resolved and before any other await, and is held
     through artifact delivery and the closing note. Messages after the
     finalization boundary become follow-ups (⏳), saved with the thread's
     conversation record and offered to the next run there.
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

8. What's New and /help: after the first successful reply in a guild on a
   new version, `main.maybe_announce_whats_new` posts that version's feature
   notes (`core/whats_new.md`) as one embed in the same channel. One atomic
   Redis `SET dcb:{guild}:whats_new_seen <version> GET` claims it, so it is
   shown once per guild per version even with several workers, and since only
   the latest version is stored a version a guild never triggered on is
   skipped, never shown late. The embed's footer marks it, and
   `MessageHandler._format_group` drops it from prompts. `/help` (from
   `classes/help_catalog.py`) and `/whats_new` reply ephemerally. See "What's
   New notes" below for how the notes are written and released.

9. Schedules and rules (`docs/automations.md`): per-guild automations stored
   under `dcb:automations:v1` in Redis. `automation_policy.py` is the pure half
   (timing, DST, matching, settings); `automation_store.py` keeps versioned
   records and does quotas, optimistic-revision edits and claims atomically
   (WATCH/MULTI and Lua). Automatic work rides the SAME bounded queue as chat,
   as `AutomationJob` items, so `WORKER_COUNT` and the channel locks still
   bound it. `schedule_forever` (main.py) polls due schedules every 15 seconds
   and admits only what the queue has room for; rule matching runs in
   `on_message` after sandbox steering and, when a rule is admitted, replaces
   the ordinary reply. An automatic run builds a `TextLLMHandler` with
   `automatic=True`, `actor_id` (the entry's last editor) and `channel`
   instead of a source message: memory and automation tools are left off, no
   personal memory is loaded, and double replies are off. A run is never
   replayed after it starts (a `started` marker survives the claim), since
   tools may already have posted; the next poll records it as `interrupted`.
   A lost channel, a departed editor or a bot without send permission
   suspends the entry instead of failing every occurrence. Interval schedules
   store their anchor (`timing.start`) at creation, so the grid never drifts.

### What's New notes (write these in every feature PR)

`core/whats_new.md` holds the user-facing features of the NEXT release. Every
push to `main` is its own release (auto-tag), so the file belongs to the PR
being merged:

- **AI agents: when a PR adds or changes something a Discord user can see or
  do, REPLACE the whole file** with one `## Feature name` section per feature,
  written from the PR's contents: one or two plain sentences on what it does,
  then a `How to use:` line (the command to type or what to ask). Never append
  to the previous release's notes.
- Features only. No bug fixes, security, performance, refactors, CI or docs.
- **No user-facing features? Leave the file untouched.** Do not clear it:
  `release.yaml` diffs it against the previous tag and, when unchanged, blanks
  it in the image and leaves it off the GitHub release, so old notes are never
  re-announced. (Editing it, even for a typo, makes it this release's notes.)
- Limits are tested (`core/tests/whats_new_tests.py`): max 25 sections,
  1024 characters per section, 6000 overall.
- The version is not in the file: `release.yaml` bakes the tag into the core
  image as `APP_VERSION`. Unset (local compose, PR builds) means no
  announcement; set it in `.env` to try one.
- A changed file also goes at the top of the GitHub release page.

**/help is hand-written** (`classes/help_catalog.py`). A new agent tool
(`text_llm_handler.agent_tools`) or slash command (`main.register_commands`)
needs an entry there, gated with `requires` when it can be switched off;
`core/tests/help_catalog_tests.py` fails until it has one.

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
   Sizes worth re-measuring before adding: `SANDBOX_INSTRUCTIONS` ~3850 chars
   formatted (pinned by a test, ceiling 3950), `run_code_sandbox` ~1750
   including its JSON schema.

   **Prefer a positive example over a list of prohibitions.** The design-
   ownership rule once carried a 350-character negative list ("no dimensions,
   colours or RGB values, no frame counts, no library choices…"); the model
   then retried a failed run with an invented canvas size, frame count and
   library — three items straight off that list. It now shows the shape
   instead: `task="Generate a gif of a cow doing a backflip."`

### Environment variables

| Var | Purpose |
|---|---|
| `DISCORD_TOKEN` | required; bot token. Chart: `discordToken`, or bring your own Secret via `existingSecret` (keys = these env var names: `DISCORD_TOKEN`, `OPENAI_API_KEY`, `SANDBOX_LLM_API_KEY`, `IMAGE_PROMPT_LLM_API_KEY`); `llamacpp.existingSecret` does the same for the llamacpp pod's `HF_TOKEN` |
| `REDIS_HOST` | required; Redis host |
| `LLM_HOST` | llama.cpp server base URL (OpenAI-compat; core appends `/v1`). Points at the `llamacpp` service on :8081 in docker-compose (dev) and :8080 in the helm chart; the in-code fallback is `http://llamacpp:8080` |
| `LLM_PASS` | placeholder key — llama.cpp does not authenticate, but the OpenAI client requires a non-empty key |
| `MODEL` | the model, single source of truth: the bot requests this name AND the compose `llamacpp` service serves it (as `LLAMA_ARG_HF_REPO`); in the Helm chart the one `model` value feeds both. In-code default `qwen3:4b` |
| `OPENAI_API_KEY` | API key for the free OpenAI Moderations endpoint (web-tool guard); fail-open if unset |
| `OPENAI_BASE_URL` | moderation API base (default `https://api.openai.com/v1`); any OpenAI-compatible `/v1/moderations` endpoint works. Chart: `openaiBaseUrl` |
| `MODERATION_MODEL` | moderation model name, e.g. `text-moderation-latest`; unset omits the parameter and uses the server default. Chart: `moderationModel` |
| `CONTENT_GUARD_ENABLED` | `0`/`false` disables the content guard on web tools (default: on) |
| `CONTENT_GUARD_DEBUG` | `0`/`false` silences content-guard debug logging (default: on) |
| `METRICS_PORT` | port to serve the Prometheus `/metrics` endpoint on (default 9464); empty/`0` disables. Chart: `metrics.enabled`/`metrics.port` also add a ClusterIP Service, the pod port, and a kube-prometheus-stack ServiceMonitor (labelled `release: kube-prometheus-stack` — the operator only imports ServiceMonitors with that label). The same `metrics.enabled` switch turns on **llama.cpp's own** `/metrics` (`LLAMA_ARG_ENDPOINT_METRICS`, off upstream by default) plus a second ServiceMonitor for it |
| `LLM_CONTEXT_LENGTH` | the context window the LLM server was started with (chart: `llamacppContextLength`, which also sets the server's `LLAMA_ARG_CTX_SIZE`), exported as the `discord_bot_llm_context_window_tokens` gauge so a dashboard can express prompt sizes as a fraction of it. Unset leaves the gauge at 0 |
| `LOG_LEVEL` | root log level (default `INFO`); `DEBUG` also dumps the raw agent run result. Logging replaced bare `print()` calls, which had no level to tune |
| `MSG_HISTORY_LIMIT` | messages in a (refreshed) history window, the trigger included, default 5. Chart: `message_history` |
| `MSG_HISTORY_MODE` | `sliding` (default: always the newest window) or `anchored` (the window's start stays fixed and grows, so the LLM server can reuse its cached prefix, until a refresh). Chart: `history.mode` |
| `MSG_HISTORY_REFRESH_MESSAGES` | anchored mode: new channel messages since the last refresh (bot replies included, the trigger counted) that move the anchor to the newest window, default 10. Chart: `history.refreshMessages` |
| `MSG_HISTORY_RESERVE_TOKENS` | tokens of the llama.cpp per-slot context (its `/props`, re-read every 5 minutes) kept free for the system prompt, tool schemas/results and the answer; older history messages are dropped whole past that, default 12000. Nothing is trimmed while the per-slot context is unknown. Chart: `history.reserveTokens` |
| `REASONING_EFFORT` | sent to the LLM as the OpenAI-compat `reasoning_effort` field (low/medium/high, default medium). Chart: `reasoningEffort` |
| `LLM_MAX_TURNS` | max model turns for ONE reply from the main agent (helm: `llmMaxTurns`), default 20. A turn is one model response, however many tool calls it carries. Passed explicitly to `Runner.run` because the SDK's own default of 10 is easily overrun by a reply that chains several sandbox/image calls — and overrunning raises `MaxTurnsExceeded`, which costs the whole answer |
| `LLM_CALL_TIMEOUT_SECONDS` | wall-clock seconds ONE model call from the main agent may take, the OpenAI client's own retries included (helm: `llmCallTimeout`), default 600 — the same as the client's own timeout, so a slow local generation is no more likely to be cut off than before. Applied as `ModelSettings.timeout`, which the SDK runner enforces by cancelling the call. Needed because the client's timeout is per-read: a server that keeps the connection warm while it works (OpenRouter pads non-streaming responses with whitespace) resets it on every byte and was previously unbounded |
| `WHATS_NEW_ENABLED` | `0`/`false` stops the automatic What's New embed (default: on); `/whats_new` still works. Chart: `whatsNew.enabled` |
| `APP_VERSION` | the release tag, baked into the core image by `release.yaml` (`--build-arg`), not configured. Unset = no What's New announcement |
| `SHOW_THINKING` | `1`/`true` sends the model's reasoning as spoiler-hidden follow-up message(s); default (off) drops it entirely. Chart: `showThinking` |
| `DOUBLE_REPLY_CHANCE` | Probability of offering two short chat messages from one generation (default `0.08`, range 0–1; `0` disables). The model may choose one. Chart: `doubleReply.chance` |
| `DOUBLE_REPLY_COOLDOWN_REPLIES` | Successful replies per channel required after a pair (default `10`, nonnegative). Process-local state, capped at 4096 channels; resets on restart/eviction. Chart: `doubleReply.cooldownReplies` |
| `AUTOMATIONS_ENABLED` | `0`/`false` stops schedules and rules from running and hides `/schedule`, `/rule` and their chat tools; stored entries are kept (default: on). Chart: `automations.enabled` |
| `AUTOMATIONS_TIMEZONE` | default IANA timezone for new schedules (default `Europe/London`); validated at startup. Chart: `automations.timezone` |
| `SCHEDULE_MIN_INTERVAL_HOURS` | minimum hours between runs of a recurring schedule, catch-ups included (default 4, min 1). One-offs are exempt. Chart: `schedules.minIntervalHours` |
| `SCHEDULE_MAX_PER_GUILD` / `RULE_MAX_PER_GUILD` | active entries per server (defaults 1 / 4; 0 blocks new ones). Lowering a limit keeps existing entries. Chart: `schedules.maxPerGuild` / `rules.maxPerGuild` |
| `RULE_COOLDOWN_SECONDS` | seconds before the same rule can fire again (default 60). Chart: `rules.cooldownSeconds` |
| `WORKER_COUNT` | queue worker tasks (default 2, min 1); each handles one message at a time, a per-channel lock keeps same-channel order. Chart: `worker_count` |
| `QUEUE_MAX_SIZE` | max messages waiting on the bounded queue (default 10, min 1); when full new messages are dropped (a mention gets a short "busy" reply). Chart: `queue_max_size` |
| `LLAMA_ARG_CACHE_TYPE_K`, `LLAMA_ARG_CACHE_TYPE_V` | optional; compose `llamacpp` service only: KV cache quantization type (llama.cpp `-ctk`/`-ctv`), default `q4_0`; in the Helm chart set via `llamacpp.cacheTypeK`/`cacheTypeV` |
| `IMAGE_GEN_ENABLED` | `0`/`false` removes the `generate_image` tool from the LLM (default: on). Chart: `diffusion.enabled` also removes the diffusion pod/PVC |
| `DIFFUSION_URL` | base URL of the diffusion service (core appends `/generate`); compose `diffusion` service on :8000 in dev, in-cluster `*-diffusion-service` in the chart; in-code fallback `http://diffusion:8000` |
| `IMAGE_MODEL` | HF repo id for the diffusion service (default `stabilityai/sd-turbo` — smallest practical model); the service downloads it into its `HF_HOME` volume on first boot |
| `IMAGE_STEPS` / `IMAGE_WIDTH` / `IMAGE_HEIGHT` | generation settings for the diffusion service (defaults: 4 steps, 512x512) |
| `IMAGE_GUIDANCE` | CFG scale; unset leaves the pipeline's own (7.5 SD1.5 / 5.0 SDXL). Pinned on distilled models (0.0 SD family, 1.0 FLUX.2 [klein]). Chart: `diffusion.guidance` |
| `IMAGE_NEGATIVE_PROMPT` | baseline negative prompt, merged BEHIND the per-request one the rewriter produces, and the only one left when that rewrite is off or falls soft. Compose and the chart ship the same non-empty default; dropped automatically on distilled models, unused by FLUX.2. Chart: `diffusion.negativePrompt` |
| `IMAGE_QUANTIZE` | `none` (default) / `nf4`: load FLUX.2 [klein]'s transformer and text encoder in 4-bit (bitsandbytes), which is what fits it on an 8 GB card; ignored for SD/SDXL. Chart: `diffusion.quantize` |
| `IMAGE_LONG_PROMPT` | `0`/`false` reverts to truncating prompts at 77 CLIP tokens instead of chunk-encoding them (default: on). Chart: `diffusion.longPrompt` |
| `IMAGE_OFFLOAD` | `model` (default: one pipeline component on GPU at a time, text encoder in CPU RAM) / `sequential` (lowest VRAM, slowest) / `none` (all on GPU) |
| `IMAGE_QUEUE_SIZE` | max queued image requests in the diffusion service (default 16); over that it returns 503 |
| `IMAGE_GEN_TIMEOUT` | seconds core waits on the diffusion service (default 300) |
| `IMAGE_PROMPT_REWRITE_ENABLED` | `0`/`false` sends image requests to the service verbatim instead of rewriting them first (default: on). Chart: `diffusion.promptRewrite.enabled` |
| `IMAGE_PROMPT_MODEL` / `IMAGE_PROMPT_LLM_HOST` / `IMAGE_PROMPT_LLM_API_KEY` / `IMAGE_PROMPT_TIMEOUT` | the prompt rewrite's own LLM connection; each falls back to the bot's `MODEL` / `LLM_HOST` / `LLM_PASS` (timeout default 60s), exactly like the `SANDBOX_*` equivalents. Chart: `diffusion.promptRewrite.*` (the key via secret.yaml) |
| `IMAGE_REVIEW_ENABLED` | `0`/`false` posts generated images unchecked (default: on). Chart: `diffusion.review.enabled` |
| `IMAGE_REVIEW_RETRIES` | extra attempts when the check finds something asked for missing (default 1, max 2; `0` = check only, so captions stay honest). Each costs a generation plus another check. Chart: `diffusion.review.retries` |
| `IMAGE_REVIEW_MODEL` / `IMAGE_REVIEW_LLM_HOST` / `IMAGE_REVIEW_LLM_API_KEY` / `IMAGE_REVIEW_TIMEOUT` | the check's own LLM connection — it must be a vision model; each falls back to `MODEL` / `LLM_HOST` / `LLM_PASS` (timeout default 60s). Chart: `diffusion.review.*` (the key via secret.yaml) |
| `SANDBOX_ENABLED` | `0`/`false` removes the `run_code_sandbox` tool from the LLM (default: on). Chart: `sandbox.enabled` also removes the Docker-socket hostPath mount |
| `SANDBOX_IMAGE` | container image for the sandbox workspace, pulled once onto the daemon (default `python:3.14-slim`). Chart: `sandbox.image` |
| `SANDBOX_MAX_TURNS` | max model turns for one sandbox task (default 10). Chart: `sandbox.maxTurns` |
| `SANDBOX_MODEL` | model id for the nested sandbox agent; empty (default) = the main bot's `MODEL`. Chart: `sandbox.model` |
| `SANDBOX_LLM_HOST` | base URL of the sandbox agent's LLM (core appends `/v1`); empty (default) = the main `LLM_HOST`. E.g. `https://openrouter.ai/api` for OpenRouter. Chart: `sandbox.llmHost` |
| `SANDBOX_LLM_API_KEY` | API key for the sandbox agent's LLM; empty (default) = the main `LLM_PASS` placeholder. Chart: `sandbox.apiKey` |
| `SANDBOX_ASK_USER_TIMEOUT` | max seconds the sandbox's `ask_user` tool waits for a reply in its thread before telling the model to proceed on its own (default 300); also clamped to whatever of the run's own `SANDBOX_TIMEOUT` budget remains. Chart: `sandbox.askUserTimeout` |
| `SANDBOX_PERSIST_TIMEOUT_SECONDS` | seconds allowed to persist a thread's workspace snapshot to Redis on container teardown, after `SANDBOX_TIMEOUT` has already elapsed (default 180 — generous since the `Memory` capability's own extraction runs here too). Chart: `sandbox.persistTimeout` |
| `SANDBOX_REQUEST_TIMEOUT_SECONDS` | seconds of silence on one HTTP request to the sandbox's LLM before the client gives up (default 180, down from the OpenAI client's unstated 600). This is the HTTP client's per-read timeout: it catches a hung connection or a server that sends nothing, but NOT one that dribbles keep-alive padding while it works (OpenRouter pads non-streaming responses) — that case is caught by the wall-clock bound below. Chart: `sandbox.requestTimeout` |
| `SANDBOX_MAX_RETRIES` | how many times that client retries a failed request (default 2; 0 disables). One model call is also wall-clock bounded (`ModelSettings.timeout`, see `sandbox_model_call_timeout()`) at (1 + this) x `SANDBOX_REQUEST_TIMEOUT_SECONDS` — the figure that was already the documented worst case for a silent server, now enforced for a padding one too. Chart: `sandbox.maxRetries` |
| `SANDBOX_SNAPSHOT_MAX_BYTES` | max size of one thread's stored workspace snapshot in Redis (default 50MB). Chart: `sandbox.snapshotMaxBytes` |
| `SANDBOX_SNAPSHOT_TTL_SECONDS` | how long an unused thread's workspace snapshot survives in Redis (default 604800 = 7 days). Chart: `sandbox.snapshotTtlSeconds` |

## Releasing

- Every push to `main` runs `auto-tag.yaml`, which bumps the highest `vMAJOR.MINOR`
  tag by one minor and pushes it. That tag push is what starts `release.yaml`
  (tests -> two images -> chart -> GitHub release). Nothing dispatches
  `release.yaml`; its `workflow_dispatch` trigger is a manual escape hatch for
  re-running a release against an existing tag.
- The packaged chart is published twice from that one `.tgz`: as a GitHub
  release asset (`dchatbot-vX.Y.tgz`) and as an OCI artifact at
  `oci://ghcr.io/dgowing95/charts/dchatbot` (tag = the git tag, e.g. `v2.29`).
  The OCI copy is what lets a Helm-repo-aware consumer (ArgoCD, `helm install
  oci://...`) resolve versions and semver ranges without a chart index; both
  carry `appVersion` = the tag, which is where the `core-<tag>` /
  `diffusion-<tag>` image tags in the templates come from.
- **`TAG_PUSH_TOKEN` (the `release-tag` environment secret) creates the tag**, and it expires.
  `GITHUB_TOKEN` cannot do the job for two independent reasons: the "Restrict
  Tagging" ruleset allows ref creation only for repo admins, and it acts as the
  GitHub Actions app rather than a user; and a ref pushed with it deliberately
  fires no `push` event, so `release.yaml` would never start.
- To mint a replacement: your **account** settings (not the repo's) → Developer
  settings → Personal access tokens → Fine-grained tokens → Generate new token,
  scoped to `dgowing95/DiscordChatBot` only, with **Repository permissions →
  Contents: Read and write** (nothing else; `workflow` is not needed, the job
  pushes a tag and never touches `.github/workflows`). Then store it in the
  **repo's** settings → Environments → `release-tag` → Environment secrets, as
  `TAG_PUSH_TOKEN`. The environment is restricted to the protected `main` branch.
- How the Auto Tag job fails tells you which half is wrong. Secret unset: the
  `gh api` step fails authentication. PAT expired or revoked: the same step gets
  a 401. A ruleset rejection while creating `refs/tags/<version>` means the token
  is valid but its owner is not a bypass actor on the tag ruleset.

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
  there is no list to keep in step. ~650 tests, roughly 20 seconds.

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
- History selection lives in `core/classes/history_policy.py` — keep it pure
  (stdlib only). Its per-channel state is two message ids, never content:
  every build re-reads Discord, so edits and deletes need no event handling
  and eviction only costs a cold refresh. Cover policy changes in
  `core/tests/history_policy_tests.py` and the Discord-side behaviour
  (queued triggers, ordering, anchored prefixes, resets, out-of-order
  workers) in `core/tests/message_history_tests.py`, whose `FakeChannel`
  mirrors discord.py's `history()` semantics (`after=` keeps the OLDEST
  `limit` messages). A passing test shows the message list is stable, not
  that llama.cpp reuses its cache — check `llamacpp:prompt_tokens_cached_total`
  and the server's checkpoint-restore log lines for that.
- The attachment cache (`core/classes/attachment_cache.py`) is pure too
  (stdlib only). It keeps only the final base64 data URL, so a cached build
  sends exactly the bytes an uncached one would (prefix reuse depends on it).
  Keys are ids, never the signed CDN URL, which expires. Failed or
  undecodable downloads are never stored, so the next build retries them.
  Cover it in `core/tests/attachment_cache_tests.py` and the download path
  (lock not held, concurrency limit, deletion during a download) in
  `core/tests/message_handler_tests.py`.
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
- Sandbox thread-message state lives in `core/classes/sandbox_thread_inbox.py`
  — keep it pure (stdlib only) and keep every mutating function free of
  `await`: that is what makes `claim_run` / `deliver` / `finalize` atomic on
  the one event loop. Cover it in `core/tests/sandbox_thread_inbox_tests.py`.
  Runner-boundary behaviour (where filter-injected messages land, the tool
  gate, continuations) is tested against the REAL SDK Runner with
  `agents.testing.ScriptedModel` in `core/tests/sandbox_conversation_tests.py`;
  extend those rather than mocking `Runner`, since the point is to catch an
  SDK upgrade that changes the boundary. The `on_message` routing into a
  running sandbox is tested in `core/tests/message_queue_tests.py`.
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
