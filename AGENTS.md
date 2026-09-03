# AGENTS.md

Overview of this repository and how to work with it (for humans and AI coding agents).

## What this is

A Python Discord bot that answers messages using a local LLM (llama.cpp server serving a GGUF model, e.g. `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`).
It is primarily deployed on **Kubernetes** via the Helm chart in `charts/dis-ai-bot`
(releases are cut by `.github/workflows/auto-tag.yaml`, which bumps a `vMAJOR.MINOR` tag on every
push to `main` and dispatches `release.yaml`; a prebuilt chart is downloadable from GitHub releases).
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
    content_guard.py       # OpenAI Moderations-based safety guard for web_search/fetch_url
    metrics.py             # PURE (stdlib + prometheus_client) Prometheus metrics:
                           #   all metric definitions + /metrics HTTP server (METRICS_PORT)
    user_memory.py         # JSON lists in Redis per (guild, user)
    config_manager.py      # per-guild settings in Redis (system prompt, temperature, ...)
    tool_functions.py      # agent function tools: web_search, fetch_url, memory tools, generate_image, run_code_sandbox
    image_generation.py    # client for the diffusion service + IMAGE_GEN_ENABLED flag
    sandbox_agent.py       # nested SandboxAgent + run_sandbox_task (throwaway Docker sandbox)
    sandbox_progress.py    # streams sandbox commands/output to one edited Discord message
                           #   (single embed: one field per command, state-coloured)
    sandbox_snapshot_store.py  # Redis-backed workspace snapshots, keyed by thread id
    sandbox_thread_inbox.py    # routes thread messages to a sandbox run in flight
    common.py              # shared helpers (Discord tool embeds)
  tests/               # pytest suite (see Testing below)
  Dockerfile           # python:3.13-slim image, runs main.py
  requirements.txt
diffusionservice/      # standalone image service (text->image; FastAPI + diffusers,
                       #   queued single-worker, sd-turbo by default,
                       #   CPU-offloaded for low VRAM)
charts/dis-ai-bot/     # Helm chart
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
   `IMAGE_WIDTH`/`HEIGHT`, `IMAGE_OFFLOAD`, `IMAGE_QUEUE_SIZE`) live in the same
   configmap/env the diffusion pod reads.
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
   every call, so no CONTAINER outlives a run — but in a thread its WORKSPACE
   does, snapshotted to Redis and restored on the next call there (see the
   snapshot/resume paragraphs below). A task must be self-contained unless it
   is resuming a thread it already ran in. Live
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
   see `sandbox_progress.py`). The run's final state (done/timeout/turns
   exhausted/model error/failed) is flushed via `finalize()` before the
   tool returns. `run_sandbox_task` never lets `asyncio.TimeoutError`,
   `MaxTurnsExceeded` or `ModelBehaviorError`/`ModelRefusalError` propagate:
   each becomes `SandboxResult(ok=False, error="timeout"|"max_turns"|
   "model_error")`, and the container is still read before teardown in all
   three cases — whatever the run had already attached (or, failing that,
   whatever the `out/` sweep finds) is recovered and delivered instead of
   being discarded with the container. `run_code_sandbox` (tool_functions.py) turns `error` into
   reason-specific guidance for the calling agent instead of one
   generic "may be unavailable" message for every failure — that generic
   message is now reserved for errors that actually propagate (a dead
   container/Docker daemon, `UserError`, ...), which mean the sandbox
   itself is unusable rather than a task that ran out of budget. Output
   files skipped by the artifact caps (`MAX_ARTIFACT_FILES`/
   `MAX_ARTIFACT_BYTES`) are reported back the same way, on both success
   and failure, instead of only being logged.

   **No stopped run tells the outer model to retry.** A retry starts the task
   over in a fresh container, throwing away the partial workspace teardown
   just persisted — and in practice the outer model answers "retry, more
   focused" by writing exactly the implementation spec `run_code_sandbox`'s
   docstring exists to suppress (observed: a timed-out cow GIF retried with
   an invented canvas size, frame count and library choice). So the three
   failure strings state what happened and stop there; the "what now" clause
   is chosen from the closing embed's own answer — when the workspace really
   was saved (`_send_sandbox_closing_note` hands its live TTL back to the
   caller) the model is told to offer the user a follow-up **in the thread**,
   which resumes from where the run stopped, and otherwise just to ask. The
   two can therefore never disagree about whether a resume is available. The sandbox agent's own
   instructions state its exact turn/time budget
   (`SANDBOX_MAX_TURNS`/`SANDBOX_TIMEOUT`) and tell it to save partial
   output under `out/` before a forced cutoff rather than risk losing
   everything.

   **Elapsed time is injected, not measured by the model.** Every shell
   result carries `[time used: 312s of 1800s]`, appended by
   `_with_elapsed_note` from the `deadline`/`timeout_seconds` pair in the
   run context — the same carrier `_with_thread_messages` uses, for the same
   reason: a small local model will not remember to check a clock, but it
   cannot avoid reading the output of the command it just ran. The budget
   bullet used to teach the measurement instead ("run `date +%s` as your
   very first command … re-run it every few turns and subtract"), which
   asked a 27B Q4 model to carry an epoch integer across turns, spent one of
   a budget as low as 10 turns, and fought `RESUMED_TASK_PREAMBLE` over
   which command gets to be "first". Composition order is tolerant →
   elapsed → thread messages, so a steering message stays last in the
   result, where the model is most likely to act on it.

   **The sandbox agent chooses what is delivered.** `out/` used to mean
   "everything in here is sent", so a model that iterated — saving v1, v2,
   v3 — delivered three near-identical images for one request (observed
   live). It now calls `attach_file(path, caption)` on the finished file;
   only attached files are sent. The list lives in the nested run context
   (`deliverables`), written DURING the run so it survives the paths where
   `Runner.run` never returns (timeout, max_turns). `_deliver` prefers it and
   falls back to the blind `out/` sweep (`_collect_artifacts`) only when
   nothing was attached — a run cut off before it could attach, or a model
   that finished without calling the tool. That sweep is also filtered with
   `find -newer` against a run marker touched beside (never inside) `out/` at
   the start of every run: a thread's snapshot carries `out/` with it and a
   resumed run only `mkdir -p`s it, so without the filter run #2 re-sent
   run #1's files. Attached files arrive under their basename
   (`_attachment_name`), not the sweep's flattened `out_plot.png` form.

   **The sandbox agent also writes the message the user reads.** Its final
   output is no longer only handed to the outer model: `run_code_sandbox`
   posts it into the thread as the `content` of the first attached file (one
   message, summary plus image), or on its own when there are no files, and
   `SANDBOX_INSTRUCTIONS`' closing bullet is aimed at the user rather than at
   "the caller". The outer model is told, in the tool result, that the user
   has already read it and to add at most one short sentence.

   The output dir is also resolved
   (via `pwd`) and mkdir -p'd before the run starts, but the sandbox agent
   is only ever told the RELATIVE `out/` dirname, never the resolved
   absolute path: exec_command's own `workdir` argument is validated by
   the SDK's manifest system, which rejects any absolute path outright
   (confirmed in production — a model told the absolute path reused it as
   `workdir` and the call failed). That rejection — not the older claim that
   `/workspace` does not exist — is what the instructions now state.
   `session.start()` does materialize `/workspace`, so the old wording had
   drifted into contradicting the very same bullet's closing clause ("even
   though the directory is real"). The core
   container needs the Docker daemon socket mounted (compose: socket bind
   mount; chart: hostPath volume gated on `sandbox.enabled`) plus the
   `docker`/`websocket-client` Python packages. Tasks go through the content
   guard first; `SANDBOX_MAX_TURNS`/`SANDBOX_TIMEOUT` bound each run.

   Every call runs in its own Discord **thread**
   (`sandbox_agent.ensure_sandbox_thread`): the first call off a normal
   message creates one (a note is posted in the original channel pointing to
   it) and everything else — the static/live-progress embed, output files,
   previews, HITL questions — goes to the thread instead; a call made from
   inside an existing thread just reuses it. The nested `SandboxAgent` also
   gets two extra tools alongside its shell capability: `ask_user` (post a
   question to the thread and `client.wait_for` a reply from the requesting
   user, bounded by `SANDBOX_ASK_USER_TIMEOUT` AND whatever of the run's own
   `SANDBOX_TIMEOUT` budget is left — an unanswered question never hangs or
   fails the run, it just tells the model to proceed on its own judgement)
   `send_preview_to_thread` (push an in-progress file to the thread before
   the task finishes, same size cap as final artifacts), `attach_file` (mark
   a finished file for delivery — see above), `say_in_thread`
   (post a progress note without waiting for anything) and
   `check_thread_messages`. The Discord
   `client` needed for `ask_user`'s `wait_for` is threaded all the way down
   from `main.py` through `MessageHandler` → `TextLLMHandler` (constructor
   param `client`, exposed to tool context as `"discord_client"`) → the
   nested `Runner.run`'s own `context=`.

   The container is still fully disposable — a fresh one is created and
   deleted every call — but when the call is happening in a thread, its
   workspace is persisted as a snapshot in **Redis** just before teardown
   (`session.stop()`, time-boxed by `SANDBOX_PERSIST_TIMEOUT_SECONDS` since
   that's also where the `Memory` capability's own extraction runs — two
   extra local-model calls, outside `SANDBOX_TIMEOUT`'s envelope) and
   restored into the next fresh container for a later call in the SAME
   thread, via a `classes.sandbox_snapshot_store.SandboxSnapshotStore`
   (binary Redis client, `SANDBOX_SNAPSHOT_MAX_BYTES` cap,
   `SANDBOX_SNAPSHOT_TTL_SECONDS` expiry) bound to the SDK's
   `RemoteSnapshot`/`Dependencies` mechanism and keyed by the thread's id —
   this is what lets "continue this in the thread" pick back up with the
   prior workspace instead of starting from scratch. A call with no thread
   (thread creation failed, or isn't possible on that channel type) skips
   snapshotting entirely rather than keying it on the plain channel id. The
   `Memory` capability rides on the same snapshot (its files live under the
   workspace root, which is exactly what gets persisted/restored) —
   configured `live_update=False` to stay Shell-only (matching the
   Filesystem/`apply_patch` avoidance above) and pointed at the sandbox's
   own local model for its phase-one/phase-two extraction instead of the
   SDK's hosted-OpenAI defaults, which this self-hosted bot has no key for.

   Which of the two happened is stated up front: `run_code_sandbox` asks
   `sandbox_snapshot_exists()` before the run and puts a one-line badge at
   the top of the embed — 🆕 fresh vs ♻️ resumed (`sandbox_workspace_note`,
   passed to `SandboxProgressHooks(workspace_note=...)` on the live path). The
   SDK offers no way to ask after the fact — `session.start()` returns `None`
   and `snapshot_fingerprint` is only written on persist — but its restore
   condition IS this store's `exists()`, so asking first gives the same
   answer. A resumed run also gets `RESUMED_TASK_PREAMBLE` prepended to its
   task, because the nested model won't otherwise think to `ls` the restored
   workspace. If a snapshot exists but can't actually be restored (corrupt
   tar), `_create_sandbox_session` deletes the key and retries once against
   the SAME id — never `snapshot=None`, which would silently disable
   persistence for that thread — and `SandboxResult.resumed` carries the
   ground truth back so the badge can be corrected.

   **Design ownership sits with the sandbox agent, not the outer LLM.** The
   outer model's job is to forward the user's request in their own words plus
   context the sandbox can't see; `SANDBOX_INSTRUCTIONS` tells the nested
   agent it receives "a REQUEST, not a specification" and owns every choice
   the user didn't pin down. This is enforced only by `run_code_sandbox`'s
   docstring, since the outer system prompt is per-guild Redis
   (`Answer as if you are {system}`) and not a code lever.

   **People can steer a run while it happens.** `classes/sandbox_thread_inbox.py`
   is a pure, module-level registry: `run_code_sandbox` calls `begin_run` /
   `end_run` around the run (threads only — registering a plain channel would
   swallow every message in it), and `main.py`'s `on_message` routes any
   human message in an active thread there instead of to the outer LLM,
   acknowledging with a 📨 reaction. Messages reach the model three ways, in
   order of reliability: appended to every shell command's result
   (`_with_thread_messages`, wrapped OUTSIDE `_tolerant_tool_invoke` so a
   rejected call still carries them), appended to
   `ask_user`/`say_in_thread`/`attach_file` returns, and on demand via
   `check_thread_messages`.

   They also reach the OUTER model, which otherwise cannot know the request
   changed: its history was built before the run started, and `on_message`
   routes these messages to the sandbox instead of enqueuing them. `deliver()`
   records each accepted message in `_SEEN`, `history()` renders the lot
   non-destructively (so it survives the `drain()`s the sandbox does mid-run,
   and the `consume()` `ask_user` does), and `run_code_sandbox` reads it just
   before `end_run` and appends it to the tool result with an instruction not
   to call the result a mistake. Without it the outer model answered a
   mid-run "make the milk red" by telling the user the sandbox "went a bit
   rogue". `ask_user` calls
   `consume()` on the reply it received, because discord.py's `dispatch`
   fans a message out to `wait_for` futures and `on_message` independently
   and the model would otherwise see it twice.

   That early return is also the concurrency guard: two runs in one thread
   would both persist to `dcb:sandbox_snapshot:{thread_id}` on teardown and
   the last to finish would clobber the other. `run_code_sandbox` repeats the
   guard itself (returning early and forwarding the task into the running
   sandbox), since queue lag or two tool calls in one turn can get past the
   `on_message` check.

   When the run ends, a muted grey **"Sandbox closed"** embed marks the
   boundary — after it, messages in the thread are ordinary chat again. When
   the run did not finish normally its first line is the reason (the same
   wording `finalize()` uses): with live progress off this embed is the
   thread's only end-of-run signal, so without it a timeout looked exactly
   like a success that happened to produce no file. In a
   thread it also states how much longer the workspace can be resumed, read
   as the LIVE Redis `TTL` (`SandboxSnapshotStore.ttl` →
   `sandbox_snapshot_remaining_seconds`) rather than echoing the configured
   `SANDBOX_SNAPSHOT_TTL_SECONDS`, because persisting is best-effort and
   swallowed — the configured value would happily promise a resume for a
   snapshot that was never written (Redis answers `-2`, which becomes "could
   not be saved"). One accepted imprecision: it reports that *a* workspace is
   resumable, not that *this run's* is — if a second-or-later run's persist
   fails, the window shown belongs to the previous run's snapshot, which is
   still resumable, just to an earlier state. It is posted after the artifacts on every path where a
   "Running in sandbox" embed already went out, including the
   sandbox-unavailable one, and never on the early returns (content guard,
   forwarded-to-a-running-run) where nothing was opened.

   **Resume is thread-local by construction, not by policy.** The snapshot id
   is the thread's own Discord id, and `ensure_sandbox_thread` creates a NEW
   thread for any request that doesn't already come from inside one — so a
   request in the parent channel gets a new id and starts fresh. There is no
   code path that resumes a thread's workspace from outside that thread.

   **A second call in the same outer turn reuses that turn's thread.** Discord
   allows exactly one thread per message, so asking `create_thread` for a
   second one fails with 160004 and the old code fell back to the parent
   channel — which is how a retried run ended up running in the main channel,
   away from the thread holding its work and with no snapshot id, silently
   abandoning the partial workspace the stopped run had just saved. Two
   defences: `run_code_sandbox` prefers a `discord.Thread` already recorded in
   `wrapper.context["sandbox_thread"]` (one context object per outer
   `Runner.run`, so the first call's write is what the retry reads), and
   `ensure_sandbox_thread` falls back to `message.thread` for a retry in a
   later turn. Both are `isinstance`-guarded: the context is seeded with
   `None`, the no-thread fallback path writes a plain channel there, and
   `MagicMock` messages in the tests have a truthy `.thread`. Residual gap,
   pre-existing and not closed by either defence: the SDK runs one turn's
   tool calls concurrently, so if the outer model emits TWO
   `run_code_sandbox` calls in a single response, both read
   `sandbox_thread=None` and race — the loser gets 160004 with `.thread` not
   yet in the guild cache (THREAD_CREATE hasn't landed) and still falls back
   to the parent channel, where `in_thread` is False and so the
   `is_run_active` guard cannot see it either.

   Known failure mode: the registry is plain process memory, deregistered in
   a `finally`. A process restart clears it (fine), but a worker torn down in
   a way that skips that `finally` would leave the thread registered — every
   later message in it then gets a 📨 and goes nowhere until the bot
   restarts. If a thread ever "goes deaf" with 📨 reactions and no replies,
   that is what happened.

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
   was ~5000 characters with `out/` explained in five bullets, and
   `run_code_sandbox`'s description alone was 37% of the entire outer tool
   payload. Two bullets had drifted into outright contradiction (an "your
   ONLY tools are the shell tools" claim four bullets above four other
   tools; a `/workspace` warning that contradicted its own closing clause).
   Duplicated guidance is worse than terse guidance on a 27B Q4 model:
   copies drift, and a model handed two versions of a rule follows neither
   reliably. So when adding guidance, edit the existing home rather than
   restating it nearby. Current sizes, worth re-measuring before adding:
   `SANDBOX_INSTRUCTIONS` ~3700 chars (pinned by a test), `run_code_sandbox`
   ~1750 including its JSON schema.

   Prefer a positive example over a list of prohibitions. The design-
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
| `IMAGE_OFFLOAD` | `model` (default: one pipeline component on GPU at a time, text encoder in CPU RAM) / `sequential` (lowest VRAM, slowest) / `none` (all on GPU) |
| `IMAGE_QUEUE_SIZE` | max queued image requests in the diffusion service (default 16); over that it returns 503 |
| `IMAGE_GEN_TIMEOUT` | seconds core waits on the diffusion service (default 300) |
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
  registries) exists twice. `grep -rn "core\.classes" core/` should stay empty.
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
