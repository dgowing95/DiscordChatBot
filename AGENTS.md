# AGENTS.md

Overview of this repository and how to work with it (for humans and AI coding agents).

## What this is

A Python Discord bot that answers messages using a local LLM: a llama.cpp server serving a GGUF model
(e.g. `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M`) or, with the chart's `llm.engine: strata`, a
[Strata](https://github.com/Niko1221/Strata) server for large mixture-of-experts models. It is deployed
on **Kubernetes** with the Helm chart in `charts/dis-ai-bot` (see Releasing). Redis holds settings,
memories and the other per-guild state. It works in servers only, never in direct messages.

This file says what lives where and the rules that break something when missed. **Why** each part works
the way it does is in its module's docstring: read that before changing it.

## Repository layout

`P` = pure: stdlib only (plus `prometheus_client` for metrics), so it is unit-tested without the
discord or agents SDKs. Keep those modules pure.

```
core/                    # the bot (what runs in production)
  main.py                # entrypoint: discord.Client, on_message, queue workers, slash commands
  whats_new.md           # the NEXT release's user-facing features (see "What's New notes")
  classes/
    # messages and the persona
    message_handler.py     # one message: history, prompt build, generate, send
    history_policy.py      # P which history goes in a prompt: sliding / anchored, token budget
    attachment_cache.py    # P encoded images kept between prompt builds
    message_queue.py       # P queue sizing, elastic worker pool, per-channel locks, in-flight hint
    reply_policy.py        # P occasional double replies
    text_llm_handler.py    # the persona: an agents-SDK Agent with one tool, ask_brain
    response_filter.py     # P answer cleaning, reasoning extraction, Discord chunking
    content_guard.py       # moderation check for web_search / fetch_url / the brain's workspace
    user_memory.py         # saved memories: JSON lists in Redis per (guild, user)
    config_manager.py      # per-guild settings in Redis (system prompt, temperature, ...)
    redis_client.py        # the two shared, lazily built Redis clients (str + binary)
    settings_check.py      # P settings removed in v3, and the startup check that exits on them
    metrics.py             # P Prometheus metrics and the /metrics server
    common.py              # shared Discord helpers (tool embeds)
    tool_embeds.py         # P how tool embeds read: titles, colours, link labels
    whats_new.py           # P What's New notes: parse, limits, embed
    help_catalog.py        # P the hand-written /help listing
    # LLM connections
    llm_roles.py           # P model / host / key / effort / timeout per role and side job
    llm_lanes.py           # P priority lanes over a self-hosted server's slots
    laned_model.py         # SDK Model wrapper: lane slot, call timeout, one retry per model call
    side_llm.py            # shared client for side calls (research brief, image rewrite / check)
    # the brain (runtime item 7)
    tool_functions.py      # ask_brain (the persona's tool) and most of the brain's tools
    brain.py               # run_brain, its SandboxAgent, BRAIN_INSTRUCTIONS, report
    brain_tools.py         # the one list of the brain's tools, gated per kind of run
    brain_result.py        # P BrainReport and what the persona is told
    brain_policy.py        # P BRAIN_* limits (unprompted replies, questions, slots)
    brain_inbox.py         # P questions the brain is waiting on, one per question message
    brain_question.py      # the question in Discord: choice / Answer… / You decide buttons
    persona_voice.py       # the brain's question reworded in the persona's voice
    skill_format.py        # P skills: SKILLS_* settings, name/limit checks, SKILL.md, the index and its ranking
    skill_store.py         # skills in Redis: every version kept, soft delete, revert, export
    skill_tools.py         # the brain's use_skill / save_skill / search_skills, and every skill embed
    skill_commands.py      # the /skills group (list, show, history, edit, delete, restore, revert, export)
    skill_reflection.py    # after a long job, one background call decides whether to save a skill
    loop_guard.py          # P stops a brain run that repeats itself that repeats itself
    context_trim.py        # P cuts old tool outputs in a long brain run
    brain_progress.py      # the "Working on it" status embed and live command output
    workspace_tools.py     # shell wrappers, attach_file and delivery, in/ files, thread tools
    workspace_inputs.py    # P which chat files go to a brain call's in/
    workspace_lifecycle.py # P the per-guild workspace state machine, WORKSPACE_* settings
    workspace_manager.py   # one Docker container per server: start, resume, snapshot, stop, reap
    workspace_snapshot_store.py    # workspace snapshots in Redis, one per guild
    workspace_commands.py  # /workspace (status, reset, progress)
    sandbox_thread_inbox.py        # P thread-message ledger    | kept for Tier 3,
    sandbox_conversation.py        # thread steering, ask_user   | unused until then
    sandbox_conversation_store.py  # per-thread loose ends       |
    # tools and features
    image_generation.py    # create_image: prompt, generate, check, retry; which image to edit
    image_prompt.py        # request -> SDXL prompt + negative prompt
    image_review.py        # vision check of a generated image
    poll_format.py         # P create_poll's checks, how a poll reads in a prompt
    polls.py               # reading polls (voters), recording each one for check_polls
    poll_store.py          # each channel's polls, in Redis
    research_format.py     # P web_search's settings, which results are read, the brief prompt
    web_research.py        # web_search: DuckDuckGo, read the results, write the brief
    web_policy.py          # P fetch_url's rules: URL routing, cache keys, public-address check
    site_format.py         # P how Reddit / Steam / YouTube read in a prompt
    web_reader.py          # fetch_url's reader: cache, site readers, browser fallback
    message_search_format.py  # P Discord message search: params, results, quirks
    message_search_tools.py   # search_messages, read_messages_around, save_messages_to_file
    event_format.py        # P Discord events: checks, time parsing, prompt text
    event_tools.py         # create_event / list_events / update_event
    automation_policy.py   # P schedule timing, DST, rule matching, AUTOMATIONS_* settings
    automation_store.py    # versioned Redis records, quotas, atomic claims
    automation_runner.py   # rule admission, schedule poll, automatic runs
    automation_commands.py # /schedule and /rule
    automation_tools.py    # the matching brain tools, schedule_followup
    voice_policy.py        # P voice settings, wake phrase, commands, sentences, voice prompts
    voice_gate.py          # P which guilds are in a call
    voice_bridge.py        # WebSocket client to the voice sidecar, discord.py's voice handshake
    voice_session.py       # one call: utterance -> turn -> speech, prefill
    voice_commands.py      # /voice
  tests/                 # pytest suite (see Testing)
  Dockerfile, requirements.txt, requirements-dev.txt
diffusionservice/        # image service (FastAPI + diffusers, one image at a time);
                         #   generation_params.py is its pure, tested half
browserservice/          # headless Chromium for fetch_url; egress_proxy.py and
                         #   render_policy.py are its pure, tested halves
speechservice/           # speech-to-text (faster-whisper) and text-to-speech (Kokoro), CPU only;
                         #   speech_params.py is its pure half
voicesidecar/            # Node: Discord voice I/O (@discordjs/voice, DAVE); README has the protocol
docs/                    # user guides (automations, events, voice) and upgrading-to-v3.md
e2e/                     # end-to-end CI (README.md there)
charts/dis-ai-bot/       # Helm chart: credentials in templates/secret.yaml, the rest in
                         #   configmap.yaml; llm.engine picks llamacpp.yaml (default) or strata.yaml
docker-compose.yaml      # local dev: redis, llamacpp or strata (COMPOSE_PROFILES), diffusion,
                         #   core (mounts ./core and the Docker socket), browser, speech, voice
pyproject.toml           # pytest configuration (why bare `pytest` works from the repo root)
.env / .env.example      # environment configuration (never commit .env)
```

## Runtime architecture

1. **Messages in** (`main.py`, `message_queue.py`). `on_message` runs `should_handle_message()`: a
   server message with content, embeds or attachments, not from the bot, not `!reset_history`, and
   either mentioning the bot or winning the per-guild reply-chance roll. Those go on a bounded queue
   (`QUEUE_MAX_SIZE`; when it is full a mention gets a short "busy" reply) served by `QUEUE_WORKERS`
   workers. A brain call on a worker starts a spare one (up to `BRAIN_MAX_CONCURRENT`), so chat keeps
   `QUEUE_WORKERS` free; the pool shrinks back as workers finish. A message let in by the reply chance
   is marked unprompted, and its brain call gets the `BRAIN_UNPROMPTED_*` limits. A per-channel lock covers only the two fast phases, the prompt build and the send. The slow
   LLM and tool phase runs unlocked, so channels run side by side and a new message can be answered
   while an earlier one in the same channel waits on the brain. **Never hold the lock across an LLM or
   tool await.** A brain call that has run 10 s shows in that channel's newer prompts as a one-line hint;
   finished calls are never listed (the persona read them as "done").
2. **The prompt** (`MessageHandler`, `history_policy.py`, `attachment_cache.py`). History is the
   messages before the trigger's id (a queued trigger has newer messages above it), up to
   `MSG_HISTORY_LIMIT`, each formatted by the one `_format_group`, so a trigger reads the same later as
   history. `MSG_HISTORY_MODE=anchored` keeps the window's start fixed so the LLM server reuses its
   cached prefix. Older history is dropped whole past the per-slot context (`/props`) minus
   `MSG_HISTORY_RESERVE_TOKENS`. The build is `select_messages` (under the lock: reads Discord) then
   `prepare_messages` (after it: downloads and encodes images, cached by ids, never the signed URL, and
   dropped on Discord's delete/edit events). Anything in a history message must stay byte-stable between
   builds: a poll's entry (`poll_format.py`) gives an absolute end time, never "ends in 3h". Polls that
   scrolled out are found with `check_polls` (`poll_store.py`).
3. **The persona** (`TextLLMHandler`). An agents-SDK Agent on the LLM server's OpenAI-compatible API
   (`LLM_HOST/v1`) with ONE tool, `ask_brain` (item 7), in chat, voice turns and automatic runs. The asker's
   saved memories (newest 20, at most 1,500 characters) go in the trailing message with the date and
   time, after the history, so a memory change never breaks the cached prefix; automatic runs get none.
   `PERSONA_MAX_TURNS` is 4, and running out of turns is answered, not failed (`_wrap_up`: one more
   call with `tool_choice="none"`; tool-call text it writes anyway is stripped). Strata is treated like llama.cpp; it differs only in checking
   `LLM_API_KEY` (core's own `/props` and `/v1/models` calls send it, `RoleConfig.auth_headers`), listing
   its model only in the OpenAI `data` shape, and answering any model name.

   **Roles and lanes.** Every model call has a role (`llm_roles.py`: persona, brain, later subagent;
   each with its own model, host, key, effort and call timeout, falling back to `LLM_*`) and a lane
   (`llm_lanes.py`: chat > brain > subagent > background over a self-hosted server's slots, with
   `LLM_CHAT_RESERVED` slots for chat only). Side jobs (research brief, image rewrite and check) fall back
   to the brain's connection. A key only ever falls back to a key for the same host, so a brain on a
   hosted API never gets the local Strata key. A server is laned when its `/props` answers (sized by
   `total_slots`, else `LLM_SLOTS`); a remote API never is. `LanedModel` takes the slot for one model
   call, never across a tool (a side call inside a tool would deadlock against its own run), and applies
   the call timeout after the slot is taken, so laned agents leave `ModelSettings.timeout` unset (it
   would count the wait). It also asks once more when a call is cut off before any output. A side call
   takes the lane of the work making it (`llm_lanes.current_lane`).
4. **The answer** (`response_filter.py`). Cleaned and sent in 2000-character chunks through
   `chunk_for_discord`; never call `textwrap.wrap` directly, as it returns a long unbroken word as one
   oversized chunk that Discord rejects. Reasoning comes out of band (`reasoning_content`, which the SDK
   makes a `reasoning_item`; `extract_reasoning_items`) or, with `--reasoning-format none`, as inline
   think tags (`extract_thinking`). It is dropped, or with `SHOW_THINKING=1` sent as spoiler-hidden
   follow-ups (`MAX_THINKING_CHUNKS`). A failed run returns the `"Error"` sentinel (a ❌), and its
   reasoning is still recovered from the exception's `run_data` and sent.
5. **Redis keys.** Per-guild settings under the `dcb` namespace; memories under
   `guild:<id>:user:<id>`.
6. **Images** (`IMAGE_GEN_ENABLED`: the brain's `generate_image` and `/generate_image`). Both go through
   `image_generation.create_image` to the diffusion service (`DIFFUSION_URL/generate`, queued, one at a
   time), which reports its model family on `/health` (`prompt_style`, `edits`):

   | | SD 1.5 / SDXL | FLUX.2 [klein] 4B (prod) |
   |---|---|---|
   | Prompt | rewritten into an SDXL prompt + negative prompt (`image_prompt.py`; verbatim on failure) | sent as written |
   | Long prompts | chunk-encoded by compel past CLIP's 77 tokens | n/a |
   | Editing | no | yes (`edit_previous`) |
   | Distilled pin | `guidance_scale=0.0`, no negative prompt | `guidance_scale=1.0` |
   | 8 GB card | fp16 + model CPU offload | `IMAGE_QUANTIZE=nf4` + model CPU offload |

   Every image is looked at before it is posted (`image_review.py`, one vision call on the side-job
   connection): something missing earns a retry (`IMAGE_REVIEW_RETRIES`), the better attempt is posted,
   and the tool tells the model what the image really shows. `MAX_IMAGE_CALLS` (3) per reply. Only
   images the bot generated can be edited (`pick_edit_source`), never an uploaded photo. The rewrite
   lives in core, not the tool docstring, because the slash command never reads that docstring.
7. **The brain.** The persona has ONE tool, `ask_brain(request, files)`, in chat, voice turns and
   automatic runs alike, and every other tool is the brain's (`brain_tools.py`; maintainer, 2026-11-02:
   with one tool the persona's choice is "answer, or ask the brain", which fixed its habit of trying jobs
   it could not do). A brain call (`brain.run_brain`) runs a `SandboxAgent` in the server's workspace
   that does the work with its tools and ends by calling `report`; the persona gets the facts back
   (`brain_result.tool_result_for_persona`) and puts them in its own voice. **Docker is required**: core
   needs the daemon's socket (compose: a bind mount; chart: a hostPath) and exits at startup when it
   cannot reach it (`workspace_manager.check_docker`). With Docker down later, a brain call fails and
   says so; there is no brain without the workspace.

   | Concern | Where it lives |
   |---|---|
   | The persona's one tool | `tool_functions.ask_brain` |
   | One brain call: workspace, task, run, files, report | `brain.run_brain` |
   | The brain's agent, its prompt, `report` | `brain.py` (`build_brain_agent`, `BRAIN_INSTRUCTIONS`) |
   | Which tools a call gets | `brain_tools.py` |
   | What the persona is told; must-keep facts | `brain_result.py` |
   | Loop guard, context trimming, limits | `loop_guard.py`, `context_trim.py`, `brain_policy.py` |
   | Status embed, live command output | `brain_progress.py` |
   | Shell wrappers, `attach_file`, delivery, `in/` files | `workspace_tools.py` |
   | The server's container | `workspace_manager.py` (rules: `workspace_lifecycle.py`) |
   | Thread steering, kept for Tier 3 | `sandbox_thread_inbox.py`, `sandbox_conversation*.py` |

   Which tools a brain call gets, in a fixed order so its tool schemas stay a cached prefix:

   | Tools | Chat | Automatic | Voice |
   |---|---|---|---|
   | web, message search/read, polls, events, `change_personality` | yes | yes | yes |
   | memory, schedules / rules / follow-ups | yes | no | yes |
   | `join_voice_channel` / `leave_voice_channel` (`VOICE_ENABLED`) | join | no | leave |
   | `generate_image` (`IMAGE_GEN_ENABLED`) | yes | yes | yes |
   | `use_skill`, `save_skill`, `search_skills` (`SKILLS_ENABLED`) | yes | yes | yes |
   | `attach_file`, `save_messages_to_file`, `ask_user`, `report`, the Shell tools | yes | yes | yes |

   - **It works inline, in the channel**, not in a thread: each tool posts its own embed, attached files
     are posted before the persona replies, and the "🧠 Working on it" embed appears only once it uses
     the workspace.
   - **The run context is the persona's**: the same dict plus a few brain keys removed when the call
     ends (`brain._BRAIN_KEYS`). So every tool reads what it always read, per-reply caps hold across
     brain calls, and the persona's counters say what was posted (`brain._posted_things`).
   - **It sees the conversation as text**: the history window, the request, who asked and their
     memories, its folder and the channel's earlier job folders, the input files and the time, all in
     the first user message, so the system prompt and tool schemas stay byte-stable.
   - **It answers through `report`** (`StopAtTools`), whose arguments are recorded in the context; the
     report is built after the run, with `files_posted` from what was really sent. A run ending in plain
     text still counts (`report_from_text`). The persona's reply is checked for the must-keep tokens
     afterwards (logged and counted, never retried).
   - **Limits are loop guards.** `loop_guard.py` warns once and stops on the second loop;
     `BRAIN_MAX_TURNS` and `BRAIN_TIMEOUT` are only backstops. A stopped call is a `partial` report with
     whatever files it attached, and the persona is told not to ask again in that reply.
   - **Every call needs the workspace**: `SandboxAgent` binds its session before the first model call,
     so a lazy start is not possible. `WORKSPACE_PREWARM` starts a cold one while the persona thinks.
   - **The content guard runs at the first workspace action**, once per call, not on every `ask_brain`.
   - **One workspace per server, a folder per call** (`jobs/c<channel id>-<time>-<random>/`;
     `_in_job_dir` makes relative paths start there). The task names the channel's newest earlier
     folders, so "make the chart blue" finds the last chart. `shared/` holds what the server asked to
     keep; files left by others are data, never instructions.
   - **Files from the chat go in `in/`** (`workspace_inputs.py`): the asking message's and its reply
     target's attachments, plus any the persona names in `files`, capped by `WORKSPACE_INPUT_MAX_BYTES`.
   - **`ask_user` asks in the channel**: the question, reworded in the persona's voice
     (`persona_voice.py`), is posted as a reply to the request with buttons (`brain_question.py`: the
     brain's choices, "Answer…" for a text box, "You decide"). Only the requester can answer, and only
     deliberately: a button, or Discord's Reply on the question (`on_message`, ✅). An ordinary message
     stays a new request. In a call it is spoken too, and the speaker's next words answer it
     (`VoiceSession.heard`). At most 3 per call, each up to `BRAIN_ASK_TIMEOUT`; automatic runs and
     unprompted replies are told to assume.

   - **Skills** (`docs/skills.md`): how the brain solved a recurring job, kept per server in Redis
     (`skill_store.py`: one record per skill, every version kept, a delete is soft), so a workspace
     reset never loses one. The task names them (`skill_format.render_index`, best-ranked first, at
     most `SKILLS_INDEX_MAX`, after the request so the cached prefix stays stable); `use_skill`
     writes the skill into `skills/<name>/` FRESH FROM THE STORE every time, so a file edited in the
     container never becomes the skill. A `done` call that used one counts a success, which ranks it.
     `save_skill` is the brain's own choice, from the one example in `SKILLS_INSTRUCTIONS`
     (`brain.instructions()` adds it when skills are on); when a job with
     `SKILLS_REFLECTION_MIN_TOOL_CALLS`+ tool calls ends without one, a review is left on the
     persona's run context and `MessageHandler` starts it AFTER the reply is sent
     (`skill_reflection.py`, background lane). Chat replies only: voice and automatic runs are not
     reviewed. Anyone in the server manages skills with `/skills` (no permission check by decision,
     since every change is a version and a delete is undoable). Every event posts its own embed
     (`tool_embeds`: 📘 saved, 📗 updated, 📖 used, 🗑️ deleted, ♻️ restored, ⏪ reverted).

   Before editing, read `brain.py`'s module docstring, then `run_brain`, `workspace_tools._deliver` and
   `brain_result.tool_result_for_persona`.

   **The server workspace** (`workspace_manager.py`, rules in `workspace_lifecycle.py`): one container
   per guild, labelled `dcb.role=workspace`, `dcb.guild` and `dcb.instance` (`WORKSPACE_INSTANCE`), so
   the reaper only removes its own.
   - **Getting it**: warm in this process, else `DockerSandboxClient.resume` from the Redis record
     `dcb:workspace:v1:guild:{id}` (survives a core restart), else a new container, filled from the
     guild's snapshot if there is one. The session has **no SDK `RemoteSnapshot`**: with one, `start()`
     after a resume restores the snapshot over the live files.
   - **Many calls share it** (a refcount). Nothing stops or resets it while a call holds it;
     `WORKSPACE_MAX_WARM` evicts the least recently used idle one or makes the new call wait (120 s).
     When the last call ends, everything it left running is killed.
   - **Snapshots** on a timer when changed, before an idle stop and at core shutdown (containers are
     left running). One over `WORKSPACE_SNAPSHOT_MAX_BYTES` keeps the previous snapshot and the
     container is only stopped, not removed.
   - **Limits** (`WORKSPACE_MEMORY_LIMIT` / `CPU` / `PIDS`) go on every container through one wrapper
     of the docker client's `containers.create`, since the SDK has no option for them.
   - **Kubernetes**: core uses its node's daemon. A core pod that moves node starts from the snapshot;
     the old container is reaped when core runs on that node again.

8. **What's New and /help.** After the first successful reply in a guild on a new version,
   `main.maybe_announce_whats_new` posts that version's notes (`core/whats_new.md`) as one embed. One
   atomic Redis `SET ... GET` claims it, so it shows once per guild per version even with several
   workers. `_format_group` drops the embed from prompts. `/help` and `/whats_new` reply ephemerally.
9. **Schedules and rules** (`docs/automations.md`), stored under `dcb:automations:v1`. Automatic work
   rides the same queue as chat (`AutomationJob`), so `QUEUE_WORKERS` and the channel locks bound it.
   `schedule_forever` polls due schedules every 15 s and admits only what the queue has room for; an
   admitted rule replaces the ordinary reply. An automatic run has no source message (`automatic=True`,
   `actor_id` = the entry's last editor): no memory or automation tools, no memories, no double replies.
   A started run is never replayed, since tools may already have posted. A lost channel, a departed
   editor or a missing permission suspends the entry. **Follow-ups** (`schedule_followup`) are one-off
   schedules with their own quota, at least `SCHEDULE_MIN_INTERVAL_HOURS` apart per person
   (`automation_policy.followup_conflict`); a follow-up cannot set another.
10. **Voice calls** (`docs/voice.md`, off unless `VOICE_ENABLED`). `/voice join` or `join_voice_channel`
    starts a `VoiceSession`; `end()` is the only way out.

    | Concern | Where it lives |
    |---|---|
    | Discord voice I/O (DAVE), per-speaker segmenting, playback | `voicesidecar/` (Node) |
    | Speech-to-text / text-to-speech, CPU only | `speechservice/` |
    | discord.py's half of the handshake, the bridge WebSocket | `voice_bridge.py` |
    | Wake phrase, commands, history window, sentences | `voice_policy.py` |
    | What a call does with each utterance; turns; prefill | `voice_session.py` |

    - **Why a Node sidecar**: discord.py cannot receive DAVE-encrypted audio, which Discord made
      mandatory. discord.py keeps the gateway and forwards the voice handshake
      (`SidecarVoiceProtocol`); audio never reaches Python, only transcripts and sentences.
    - **Text stops in a guild while it is in a call** (`voice_gate`): mentions get a rate-limited
      notice, queued messages are dropped, automation jobs are given back, due schedules stay due and
      `/generate_image` refuses. In-process only: a restart ends every call.
    - The wake phrase is matched fuzzily in the transcript (`voice_policy.match_wake`); Whisper gets no
      prompt (given one, it leaves those words out). "leave" / "stop" never go to the LLM.
    - Turns stream sentence by sentence with thinking off (`VOICE_THINKING`). A tool call is covered by
      a spoken hold-on line, written by a side call (`HOLD_ON_PROMPT`) if the model wrote none.
    - **The cache is kept warm**: anchored history (`VOICE_HISTORY_*`) and a rate-limited `prefill()`
      per transcript with exactly a turn's prefix (pinned by `voice_prefill_tests.py`).
    - Output (images, brain files) goes to a session thread off the channel the join came from.
      `original_message` is None in a voice turn: tools read `context["channel"]`, `user_id` and
      `request_text`.
11. **Server events** (`docs/events.md`, `event_tools.py`): create, list and move/cancel Discord
    scheduled events. **The person asking must hold Create Events or Manage Events**, not just the bot,
    resolved like `automation_tools._scope`, so the check also runs in voice turns and automatic runs
    (as the entry's editor). A follow-up skips the person's check, not the bot's
    (`event_tools._checks_requester`). Only events the bot made can be changed. Naive times are read in
    `AUTOMATIONS_TIMEZONE`; results give `<t:unix:F>`. Finding a time is a poll plus a follow-up.
12. **Links and message search.**
    - `fetch_url(url, part)` (`web_reader.py`): `web_policy.route` picks YouTube captions, Reddit's RSS
      feeds, Steam's `appdetails`, else a plain GET read by trafilatura, with the browser service
      (`BROWSER_URL`) when that text is too thin. Cached in Redis (`dcb:webcache:v1:<sha256>`), one load
      per URL at a time; long pages come in parts (`WEB_FETCH_PART_CHARS`).
    - **It never reaches private addresses**: one aiohttp session whose resolver drops non-public
      addresses, plus a check of every redirect hop for a private IP literal. Chromium's only way out is
      the browser service's egress proxy (`browserservice/egress_proxy.py`), because Playwright's route
      handlers never see a redirect's later hops. The chart adds a NetworkPolicy.
    - `web_search` (`web_research.py`): DuckDuckGo, the top `RESEARCH_PAGES` read through the same
      reader, and a side call writes a brief of at most 200 words; the source list is built from the
      real results. On failure it returns the plain results.
    - `search_messages` / `read_messages_around` (`message_search_tools.py`): Discord's Search Guild
      Messages endpoint (not wrapped by discord.py; a 202 means not indexed yet) and the messages around
      one link. Both read what the bot can see, in this server only. `save_messages_to_file` writes every
      match (up to 10000, 20 MB) as JSON lines into the brain's `in/`, so code can count or chart them.

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

**/help is hand-written** (`classes/help_catalog.py`). A new brain tool
(`brain_tools.brain_tools`) or slash command (`main.register_commands`)
needs an entry there, gated with `requires` when it can be switched off;
`core/tests/help_catalog_tests.py` fails until it has one.

### Prompt surface

Everything the models are told lives in code. The persona sees one tool: the `ask_brain` docstring
(its whole routing rule, with one positive example per kind) and its return string,
`brain_result.tool_result_for_persona`. The brain sees `BRAIN_INSTRUCTIONS` (`brain.py`), every other
tool's `@function_tool` docstring and each tool's return strings (read at reply-writing time; these
are the ones that actually change behaviour). The persona's *system* prompt is not a lever: it is
`f"Answer as if you are {redis['dcb:{guild}:system']}"`, user-owned via `/system` and
`change_personality`. The one exception is a voice turn, which appends
`voice_policy.VOICE_INSTRUCTIONS`; `voice_policy` is also the one home of `GREETING_PROMPT` and
`HOLD_ON_PROMPT`.

- **One home per concept.** When adding guidance, edit its existing home rather than restating it
  nearby: a small model handed two versions of a rule follows neither reliably.
- **Prefer a positive example over a list of prohibitions.** A model given a list of things not to
  choose went on to choose items straight off the list. The prompts show the shape instead
  (`ask_brain`'s examples, the `report(...)` call in `BRAIN_INSTRUCTIONS`).
- Sizes are pinned by `brain_tests.py`: `BRAIN_INSTRUCTIONS` ~2230 characters (ceiling 2300),
  `ask_brain` ~1570 including its JSON schema (ceiling 1600). Re-measure before adding. Skills have
  their own homes, pinned by `skill_tools_tests.py`: `SKILLS_INSTRUCTIONS` ~270 characters (ceiling
  400, added to the brain's prompt only when skills are on) and `REFLECTION_PROMPT` ~730 (ceiling
  1000, the one prompt of the after-job review).

### Environment variables

v3 keeps no aliases for renamed settings: an old name in the environment makes core log it with its
replacement and exit (`settings_check.py`), and an old values key fails `helm template`
(`templates/_removed.tpl`). Every rename goes in `settings_check.REMOVED`, `_removed.tpl` (values keys)
and `docs/upgrading-to-v3.md` in the same PR. `settings_docs_tests.py` checks that every env var core
reads has a row here and a line in `.env.example`, that every row is read by something, and that every
`values.yaml` key is used by a template. Defaults are in brackets.

| Var | Purpose |
|---|---|
| `DISCORD_TOKEN` | required; bot token. Chart: `discordToken`, or bring your own Secret via `existingSecret` (keys = env var names: `DISCORD_TOKEN`, `LLM_API_KEY`, `CONTENT_GUARD_API_KEY`, `BRAIN_LLM_API_KEY`, `IMAGE_PROMPT_LLM_API_KEY`, `IMAGE_REVIEW_LLM_API_KEY`, `RESEARCH_LLM_API_KEY`, `PERSONA_LLM_API_KEY`); `llamacpp.existingSecret` does the same for `HF_TOKEN` |
| `REDIS_HOST` | required; Redis host |
| `LLM_HOST` | the bot's own LLM server (core appends `/v1`), every role's fallback. Compose: the `llamacpp` service (:8081) or `http://strata:8080`. Chart: `llm.host`, empty = the in-cluster service of `llm.engine` [`http://llamacpp:8080`] |
| `LLM_API_KEY` | that server's key: a placeholder for llama.cpp (no auth, but the client needs one); the real key for Strata, which checks it (chart: `strata.apiKey` / `strata.existingSecret`, key `STRATA_API_KEY`; compose gives `strata` the same value). Chart: `llm.apiKey` for a server outside the chart |
| `LLM_EXTRA_HEADERS` | comma-separated `name=value` headers on every request to `LLM_HOST` (none), e.g. OpenCode Go's required `x-opencode-session` (E2E sets one per run). Never sent to another host; `Authorization` is skipped |
| `LLM_SLOTS` | slots of a self-hosted server whose `/props` has no `total_slots` [1]. Chart: `llm.slots`, empty = `llamacpp.parallel` or `strata.parallel` |
| `LLM_CHAT_RESERVED` | slots per self-hosted server only chat replies may use [1]. Chart: `llm.chatReserved` |
| `PERSONA_MODEL` / `PERSONA_LLM_HOST` / `PERSONA_LLM_API_KEY` | the persona's own connection; empty = `LLM_*` (the key only when the host is `LLM_HOST`). Chart: `persona.model` / `llmHost` / `apiKey` |
| `LLM_MODEL` | **required**; the model the bot requests and compose's `llamacpp` serves (`LLAMA_ARG_HF_REPO`); the chart's one `llm.model` feeds both. With Strata only the requested name (Strata serves `strata.model` / `STRATA_MODEL`) |
| `CONTENT_GUARD_API_KEY` | key for the moderation call (an OpenAI-compatible Moderations endpoint; OpenAI's is free); unset = fail-open. Chart: `contentGuard.apiKey` |
| `CONTENT_GUARD_BASE_URL` | moderation API base [`https://api.openai.com/v1`]. Chart: `contentGuard.baseUrl` |
| `CONTENT_GUARD_MODEL` | moderation model; unset = the server's default. Chart: `contentGuard.model` |
| `CONTENT_GUARD_CACHE_TTL_SECONDS` | seconds a verdict stays cached [600; 0 = off]. Chart: `contentGuard.cacheTtlSeconds` |
| `CONTENT_GUARD_MAX_RETRIES` / `CONTENT_GUARD_RETRY_BACKOFF` | retries on 429/5xx [2] and base backoff seconds, doubled per retry [2]. Chart: `contentGuard.maxRetries` / `retryBackoff` |
| `CONTENT_GUARD_ENABLED` | `0` turns the guard off [on]. Chart: `contentGuard.enabled` |
| `CONTENT_GUARD_DEBUG` | `1` logs per-request detail [off]. Chart: `contentGuard.debug` |
| `WEB_FETCH_PART_CHARS` / `WEB_FETCH_MAX_CHARS` | characters per `fetch_url` part / most kept per page [10000 / 200000]. Chart: `webFetch.partChars` / `maxChars` |
| `WEB_FETCH_MAX_BYTES` / `WEB_FETCH_TIMEOUT` | most bytes downloaded per page / seconds to wait [5000000 / 15]. Chart: `webFetch.maxBytes` / `timeout` |
| `WEB_CACHE_TTL_SECONDS` / `WEB_CACHE_VIDEO_TTL_SECONDS` / `WEB_CACHE_FAILURE_TTL_SECONDS` | Redis cache life of a page / a YouTube transcript / a failure [21600 / 2592000 / 300]. Chart: `webFetch.*TtlSeconds` |
| `BROWSER_URL` | the browser service for thin pages; empty = none. Compose: `http://browser:8000`; chart: `browser.url`, else the in-cluster service when `browser.enabled` |
| `BROWSER_MIN_CHARS` | a plain fetch with less text goes to the browser [400]. Chart: `webFetch.browserMinChars` |
| `RESEARCH_BRIEF_ENABLED` | `0`: `web_search` returns the plain results, no pages read and no brief [on]. Chart: `research.enabled` |
| `RESEARCH_PAGES` / `RESEARCH_PAGE_CHARS` | results read in full [3, max 5] and characters of each given to the brief [6000]. Chart: `research.pages` / `pageChars` |
| `RESEARCH_FETCH_TIMEOUT` / `RESEARCH_TIMEOUT` | seconds for those pages [25] and for the brief [90]. Chart: `research.fetchTimeout` / `timeout` |
| `RESEARCH_MODEL` / `RESEARCH_LLM_HOST` / `RESEARCH_LLM_API_KEY` | the brief's connection; empty = the brain's, then `LLM_*` (the key only for the same host). Chart: `research.*` |
| `BROWSER_CONCURRENCY` / `BROWSER_QUEUE_SIZE` | browser service: pages rendered at once / renders waiting before a 503 [2 / 8]. Chart: `browser.concurrency` / `queueSize` |
| `BROWSER_TIMEOUT` / `BROWSER_SETTLE_SECONDS` | browser service: seconds per render / for scripts to settle [30 / 5]; core reads both to know how long to wait. Chart: `browser.timeout` / `settleSeconds` |
| `METRICS_PORT` | Prometheus `/metrics` port [9464]; empty/`0` = off. Chart: `metrics.enabled` / `metrics.port` add a Service and a ServiceMonitor (label `release: kube-prometheus-stack`), and turn on llama.cpp's or Strata's own `/metrics` with a second ServiceMonitor |
| `LOG_LEVEL` | root log level [`INFO`]; `DEBUG` also dumps raw agent run results |
| `MSG_HISTORY_LIMIT` | messages in a (refreshed) history window, the trigger included [5]. Chart: `history.limit` |
| `MSG_HISTORY_MODE` | `sliding` [default] or `anchored` (the window's start stays fixed, for prefix reuse). Chart: `history.mode` |
| `MSG_HISTORY_REFRESH_MESSAGES` | anchored mode: new messages that move the anchor [10]. Chart: `history.refreshMessages` |
| `MSG_HISTORY_RESERVE_TOKENS` | per-slot context kept free for the system prompt, tools and answer; older history is dropped past it [12000]. Chart: `history.reserveTokens` |
| `PERSONA_REASONING_EFFORT` | the persona's `reasoning_effort` [medium]. Chart: `persona.reasoningEffort` |
| `PERSONA_MAX_TURNS` | model turns per persona reply [4]: ask the brain, maybe again, answer. An overrun is still answered. Chart: `persona.maxTurns` |
| `PERSONA_CALL_TIMEOUT_SECONDS` | wall-clock seconds per persona model call, client retries included [600], applied after the lane wait; also the per-read timeout. A per-read timeout alone never trips on a server that pads its responses (OpenRouter). Chart: `persona.callTimeout` |
| `WHATS_NEW_ENABLED` | `0` stops the automatic What's New embed; `/whats_new` still works [on]. Chart: `whatsNew.enabled` |
| `APP_VERSION` | the release tag, baked into the core image by `release.yaml`; unset = no announcement |
| `SHOW_THINKING` | `1` sends the model's reasoning as spoiler-hidden follow-ups [off]. Chart: `showThinking` |
| `DOUBLE_REPLY_CHANCE` | chance of offering two short messages from one generation [0.08; 0 = off]. Chart: `doubleReply.chance` |
| `DOUBLE_REPLY_COOLDOWN_REPLIES` | replies per channel needed after a pair [10]; in-process only. Chart: `doubleReply.cooldownReplies` |
| `AUTOMATIONS_ENABLED` | `0` stops schedules and rules and hides their commands and tools; entries are kept [on]. Chart: `automations.enabled` |
| `AUTOMATIONS_TIMEZONE` | default IANA timezone for new schedules [`Europe/London`]. Chart: `automations.timezone` |
| `SCHEDULE_MIN_INTERVAL_HOURS` | minimum hours between runs of a recurring schedule [4, min 1]; one-offs are exempt. Chart: `schedules.minIntervalHours` |
| `SCHEDULE_MAX_PER_GUILD` / `RULE_MAX_PER_GUILD` | active entries per server [1 / 4; 0 blocks new ones]. Chart: `schedules.maxPerGuild` / `rules.maxPerGuild` |
| `RULE_COOLDOWN_SECONDS` | seconds before the same rule can fire again [60]. Chart: `rules.cooldownSeconds` |
| `FOLLOWUP_MAX_PER_GUILD` | follow-ups waiting at once per server [3; 0 = off]. Chart: `followups.maxPerGuild` |
| `EVENTS_ENABLED` | `0` removes the event tools [on]; the bot's role needs Create Events. Chart: `events.enabled` |
| `QUEUE_WORKERS` | queue workers [2, min 1]. Chart: `queue.workers` |
| `QUEUE_MAX_SIZE` | messages waiting on the queue [10, min 1]. Chart: `queue.maxSize` |
| `CORE_ENV_FILE` | compose only: the env file handed to core [`.env`]. A second env file (`.env.v3`) sets it to its own name, since `--env-file` alone only changes compose's own values |
| `LLAMA_CTX_SIZE` / `LLAMA_PARALLEL` / `LLAMA_CACHE_RAM_MIB` | compose `llamacpp` only: total context over all slots [80000; keep 32000+ per slot], slots [1], host-RAM prompt cache in MiB [8192]. Chart: `llamacpp.contextPerSlot` (x `parallel`), `llamacpp.parallel`, `llamacpp.cacheRamMiB` |
| `LLAMA_CACHE_TYPE_K`, `LLAMA_CACHE_TYPE_V` | compose `llamacpp` only: KV cache type [`q4_0`]. Chart: `llamacpp.cacheTypeK` / `cacheTypeV` |
| `IMAGE_GEN_ENABLED` | `0` removes `generate_image` [on]. Chart: `diffusion.enabled`, which also removes the diffusion pod and PVC |
| `DIFFUSION_URL` | the diffusion service (core appends `/generate`) [`http://diffusion:8000`]; in-cluster service in the chart |
| `IMAGE_MODEL` | HF repo for the diffusion service [`stabilityai/sd-turbo`], downloaded on first boot |
| `IMAGE_STEPS` / `IMAGE_WIDTH` / `IMAGE_HEIGHT` | generation settings [4 steps, 512x512] |
| `IMAGE_GUIDANCE` | CFG scale; unset = the pipeline's own. Pinned on distilled models. Chart: `diffusion.guidance` |
| `IMAGE_NEGATIVE_PROMPT` | baseline negative prompt, behind the rewriter's; dropped on distilled models, unused by FLUX.2. Chart: `diffusion.negativePrompt` |
| `IMAGE_QUANTIZE` | `none` [default] / `nf4`: FLUX.2 [klein] in 4-bit, to fit an 8 GB card; ignored for SD/SDXL. Chart: `diffusion.quantize` |
| `IMAGE_LONG_PROMPT` | `0` truncates prompts at 77 CLIP tokens instead of chunk-encoding them [on]. Chart: `diffusion.longPrompt` |
| `IMAGE_OFFLOAD` | `model` [default] / `sequential` (lowest VRAM, slowest) / `none` |
| `IMAGE_QUEUE_SIZE` | image requests queued before a 503 [16] |
| `IMAGE_GEN_TIMEOUT` | seconds core waits on the diffusion service [300] |
| `IMAGE_PROMPT_REWRITE_ENABLED` | `0` sends image requests verbatim [on]. Chart: `diffusion.promptRewrite.enabled` |
| `IMAGE_PROMPT_MODEL` / `IMAGE_PROMPT_LLM_HOST` / `IMAGE_PROMPT_LLM_API_KEY` / `IMAGE_PROMPT_TIMEOUT` | the rewrite's connection; empty = the brain's, then `LLM_*` (the key only for the same host); timeout [60]. Chart: `diffusion.promptRewrite.*` |
| `IMAGE_REVIEW_ENABLED` | `0` posts generated images unchecked [on]. Chart: `diffusion.review.enabled` |
| `IMAGE_REVIEW_RETRIES` | extra attempts when something asked for is missing [1, max 2; 0 = check only]. Chart: `diffusion.review.retries` |
| `IMAGE_REVIEW_MODEL` / `IMAGE_REVIEW_LLM_HOST` / `IMAGE_REVIEW_LLM_API_KEY` / `IMAGE_REVIEW_TIMEOUT` | the check's connection (a vision model), falling back like `IMAGE_PROMPT_*`. Chart: `diffusion.review.*` |
| `BRAIN_MAX_TURNS` | backstop on model turns per brain call [100]. Chart: `brain.maxTurns` |
| `BRAIN_TIMEOUT` | backstop on a brain call's seconds, lane waits included [1800]. Chart: `brain.timeout` |
| `BRAIN_REPEAT_LIMIT` | loop guard: the same call this many times in a row, or eight calls that bring nothing new, warns; a second time stops the call [3, min 2]. Chart: `brain.repeatLimit` |
| `BRAIN_UNPROMPTED_MAX_TURNS` / `BRAIN_UNPROMPTED_TIMEOUT` | backstops for a brain call in a reply the bot was not asked for (the random reply chance): nobody is waiting on it [40 / 180]. Chart: `brain.unprompted.maxTurns` / `timeout` |
| `BRAIN_ASK_TIMEOUT` | seconds `ask_user` waits for the requester's answer in the channel, cut to what the call has left [120]. Chart: `brain.askTimeout` |
| `SKILLS_ENABLED` | `0`/`false` turns skills off: no skill tools for the brain, no `/skills`, no index in its task; saved skills are kept (default: on). Chart: `skills.enabled` |
| `SKILLS_MAX_PER_GUILD` / `SKILLS_MAX_VERSIONS` | live skills per server (default 100; deleted ones do not count), and versions kept per skill (default 50; past it the oldest goes). Chart: `skills.maxPerGuild` / `maxVersions` |
| `SKILLS_MAX_BYTES` / `SKILLS_MAX_BODY_CHARS` / `SKILLS_MAX_FILES` | limits on one skill: all its text together (default 65536), its steps (8000 characters), files kept with it (10). Chart: `skills.maxBytes` / `maxBodyChars` / `maxFiles` |
| `SKILLS_INDEX_MAX` | skills named in the brain's task (default 20, best ranked first); the rest are found with `search_skills`. Chart: `skills.indexMax` |
| `SKILLS_REFLECTION_ENABLED` / `SKILLS_REFLECTION_MIN_TOOL_CALLS` | after a brain job with at least this many tool calls (default 6) that ended done or partial without saving a skill, one background call decides whether to save one (default: on). Chart: `skills.reflection.enabled` / `minToolCalls` |
| `BRAIN_MAX_CONCURRENT` / `BRAIN_QUEUE_WAIT` | brain calls at once (each on a queue worker starts a spare worker), and seconds a call waits for a slot before it is told the bot is busy [3 / 120]. Chart: `brain.maxConcurrent` / `queueWait` |
| `BRAIN_TRIM_KEEP_RECENT` | tool outputs kept whole in a long brain call; older ones are cut to their first and last 400 characters [6]. Chart: `brain.trimKeepRecent` |
| `WORKSPACE_IMAGE` | the workspace's container image, pulled at startup if missing [`python:3.14-slim`]. Chart: `brain.workspace.image` |
| `WORKSPACE_IDLE_SECONDS` | seconds after the last task before a workspace is saved and stopped [900]. Chart: `brain.workspace.idleSeconds` |
| `WORKSPACE_SNAPSHOT_INTERVAL_SECONDS` | seconds between saves of a running workspace that changed; 0 = only when it stops [1800]. Chart: `brain.workspace.snapshotIntervalSeconds` |
| `WORKSPACE_MAX_WARM` | workspaces running at once per core [4]; the least recently used idle one stops first. Chart: `brain.workspace.maxWarm` |
| `WORKSPACE_JOB_RETENTION_HOURS` | job folders older than this are deleted before each save [72]. Chart: `brain.workspace.jobRetentionHours` |
| `WORKSPACE_MEMORY_LIMIT` / `WORKSPACE_CPU_LIMIT` / `WORKSPACE_PIDS_LIMIT` | per container: memory (`4g`; swap capped the same), CPUs, processes [512; 0 = none]; empty = no cap. Outside Kubernetes accounting. Chart: `brain.workspace.memoryLimit` / `cpuLimit` / `pidsLimit` |
| `WORKSPACE_INSTANCE` | label on this bot's containers, so two bots on one daemon never remove each other's [`dcb-dev`]. The chart sets `<namespace>-<release name>` |
| `WORKSPACE_PREWARM` | start a server's workspace when the bot is about to answer there [1; 0 = off]. Chart: `brain.workspace.prewarm` |
| `BRAIN_MODEL` | the brain's model; empty = `LLM_MODEL`. Chart: `brain.model` |
| `BRAIN_LLM_HOST` | the brain's server (core appends `/v1`), e.g. `https://openrouter.ai/api`; empty = `LLM_HOST`. Chart: `brain.llmHost` |
| `BRAIN_LLM_API_KEY` | the brain's key; empty = `LLM_API_KEY`, but only when the brain's host is `LLM_HOST` (else none; startup warns). Chart: `brain.apiKey` |
| `BRAIN_REASONING_EFFORT` | the brain's `reasoning_effort` [medium]. Chart: `brain.reasoningEffort` |
| `SANDBOX_ASK_USER_TIMEOUT` | seconds the thread `ask_user` (kept for Tier 3, unused) waits for a reply [300]. Chart: `sandbox.askUserTimeout` |
| `WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS` | seconds for one workspace save, and for all of them at shutdown [120]. Chart: `brain.workspace.snapshotTimeout` |
| `BRAIN_CALL_TIMEOUT_SECONDS` | like `PERSONA_CALL_TIMEOUT_SECONDS`, for the brain [600]. Chart: `brain.callTimeout` |
| `WORKSPACE_SNAPSHOT_MAX_BYTES` | most bytes of one saved workspace [50MB]; over it the previous save is kept and the container is stopped, not removed. Chart: `brain.workspace.snapshotMaxBytes` |
| `WORKSPACE_INPUT_MAX_BYTES` | chat files copied into a task's `in/`, all together [25MB, at most 10 files]. Chart: `brain.workspace.inputMaxBytes` |
| `WORKSPACE_SNAPSHOT_TTL_SECONDS` | how long an unused server's saved workspace lasts [604800 = 7 days]. Chart: `brain.workspace.snapshotTtlSeconds` |
| `VOICE_ENABLED` | `1` turns voice calls on [off in code]. The chart sets it from `voice.enabled` [true], which also adds the sidecar container and the speech pod |
| `VOICE_BRIDGE_URL` | the voice sidecar's WebSocket [`ws://voice:8765`; chart: `ws://127.0.0.1:<voice.bridgePort>`]. The sidecar binds `VOICE_BRIDGE_HOST` / `VOICE_BRIDGE_PORT` |
| `VOICE_BRIDGE_TOKEN` | optional shared secret the sidecar requires of a bridge connection; for compose, where the bridge listens on the whole compose network. The chart binds it to localhost and leaves it unset |
| `SPEECH_URL` | the speech service, used by the sidecar and `/voice voice`'s autocomplete [`http://speech:8000`]; chart: the in-cluster service, or `voice.speechUrl` |
| `VOICE_IDLE_LEAVE_SECONDS` | seconds the bot stays after the last person leaves [30]. Chart: `voice.idleLeaveSeconds` |
| `VOICE_HISTORY_LIMIT` / `VOICE_HISTORY_REFRESH` | clips kept in a call's prompt, and how many more before rebasing [10 / 10]. Chart: `voice.historyLimit` / `voice.historyRefresh` |
| `VOICE_FOLLOWUP_SECONDS` | after the wake phrase alone, how long that speaker's next words count as the request [8]. Chart: `voice.followupSeconds` |
| `VOICE_THINKING` | `1` lets the model think before a spoken reply [off]. Chart: `voice.thinking` |
| `VOICE_DEFAULT_VOICE` | Kokoro voice for servers that have not picked one [`af_heart`]; read by core and the speech service. Chart: `voice.defaultVoice` |
| `VOICE_SILENCE_MS` | silence that ends an utterance, read by the sidecar [800]. Chart: `voice.silenceMs` |
| `VOICE_PREFILL_MIN_SECONDS` | minimum seconds between two prefills of a call's prompt [5]. Chart: `voice.prefillMinSeconds` |
| `VOICE_DEBUG` | `1` logs every transcript and spoken sentence (core) and the voice connection's debug output (sidecar) [off: it is everything said in a call] |
| `VOICE_DEV_INJECT` | local testing only, never in the chart: `!voice_join <channel id>` / `!voice_say <text>` drive a call from text |
| `STT_MODEL` / `STT_THREADS` / `STT_CONCURRENCY` | speech service: faster-whisper model [`small.en`, int8], CPU threads per transcription [4], transcriptions at once [2]. Chart: `voice.stt*` |
| `TTS_THREADS` / `TTS_CONCURRENCY` / `SPEECH_QUEUE_SIZE` | speech service: Kokoro threads per sentence [4], sentences at once [2], requests waiting per kind before a 503 [16]. Chart: `voice.tts*`, `voice.queueSize` |

## Releasing

- Every push to `main` runs `auto-tag.yaml`, which bumps the highest `vMAJOR.MINOR` tag by one minor and
  pushes it. That tag push starts `release.yaml` (tests -> five images -> chart -> GitHub release); its
  `workflow_dispatch` trigger is only for re-running a release against an existing tag. Dependabot's
  auto-merged PRs are releases like any other.
- The chart `.tgz` is published as a GitHub release asset (`dchatbot-vX.Y.tgz`) and as an OCI artifact
  at `oci://ghcr.io/dgowing95/charts/dchatbot` (tag = the git tag), which is what ArgoCD and
  `helm install oci://...` resolve. Both carry `appVersion` = the tag, which picks the `core-<tag>` /
  `diffusion-<tag>` image tags.
- **`TAG_PUSH_TOKEN` (the `release-tag` environment secret) creates the tag**, and it expires.
  `GITHUB_TOKEN` cannot: the "Restrict Tagging" ruleset lets only repo admins create tags, and a ref
  pushed with it fires no `push` event, so `release.yaml` would never start.
- To mint a replacement: your **account** settings (not the repo's) → Developer settings → Personal
  access tokens → Fine-grained tokens → Generate new token, scoped to `dgowing95/DiscordChatBot` only,
  with **Repository permissions → Contents: Read and write** (nothing else). Store it in the **repo's**
  settings → Environments → `release-tag` → Environment secrets, as `TAG_PUSH_TOKEN`. The environment
  is restricted to the protected `main` branch.
- How Auto Tag fails tells you what is wrong: secret unset = the `gh api` step fails authentication;
  expired or revoked = a 401; a ruleset rejection on `refs/tags/<version>` = the token works but its
  owner is not a bypass actor on the tag ruleset.

## Testing

- **Framework:** `pytest`, configured in `pyproject.toml` at the repo root (`python_files = *_tests.py`,
  plus `testpaths` and `pythonpath`).
- **Prereq:** a Python 3.13+ venv with `pip install -r core/requirements-dev.txt`.
- **Run** from the repo root, no arguments and no `PYTHONPATH`: `pytest`. This is exactly what CI runs
  (`tests.yaml`, and again as `release.yaml`'s gate), so a new test file is picked up with no list to
  update. ~1500 tests, roughly 25 seconds. A new setting fails it until it has an env-table row and an
  `.env.example` line.
- The voice sidecar has its own `node:test` suite: `cd voicesidecar && npm ci && npm test` (a separate
  CI job).
- **One import name:** tests import production modules as `classes.X`, as the app does (cwd `/app`).
  Never `core.classes.X`: that is a separate module object with its own globals, so a patch on one copy
  misses the other and import-time state (metrics, lock and in-flight registries) exists twice.
  `grep -rn "core\.classes\|core\.tests" core/` should stay empty. To borrow a helper from another test
  module, import it by its bare name (`from response_filter_tests import _reasoning_item`).
- **Never write to `os.environ` in a test**; use `monkeypatch`. A leaked `REDIS_HOST` once made the whole
  suite seem to hang on Redis connect timeouts.
- Stub Redis with fixtures and `unittest.mock`. Anything worth unit-testing goes in a pure module.
- Test the SDK boundary against the **real `Runner`** with `agents.testing.ScriptedModel`
  (`brain_tests.py`, `sandbox_conversation_tests.py`, `max_turns_wrap_up_tests.py`) rather than mocking
  `Runner`, so an SDK upgrade that changes the boundary fails a test.
- Where the tests for the concurrency model live: `message_queue_tests.py` (also `main.py`'s
  `on_message` / `process_messages`; importing `main.py` starts nothing), `task_registry_tests.py`,
  `message_handler_tests.py` (scoped locks, image downloads) and `metrics_tests.py`. History:
  `history_policy_tests.py` and `message_history_tests.py`, whose `FakeChannel` mirrors discord.py's
  `history()` (`after=` keeps the OLDEST `limit` messages). A passing history test shows the message list
  is stable, not that llama.cpp reuses its cache; check `llamacpp:prompt_tokens_cached_total` for that.
  Answer cleaning and reasoning: `response_filter_tests.py`; the image cache: `attachment_cache_tests.py`.

### Manual/live testing via a Discord webhook

The bot can be driven end to end against a running deployment by POSTing through a Discord webhook:

```bash
curl -sS -X POST "$TEST_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"content": "<@BOT_USER_ID> your test message here"}'
```

- The content needs a real `<@id>` mention of the bot's user ID (a plain `@botname` does nothing).
- `TEST_WEBHOOK_URL` (`.env`, local dev only) is not read by the app; it is there so it need not be
  found again each session. **Never** commit a webhook URL, and never point one at a production channel:
  a webhook post is indistinguishable from real user traffic.
- A webhook author is not a member, so `schedule_followup` refuses it.
- Watch it: `docker compose logs core -f`, or `kubectl -n <namespace> logs
  deployment/<release>-core-deployment -f`. Compose bind-mounts `./core`, so `docker compose restart
  core` picks up code edits.
- Voice without anyone speaking: `VOICE_ENABLED=1 VOICE_DEV_INJECT=1` (and `docker compose up -d speech
  voice core`), then POST `!voice_join <voice channel id>` and `!voice_say hey <name>, ...`;
  `VOICE_DEBUG=1` logs what is heard and said.

### End-to-end CI (and Dependabot auto-merge)

`.github/workflows/e2e.yaml` runs on every PR touching `core/`, `speechservice/`, `voicesidecar/`,
`diffusionservice/` or `e2e/`. It starts the PR's bot from source (`e2e/start_stack.sh`) against a real
LLM and its own CPU image service, and talks to it through a webhook in a private test server
(`e2e/run_checks.py`). It checks structure (a reply arrived, a fact is in Redis, a PNG was attached),
never wording. `e2e/README.md` lists every check; a failure shows in the step summary and the logs are
in the `e2e-logs` artifact.

- **Runner**: `ubuntu-latest`, a VM thrown away after each job, so a PR's unreviewed dependencies never
  run on a machine that is kept. The LLM is a hosted API (named in `e2e.yaml`), not prod's model. The
  image service runs on CPU with a tiny model and the vision check off, so the CUDA/nf4 paths are never
  covered and `diffusionservice/` PRs are never auto-merged.
- **Secrets** `E2E_DISCORD_TOKEN`, `E2E_WEBHOOK_URL` and `E2E_VOICE_CHANNEL_ID` belong to a bot and
  server made only for testing. `OPENCODE_API_KEY` is NOT test-only: it is `review.yml`'s key too, by the
  owner's choice. Each run sends its own `x-opencode-session` through `LLM_EXTRA_HEADERS`. Every secret
  goes in **both** the Actions and the Dependabot secrets (a Dependabot run sees only the latter).
- **Security rules** (the repo is public and the job runs a PR's new dependencies before review):
  - Keep "Require approval for all external contributors" on (Settings → Actions). A fork edits its own
    copy of the workflow, so `runs-on:` and `if:` are not a boundary.
  - Stay on GitHub-hosted runners; the job has sudo and Docker, which is fine only because the VM is
    thrown away. Everything the job can read, the PR's code can read. If the OpenCode key is ever
    misused, rotate it in both secret stores and consider a key used only here.
  - Never switch to `pull_request_target` to reach secrets.
  - One run at a time across the repo (the `concurrency` group): two bots on one token would both
    answer. A cancelled `e2e` blocks its PR until it is re-run.
- **Required checks**: both `e2e` and `e2e-changes` in "Protect main". `e2e` alone is not enough: if
  `e2e-changes` fails, `e2e` is skipped, and a skipped job counts as passing.
- **Auto-merge** (`dependabot-auto-merge.yaml`) approves a Dependabot PR and sets it to merge once the
  required checks pass. It is an **allowlist**: patch/minor, ecosystem pip or docker, directory `/core`,
  `/speechservice` or `/voicesidecar`, and ungrouped or the `core-minor` group. Everything else waits for
  a person (majors, `github_actions`, npm, `/diffusionservice`, `/`, and the `openai` / `huggingface` /
  `speech` / `voice` groups). A new ecosystem, directory or group stays manual until added there. Only
  Dependabot's own pushes are approved. When `fetch-metadata` reports no update type (single-package
  PRs), it is read from the title, plain numeric versions only, else the PR stays manual.
  - It uses `AUTOMERGE_TOKEN`, a fine-grained PAT of the code owner (Contents + Pull requests write) in
    the Dependabot secrets: "Protect main" needs a code-owner approval, and only a user token's merge
    fires the push that releases. It expires: mint it like `TAG_PUSH_TOKEN`, adding **Pull requests:
    Read and write**. If it is missing, the job leaves the PR for a person.
  - **That token can also create release tags** (its owner is an admin and a tag-ruleset bypass actor),
    which is why that job runs no third-party action and no repository code, and why bumped actions are
    never auto-merged. Auto-merge still waits for the required checks for an admin's token.
  - Merging is deploying: every merge is a release, and prod follows releases.
- **Rebase train** (`dependabot-rebase-train.yaml`): "Protect main" merges only up-to-date PRs and
  Dependabot rebases only on conflicts, so after each push to main (and every 3 hours) it posts
  `@dependabot rebase` on the oldest auto-merge PR that is behind, skipping one already nudged since its
  last commit. It never uses `gh pr update-branch`, which would void the PAT owner's approval.
- `review.yml` skips Dependabot's PRs: E2E is their check, and an open review thread would hold up
  auto-merge.

## Conventions & gotchas

- Docker is a hard requirement (runtime item 7); there is no fallback without it.
- `sandbox_thread_inbox.py` (kept for Tier 3): keep every mutating function free of `await`; that is
  what makes `claim_run` / `deliver` / `finalize` atomic on the one event loop.
- The free OpenAI Moderations endpoint rate-limits hard (429): `content_guard.py` retries with backoff,
  caches verdicts and fails open.
- llama.cpp has no pull API: the `llamacpp` container downloads `LLM_MODEL` on boot into its cache
  volume, so a model change needs `docker compose up -d` (not `restart`, which does not re-read `.env`)
  or `helm upgrade`. At startup `main.py` checks the model is listed on `/v1/models` (fail-soft).
- The chart's LLM volume is still named `*-pvc-ollama` / `ollama-pv-claim` on purpose, so the downloaded
  model survives upgrades. **Do not rename it.** It is rendered whatever `llm.engine` is.
- Strata (`templates/strata.yaml`) needs three things llama.cpp does not: unlimited locked memory (the
  pod gets `SYS_RESOURCE` and starts under `prlimit`, `strata.raiseLimits`); an API key (the render fails
  without `strata.apiKey` / `existingSecret`); and a clean start (the wrapper deletes its saved `docker`
  preset first). Its `/data` must be on an SSD. Kubernetes turns `$$` into `$` in `args`, so the wrapper
  avoids `$$`.
- One switch, `diffusion.enabled` (→ `IMAGE_GEN_ENABLED`), gates the diffusion pod, its PVC and the
  `generate_image` tool. The first start downloads the model (readiness allows ~15 minutes); changing
  `IMAGE_MODEL` needs a pod restart.
- Never commit `.env`; copy `.env.example` and fill it in locally.
