"""Prometheus metrics for the bot.

Pure module (stdlib + prometheus_client only — no discord/agents imports),
following the response_filter.py pattern so it can be unit tested without the
bot's runtime dependencies. All metrics are defined at import time in the
default prometheus_client REGISTRY (which also auto-exposes Python process
metrics: memory, CPU seconds, GC, thread count).

Metrics (scraped by Prometheus from the /metrics HTTP endpoint):

  Counter  discord_bot_messages_received_total{guild_id}
           Messages that passed the reply filter (bot mentioned, or the
           random reply chance rolled successfully) and were enqueued for
           handling (messages dropped for a full queue are not counted here;
           see discord_bot_message_queue_drops_total).
  Counter  discord_bot_messages_processed_total{guild_id}
           Messages whose handler completed without raising.
  Counter  discord_bot_llm_errors_total{guild_id}
           Times the LLM run failed and the bot reacted ❌.
  Histogram discord_bot_response_generation_seconds{guild_id}
           Wall-clock time of a full handle_message() (prompt build + LLM
           run + chunked send).
  Counter  discord_bot_tool_calls_total{tool,guild_id}
  Histogram discord_bot_tool_duration_seconds{tool,guild_id}
  Counter  discord_bot_tool_errors_total{tool,guild_id}
           Per-agent-tool invocation count / duration / failures (the
           generate_image tool also shows up here; image_generation_seconds
           additionally measures the raw diffusion-service call, including
           the /generate_image slash command that bypasses the LLM).
  Histogram discord_bot_image_generation_seconds{mode}
           Duration of one diffusion-service call; mode: text_to_image /
           edit (a reference image was sent).
  Counter  discord_bot_image_review_total{outcome}
           Generated images checked before posting (classes/image_review.py).
           outcome: match / mismatch (something asked for was missing, which
           earns a retry) / error (not checked; posted unchecked).
  Gauge    discord_bot_message_queue_size
           Current number of messages waiting on the asyncio queue.
  Counter  discord_bot_message_queue_drops_total{guild_id}
           Messages dropped because the processing queue was full (bounded
           queue, QUEUE_MAX_SIZE in main.py).
  Histogram discord_bot_llm_prompt_tokens{guild_id}
           Prompt (input) tokens of ONE model call - a single reply makes
           several of these (one per turn), and each is a separate
           observation, because each is a prompt that had to fit in the
           context window. This is the answer to "how much of the configured
           context are we actually using"; divide by the gauge below.
  Gauge    discord_bot_llm_context_window_tokens{host}
           Each self-hosted LLM server's TOTAL context (per-slot n_ctx x its
           slots, from its /props, refreshed with the per-slot gauge below),
           so a dashboard has the denominator without hard-coding it. With
           several slots one request only gets its share, so use the
           per-slot gauge as a request's capacity.
  Gauge    discord_bot_llm_slots{host,source}
           Parallel slots core lanes each self-hosted server by: source
           "props" (its /props total_slots), "env" (LLM_SLOTS) or "default"
           (1). One series per host; a changed source replaces it.
  Histogram discord_bot_llm_lane_wait_seconds{lane}
           Seconds one model call waited for a slot on its server before it
           was sent (classes/llm_lanes.py). Not part of
           discord_bot_llm_call_seconds, which stays model time. A remote
           API has no lanes and records nothing here.
  Counter  discord_bot_llm_empty_retries_total{lane}
           Model calls cut off before they said anything (finish_reason
           "length", no text or tool call) and so asked for once more
           (classes/laned_model.py). A rising count means the model spends
           its whole output on thinking.
  Gauge    discord_bot_llm_lane_in_use{host,lane} / discord_bot_llm_lane_waiting{host,lane}
           Slots held, and calls waiting for one, per server and lane, read
           at scrape time.
  Gauge    discord_bot_llm_slot_context_tokens
           Per-slot context of the running llama.cpp server, read from its
           /props (default_generation_settings.n_ctx) at startup and every few
           minutes after, so a server restart/reconfigure is picked up.
  Gauge    discord_bot_llm_slot_context_available
           1 when the value above was read on the last attempt, 0 when /props
           could not be read (the value above is then the last one seen, or 0
           if it never was) - so "unknown" never passes for a measurement.

  Latency breakdown (receipt -> reply):
  Histogram discord_bot_queue_wait_seconds{guild_id}
           Enqueued (on_message) to picked up by a worker.
  Histogram discord_bot_reply_latency_seconds{guild_id,outcome}
           Enqueued to handling finished - what the user waits, queue
           included. outcome: replied / llm_error (the ❌ path) / exception.
  Histogram discord_bot_stage_seconds{stage,guild_id}
           One phase of a handle: history_fetch (Discord history reads),
           attachments (image download + encoding), lock_wait (waiting for
           the channel lock), send (the chunked reply and reasoning sends).
  Histogram discord_bot_llm_call_seconds{caller,outcome}
           ONE model request (a reply makes one per turn). caller: main (the
           reply agent) / image_prompt (the image-request rewrite) /
           image_review (the look at a generated image), the last two on the
           same local server by default. outcome: ok / error / cancelled.
  Histogram discord_bot_llm_completion_tokens{caller}
           Generated tokens of one model request, when the server reported it
           (a missing figure is skipped, never recorded as 0).

  History (classes/history_policy.py):
  Counter  discord_bot_history_refreshes_total{reason}
           Windows re-taken from the newest messages instead of reused.
           reason: cold / threshold / burst / reset / token_budget, plus
           stale_trigger for a private window built for an out-of-order
           message (never committed).
  Counter  discord_bot_history_trimmed_messages_total
           Oldest history messages dropped to fit the per-slot token budget.

  Attachments (classes/attachment_cache.py):
  Counter  discord_bot_attachment_cache_lookups_total{result}
           Image attachments looked up while building a prompt. result: hit
           (already encoded) / miss (downloaded now) / coalesced (joined a
           download another build had already started).
  Gauge    discord_bot_attachment_cache_bytes
           Bytes held by the cache (the base64 data URLs it keeps).
  Counter  discord_bot_attachment_download_failures_total{reason}
           Images left out of a prompt: http (non-200) / error (network,
           timeout) / undecodable (not an image). Never cached, so the next
           build tries again.

  Links (classes/web_reader.py):
  Counter  discord_bot_web_fetch_total{source,cache}
           fetch_url reads. source: plain / browser / youtube /
           youtube_nocaptions / reddit / steam ("none" for a coalesced or
           failed lookup); cache: hit / miss (fetched now) / coalesced (joined
           a fetch already running) / failure_hit (a recent failure, cached).
  Counter  discord_bot_web_fetch_failures_total{reason}
           Links that could not be read, by reason (network / timeout /
           blocked / http / unsupported / rate_limited / blocked_by_site /
           unavailable / ...). Cached briefly, so each is counted once.

  Tier 3 workflows (classes/workflows.py):
  Counter  discord_bot_workflows_total{trigger,end_state}
           Workflows ended, by how they started (brain / slash / automation)
           and how they ended (done / partial / failed / cancelled).
  Histogram discord_bot_workflow_seconds{end_state}
           A workflow's wall-clock time.
  Histogram discord_bot_workflow_subagents
           Steps (subagent runs) per workflow.
  Histogram discord_bot_workflow_turns
           Model turns per workflow, every agent together.
  Counter  discord_bot_workflow_plan_approval_total{outcome}
           How plans were approved (approved / edited / cancelled / timeout /
           auto).
  Counter  discord_bot_workflow_scouts_total{outcome}
           How scouts ended (done / reported_at_limit / stopped).
  Counter  discord_bot_workflow_plan_questions_total{outcome}
           Questions before the plan (answered / unanswered).
  Gauge    discord_bot_workflows_running
           Workflows running in this process.

  Thread conversation (classes/workflow_conversation.py; labels are bounded
  outcome names — never message ids, thread ids or user text):
  Counter  discord_bot_workflow_thread_messages_total{outcome}
           Thread messages offered to a running workflow, by delivery result
           (accepted / duplicate / too_long / full / attachment_only /
           finishing / ...).
  Histogram discord_bot_workflow_input_presentation_seconds
           Receipt of a thread message to its first completed model call.
  Histogram discord_bot_workflow_input_response_seconds
           Receipt of a thread message to the model's respond_to_updates.
  Counter  discord_bot_workflow_stale_actions_blocked_total
           Tool calls refused because thread input was still unanswered.
  Counter  discord_bot_workflow_questions_total{mode,outcome}
           ask_user questions by mode (blocking/optional) and outcome
           (posted / replied / timeout / send_failed).
  Counter  discord_bot_workflow_unresolved_input_total{status}
           Messages still unresolved when a run ended, by status
           (accepted / unacknowledged / unseen / late).
  Counter  discord_bot_workflow_continuations_total
           Extra runner passes started because a run tried to finish with
           thread input unanswered (or outdated files attached).

  Server workspace (classes/workspace_manager.py):
  Counter  discord_bot_workspace_starts_total{kind}
  Histogram discord_bot_workspace_start_seconds{kind}
           Each acquisition of a server's workspace, and how long it took to
           be ready. kind: cached (already warm here) / resumed (re-attached
           to its running container) / snapshot (a new container filled from
           the snapshot) / fresh (a new, empty one).
  Gauge    discord_bot_workspaces_warm
           Server workspaces warm in this core process (WORKSPACE_MAX_WARM caps it).
  Histogram discord_bot_workspace_snapshot_seconds / discord_bot_workspace_snapshot_bytes
           Each saved snapshot's time and size.
  Counter  discord_bot_workspace_snapshot_failures_total{reason}
           Snapshots not saved: too_large (over WORKSPACE_SNAPSHOT_MAX_BYTES;
           the previous one is kept) / timeout / error.
  Counter  discord_bot_workspace_reaped_total
           Workspace containers removed because no live record names them.

  The brain (classes/brain.py; labels are bounded: tier is 2 until Tier 3,
  status is done / partial / needs_input / failed):
  Counter  discord_bot_brain_calls_total{tier,status}
  Histogram discord_bot_brain_call_seconds{tier,status}
           Each ask_brain call, and how long it took end to end.
  Histogram discord_bot_brain_turns / discord_bot_brain_tool_calls
           Model turns and tool calls per brain call: how far real calls are
           from the loop guard's backstops, and how many are one-tool jobs.
  Counter  discord_bot_brain_loop_stops_total{kind}
           Brain calls the loop guard stopped (repeat / stale).
  Counter  discord_bot_brain_facts_dropped_total
           Must-keep facts the persona's reply left out.
  Counter  discord_bot_persona_escalations_total{result}
           Persona replies that called the brain (brain) or answered alone
           (answered).
  Counter  discord_bot_brain_questions_total{outcome}
  Counter  discord_bot_skills_saved_total{source,result}
  Counter  discord_bot_skills_used_total
  Counter  discord_bot_skill_success_total
  Counter  discord_bot_skill_reflection_total{action}
           The brain's ask_user: answered / timeout (no answer in time) /
           skipped (no one to ask: automatic runs, unprompted replies, or
           the per-call cap).
  Histogram discord_bot_brain_slot_wait_seconds
           Seconds a brain call waited for one of BRAIN_MAX_CONCURRENT slots.
  Gauge    discord_bot_queue_workers
           Queue worker tasks running: QUEUE_WORKERS plus one spare per brain
           call holding a worker (classes/message_queue.py).

Context-window note: this is the PROMPT side, which is what has to fit before
generation starts. The server-side counterpart is llama.cpp's own
`llamacpp:n_tokens_max` - "high watermark of the context size observed", which
the chart enables alongside these (metrics.enabled ->
LLAMA_ARG_ENDPOINT_METRICS). Upstream does not document whether that watermark
is taken before or after generation, so treat it as >= this histogram, not as
a known prompt+completion total.

Environment:
  METRICS_PORT  port to serve /metrics on (default 9464). Empty or 0
                disables the server (metrics are still defined, just not
                exported).
  LLM_CONTEXT_LENGTH
                the llama.cpp --ctx-size the server was started with (helm:
                llamacppContextLength), exported as the gauge above. Unset or
                unparseable leaves the gauge at 0.
"""

import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import REGISTRY, Counter, Gauge, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily

from classes import llm_lanes

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Buckets are generous: LLM responses take seconds to minutes on local
# llama.cpp, and image generation can queue for a while on a busy GPU.
RESPONSE_GEN_BUCKETS = (0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
TOOL_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
IMAGE_GEN_BUCKETS = (1, 5, 10, 30, 60, 120, 300)
# Prompt sizes span the chart default context (40000) and prod's (80000), with
# headroom above so an over-budget prompt is still visible rather than lost in
# +Inf.
PROMPT_TOKEN_BUCKETS = (500, 1000, 2000, 4000, 8000, 16000, 24000, 32000,
                        48000, 64000, 96000, 128000)

# Labelled by guild only. user_id was a label here too, which is unbounded
# cardinality: every distinct Discord user who ever triggered the bot became a
# permanent extra time series in the scrape output and in Prometheus' index.
# Guild count is bounded and is what the dashboards actually group by; per-user
# detail belongs in logs, not in a metric label.
messages_received_total = Counter(
    "discord_bot_messages_received_total",
    "Messages that passed the reply filter (mentioned or random-chance hit) and entered handling",
    ["guild_id"],
)

messages_processed_total = Counter(
    "discord_bot_messages_processed_total",
    "Messages whose handler completed without error",
    ["guild_id"],
)

llm_errors_total = Counter(
    "discord_bot_llm_errors_total",
    "LLM runs that failed (bot reacted ❌)",
    ["guild_id"],
)

response_generation_seconds = Histogram(
    "discord_bot_response_generation_seconds",
    "Wall-clock time of a full message handling (prompt build + LLM run + send)",
    ["guild_id"],
    buckets=RESPONSE_GEN_BUCKETS,
)

tool_calls_total = Counter(
    "discord_bot_tool_calls_total",
    "Agent tool invocations",
    ["tool", "guild_id"],
)

tool_duration_seconds = Histogram(
    "discord_bot_tool_duration_seconds",
    "Wall-clock time of one agent tool invocation",
    ["tool", "guild_id"],
    buckets=TOOL_BUCKETS,
)

tool_errors_total = Counter(
    "discord_bot_tool_errors_total",
    "Agent tool invocations that failed (SDK default failure handler result)",
    ["tool", "guild_id"],
)

image_generation_seconds = Histogram(
    "discord_bot_image_generation_seconds",
    "Duration of one diffusion-service image generation call",
    ["mode"],
    buckets=IMAGE_GEN_BUCKETS,
)

image_review_total = Counter(
    "discord_bot_image_review_total",
    "Generated images checked before posting, by outcome",
    ["outcome"],
)

message_queue_size = Gauge(
    "discord_bot_message_queue_size",
    "Number of messages waiting on the processing queue",
)

automation_runs_total = Counter(
    "discord_bot_automation_runs_total",
    "Automatic runs by admission or terminal outcome",
    ["kind", "outcome"],
)
automation_queue_wait_seconds = Histogram(
    "discord_bot_automation_queue_wait_seconds", "Automatic run queue wait", ["kind"],
    buckets=(0.1, 1, 5, 15, 30, 60, 120, 300, 600),
)
automation_execution_seconds = Histogram(
    "discord_bot_automation_execution_seconds", "Automatic run duration", ["kind", "outcome"],
    buckets=(1, 5, 15, 30, 60, 120, 300, 600, 1200),
)


# Voice calls (classes/voice_session.py). Speech timings come from the voice
# sidecar, which measures them against the speech service.
voice_sessions_active = Gauge(
    "discord_bot_voice_sessions_active", "Guilds the bot is in a voice call in",
)
voice_utterances_total = Counter(
    "discord_bot_voice_utterances_total",
    "Transcribed utterances heard in voice calls, by what they led to",
    ["outcome"],
)
voice_turns_total = Counter(
    "discord_bot_voice_turns_total", "Voice turns (wake phrase + request), by outcome", ["outcome"],
)
voice_first_audio_seconds = Histogram(
    "discord_bot_voice_first_audio_seconds",
    "From a request being heard to its first spoken sentence being sent to the voice service",
    buckets=(0.25, 0.5, 1, 1.5, 2, 3, 5, 10, 30),
)
voice_speech_seconds = Histogram(
    "discord_bot_voice_speech_seconds",
    "Speech service time per clip: stt = one transcription, tts = one sentence",
    ["kind"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)


def set_voice_sessions(n: int) -> None:
    voice_sessions_active.set(n)


def inc_voice_utterance(outcome: str) -> None:
    voice_utterances_total.labels(outcome=str(outcome)).inc()


def inc_voice_turn(outcome: str) -> None:
    voice_turns_total.labels(outcome=str(outcome)).inc()


def observe_voice_first_audio(seconds: float) -> None:
    voice_first_audio_seconds.observe(max(0.0, seconds))


def observe_voice_speech(kind: str, seconds) -> None:
    try:
        voice_speech_seconds.labels(kind=str(kind)).observe(max(0.0, float(seconds)))
    except (TypeError, ValueError):
        pass


def inc_automation(kind, outcome):
    automation_runs_total.labels(kind=kind, outcome=outcome).inc()


def observe_automation_wait(kind, seconds):
    automation_queue_wait_seconds.labels(kind=kind).observe(max(0, seconds))


def observe_automation_execution(kind, outcome, seconds):
    automation_execution_seconds.labels(kind=kind, outcome=outcome).observe(max(0, seconds))

queue_drops_total = Counter(
    "discord_bot_message_queue_drops_total",
    "Messages dropped because the processing queue was full",
    ["guild_id"],
)

llm_prompt_tokens = Histogram(
    "discord_bot_llm_prompt_tokens",
    "Prompt (input) tokens sent to the LLM on one model call",
    ["guild_id"],
    buckets=PROMPT_TOKEN_BUCKETS,
)

llm_context_window_tokens = Gauge(
    "discord_bot_llm_context_window_tokens",
    "Total context of a self-hosted LLM server in tokens (per-slot n_ctx x slots, from its /props)",
    ["host"],
)

llm_slots = Gauge(
    "discord_bot_llm_slots",
    "Parallel slots of a self-hosted LLM server, and where the number came from",
    ["host", "source"],
)

llm_slot_context_tokens = Gauge(
    "discord_bot_llm_slot_context_tokens",
    "Per-slot context of the LLM server in tokens (from its /props)",
)

llm_slot_context_available = Gauge(
    "discord_bot_llm_slot_context_available",
    "1 when the per-slot context was read from the LLM server on the last attempt, else 0",
)

# Queue waits and full replies span a quick chat answer to a long brain call.
LATENCY_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600)
# Fast phases (history reads, downloads, lock waits, sends).
STAGE_BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
COMPLETION_TOKEN_BUCKETS = (16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)

queue_wait_seconds = Histogram(
    "discord_bot_queue_wait_seconds",
    "Seconds a message waited on the queue before a worker picked it up",
    ["guild_id"],
    buckets=LATENCY_BUCKETS,
)

reply_latency_seconds = Histogram(
    "discord_bot_reply_latency_seconds",
    "Seconds from enqueueing a message to finishing its handling, queue wait included",
    ["guild_id", "outcome"],
    buckets=LATENCY_BUCKETS,
)

stage_seconds = Histogram(
    "discord_bot_stage_seconds",
    "Seconds spent in one phase of handling a message",
    ["stage", "guild_id"],
    buckets=STAGE_BUCKETS,
)

llm_call_seconds = Histogram(
    "discord_bot_llm_call_seconds",
    "Seconds of one model request",
    ["caller", "outcome"],
    buckets=LATENCY_BUCKETS,
)

llm_lane_wait_seconds = Histogram(
    "discord_bot_llm_lane_wait_seconds",
    "Seconds one model call waited for a slot on its LLM server",
    ["lane"],
    buckets=LATENCY_BUCKETS,
)

llm_empty_retries_total = Counter(
    "discord_bot_llm_empty_retries_total",
    "Model calls cut off before any output and asked for once more",
    ["lane"],
)

llm_completion_tokens = Histogram(
    "discord_bot_llm_completion_tokens",
    "Generated tokens of one model request",
    ["caller"],
    buckets=COMPLETION_TOKEN_BUCKETS,
)

history_refreshes_total = Counter(
    "discord_bot_history_refreshes_total",
    "History windows re-taken from the newest messages, by reason",
    ["reason"],
)

history_trimmed_messages_total = Counter(
    "discord_bot_history_trimmed_messages_total",
    "Oldest history messages dropped to fit the per-slot token budget",
)

attachment_cache_lookups_total = Counter(
    "discord_bot_attachment_cache_lookups_total",
    "Image attachment lookups while building a prompt, by result",
    ["result"],
)

attachment_cache_bytes = Gauge(
    "discord_bot_attachment_cache_bytes",
    "Bytes held by the attachment cache",
)

attachment_download_failures_total = Counter(
    "discord_bot_attachment_download_failures_total",
    "Image attachments that could not be downloaded or decoded, by reason",
    ["reason"],
)

web_fetch_total = Counter(
    "discord_bot_web_fetch_total",
    "fetch_url reads, by where the text came from and the cache result",
    ["source", "cache"],
)

web_fetch_failures_total = Counter(
    "discord_bot_web_fetch_failures_total",
    "Links fetch_url could not read, by reason",
    ["reason"],
)

# Receipt -> presentation is bounded by one model call plus any running
# command's checkpoint; receipt -> response adds the model's own turn.
WORKFLOW_INPUT_BUCKETS = (1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600)

workflow_thread_messages_total = Counter(
    "discord_bot_workflow_thread_messages_total",
    "Thread messages offered to a running workflow, by delivery result",
    ["outcome"],
)

workflow_input_presentation_seconds = Histogram(
    "discord_bot_workflow_input_presentation_seconds",
    "Seconds from receiving a thread message to its first completed model call",
    buckets=WORKFLOW_INPUT_BUCKETS,
)

workflow_input_response_seconds = Histogram(
    "discord_bot_workflow_input_response_seconds",
    "Seconds from receiving a thread message to the workflow's model responding to it",
    buckets=WORKFLOW_INPUT_BUCKETS,
)

workflow_stale_actions_blocked_total = Counter(
    "discord_bot_workflow_stale_actions_blocked_total",
    "Workflow tool calls refused because thread input was still unanswered",
)

workflow_questions_total = Counter(
    "discord_bot_workflow_questions_total",
    "Questions asked in a workflow's thread, by mode and outcome",
    ["mode", "outcome"],
)

workflow_unresolved_input_total = Counter(
    "discord_bot_workflow_unresolved_input_total",
    "Thread messages still unresolved when a workflow ended, by status",
    ["status"],
)

workflow_continuations_total = Counter(
    "discord_bot_workflow_continuations_total",
    "Extra workflow runner passes started to account for thread input",
)

# The server workspace (classes/workspace_manager.py).
workspace_starts_total = Counter(
    "discord_bot_workspace_starts_total",
    "Server workspace acquisitions, by how the workspace was made ready",
    ["kind"],
)
workspace_start_seconds = Histogram(
    "discord_bot_workspace_start_seconds",
    "Seconds to make a server workspace ready, by kind",
    ["kind"],
    buckets=(0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
workspaces_warm = Gauge(
    "discord_bot_workspaces_warm", "Server workspaces warm in this core process",
)
workspace_snapshot_seconds = Histogram(
    "discord_bot_workspace_snapshot_seconds",
    "Seconds to save one server workspace snapshot",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)
workspace_snapshot_bytes = Histogram(
    "discord_bot_workspace_snapshot_bytes",
    "Size of each saved server workspace snapshot",
    buckets=(10_000, 100_000, 1_000_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000,
             100_000_000),
)
workspace_snapshot_failures_total = Counter(
    "discord_bot_workspace_snapshot_failures_total",
    "Server workspace snapshots that were not saved, by reason",
    ["reason"],
)
workspace_reaped_total = Counter(
    "discord_bot_workspace_reaped_total",
    "Workspace containers removed by the reaper (no live record names them)",
)


# The brain (classes/brain.py).
brain_calls_total = Counter(
    "discord_bot_brain_calls_total", "Brain calls, by tier and how they ended",
    ["tier", "status"],
)
brain_call_seconds = Histogram(
    "discord_bot_brain_call_seconds", "Seconds one brain call took, by tier and how it ended",
    ["tier", "status"],
    buckets=(1, 2.5, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800),
)
brain_turns = Histogram(
    "discord_bot_brain_turns", "Model turns per brain call",
    buckets=(1, 2, 3, 5, 8, 12, 20, 30, 50, 75, 100),
)
brain_tool_calls = Histogram(
    "discord_bot_brain_tool_calls", "Tool calls per brain call",
    buckets=(0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 100),
)
brain_loop_stops_total = Counter(
    "discord_bot_brain_loop_stops_total", "Brain calls stopped by the loop guard, by kind",
    ["kind"],
)
brain_facts_dropped_total = Counter(
    "discord_bot_brain_facts_dropped_total",
    "Must-keep facts from the brain that the persona's reply left out",
)
persona_escalations_total = Counter(
    "discord_bot_persona_escalations_total",
    "Persona replies, by whether they called the brain",
    ["result"],
)
brain_questions_total = Counter(
    "discord_bot_brain_questions_total",
    "Questions the brain asked the requester, by outcome",
    ["outcome"],
)
skills_saved_total = Counter(
    "discord_bot_skills_saved_total",
    "Skill saves, by who made them (brain, reflection, command, curator, revert) and result",
    ["source", "result"],
)
skills_used_total = Counter("discord_bot_skills_used_total", "use_skill calls that found a skill")
skill_success_total = Counter(
    "discord_bot_skill_success_total", "Brain calls that used a skill and ended done")
skill_reflection_total = Counter(
    "discord_bot_skill_reflection_total",
    "Reviews of a finished job, by what they decided (none, create, update, failed)", ["action"])
skill_curator_changes_total = Counter(
    "discord_bot_skill_curator_changes_total",
    "The daily tidy-up's changes to skills, by kind (merge, retire, describe, refused)", ["kind"])
skills_live_gauge = Gauge("discord_bot_skills_live", "Live skills, every server together")
brain_slot_wait_seconds = Histogram(
    "discord_bot_brain_slot_wait_seconds",
    "Seconds a brain call waited for a brain slot",
    buckets=(0.1, 1, 5, 15, 30, 60, 120, 300),
)
queue_workers_gauge = Gauge(
    "discord_bot_queue_workers",
    "Queue worker tasks running (base pool plus spares for brain calls)",
)


def observe_brain_call(tier, status: str, seconds: float, turns: int, tool_calls: int) -> None:
    brain_calls_total.labels(tier=str(tier), status=str(status)).inc()
    brain_call_seconds.labels(tier=str(tier), status=str(status)).observe(max(0.0, seconds))
    brain_turns.observe(max(0, turns))
    brain_tool_calls.observe(max(0, tool_calls))


def inc_brain_loop_stop(kind: str) -> None:
    brain_loop_stops_total.labels(kind=str(kind)).inc()


def inc_brain_facts_dropped(count: int) -> None:
    if count > 0:
        brain_facts_dropped_total.inc(count)


def inc_persona_escalation(result: str) -> None:
    persona_escalations_total.labels(result=str(result)).inc()


def inc_brain_question(outcome: str) -> None:
    brain_questions_total.labels(outcome=str(outcome)).inc()


def inc_skill_saved(source: str, result: str) -> None:
    skills_saved_total.labels(source=str(source), result=str(result)).inc()


def inc_skill_used() -> None:
    skills_used_total.inc()


def inc_skill_success() -> None:
    skill_success_total.inc()


def inc_skill_reflection(action: str) -> None:
    skill_reflection_total.labels(action=str(action)).inc()


def inc_skill_curator_change(kind: str) -> None:
    skill_curator_changes_total.labels(kind=str(kind)).inc()


def set_skills_live(count: int) -> None:
    skills_live_gauge.set(max(0, int(count)))


def observe_brain_slot_wait(seconds: float) -> None:
    brain_slot_wait_seconds.observe(max(0.0, seconds))


def set_queue_workers(n: int) -> None:
    queue_workers_gauge.set(n)


def observe_workspace_start(kind: str, seconds: float) -> None:
    workspace_starts_total.labels(kind=kind).inc()
    workspace_start_seconds.labels(kind=kind).observe(seconds)


def set_workspaces_warm(n: int) -> None:
    workspaces_warm.set(n)


def observe_workspace_snapshot(seconds: float, size: int) -> None:
    workspace_snapshot_seconds.observe(seconds)
    workspace_snapshot_bytes.observe(size)


def inc_workspace_snapshot_failure(reason: str) -> None:
    workspace_snapshot_failures_total.labels(reason=reason).inc()


def inc_workspace_reaped(n: int = 1) -> None:
    workspace_reaped_total.inc(n)

# ---------------------------------------------------------------------------
# Convenience helpers (labels are always strings; IDs come in as ints)
# ---------------------------------------------------------------------------


def _guild_label(guild_id) -> str:
    return str(guild_id) if guild_id else "unknown"


def inc_messages_received(guild_id) -> None:
    messages_received_total.labels(guild_id=_guild_label(guild_id)).inc()


def inc_messages_processed(guild_id) -> None:
    messages_processed_total.labels(guild_id=_guild_label(guild_id)).inc()


def inc_llm_error(guild_id) -> None:
    llm_errors_total.labels(guild_id=_guild_label(guild_id)).inc()


def observe_response_generation(guild_id, seconds: float) -> None:
    response_generation_seconds.labels(guild_id=_guild_label(guild_id)).observe(seconds)


def inc_tool_call(tool: str, guild_id) -> None:
    tool_calls_total.labels(tool=str(tool), guild_id=_guild_label(guild_id)).inc()


def observe_tool_duration(tool: str, guild_id, seconds: float) -> None:
    tool_duration_seconds.labels(tool=str(tool), guild_id=_guild_label(guild_id)).observe(seconds)


def inc_tool_error(tool: str, guild_id) -> None:
    tool_errors_total.labels(tool=str(tool), guild_id=_guild_label(guild_id)).inc()


def observe_image_generation(mode: str, seconds: float) -> None:
    image_generation_seconds.labels(mode=str(mode)).observe(seconds)


def inc_image_review(outcome: str) -> None:
    image_review_total.labels(outcome=str(outcome)).inc()


def set_message_queue_size(n: int) -> None:
    message_queue_size.set(n)


def inc_queue_drop(guild_id) -> None:
    queue_drops_total.labels(guild_id=_guild_label(guild_id)).inc()


def observe_llm_prompt_tokens(guild_id, tokens: int) -> None:
    llm_prompt_tokens.labels(guild_id=_guild_label(guild_id)).observe(tokens)


def inc_workflow_thread_message(outcome: str) -> None:
    workflow_thread_messages_total.labels(outcome=str(outcome)).inc()


def observe_workflow_input_presentation(seconds: float) -> None:
    workflow_input_presentation_seconds.observe(max(0.0, seconds))


def observe_workflow_input_response(seconds: float) -> None:
    workflow_input_response_seconds.observe(max(0.0, seconds))


def inc_workflow_stale_action_blocked() -> None:
    workflow_stale_actions_blocked_total.inc()


def inc_workflow_question(mode: str, outcome: str) -> None:
    workflow_questions_total.labels(mode=str(mode), outcome=str(outcome)).inc()


def inc_workflow_unresolved_input(status: str) -> None:
    workflow_unresolved_input_total.labels(status=str(status)).inc()


def inc_workflow_continuation() -> None:
    workflow_continuations_total.inc()


workflows_total = Counter(
    "discord_bot_workflows_total",
    "Tier 3 workflows ended, by trigger and end state", ["trigger", "end_state"])
workflow_seconds = Histogram(
    "discord_bot_workflow_seconds", "Wall-clock seconds of one workflow", ["end_state"],
    buckets=(60, 300, 600, 1200, 1800, 3600, 7200, 14400))
workflow_subagents = Histogram(
    "discord_bot_workflow_subagents", "Steps (subagent runs) per workflow",
    buckets=(0, 1, 2, 3, 4, 6, 8))
workflow_turns = Histogram(
    "discord_bot_workflow_turns", "Model turns per workflow, every agent together",
    buckets=(5, 10, 25, 50, 100, 200, 350, 500))
workflow_plan_approval_total = Counter(
    "discord_bot_workflow_plan_approval_total", "How workflow plans were approved", ["outcome"])
workflow_scouts_total = Counter(
    "discord_bot_workflow_scouts_total", "Scouts sent before a workflow's plan", ["outcome"])
workflow_plan_questions_total = Counter(
    "discord_bot_workflow_plan_questions_total", "Questions asked before a workflow's plan",
    ["outcome"])
workflows_running_gauge = Gauge("discord_bot_workflows_running", "Workflows running in this process")


def observe_workflow(trigger: str, end_state: str, seconds: float, subagents: int,
                     turns: int) -> None:
    workflows_total.labels(trigger=str(trigger), end_state=str(end_state)).inc()
    workflow_seconds.labels(end_state=str(end_state)).observe(max(0.0, seconds))
    workflow_subagents.observe(max(0, subagents))
    workflow_turns.observe(max(0, turns))


def inc_workflow_plan_approval(outcome: str) -> None:
    workflow_plan_approval_total.labels(outcome=str(outcome)).inc()


def inc_workflow_scout(outcome: str) -> None:
    workflow_scouts_total.labels(outcome=str(outcome)).inc()


def inc_workflow_plan_question(outcome: str) -> None:
    workflow_plan_questions_total.labels(outcome=str(outcome)).inc()


def set_workflows_running(count: int) -> None:
    workflows_running_gauge.set(max(0, count))


def set_slot_context(tokens: int | None) -> None:
    """Publish the per-slot context; None marks it unavailable (value kept)."""
    if tokens is None:
        llm_slot_context_available.set(0)
        return
    llm_slot_context_tokens.set(tokens)
    llm_slot_context_available.set(1)


def observe_queue_wait(guild_id, seconds: float) -> None:
    queue_wait_seconds.labels(guild_id=_guild_label(guild_id)).observe(max(0.0, seconds))


def observe_reply_latency(guild_id, outcome: str, seconds: float) -> None:
    reply_latency_seconds.labels(
        guild_id=_guild_label(guild_id), outcome=str(outcome)).observe(max(0.0, seconds))


def observe_stage(stage: str, guild_id, seconds: float) -> None:
    stage_seconds.labels(stage=str(stage), guild_id=_guild_label(guild_id)).observe(max(0.0, seconds))


def observe_llm_call(caller: str, outcome: str, seconds: float) -> None:
    llm_call_seconds.labels(caller=str(caller), outcome=str(outcome)).observe(max(0.0, seconds))


def observe_llm_completion_tokens(caller: str, tokens) -> None:
    """Skips a missing/zero figure: the SDK reports 0 when the server sent no
    usage, and recording that would drag the distribution toward zero."""
    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) and tokens > 0:
        llm_completion_tokens.labels(caller=str(caller)).observe(tokens)


def inc_history_refresh(reason: str) -> None:
    history_refreshes_total.labels(reason=str(reason)).inc()


def inc_history_trimmed(count: int) -> None:
    if count > 0:
        history_trimmed_messages_total.inc(count)


def inc_attachment_cache_lookup(result: str) -> None:
    attachment_cache_lookups_total.labels(result=str(result)).inc()


def set_attachment_cache_bytes(n: int) -> None:
    attachment_cache_bytes.set(n)


def inc_attachment_download_failure(reason: str) -> None:
    attachment_download_failures_total.labels(reason=str(reason)).inc()


def inc_web_fetch(source: str, cache: str) -> None:
    web_fetch_total.labels(source=str(source), cache=str(cache)).inc()


def inc_web_fetch_failure(reason: str) -> None:
    web_fetch_failures_total.labels(reason=str(reason)).inc()


def set_context_window(host: str, tokens: int | None) -> None:
    """Publish a server's total context; None (not read) keeps the last value."""
    if tokens:
        llm_context_window_tokens.labels(host=str(host)).set(tokens)


_slot_sources: dict[str, str] = {}


def set_llm_slots(host: str, slots: int, source: str) -> None:
    """Publish a server's slot count; one series per host."""
    host = str(host)
    old = _slot_sources.get(host)
    if old is not None and old != source:
        try:
            llm_slots.remove(host, old)
        except KeyError:
            pass
    _slot_sources[host] = source
    llm_slots.labels(host=host, source=source).set(slots)


def inc_empty_retry(lane: str) -> None:
    llm_empty_retries_total.labels(lane=str(lane)).inc()


def observe_lane_wait(lane: str, seconds: float) -> None:
    llm_lane_wait_seconds.labels(lane=str(lane)).observe(max(0.0, seconds))


class _LaneCollector:
    """In use / waiting per server and lane, read from llm_lanes at scrape
    time (the scrape runs on the metrics thread; snapshot() only reads ints
    and deque lengths, so it never iterates a deque the loop is changing)."""

    def collect(self):
        in_use = GaugeMetricFamily("discord_bot_llm_lane_in_use",
                                   "LLM slots held, per server and lane", labels=["host", "lane"])
        waiting = GaugeMetricFamily("discord_bot_llm_lane_waiting",
                                    "Model calls waiting for an LLM slot, per server and lane",
                                    labels=["host", "lane"])
        for host, lanes in llm_lanes.all_lanes().items():
            for lane, counts in lanes.snapshot().items():
                in_use.add_metric([host, lane], counts["in_use"])
                waiting.add_metric([host, lane], counts["waiting"])
        yield in_use
        yield waiting


REGISTRY.register(_LaneCollector())


# ---------------------------------------------------------------------------
# /metrics HTTP server
# ---------------------------------------------------------------------------


class _MetricsHandler(BaseHTTPRequestHandler):
    """Serves the Prometheus scrape endpoint on GET /metrics (404 otherwise)."""

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler naming)
        if self.path.split("?", 1)[0] != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence per-request logging
        pass


def start_metrics_server(port: int) -> ThreadingHTTPServer:
    """Start the /metrics server in a daemon thread.

    port=0 binds an ephemeral port (useful for tests); returns the server
    (the actually-bound port is server.server_address[1])."""
    server = ThreadingHTTPServer(("0.0.0.0", port), _MetricsHandler)
    threading.Thread(target=server.serve_forever, name="metrics-server", daemon=True).start()
    return server


def start_metrics_server_from_env() -> ThreadingHTTPServer | None:
    """Start the server on METRICS_PORT (default 9464); empty/0 disables."""
    raw = os.environ.get("METRICS_PORT", "9464").strip()
    if raw in ("", "0"):
        logger.info("Metrics server disabled (METRICS_PORT is empty/0)")
        return None
    server = start_metrics_server(int(raw))
    logger.info(f"Metrics server listening on :{server.server_address[1]}/metrics")
    return server
