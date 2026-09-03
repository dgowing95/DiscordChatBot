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
           Duration of one diffusion-service call; mode is always
           text_to_image.
  Gauge    discord_bot_message_queue_size
           Current number of messages waiting on the asyncio queue.
  Counter  discord_bot_message_queue_drops_total{guild_id}
           Messages dropped because the processing queue was full (bounded
           queue, QUEUE_MAX_SIZE in main.py).

Environment:
  METRICS_PORT  port to serve /metrics on (default 9464). Empty or 0
                disables the server (metrics are still defined, just not
                exported).
"""

import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from prometheus_client import Counter, Gauge, Histogram, generate_latest

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# Buckets are generous: LLM responses take seconds to minutes on local
# llama.cpp, and image generation can queue for a while on a busy GPU.
RESPONSE_GEN_BUCKETS = (0.5, 1, 2.5, 5, 10, 30, 60, 120, 300)
TOOL_BUCKETS = (0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600)
IMAGE_GEN_BUCKETS = (1, 5, 10, 30, 60, 120, 300)

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

message_queue_size = Gauge(
    "discord_bot_message_queue_size",
    "Number of messages waiting on the processing queue",
)

queue_drops_total = Counter(
    "discord_bot_message_queue_drops_total",
    "Messages dropped because the processing queue was full",
    ["guild_id"],
)

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


def set_message_queue_size(n: int) -> None:
    message_queue_size.set(n)


def inc_queue_drop(guild_id) -> None:
    queue_drops_total.labels(guild_id=_guild_label(guild_id)).inc()


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
        print("Metrics server disabled (METRICS_PORT is empty/0)")
        return None
    server = start_metrics_server(int(raw))
    print(f"Metrics server listening on :{server.server_address[1]}/metrics")
    return server