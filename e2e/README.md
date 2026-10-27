# End-to-end checks

`.github/workflows/e2e.yaml` starts the PR's bot for real and talks to it
through Discord. It is what lets a dependency bump merge without someone
trying it by hand. The design and the security rules are in AGENTS.md ("End-to-end
CI"); this file covers how it runs.

| File | What it does |
|---|---|
| `start_stack.sh` | Starts Redis, the speech service, the voice sidecar, core and (for image-service PRs) a CPU image service with a tiny model, all from the checked-out source. Logs go to `e2e/logs/`. |
| `run_checks.py` | Talks to the bot through a webhook in a private test server and checks the replies through Discord's REST API. |
| `stop_stack.sh` | Stops everything `start_stack.sh` started. |

## What is checked

Structure, never wording:

| Check | Passes when |
|---|---|
| speech service | A sentence sent to `/speak` and back through `/transcribe` comes back with its words |
| slash commands | core's log shows `help`, `whats_new`, `system`, `generate_image` (and `voice`) synced |
| chat reply | The bot answers a mention with text, and does not mark it ❌ |
| memory tool | After "remember my favourite colour is teal", Redis holds a memory with "teal" |
| image | The reply has a `generated-image.png` PNG attachment |
| poll | The bot posts a real Discord poll with at least two answers |
| voice call | `!voice_join` joins the voice channel (Discord shows it), the greeting is spoken, a spoken question is answered aloud, and "leave" ends the call |
| no tracebacks | core's log has no Python traceback |

Each case waits on core's own metrics (`reply_latency_seconds` by outcome, the
voice counters) rather than sleeping. Every run uses its own thread.

## Configuration

Secrets. Add each one to **both** the repository's Actions secrets and its
Dependabot secrets: Dependabot's runs can only see the second store.

| Secret | What |
|---|---|
| `E2E_DISCORD_TOKEN` | Token of a bot made only for testing. Never the real bot's. |
| `E2E_WEBHOOK_URL` | Webhook in the test server's text channel. The server and channel are read from it. |
| `E2E_VOICE_CHANNEL_ID` | A voice channel in the test server. Empty skips the voice check. |

The runner (label `discordchatbot-e2e`) provides these as environment
variables, so the endpoints never appear in this repository:

| Variable | What |
|---|---|
| `E2E_LLM_HOST` | LLM server (OpenAI-compatible: llama.cpp or Strata) the bot uses, without `/v1` |
| `E2E_LLM_API_KEY` | Optional: that server's API key (Strata needs one), passed to core only as `LLM_PASS`. Empty = core's placeholder |
| `E2E_DIFFUSION_URL` | Image service the bot uses (unused when the PR's own runs). Optional: unset = every run starts the local CPU image service, as image-service PRs do (for a deployment with image generation off) |
| `E2E_MODEL` | Model name to request |

## Running it yourself

On Linux, with Python 3.13 and 3.14, Node 22 and `redis-server` installed:

```bash
export E2E_DISCORD_TOKEN=... E2E_WEBHOOK_URL=... E2E_VOICE_CHANNEL_ID=...
export E2E_LLM_HOST=http://localhost:8081 E2E_MODEL=... E2E_DIFFUSION_URL=http://localhost:8000
export CORE_PYTHON=python3.14 SPEECH_PYTHON=python3.13
e2e/start_stack.sh && .e2e/core/bin/python e2e/run_checks.py; e2e/stop_stack.sh
```

Use a test bot and a test server, never the real ones. While it runs, the bot
answers every mention in that server.
