# End-to-end checks

`.github/workflows/e2e.yaml` starts the PR's bot for real and talks to it
through Discord. It is what lets a dependency bump merge without someone
trying it by hand. The design and the security rules are in AGENTS.md ("End-to-end
CI"); this file covers how it runs.

| File | What it does |
|---|---|
| `start_stack.sh` | Starts Redis, the speech service, the voice sidecar, core and (in CI, always) a CPU image service with a tiny model, all from the checked-out source. Core's server workspaces run on the runner's own Docker (labelled `dcb.instance=e2e`). Logs go to `e2e/logs/`. |
| `run_checks.py` | Talks to the bot through a webhook in a private test server and checks the replies through Discord's REST API. |
| `stop_stack.sh` | Stops everything `start_stack.sh` started. |

## What is checked

Structure, never wording:

| Check | Passes when |
|---|---|
| speech service | A sentence sent to `/speak` and back through `/transcribe` comes back with its words |
| slash commands | core's log shows `help`, `whats_new`, `system`, `generate_image`, `workspace` (and `voice`) synced |
| chat reply | The bot answers a mention with text, and does not mark it ❌ |
| memory tool | After "remember my favourite colour is teal" (a brain call), Redis holds a memory with "teal" |
| memory recall | After `!reset_history` (so the history no longer has it), asked "what is my favourite colour?", the reply says teal and `discord_bot_brain_calls_total` did not go up: saved memories are in the persona's prompt |
| image | The reply has a `generated-image.png` PNG attachment |
| poll | The bot posts a real Discord poll with at least two answers |
| workspace task | Asked to make a CSV, the bot asked the brain (`discord_bot_brain_calls_total` went up), which posted a `.csv` attachment in the channel |
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
| `OPENCODE_API_KEY` | OpenCode Go key for the LLM below, given to `start_stack.sh` as `E2E_LLM_API_KEY` and to core only as `LLM_API_KEY`. Not test-only: it is also `review.yml`'s key, so the PR code this job runs can read it (see AGENTS.md's security rules). Dependabot's runs need it in the Dependabot store too. |

The job runs on a GitHub-hosted runner (`ubuntu-latest`). The rest of its
settings are plain values in `e2e.yaml`:

| Variable | What |
|---|---|
| `E2E_LLM_HOST` | OpenAI-compatible LLM the bot uses, without `/v1` (OpenCode Go) |
| `E2E_MODEL` | Model name to request (`deepseek-v4.1-flash`) |
| `E2E_LLM_EXTRA_HEADERS` | core's `LLM_EXTRA_HEADERS`: OpenCode Go refuses requests without an `x-opencode-session` header, so each run sends its own |
| `E2E_LOCAL_DIFFUSION` | `1`: start the PR's own image service on CPU with a tiny model. The vision check of the picture is off on this path (a tiny model draws noise); `start_stack.sh` also takes `E2E_DIFFUSION_URL` for a real image service when run by hand |

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
