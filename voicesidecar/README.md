# Voice sidecar

Discord voice I/O for the bot's core process. See `docs/voice.md` for the feature and `core/classes/voice_bridge.py` for core's side.

## Why it exists

Since March 2026 Discord requires its DAVE end-to-end encryption on every voice call. discord.py can *send* DAVE audio, but nothing maintained for discord.py can *receive* it. The only option was replacing discord.py with a one-person fork. `@discordjs/voice` (0.19.1+) decrypts received audio, so this small Node process holds the voice connections, and core keeps official discord.py for everything else.

## How it fits

- **core keeps the gateway.** discord.py routes the bot's VOICE_STATE_UPDATE and VOICE_SERVER_UPDATE to `SidecarVoiceProtocol`, which forwards them here (`gateway`). The `@discordjs/voice` adapter's op-4 payloads go back as `payload`, and core sends them on its gateway. This is the Lavalink/wavelink pattern.
- **Audio never reaches Python.** Each speaker's audio is cut at `VOICE_SILENCE_MS` of silence, decoded and downmixed, then sent to the speech service. Only the transcript goes to core (`utterance`). Core sends sentences to speak (`speak`). They are synthesised one ahead of playback and played in order.
- **One connection carries every guild.** Several guilds can be in calls at once. When core disconnects, every voice connection is dropped, because core's in-call state died with it.

## Protocol

One JSON object per WebSocket text frame. Every message names its `guild_id`, and snowflakes are strings. `src/protocol.js` validates what core sends.

| Direction | Type | Fields |
|---|---|---|
| core → sidecar | `join` | `channel_id` |
| | `gateway` | `t` (`VOICE_STATE_UPDATE` / `VOICE_SERVER_UPDATE`), `d` |
| | `config` | `ignore_user_ids` (other bots in the call) |
| | `speak` | `id` (the turn), `text` (one sentence), `voice`, `speed` |
| | `chime` | queues the built-in "I heard you" tone behind any speech |
| | `stop` / `leave` | |
| sidecar → core | `hello` | `version` |
| | `payload` | `d`: an op-4 voice state for core's gateway |
| | `ready` / `disconnected` | `reason` |
| | `utterance` | `user_id`, `text`, `duration_ms`, `ended_at`, `stt_seconds` |
| | `speech` | `id`, `state` (`started` / `done` / `failed`), `synth_seconds` |

## Gotchas

- `joinVoiceChannel` defaults to `selfDeaf: true` *and* `selfMute: true`. Deafened, no audio arrives. Muted, Discord drops what the bot plays.
- The server binds `VOICE_BRIDGE_HOST` (default `127.0.0.1`). In k8s it runs in the core pod on localhost. In compose it binds `0.0.0.0` on the internal network with no published port, so any container there could connect: set `VOICE_BRIDGE_TOKEN` and core must send `Authorization: Bearer <token>`. A wrong or missing token is closed with 4001 before it can replace the connected core. `/health` needs no token.
- Opus comes from `opusscript` (WebAssembly), not the native `@discordjs/opus`, so the image needs no compiler and runs no install scripts.

## Develop

```bash
npm ci
npm test          # node:test, pure helpers only (audio, protocol, playback queue)
SPEECH_URL=http://localhost:8002 VOICE_BRIDGE_HOST=0.0.0.0 npm start
```

`VOICE_DEBUG=1` logs the voice connection's own debug output, and `LOG_LEVEL=debug` logs every transcript.
