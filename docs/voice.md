# Voice calls

The bot can join a voice channel and talk out loud. Ask it in chat ("join voice", "come talk to us") and it joins the voice channel you are in, or use `/voice join` (optionally naming a channel). It opens a thread called "🎙️ Voice: <channel>" in the text channel you asked from, and reuses that thread on later calls. Pictures, code sandbox runs and other tool output from the call go there.

## Talking to it

It only answers when addressed. Start with the wake phrase, which by default is "hey" and the first word of its display name, with CamelCase split ("CleverHelperBot" is "hey clever helper bot"). Say the phrase and your question in one go ("hey sparky, what's the weather in London?"), or say the phrase alone, wait for the chime, then ask. Speech recognition sometimes clips a leading "hey", so a name of two or more words that starts what you say also counts on its own; a one-word name, or the name later in a sentence, needs the greeting. Everything else said in the call is heard but not answered. It stays in the conversation as context, so "hey sparky, what do you think?" works.

Before a tool runs (a web search, a picture, a code run) the bot says a short line about what it is doing, then gives the answer. Answers are spoken a sentence at a time while the rest is still being written.

- "hey sparky, stop" (or "shut up", "be quiet") stops it talking. Work already started, such as a picture or a code run, still finishes and appears in the thread.
- "hey sparky, leave" (or "goodbye", "go away"), `/voice leave`, or asking it to leave in other words ends the call.
- It leaves on its own 30 seconds after the last person does.

## Settings (per server)

| Command | What it sets |
|---|---|
| `/voice wake_word <phrase>` | The phrase that gets its attention, up to five words; `default` goes back to "hey <name>" |
| `/voice voice <name>` | Its voice. Start typing to pick from the list; blend voices with weights, e.g. `af_bella:0.6,am_adam:0.4` (up to four) |
| `/voice speed <0.5-2>` | How fast it talks |
| `/voice settings` | Shows the current settings (only to you) |

Voice ids start with a language and gender letter: `a` American English, `b` British English, then `f`/`m`. Others exist (Spanish, French, Japanese and more) and speak that language.

## While it is in a call

While the bot is in a call in a server, it does not reply in text there. @mentions get a short "I'm in a voice call" note, messages that were waiting in the queue are dropped, rules don't fire, and schedules wait until it leaves. `/generate_image` is refused until then. Other servers are unaffected. A sandbox run in the call's thread can still be steered by typing in the thread.

## How it works (for operators)

Voice is on by default in the Helm chart (`voice.enabled: true`); outside the chart it is off unless `VOICE_ENABLED` is set. It adds two CPU-only services:

- **voice sidecar** (`voicesidecar/`, Node, in the core pod). Discord voice I/O through `@discordjs/voice`, the library that can hear under Discord's DAVE end-to-end encryption. discord.py keeps the gateway connection and hands the sidecar the voice handshake. The sidecar splits audio per speaker at `VOICE_SILENCE_MS` of silence, gets it transcribed, and plays the sentences core sends.
- **speech** (`speechservice/`, Python). faster-whisper (`STT_MODEL`, default `small.en`) for speech-to-text and Kokoro-82M for text-to-speech, both on CPU. Models download into its volume on first boot.

Everything people say in a call is transcribed (that is how the wake phrase is found) and kept in the call's prompt, up to `VOICE_HISTORY_LIMIT` clips. Transcripts are not stored anywhere. They are only logged with `VOICE_DEBUG=1`. Thinking is off for voice turns (`VOICE_THINKING=1` turns it on) because a spoken reply waits for every reasoning token. Between turns the conversation is prefilled into llama.cpp so a turn only processes its last few tokens.
