// One guild's voice connection: join through core's gateway, hear each
// speaker, transcribe what they say, and play queued speech.

import { Readable } from "node:stream";
import {
  AudioPlayerStatus,
  EndBehaviorType,
  NoSubscriberBehavior,
  StreamType,
  VoiceConnectionStatus,
  createAudioPlayer,
  createAudioResource,
  entersState,
  joinVoiceChannel,
} from "@discordjs/voice";
import prism from "prism-media";

import { chime, downmixToMono, stereoBytes, worthTranscribing } from "./audio.js";
import { SpeechQueue } from "./playback.js";
import { synthesize, transcribe } from "./speech.js";
import { log } from "./log.js";

const SILENCE_MS = Number(process.env.VOICE_SILENCE_MS) || 800;
const MIN_SECONDS = 0.4;
// Past this an utterance is cut and transcribed as it stands, so one person
// talking non-stop cannot hold a transcription back forever.
const MAX_SECONDS = 30;
// How long a dropped connection may take to come back (Discord moving the
// bot between servers, someone dragging it to another channel) before it is
// given up on.
const RECONNECT_MS = 5_000;

export class GuildVoice {
  /**
   * @param {string} guildId
   * @param {string} channelId
   * @param {(message: object) => boolean} send  to core; false when it is gone
   * @param {() => void} onGone  called once the connection is destroyed
   */
  constructor(guildId, channelId, send, onGone) {
    this.guildId = guildId;
    this.channelId = channelId;
    this.send = send;
    this.onGone = onGone;
    this.adapter = null;
    this.ignored = new Set();
    this.listening = new Set();
    this.gone = false;
    this.player = createAudioPlayer({ behaviors: { noSubscriber: NoSubscriberBehavior.Play } });
    this.player.on("error", (error) => log.warn(`guild ${guildId}: player error: ${error.message}`));
    this.speech = new SpeechQueue({
      synthesize,
      play: (pcm) => this.#play(pcm),
      interrupt: () => this.player.stop(true),
      emit: (event) => this.send({ type: "speech", guild_id: guildId, ...event }),
    });
  }

  connect() {
    // selfDeaf AND selfMute default to true in @discordjs/voice: deafened,
    // the bot receives no audio at all; muted, Discord drops what it plays.
    this.connection = joinVoiceChannel({
      guildId: this.guildId,
      channelId: this.channelId,
      selfDeaf: false,
      selfMute: false,
      debug: process.env.VOICE_DEBUG === "1",
      adapterCreator: (methods) => {
        this.adapter = methods;
        return {
          // The op-4 payload goes to core, which sends it on discord.py's
          // gateway connection: this process has no gateway of its own.
          sendPayload: (payload) => this.send({ type: "payload", guild_id: this.guildId, d: payload.d }),
          destroy: () => {
            this.adapter = null;
          },
        };
      },
    });
    const connection = this.connection;
    connection.on("debug", (message) => log.debug(`guild ${this.guildId}: ${message}`));
    connection.on("stateChange", (oldState, newState) => {
      log.info(`guild ${this.guildId}: ${oldState.status} -> ${newState.status}`);
    });
    connection.on(VoiceConnectionStatus.Ready, () => {
      this.send({ type: "ready", guild_id: this.guildId });
    });
    connection.on(VoiceConnectionStatus.Disconnected, async () => {
      try {
        await Promise.race([
          entersState(connection, VoiceConnectionStatus.Signalling, RECONNECT_MS),
          entersState(connection, VoiceConnectionStatus.Connecting, RECONNECT_MS),
        ]);
      } catch {
        this.destroy("disconnected");
      }
    });
    connection.on(VoiceConnectionStatus.Destroyed, () => this.#gone("destroyed"));
    connection.on("error", (error) => log.warn(`guild ${this.guildId}: connection error: ${error.message}`));
    connection.subscribe(this.player);
    connection.receiver.speaking.on("start", (userId) => this.#listen(userId));
  }

  gateway(t, d) {
    if (!this.adapter) return;
    if (t === "VOICE_STATE_UPDATE") this.adapter.onVoiceStateUpdate(d);
    else this.adapter.onVoiceServerUpdate(d);
  }

  configure({ ignore_user_ids: ignored }) {
    if (ignored !== undefined) this.ignored = new Set(ignored);
  }

  speak({ id, text, voice, speed }) {
    this.speech.enqueue({ id, text, voice, speed });
  }

  chime() {
    this.speech.enqueue({ id: "chime", pcm: chime() });
  }

  stop() {
    this.speech.stop();
  }

  destroy(reason = "left") {
    this.speech.stop();
    if (this.connection && this.connection.state.status !== VoiceConnectionStatus.Destroyed) {
      this.connection.destroy();
    }
    this.#gone(reason);
  }

  #gone(reason) {
    if (this.gone) return;
    this.gone = true;
    this.speech.stop();
    this.send({ type: "disconnected", guild_id: this.guildId, reason });
    this.onGone();
  }

  #play(pcm) {
    return new Promise((resolve) => {
      const resource = createAudioResource(Readable.from([pcm]), { inputType: StreamType.Raw });
      const finish = () => {
        this.player.off(AudioPlayerStatus.Idle, finish);
        this.player.off("error", finish);
        resolve();
      };
      this.player.on(AudioPlayerStatus.Idle, finish);
      this.player.on("error", finish);
      this.player.play(resource);
    });
  }

  #listen(userId) {
    if (this.gone || this.ignored.has(userId) || this.listening.has(userId)) return;
    this.listening.add(userId);
    const stream = this.connection.receiver.subscribe(userId, {
      end: { behavior: EndBehaviorType.AfterSilence, duration: SILENCE_MS },
    });
    const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
    const chunks = [];
    let bytes = 0;
    let finished = false;
    const maxBytes = stereoBytes(MAX_SECONDS);

    const finish = () => {
      if (finished) return;
      finished = true;
      this.listening.delete(userId);
      stream.unpipe(decoder);
      if (!stream.destroyed) stream.destroy();
      decoder.destroy();
      this.#transcribe(userId, Buffer.concat(chunks));
    };
    decoder.on("data", (chunk) => {
      chunks.push(chunk);
      bytes += chunk.length;
      if (bytes >= maxBytes) finish();
    });
    // A corrupt packet (or one that failed DAVE decryption and slipped
    // through) must cost that utterance at most, never the listener.
    decoder.on("error", (error) => {
      log.debug(`guild ${this.guildId}: opus decode error for ${userId}: ${error.message}`);
    });
    stream.on("error", (error) => {
      log.debug(`guild ${this.guildId}: receive error for ${userId}: ${error.message}`);
      finish();
    });
    stream.on("end", finish);
    stream.on("close", finish);
    stream.pipe(decoder);
  }

  async #transcribe(userId, stereo) {
    const mono = downmixToMono(stereo);
    if (!worthTranscribing(mono, MIN_SECONDS)) return;
    const endedAt = Date.now();
    try {
      const result = await transcribe(mono);
      const text = (result.text || "").trim();
      log.debug(`guild ${this.guildId}: ${userId} said ${JSON.stringify(text)} (${result.seconds}s)`);
      if (!text || this.gone) return;
      this.send({
        type: "utterance",
        guild_id: this.guildId,
        user_id: userId,
        text,
        duration_ms: Math.round((mono.length / 96000) * 1000),
        ended_at: endedAt,
        stt_seconds: result.seconds,
      });
    } catch (error) {
      log.warn(`guild ${this.guildId}: transcription failed: ${error.message}`);
    }
  }
}
