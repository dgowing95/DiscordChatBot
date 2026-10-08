// PCM helpers. Pure (no I/O), so they are unit-tested in test/audio.test.js.
//
// Discord voice is 48 kHz stereo signed 16-bit little-endian once decoded,
// and that is also what @discordjs/voice plays as StreamType.Raw.

export const SAMPLE_RATE = 48000;
const BYTES_PER_STEREO_SECOND = SAMPLE_RATE * 2 * 2;
const BYTES_PER_MONO_SECOND = SAMPLE_RATE * 2;

/** Seconds of audio in a mono 48 kHz s16le buffer. */
export function monoSeconds(buffer) {
  return buffer.length / BYTES_PER_MONO_SECOND;
}

/** Bytes of decoded stereo PCM that make up `seconds` of audio. */
export function stereoBytes(seconds) {
  return Math.floor(seconds * SAMPLE_RATE) * 4;
}

/**
 * Stereo s16le -> mono s16le by averaging the channels. Halves what goes to
 * the speech service; Whisper only reads one channel anyway. A trailing
 * partial frame is dropped.
 */
export function downmixToMono(stereo) {
  const frames = Math.floor(stereo.length / 4);
  const mono = Buffer.alloc(frames * 2);
  for (let i = 0; i < frames; i++) {
    const left = stereo.readInt16LE(i * 4);
    const right = stereo.readInt16LE(i * 4 + 2);
    mono.writeInt16LE((left + right) >> 1, i * 2);
  }
  return mono;
}

/**
 * Whether a finished utterance is worth transcribing: a cough or a click is
 * shorter than any request, and every clip costs a CPU transcription.
 */
export function worthTranscribing(mono, minSeconds) {
  return monoSeconds(mono) >= minSeconds;
}

/**
 * A short rising two-note chime (48 kHz stereo s16le): the "I heard you" cue,
 * played when someone says the wake phrase and again when their request
 * starts being answered. Generated rather than synthesised so it plays at
 * once, with no TTS round trip. A fundamental plus a quieter octave, struck
 * and decaying like a bell, so it stands out from speech without being harsh.
 */
const CHIME = (() => {
  const notes = [[784, 0.13], [1175, 0.24]]; // G5 then D6
  const parts = notes.map(([freq, seconds]) => {
    const frames = Math.floor(SAMPLE_RATE * seconds);
    const buffer = Buffer.alloc(frames * 4);
    for (let i = 0; i < frames; i++) {
      const t = i / SAMPLE_RATE;
      // Short fade in/out so the tone does not click, and a bell-like decay.
      const edge = Math.min(1, i / 240, (frames - i) / 480);
      const decay = Math.exp((-3 * t) / seconds);
      const wave = Math.sin(2 * Math.PI * freq * t) + 0.35 * Math.sin(4 * Math.PI * freq * t);
      const value = Math.round((wave / 1.35) * 14000 * decay * edge);
      buffer.writeInt16LE(value, i * 4);
      buffer.writeInt16LE(value, i * 4 + 2);
    }
    return buffer;
  });
  return Buffer.concat(parts);
})();

export function chime() {
  return CHIME;
}

export { BYTES_PER_STEREO_SECOND };
