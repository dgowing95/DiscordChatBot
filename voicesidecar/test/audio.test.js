import assert from "node:assert/strict";
import { test } from "node:test";

import { chime, downmixToMono, monoSeconds, stereoBytes, worthTranscribing } from "../src/audio.js";

function stereo(frames) {
  const buffer = Buffer.alloc(frames.length * 4);
  frames.forEach(([left, right], i) => {
    buffer.writeInt16LE(left, i * 4);
    buffer.writeInt16LE(right, i * 4 + 2);
  });
  return buffer;
}

test("downmix averages the two channels", () => {
  const mono = downmixToMono(stereo([[100, 300], [-32768, -32768], [32767, 32767], [-1, 1]]));
  assert.deepEqual([0, 1, 2, 3].map((i) => mono.readInt16LE(i * 2)), [200, -32768, 32767, 0]);
});

test("downmix drops a trailing partial frame", () => {
  assert.equal(downmixToMono(Buffer.alloc(10)).length, 4);
});

test("one second of 48 kHz audio", () => {
  assert.equal(stereoBytes(1), 192000);
  assert.equal(monoSeconds(Buffer.alloc(96000)), 1);
});

test("short clips are not worth transcribing", () => {
  assert.equal(worthTranscribing(Buffer.alloc(96000 * 0.3), 0.4), false);
  assert.equal(worthTranscribing(Buffer.alloc(96000 * 0.5), 0.4), true);
});

test("the chime is short stereo PCM that starts and ends quietly", () => {
  const pcm = chime();
  assert.equal(pcm.length % 4, 0);
  const seconds = pcm.length / 192000;
  assert.ok(seconds > 0.1 && seconds < 0.5, `chime lasts ${seconds}s`);
  assert.equal(pcm.readInt16LE(0), 0);
  assert.ok(Math.abs(pcm.readInt16LE(pcm.length - 4)) < 50);
});
