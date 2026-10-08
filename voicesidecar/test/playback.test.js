import assert from "node:assert/strict";
import { test } from "node:test";

import { SpeechQueue } from "../src/playback.js";

function deferred() {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return { promise, resolve, reject };
}

/** A queue whose synthesis and playback are driven by the test. */
function harness() {
  const log = [];
  const synth = new Map();
  const playing = [];
  const queue = new SpeechQueue({
    synthesize: (item) => {
      log.push(`synth ${item.text}`);
      const d = deferred();
      synth.set(item.text, d);
      return d.promise;
    },
    play: (pcm) => {
      log.push(`play ${pcm.toString()}`);
      const d = deferred();
      playing.push(d);
      return d.promise;
    },
    interrupt: () => {
      log.push("interrupt");
      for (const d of playing.splice(0)) d.resolve();
    },
    emit: (event) => log.push(`${event.state} ${event.id}`),
  });
  const tick = () => new Promise((resolve) => setImmediate(resolve));
  const audio = (text) => ({ pcm: Buffer.from(text), seconds: 0.1 });
  return { queue, log, synth, playing, tick, audio };
}

test("plays in order and synthesises only one sentence ahead", async () => {
  const { queue, log, synth, playing, tick, audio } = harness();
  for (const text of ["one", "two", "three"]) queue.enqueue({ id: "t", text });
  await tick();
  assert.deepEqual(log, ["synth one", "synth two"]);
  synth.get("two").resolve(audio("two"));
  synth.get("one").resolve(audio("one"));
  await tick();
  assert.deepEqual(log.slice(2), ["synth three", "started t", "play one"]);
  synth.get("three").resolve(audio("three"));
  playing.shift().resolve();
  await tick();
  playing.shift().resolve();
  await tick();
  playing.shift().resolve();
  await tick();
  assert.deepEqual(log.filter((l) => l.startsWith("play")), ["play one", "play two", "play three"]);
  assert.equal(queue.busy, false);
});

test("stop drops queued speech and cuts off the current sentence", async () => {
  const { queue, log, synth, tick, audio } = harness();
  queue.enqueue({ id: "a", text: "one" });
  queue.enqueue({ id: "a", text: "two" });
  synth.get("one").resolve(audio("one"));
  await tick();
  queue.stop();
  synth.get("two").resolve(audio("two"));
  await tick();
  assert.ok(log.includes("interrupt"));
  assert.ok(!log.includes("play two"));
  // A new answer after the stop plays normally.
  queue.enqueue({ id: "b", text: "three" });
  synth.get("three").resolve(audio("three"));
  await tick();
  assert.ok(log.includes("play three"));
});

test("a failed synthesis is reported and the next sentence still plays", async () => {
  const { queue, log, synth, playing, tick, audio } = harness();
  queue.enqueue({ id: "t", text: "bad" });
  queue.enqueue({ id: "t", text: "good" });
  synth.get("bad").reject(new Error("503 busy"));
  synth.get("good").resolve(audio("good"));
  await tick();
  await tick();
  assert.ok(log.includes("failed t"));
  assert.ok(log.includes("play good"));
  playing.shift().resolve();
});

test("pre-made audio (the chime) needs no synthesis", async () => {
  const { queue, log, tick } = harness();
  queue.enqueue({ id: "chime", pcm: Buffer.from("ding") });
  await tick();
  assert.deepEqual(log, ["started chime", "play ding"]);
});
