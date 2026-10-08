// One guild's spoken output: a FIFO of sentences, synthesised one ahead of
// playback. Pure apart from the injected functions, so the ordering and stop
// rules are unit-tested in test/playback.test.js.
//
// Core streams the model's answer a sentence at a time, so the first sentence
// can be playing while the model is still writing the third. Fetching the
// NEXT sentence's audio while the current one plays hides the synthesis time
// of every sentence but the first. Looking further ahead would only queue
// more work on the CPU speech service, and a `stop` would then waste it.

export class SpeechQueue {
  /**
   * @param {object} deps
   * @param {(item) => Promise<{pcm: Buffer, seconds: number}>} deps.synthesize
   * @param {(pcm: Buffer) => Promise<void>} deps.play  resolves when playback ends
   * @param {() => void} deps.interrupt  stops whatever is playing now
   * @param {(event) => void} deps.emit  started / done / failed per item
   */
  constructor({ synthesize, play, interrupt, emit }) {
    this.synthesize = synthesize;
    this.play = play;
    this.interrupt = interrupt;
    this.emit = emit;
    this.items = [];
    this.running = false;
    // Bumped by stop(): a loop iteration from before the stop must not play
    // the audio it was waiting for.
    this.generation = 0;
  }

  /** Queue one item: {id, text, voice, speed} or {id, pcm} (pre-made audio). */
  enqueue(item) {
    this.items.push({ ...item, audio: null });
    this.#prefetch();
    if (!this.running) this.#run();
  }

  /** Drop everything queued and cut off what is playing. */
  stop() {
    this.generation += 1;
    const dropped = this.items;
    this.items = [];
    for (const item of dropped) {
      // Swallow the rejection of a prefetch nobody will await any more.
      item.audio?.catch(() => {});
      // Core counts every sentence it sends and waits for each to end; one
      // dropped here never plays, so it ends now. (The playing item is
      // already off the list and gets its own "done" from #run.)
      this.emit({ id: item.id, state: "failed", error: "stopped" });
    }
    this.interrupt();
  }

  get busy() {
    return this.running;
  }

  #start(item) {
    if (!item.audio) {
      item.audio = item.pcm ? Promise.resolve({ pcm: item.pcm, seconds: 0 }) : this.synthesize(item);
      item.audio.catch(() => {});
    }
  }

  #prefetch() {
    // The head (about to play) and the one after it.
    for (const item of this.items.slice(0, 2)) this.#start(item);
  }

  async #run() {
    this.running = true;
    try {
      while (this.items.length) {
        const generation = this.generation;
        const item = this.items[0];
        this.#start(item);
        let audio;
        try {
          audio = await item.audio;
        } catch (error) {
          if (generation !== this.generation) continue;
          this.items.shift();
          this.emit({ id: item.id, state: "failed", error: String(error?.message ?? error) });
          this.#prefetch();
          continue;
        }
        if (generation !== this.generation) continue;
        this.items.shift();
        this.#prefetch();
        this.emit({ id: item.id, state: "started", synth_seconds: audio.seconds });
        try {
          await this.play(audio.pcm);
        } catch {
          // Interrupted by stop(); the event below still closes the item.
        }
        this.emit({ id: item.id, state: "done" });
      }
    } finally {
      this.running = false;
    }
  }
}
