import assert from "node:assert/strict";
import { test } from "node:test";

import { MAX_SPEAK_CHARS, parseCoreMessage, tokenAccepted } from "../src/protocol.js";

const G = "123456789012345678";
const parse = (message) => parseCoreMessage(JSON.stringify(message));

test("valid messages parse", () => {
  assert.equal(parse({ type: "join", guild_id: G, channel_id: G }).type, "join");
  assert.equal(parse({ type: "gateway", guild_id: G, t: "VOICE_SERVER_UPDATE", d: { token: "x" } }).t,
    "VOICE_SERVER_UPDATE");
  assert.deepEqual(parse({ type: "config", guild_id: G, ignore_user_ids: [G] }).ignore_user_ids, [G]);
  assert.equal(parse({ type: "speak", guild_id: G, id: "t1", text: "Hello." }).text, "Hello.");
  for (const type of ["chime", "stop", "leave"]) assert.equal(parse({ type, guild_id: G }).type, type);
});

for (const [name, message, error] of [
  ["not JSON", "{", /not JSON/],
  ["numeric guild id", { type: "stop", guild_id: 123 }, /guild_id/],
  ["unknown type", { type: "explode", guild_id: G }, /unknown type/],
  ["join without channel", { type: "join", guild_id: G }, /channel_id/],
  ["other gateway events", { type: "gateway", guild_id: G, t: "MESSAGE_CREATE", d: {} }, /gateway event/],
  ["empty speech", { type: "speak", guild_id: G, id: "t1", text: "  " }, /text/],
  ["speech without id", { type: "speak", guild_id: G, text: "hi" }, /id/],
  ["huge speech", { type: "speak", guild_id: G, id: "t", text: "a".repeat(MAX_SPEAK_CHARS + 1) }, /too long/],
  ["bad ignore list", { type: "config", guild_id: G, ignore_user_ids: ["bot"] }, /ignore_user_ids/],
]) {
  test(`refuses ${name}`, () => {
    assert.throws(() => parseCoreMessage(typeof message === "string" ? message : JSON.stringify(message)), error);
  });
}

test("the bridge token is checked only when one is set", () => {
  assert.equal(tokenAccepted("", undefined), true);
  assert.equal(tokenAccepted("", "Bearer anything"), true);
  assert.equal(tokenAccepted("s3cret", "Bearer s3cret"), true);
  assert.equal(tokenAccepted("s3cret", "Bearer wrong!"), false);
  assert.equal(tokenAccepted("s3cret", "Bearer s3cre"), false);
  assert.equal(tokenAccepted("s3cret", undefined), false);
});
