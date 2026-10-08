// The bridge protocol between core (discord.py) and this sidecar: one JSON
// object per WebSocket text frame, every one naming its guild. Pure, so the
// validation is unit-tested in test/protocol.test.js.
//
// core -> sidecar
//   join     {guild_id, channel_id}         connect to a voice channel
//   gateway  {guild_id, t, d}               a VOICE_STATE_UPDATE /
//                                           VOICE_SERVER_UPDATE discord.py got
//   config   {guild_id, ignore_user_ids}
//   speak    {guild_id, id, text, voice, speed}   one sentence, queued
//   chime    {guild_id}
//   stop     {guild_id}                     drop queued speech, stop playing
//   leave    {guild_id}
// sidecar -> core
//   hello        {version}
//   payload      {guild_id, d}              an op-4 voice state update for
//                                           discord.py to send on its gateway
//   ready        {guild_id}
//   disconnected {guild_id, reason}
//   utterance    {guild_id, user_id, text, duration_ms, ended_at, stt_seconds}
//   speech       {guild_id, id, state: started|done|failed, synth_seconds?}

import { timingSafeEqual } from "node:crypto";

export const PROTOCOL_VERSION = 1;

/**
 * Whether a connecting client may be core. With VOICE_BRIDGE_TOKEN set, core
 * must send it as "Authorization: Bearer <token>"; unset, anyone who can reach
 * the port may connect (in k8s that is only the core pod, on localhost).
 */
export function tokenAccepted(expected, authorization) {
  if (!expected) return true;
  const given = Buffer.from(String(authorization ?? ""));
  const wanted = Buffer.from(`Bearer ${expected}`);
  return given.length === wanted.length && timingSafeEqual(given, wanted);
}

const SNOWFLAKE = /^\d{5,25}$/;
const GATEWAY_EVENTS = new Set(["VOICE_STATE_UPDATE", "VOICE_SERVER_UPDATE"]);
export const MAX_SPEAK_CHARS = 1000;

function snowflake(value) {
  return typeof value === "string" && SNOWFLAKE.test(value);
}

/**
 * The parsed message, or throws an Error saying what is wrong with it.
 * Snowflakes are strings on the wire (they overflow a JS number).
 */
export function parseCoreMessage(raw) {
  let message;
  try {
    message = JSON.parse(raw);
  } catch {
    throw new Error("not JSON");
  }
  if (!message || typeof message !== "object") throw new Error("not an object");
  if (!snowflake(message.guild_id)) throw new Error("bad guild_id");
  switch (message.type) {
    case "join":
      if (!snowflake(message.channel_id)) throw new Error("bad channel_id");
      return message;
    case "gateway":
      if (!GATEWAY_EVENTS.has(message.t)) throw new Error("bad gateway event");
      if (!message.d || typeof message.d !== "object") throw new Error("bad gateway data");
      return message;
    case "config":
      if (message.ignore_user_ids !== undefined
          && !(Array.isArray(message.ignore_user_ids) && message.ignore_user_ids.every(snowflake))) {
        throw new Error("bad ignore_user_ids");
      }
      return message;
    case "speak":
      if (typeof message.id !== "string" || !message.id) throw new Error("bad id");
      if (typeof message.text !== "string" || !message.text.trim()) throw new Error("bad text");
      if (message.text.length > MAX_SPEAK_CHARS) throw new Error("text too long");
      return message;
    case "chime":
    case "stop":
    case "leave":
      return message;
    default:
      throw new Error(`unknown type ${JSON.stringify(message.type)}`);
  }
}
