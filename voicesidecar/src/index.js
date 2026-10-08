// Discord voice sidecar for the bot's core process.
//
// discord.py cannot receive voice under Discord's DAVE end-to-end encryption
// (mandatory since March 2026); @discordjs/voice can. So core keeps its own
// gateway connection and this process handles the voice connections only:
// core forwards the voice handshake events here, sends on its gateway the
// op-4 payloads this process asks for, and the audio itself never touches
// Python. See README.md for the protocol.
//
// One core connection carries every guild. A second connection replaces the
// first (a restarting core), and when core goes away every voice connection
// is dropped: core's in-call state died with it.

import http from "node:http";
import { VoiceConnectionStatus } from "@discordjs/voice";
import { WebSocketServer } from "ws";

import { GuildVoice } from "./guild.js";
import { log } from "./log.js";
import { PROTOCOL_VERSION, parseCoreMessage, tokenAccepted } from "./protocol.js";

const HOST = process.env.VOICE_BRIDGE_HOST || "127.0.0.1";
const PORT = Number(process.env.VOICE_BRIDGE_PORT) || 8765;
const TOKEN = process.env.VOICE_BRIDGE_TOKEN || "";

const guilds = new Map();
let core = null;

function send(message) {
  if (!core || core.readyState !== core.OPEN) return false;
  core.send(JSON.stringify(message));
  return true;
}

function dropAll(reason) {
  for (const guild of [...guilds.values()]) guild.destroy(reason);
  guilds.clear();
}

function handle(message) {
  const guildId = message.guild_id;
  let guild = guilds.get(guildId);
  if (message.type === "join") {
    if (guild && guild.channelId === message.channel_id && !guild.gone) {
      // Already there (a core that retried its join): say so again.
      if (guild.connection?.state.status === VoiceConnectionStatus.Ready) send({ type: "ready", guild_id: guildId });
      return;
    }
    guild?.destroy("rejoin");
    guild = new GuildVoice(guildId, message.channel_id, send, () => {
      if (guilds.get(guildId) === guild) guilds.delete(guildId);
    });
    guilds.set(guildId, guild);
    guild.connect();
    return;
  }
  if (!guild) {
    // A leave for a guild already gone still owes core an answer, so its
    // VoiceProtocol can finish disconnecting.
    if (message.type === "leave") send({ type: "disconnected", guild_id: guildId, reason: "not connected" });
    return;
  }
  switch (message.type) {
    case "gateway": guild.gateway(message.t, message.d); break;
    case "config": guild.configure(message); break;
    case "speak": guild.speak(message); break;
    case "chime": guild.playChime(); break;
    case "stop": guild.stop(); break;
    case "leave": guild.destroy("left"); break;
  }
}

const server = http.createServer((request, response) => {
  if (request.url === "/health") {
    response.writeHead(200, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ status: "ok", core: Boolean(core), guilds: guilds.size }));
    return;
  }
  response.writeHead(404);
  response.end();
});

const wss = new WebSocketServer({ server, maxPayload: 1024 * 1024 });
wss.on("connection", (socket, request) => {
  // Checked first: a refused client must not get to replace the real core.
  if (!tokenAccepted(TOKEN, request.headers.authorization)) {
    log.warn("refused a bridge connection without the right token");
    socket.close(4001, "unauthorized");
    return;
  }
  if (core) {
    log.warn("a new core connection replaces the old one");
    dropAll("core reconnected");
    core.close(4000, "replaced");
  }
  core = socket;
  log.info("core connected");
  send({ type: "hello", version: PROTOCOL_VERSION });
  socket.on("message", (data) => {
    let message;
    try {
      message = parseCoreMessage(data.toString());
    } catch (error) {
      log.warn(`ignoring a bad message from core: ${error.message}`);
      return;
    }
    try {
      handle(message);
    } catch (error) {
      log.error(`handling ${message.type} for ${message.guild_id} failed: ${error.stack || error}`);
    }
  });
  socket.on("close", () => {
    if (core !== socket) return;
    core = null;
    log.info("core disconnected; leaving every voice channel");
    dropAll("core disconnected");
  });
  socket.on("error", (error) => log.warn(`core socket error: ${error.message}`));
});

server.listen(PORT, HOST, () => log.info(`voice sidecar listening on ${HOST}:${PORT}`));

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    dropAll("shutdown");
    server.close();
    process.exit(0);
  });
}
