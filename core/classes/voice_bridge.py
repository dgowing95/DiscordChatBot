"""The link to the voice sidecar (voicesidecar/), and the discord.py side of
its voice handshake.

discord.py cannot RECEIVE voice: Discord made its DAVE end-to-end
encryption mandatory in March 2026, and no maintained discord.py extension
decrypts it. @discordjs/voice does, so the voice connections live in a small
Node process and this module bridges it to discord.py, the way Lavalink and
wavelink do for playback:

  * discord.py keeps the one gateway connection. SidecarVoiceProtocol is the
    VoiceProtocol it routes this guild's VOICE_STATE_UPDATE and
    VOICE_SERVER_UPDATE to; they are forwarded raw to the sidecar.
  * The sidecar's op-4 payloads (join, move, leave) come back as `payload`
    messages and go out through Guild.change_voice_state on discord.py's
    gateway, since the sidecar has none of its own.
  * Audio never reaches Python: the sidecar sends transcripts (`utterance`)
    and plays the sentences it is sent (`speak`).

One WebSocket carries every guild. If it drops, every call ends (the sidecar
leaves its channels when core goes away), and the bridge reconnects.
"""
import asyncio
import json
import logging

import aiohttp
import discord

from classes import voice_gate

logger = logging.getLogger(__name__)

# How long a join may take before it is given up (sidecar handshake + DAVE).
CONNECT_TIMEOUT = 30.0
# How long a leave waits for the sidecar to confirm before cleaning up anyway.
LEAVE_TIMEOUT = 5.0
RECONNECT_DELAYS = (1, 2, 5, 10, 30)


class VoiceUnavailable(Exception):
    """The voice service cannot be reached right now."""


class VoiceBridge:
    def __init__(self, url: str, client: discord.Client):
        self.url = url
        self.client = client
        self.ws = None
        self.protocols: dict[int, "SidecarVoiceProtocol"] = {}
        self._session = None
        self._tasks: set = set()

    @property
    def connected(self) -> bool:
        return self.ws is not None and not self.ws.closed

    async def send(self, message: dict) -> bool:
        if not self.connected:
            return False
        try:
            await self.ws.send_str(json.dumps(message))
            return True
        except Exception as e:
            logger.warning(f"Voice bridge: send failed: {e}")
            return False

    async def run_forever(self) -> None:
        attempt = 0
        while True:
            try:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession()
                async with self._session.ws_connect(self.url, heartbeat=20, max_msg_size=1024 * 1024) as ws:
                    self.ws = ws
                    attempt = 0
                    logger.info(f"Voice bridge connected to {self.url}")
                    async for frame in ws:
                        if frame.type == aiohttp.WSMsgType.TEXT:
                            await self._dispatch(frame.data)
                        elif frame.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Voice bridge: cannot reach {self.url}: {e}")
            finally:
                was_connected = self.ws is not None
                self.ws = None
                if was_connected:
                    logger.warning("Voice bridge disconnected; ending every voice call")
                    await self._lost()
            await asyncio.sleep(RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)])
            attempt += 1

    async def close(self) -> None:
        if self.ws is not None:
            await self.ws.close()
        if self._session is not None:
            await self._session.close()

    async def _lost(self) -> None:
        for protocol in list(self.protocols.values()):
            protocol.sidecar_gone("the voice service restarted")
        for session in voice_gate.sessions():
            try:
                await session.end("I lost my connection to the voice service.")
            except Exception:
                logger.exception("Voice bridge: ending a session failed")

    async def _dispatch(self, raw: str) -> None:
        try:
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "hello":
                logger.info(f"Voice sidecar protocol v{message.get('version')}")
                return
            guild_id = int(message["guild_id"])
        except Exception as e:
            logger.warning(f"Voice bridge: bad message from the sidecar: {e}")
            return
        try:
            if kind == "payload":
                await self._send_voice_state(guild_id, message.get("d") or {})
            elif kind == "ready":
                protocol = self.protocols.get(guild_id)
                if protocol is not None:
                    protocol.sidecar_ready()
            elif kind == "disconnected":
                protocol = self.protocols.get(guild_id)
                if protocol is not None:
                    protocol.sidecar_gone(message.get("reason") or "disconnected")
                session = voice_gate.session(guild_id)
                if session is not None and not getattr(session, "ending", False):
                    self._spawn(session.end("I was disconnected from the voice channel."))
            elif kind == "utterance":
                session = voice_gate.session(guild_id)
                if session is not None:
                    # A task, not awaited here: an utterance can end the call
                    # ("hey sparky, leave"), and leaving waits for the
                    # sidecar's `disconnected` -- which this read loop could
                    # never deliver while it was blocked on that very wait.
                    self._spawn(session.on_utterance(message))
            elif kind == "speech":
                session = voice_gate.session(guild_id)
                if session is not None:
                    session.on_speech(message)
        except Exception:
            logger.exception(f"Voice bridge: handling {kind} for guild {guild_id} failed")

    def _spawn(self, coro) -> None:
        """Run a handler off the read loop, keeping a reference until it ends."""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._handler_done)

    def _handler_done(self, task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Voice bridge: a handler failed", exc_info=task.exception())

    async def _send_voice_state(self, guild_id: int, d: dict) -> None:
        """An op-4 payload the sidecar asked for, sent on discord.py's gateway."""
        guild = self.client.get_guild(guild_id)
        if guild is None:
            return
        channel_id = d.get("channel_id")
        channel = discord.Object(id=int(channel_id)) if channel_id else None
        await guild.change_voice_state(channel=channel, self_mute=bool(d.get("self_mute")),
                                       self_deaf=bool(d.get("self_deaf")))


_bridge: VoiceBridge | None = None


def get_bridge() -> VoiceBridge | None:
    return _bridge


def start_bridge(url: str, client: discord.Client) -> VoiceBridge:
    global _bridge
    if _bridge is None:
        _bridge = VoiceBridge(url, client)
    return _bridge


class SidecarVoiceProtocol(discord.VoiceProtocol):
    """This guild's voice connection, as far as discord.py is concerned.

    discord.py registers the protocol BEFORE it calls connect(), and only
    cleans up after a TimeoutError (by calling disconnect(force=True)). Any
    other failure inside connect() would leave guild.voice_client set and
    every later join failing with "Already connected" -- so connect() cleans
    up after itself on every failure, and disconnect() always ends in
    cleanup().
    """

    def __init__(self, client, channel):
        super().__init__(client, channel)
        self.guild = channel.guild
        self.bridge = get_bridge()
        self._ready = asyncio.Event()
        self._gone = asyncio.Event()
        self._cleaned = False

    # discord.py -> sidecar --------------------------------------------------

    async def on_voice_state_update(self, data, /) -> None:
        await self.bridge.send({"type": "gateway", "guild_id": str(self.guild.id),
                                "t": "VOICE_STATE_UPDATE", "d": data})
        if data.get("channel_id") is None:
            # Kicked, or left: there is no call any more.
            self.cleanup()
        else:
            channel = self.guild.get_channel(int(data["channel_id"]))
            if channel is not None:
                self.channel = channel

    async def on_voice_server_update(self, data, /) -> None:
        await self.bridge.send({"type": "gateway", "guild_id": str(self.guild.id),
                                "t": "VOICE_SERVER_UPDATE", "d": data})

    async def connect(self, *, timeout: float, reconnect: bool, self_deaf: bool = False,
                      self_mute: bool = False) -> None:
        if self.bridge is None:
            self.cleanup()
            raise VoiceUnavailable("Voice is not set up on this bot.")
        self.bridge.protocols[self.guild.id] = self
        try:
            if not await self.bridge.send({"type": "join", "guild_id": str(self.guild.id),
                                           "channel_id": str(self.channel.id)}):
                raise VoiceUnavailable("The voice service is not reachable right now.")
            ready = asyncio.create_task(self._ready.wait())
            gone = asyncio.create_task(self._gone.wait())
            done, pending = await asyncio.wait({ready, gone}, timeout=timeout,
                                               return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            if ready not in done:
                if gone in done:
                    raise VoiceUnavailable("The voice connection failed.")
                raise asyncio.TimeoutError()
        except asyncio.TimeoutError:
            raise  # discord.py calls disconnect(force=True), which cleans up
        except BaseException:
            await self.bridge.send({"type": "leave", "guild_id": str(self.guild.id)})
            self.cleanup()
            raise

    async def disconnect(self, *, force: bool = False) -> None:
        try:
            if self.bridge is not None and not self._gone.is_set():
                if await self.bridge.send({"type": "leave", "guild_id": str(self.guild.id)}):
                    try:
                        await asyncio.wait_for(self._gone.wait(), LEAVE_TIMEOUT)
                    except asyncio.TimeoutError:
                        pass
            if self.guild.me is not None and self.guild.me.voice is not None:
                # The sidecar normally asks for this itself (its destroy sends
                # a leave payload); this covers a sidecar that is gone.
                await self.guild.change_voice_state(channel=None)
        except Exception as e:
            logger.warning(f"Voice: disconnect in guild {self.guild.id} was not clean: {e}")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        if self.bridge is not None and self.bridge.protocols.get(self.guild.id) is self:
            del self.bridge.protocols[self.guild.id]
        super().cleanup()

    # sidecar -> here --------------------------------------------------------

    def sidecar_ready(self) -> None:
        self._ready.set()

    def sidecar_gone(self, reason: str) -> None:
        logger.info(f"Voice: sidecar connection for guild {self.guild.id} ended ({reason})")
        self._gone.set()
