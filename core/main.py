import logging
import discord
import os
import asyncio
import random
import time
import aiohttp
import io
from classes.message_handler import MessageHandler, close_http_session
from classes.text_llm_handler import TextLLMHandler
from classes.llm_config import llm_model
from classes.config_manager import configManager
from classes.image_generation import (
    GENERATED_IMAGE_FILENAME,
    create_image,
    image_generation_enabled,
)
from classes.sandbox_agent import sandbox_enabled
from classes import attachment_cache, help_catalog, sandbox_thread_inbox, whats_new
from classes.common import embed_from_data
from classes.redis_client import text_client
from classes import automation_runner, voice_gate
from classes.polls import remember_poll
from classes.automation_policy import settings as automation_settings
from classes.voice_policy import settings as voice_settings
from classes.message_queue import (
    get_channel_lock, make_message_queue, mark_enqueued, pop_enqueued, worker_count,
)
from classes.metrics import (
    inc_messages_processed,
    inc_messages_received,
    inc_queue_drop,
    inc_sandbox_thread_message,
    observe_queue_wait,
    observe_reply_latency,
    set_attachment_cache_bytes,
    set_context_window_from_env,
    set_message_queue_size,
    start_metrics_server_from_env,
)

logger = logging.getLogger(__name__)

# The only logging setup in the app; every module just uses
# logging.getLogger(__name__). LOG_LEVEL tunes it without a code change - these
# were all bare print() calls before, so there was no level to tune.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
# CONTENT_GUARD_DEBUG=1 (the default) means the guard's per-request detail is
# wanted, which is DEBUG on its own logger rather than on the root.
if os.environ.get("CONTENT_GUARD_DEBUG", "1").strip().lower() not in ("0", "false", "no", "off"):
    logging.getLogger("classes.content_guard").setLevel(logging.DEBUG)
# VOICE_DEBUG=1 logs what is heard and said in voice calls (every transcript,
# so off by default); the voice sidecar reads the same switch.
if os.environ.get("VOICE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on"):
    logging.getLogger("classes.voice_session").setLevel(logging.DEBUG)


intents = discord.Intents.default()
intents.message_content = True



class Bot(discord.Client):
    async def close(self):
        for task in getattr(self, "background_tasks", []):
            task.cancel()
        if getattr(self, "background_tasks", None):
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        while not message_queue.empty():
            item = message_queue.get_nowait()
            if isinstance(item, automation_runner.AutomationJob) and item.renewal_task:
                item.renewal_task.cancel()
            message_queue.task_done()
        # The image-download session is shared by every prompt build, so it
        # outlives them all and is closed here, once.
        await close_http_session()
        from classes.voice_bridge import get_bridge
        if get_bridge() is not None:
            await get_bridge().close()
        await super().close()


client = Bot(intents=intents)
config = configManager()

# Bounded queue shared by the worker pool. Sizing (WORKER_COUNT /
# QUEUE_MAX_SIZE), the per-channel locks and the concurrency model live in
# classes/message_queue.py (pure, unit-tested in core/tests/).
message_queue = make_message_queue()

# How often the LLM server's per-slot context is re-read, so a restart or a
# changed --ctx-size/--parallel reaches the history token budget and the gauge.
SLOT_CONTEXT_REFRESH_SECONDS = 300


async def refresh_slot_context_forever():
    while True:
        await TextLLMHandler.refresh_slot_context()
        await asyncio.sleep(SLOT_CONTEXT_REFRESH_SECONDS)


async def register_commands():
    logger.info("Registering commands")
    command_tree = discord.app_commands.CommandTree(client=client, fallback_to_global=True)
    if automation_settings()["enabled"]:
        from classes.automation_commands import register_automation_commands
        register_automation_commands(command_tree)
    if voice_settings()["enabled"]:
        from classes.voice_commands import register_voice_commands
        register_voice_commands(command_tree, client)

    @command_tree.command(name="system", description="Change the behaviour/personality of the bot")
    async def change_system(ctx, system: str):
        await config.update_setting("system", system, ctx.guild.id)
        await ctx.response.send_message(content=f"System updated to: \"{system}\"")

    @command_tree.command(name="get_system", description="See the existing behaviour/personality of the bot")
    async def get_system(ctx):
        system = await config.get_setting("system", ctx.guild.id)
        await ctx.response.send_message(content=f"System is currently: \"{system}\"")

    @command_tree.command(name="temperature", description="Change the randomness of responses, max of 2.0 is max random")
    async def change_temperature(ctx, temperature: float):
        await config.update_setting("temperature", temperature, ctx.guild.id)
        await ctx.response.send_message(content=f"Temperature updated to: \"{temperature}\"")

    @command_tree.command(name="chance", description="Change the chance (0-50%) that the bot replies without being mentioned, default 5%")
    async def change_chance(ctx, chance: discord.app_commands.Range[int, 0, 50]):
        await config.update_setting("response_chance", chance, ctx.guild.id)
        await ctx.response.send_message(content=f"Response chance updated to: \"{chance}%\"")

    @command_tree.command(name="get_chance", description="See the current chance that the bot replies without being mentioned")
    async def get_chance(ctx):
        chance = await config.get_setting("response_chance", ctx.guild.id) or 5
        await ctx.response.send_message(content=f"Response chance is currently: \"{chance}%\"")
    
    # Ephemeral, so neither reply ever lands in channel.history (and so in a
    # prompt), and nobody else's channel fills up with a help listing.
    @command_tree.command(name="help", description="See everything the bot can do")
    async def help_cmd(ctx):
        data = help_catalog.help_embed_data(image_generation_enabled(), sandbox_enabled())
        await ctx.response.send_message(embed=embed_from_data(data), ephemeral=True)

    @command_tree.command(name="whats_new", description="See the new features in the current version")
    async def whats_new_cmd(ctx):
        version = whats_new.app_version()
        notes = whats_new.load_notes() if version else []
        if notes:
            await ctx.response.send_message(
                embed=embed_from_data(whats_new.notes_embed_data(version, notes)), ephemeral=True)
        elif version:
            await ctx.response.send_message(
                content=f"No new features in {version}. Try /help to see everything I can do.",
                ephemeral=True)
        else:
            await ctx.response.send_message(
                content="I don't know which version I'm running (a development build), "
                        "so there are no release notes. Try /help.",
                ephemeral=True)

    # Only offered when the code sandbox is enabled (SANDBOX_ENABLED; set
    # from the helm chart's sandbox.enabled). Per-guild toggle, default off:
    # when true, run_code_sandbox streams the sandbox's commands and output
    # to the channel in a live-updating message; when false it only sends
    # the one static "Running in sandbox" embed.
    if sandbox_enabled():

        @command_tree.command(name="sandbox_progress_updates",
                              description="Enable/disable live progress updates (commands & output) for code sandbox runs in this server")
        async def change_sandbox_progress_updates(ctx, enabled: bool):
            await config.update_setting("sandbox_progress_updates", str(enabled), ctx.guild.id)
            await ctx.response.send_message(content=f"Sandbox progress updates are now: \"{enabled}\"")
    
    # Only offered when the diffusion service is enabled (IMAGE_GEN_ENABLED;
    # set from the helm chart's diffusion.enabled).
    if image_generation_enabled():

        @command_tree.command(name="generate_image", description="Generate an image from a text prompt using the image service")
        async def generate_image_cmd(ctx, prompt: str):
            # Its prompt rewrite and image check use the LLM, which is the
            # call's while the bot is in voice here.
            if voice_gate.active(ctx.guild_id):
                await ctx.response.send_message(VOICE_BUSY_REPLY, ephemeral=True)
                return
            # Image generation is slow (queue + GPU): defer first (spinner) so
            # the command doesn't time out, then respond to the deferred
            # interaction via the webhook. ctx is a raw discord.Interaction
            # (the bot is a discord.Client), so: ctx.response.defer() to defer
            # and ctx.edit_original_response() to answer it later — NOT
            # ctx.response.edit_message(), which raises InteractionResponded.
            await ctx.response.defer()
            logger.info(f"Slash command: generating image for prompt: {prompt}")
            # Raw user text, never seen by the agent: this path is the reason
            # the SDXL prompt rules live in image_prompt.py rather than in the
            # generate_image tool's docstring, which only the agent reads.
            # The typed prompt is both the description and the request the
            # result is checked against.
            try:
                result = await create_image(prompt, prompt)
            except Exception as e:
                logger.warning(f"Image generation failed: {e}")
                await ctx.edit_original_response(
                    content="❌ Image generation failed — the image service may be down or busy. Try again later."
                )
                return
            await ctx.edit_original_response(
                content="🎨",
                attachments=[discord.File(io.BytesIO(result.png), filename=GENERATED_IMAGE_FILENAME)],
            )

    synced_commands = await command_tree.sync()
    for synced_command in synced_commands:
        logger.info(f"Command '{synced_command.name}' synced")

@client.event
async def on_ready():
    logger.info(f'Logged in as {client.user}')
    # Prometheus /metrics endpoint (METRICS_PORT; empty/0 disables).
    start_metrics_server_from_env()
    # The context window the LLM server was started with (LLM_CONTEXT_LENGTH),
    # published as a gauge so a dashboard can express prompt sizes as a
    # fraction of it without hard-coding the number.
    set_context_window_from_env()
    # The per-slot context (llama.cpp /props) is what one request can actually
    # use, and what the history token budget is measured against.
    if not getattr(client, "background_tasks", None):
        client.background_tasks = [client.loop.create_task(refresh_slot_context_forever())]
        if automation_settings()["enabled"]:
            client.background_tasks.append(client.loop.create_task(schedule_forever()))
        if voice_settings()["enabled"]:
            from classes.voice_bridge import start_bridge
            bridge = start_bridge(voice_settings()["bridge_url"], client, voice_settings()["bridge_token"])
            client.background_tasks.append(client.loop.create_task(bridge.run_forever()))
    # Start the worker pool immediately so the bot still consumes messages
    # even if the model check or command sync fails (e.g. server not up yet
    # after a power cycle).
    count = worker_count()
    logger.info(f"Starting {count} queue worker(s)")
    if not getattr(client, "workers_started", False):
        client.workers_started = True
        for _ in range(count):
            client.background_tasks.append(client.loop.create_task(process_messages()))
    try:
        # Same accessor the bot itself uses, so an unset MODEL cannot make
        # the readiness check verify a different model than the one requested.
        model = llm_model()
        await TextLLMHandler.check_model_ready(model)
    except Exception as e:
        logger.warning(f"Failed to check model: {e}")
        return
    await register_commands()
    


def invalidate_attachments(channel_id, message_ids, deleted):
    """Drop cached images of deleted/edited messages, so a deleted image
    is neither kept in memory nor sent to the model again."""
    cache = attachment_cache.cache()
    for message_id in message_ids:
        cache.invalidate_message(channel_id, message_id, deleted=deleted)
    set_attachment_cache_bytes(cache.bytes)


# Raw events: they fire for messages outside discord.py's message cache too.
@client.event
async def on_raw_message_delete(payload):
    invalidate_attachments(payload.channel_id, [payload.message_id], deleted=True)


@client.event
async def on_raw_bulk_message_delete(payload):
    invalidate_attachments(payload.channel_id, payload.message_ids, deleted=True)


@client.event
async def on_raw_message_edit(payload):
    # An edit can remove an attachment; a message that is still there is
    # simply cached again on its next build.
    invalidate_attachments(payload.channel_id, [payload.message_id], deleted=False)


@client.event
async def on_message(message):
    # Every poll, the bot's own included, is recorded so check_polls can
    # find it after it scrolls out of the history window (no-op otherwise).
    await remember_poll(message)
    # A sandbox run in flight in this thread takes the message instead of
    # the outer LLM, so people can steer a run while it happens ("make it
    # blue instead") without an @mention. Returning here is also the
    # concurrency guard: a second run in the same thread would race the
    # first to persist dcb:sandbox_snapshot:{thread_id} on teardown and
    # silently clobber it. Answers to the sandbox's ask_user questions come
    # through here too — the ledger is the one pipeline for thread input.
    if (message.author != client.user and not message.author.bot
            and sandbox_thread_inbox.is_run_active(message.channel.id)):
        await route_to_sandbox(message)
        return
    if await voice_dev_inject(message):
        return
    # In a voice call in this guild: no text replies, rules or queueing
    # (classes/voice_gate.py). Sandbox steering above still works, so a run
    # started from the call can be steered in its thread.
    if message.guild is not None and voice_gate.active(message.guild.id):
        await voice_busy_notice(message)
        return
    if await automation_runner.match_message(message, message_queue):
        set_message_queue_size(message_queue.qsize())
        return

    # The reply filter runs at receive time so only messages the bot will
    # actually handle ever enter the queue. "Received" = passed this filter
    # and was enqueued (mentioned or random-chance hit); discarded and
    # queue-full messages are not counted.
    if not await should_handle_message(message):
        return
    guild_id = message.guild.id if message.guild else 0

    # Backpressure: the queue is bounded (QUEUE_MAX_SIZE). When it's full the
    # bot is behind (e.g. a 10-minute sandbox run) — dropping beats answering
    # stale messages against stale history. Mentioned messages get a short
    # busy reply so they aren't silently ignored; random-chance messages
    # drop quietly.
    if message_queue.full():
        logger.warning(f"Message queue full ({message_queue.maxsize}); dropping message {message.id}")
        inc_queue_drop(guild_id)
        if client.user in message.mentions:
            try:
                await message.reply("⏳ I'm still working through my queue — try me again in a moment.")
            except Exception as e:
                logger.warning(f"Could not send busy reply to {message.id}: {e}")
        return

    inc_messages_received(guild_id)
    mark_enqueued(message.id)
    await message_queue.put(message)
    set_message_queue_size(message_queue.qsize())


# What a person sees for each delivery result (sandbox_thread_inbox.deliver).
# A reaction is the receipt; the replies explain every case where the run
# will NOT act on the message, so nobody mistakes 🚫 for "seen".
_SANDBOX_REACTIONS = {
    sandbox_thread_inbox.ACCEPTED: "📨",
    sandbox_thread_inbox.FINISHING: "⏳",
}
_SANDBOX_REPLIES = {
    sandbox_thread_inbox.EMPTY: (
        "🚫 The running sandbox can only read text — send that as a text message."),
    sandbox_thread_inbox.FINISHING: (
        "⏳ This sandbox run is finishing, so that hasn't been applied. Once it "
        "closes, @mention me in this thread to carry on with it."),
    sandbox_thread_inbox.ATTACHMENT_ONLY: (
        "🚫 The running sandbox can't open attachments — paste the relevant part "
        "as text instead."),
    sandbox_thread_inbox.TOO_LONG: (
        f"🚫 That's too long to pass to the running sandbox (max "
        f"{sandbox_thread_inbox.MAX_MESSAGE_CHARS} characters) — shorten it or split it up."),
    sandbox_thread_inbox.FULL: (
        "🚫 The sandbox has too many unanswered messages right now — give it a "
        "moment to catch up, then send that again."),
}
# How often "received, I'll review it before my next step" may be posted
# while a command runs: rapid messages get one notice, not one each.
SANDBOX_RECEIPT_NOTICE_SECONDS = 30


async def route_to_sandbox(message) -> None:
    """Delivers one thread message to the sandbox run claimed there and
    acknowledges it accurately: 📨 means the run WILL see it, and says
    nothing about whether it has been applied — the sandbox answers that
    itself (respond_to_updates)."""
    ledger = sandbox_thread_inbox.get_run(message.channel.id)
    reference = getattr(message, "reference", None)
    outcome = sandbox_thread_inbox.deliver(
        message.channel.id, message.id,
        message.author.id, message.author.display_name, message.content,
        reply_to=getattr(reference, "message_id", None),
        has_attachments=bool(getattr(message, "attachments", None)),
    )
    inc_sandbox_thread_message(outcome)
    if outcome == sandbox_thread_inbox.DUPLICATE:
        return
    try:
        await message.add_reaction(_SANDBOX_REACTIONS.get(outcome, "🚫"))
    except Exception as e:
        logger.warning(f"Sandbox inbox: could not acknowledge {message.id}: {e}")
    reply = _SANDBOX_REPLIES.get(outcome)
    if (reply is None and outcome == sandbox_thread_inbox.ACCEPTED and ledger is not None
            and ledger.busy_since is not None
            and time.monotonic() - ledger.last_receipt_notice > SANDBOX_RECEIPT_NOTICE_SECONDS):
        ledger.last_receipt_notice = time.monotonic()
        reply = "📨 Received — a command is still running; I'll review this before my next step."
    if reply is None:
        return
    try:
        await message.reply(reply, mention_author=False)
    except Exception as e:
        logger.warning(f"Sandbox inbox: could not reply to {message.id}: {e}")


VOICE_BUSY_REPLY = "🎙️ I'm in a voice call right now — talk to me there, or use /voice leave."
# How often the "in a call" notice may be posted per channel.
VOICE_NOTICE_SECONDS = 60
_voice_notice_at: dict[int, float] = {}


async def voice_busy_notice(message) -> None:
    """A fixed reply (no LLM) to a mention while the bot is in a call."""
    if message.author == client.user or message.author.bot or client.user not in message.mentions:
        return
    now = time.monotonic()
    if now - _voice_notice_at.get(message.channel.id, -VOICE_NOTICE_SECONDS) < VOICE_NOTICE_SECONDS:
        return
    _voice_notice_at[message.channel.id] = now
    try:
        await message.reply(VOICE_BUSY_REPLY, mention_author=False)
    except Exception as e:
        logger.warning(f"Could not send the in-a-call notice: {e}")


async def voice_dev_inject(message) -> bool:
    """VOICE_DEV_INJECT=1 only (local testing, never in the chart): drive a
    call from text, so the wake -> LLM -> tools -> speech path can be tested
    without anyone speaking. "!voice_join <channel id>" joins, "!voice_say
    <text>" is heard as if the author had said it."""
    content = message.content or ""
    if not content.startswith(("!voice_join", "!voice_say")) or message.guild is None:
        return False
    if not voice_settings()["dev_inject"] or message.author == client.user:
        return False
    from classes import voice_session
    command, _, argument = content.partition(" ")
    try:
        if command == "!voice_join":
            channel = message.guild.get_channel(int(argument.strip()))
            await voice_session.start(client, channel, message.channel, message.author)
        elif command == "!voice_say":
            session = voice_gate.session(message.guild.id)
            if session is not None:
                await session.heard(message.author.id, message.author.display_name, argument)
    except Exception as e:
        logger.warning(f"Voice dev inject failed: {e}")
        await message.reply(f"❌ {e}", mention_author=False)
    return True


@client.event
async def on_voice_state_update(member, before, after):
    session = voice_gate.session(member.guild.id)
    if session is None:
        return
    if member.id == client.user.id:
        if after.channel is None:
            await session.end("I was disconnected from the voice channel.")
        else:
            session.moved(after.channel)
        return
    channel_ids = {getattr(before.channel, "id", None), getattr(after.channel, "id", None)}
    if session.voice_channel.id in channel_ids:
        session.members_changed()


async def should_handle_message(message) -> bool:
    # Decides whether the bot should reply to a message; called from
    # on_message before enqueueing, so no-reply messages never reach the
    # queue. Rules (in order): must have content/embeds/attachments, must
    # not be from the bot, must not be a history reset, mentions always
    # reply, otherwise roll the per-guild response chance (default 5%).
    if len(message.content) == 0 and len(message.embeds) == 0 and not message.attachments:
        return False
    if message.author == client.user:
        return False
    if message.content.lower() == "!reset_history":
        return False
    if client.user in message.mentions:
        return True
    return await random_chance_reply(message)


async def random_chance_reply(message) -> bool:
    guild_id = message.guild.id if message.guild else 0
    chance = await config.get_setting("response_chance", guild_id) or 5
    try:
        chance = min(max(float(chance), 0), 50)
    except (ValueError, TypeError):
        chance = 5
    return random.uniform(0, 100) < chance


async def maybe_announce_whats_new(message) -> None:
    """Posts this version's What's New embed in the message's channel the
    first time a guild triggers the bot on it (see classes/whats_new.py).

    One atomic SET ... GET claims the announcement, so two workers handling
    the same guild at once cannot both post it. Only the latest version is
    stored, which is also what skips a version a guild never triggered on.
    A release without features ships an empty notes file and posts nothing.
    Sent under the channel lock, like replies, so it never lands between
    another worker's reply chunks. A failed send gives the claim back, so a
    one-off Discord error does not cost the guild this version's notes.
    Fail-soft: an announcement must never cost a reply."""
    if message.guild is None or not whats_new.whats_new_enabled():
        return
    version = whats_new.app_version()
    if not version:
        return
    try:
        notes = whats_new.load_notes()
        if not notes:
            return
        redis = text_client()
        key = whats_new.seen_key(message.guild.id)
        previous = await redis.set(key, version, get=True)
        if previous == version:
            return
        embed = embed_from_data(whats_new.notes_embed_data(version, notes))
        try:
            async with get_channel_lock(message.channel.id):
                await message.channel.send(embed=embed)
        except Exception:
            if previous is None:
                await redis.delete(key)
            else:
                await redis.set(key, previous)
            raise
    except Exception as e:
        logger.warning(f"What's New announcement failed: {e}")


async def process_messages():
    while True:
        message = await message_queue.get()
        set_message_queue_size(message_queue.qsize())
        if isinstance(message, automation_runner.AutomationJob):
            try:
                # Its guild went into a voice call while it waited.
                if voice_gate.active(message.records[0]["guild_id"]):
                    await automation_runner.discard(message)
                else:
                    await automation_runner.execute(message, client)
            finally:
                message_queue.task_done()
            continue
        if message.guild is not None and voice_gate.active(message.guild.id):
            # Queued before the bot joined a call here; dropped, not delayed:
            # answered after the call it would be stale.
            pop_enqueued(message.id)
            message_queue.task_done()
            logger.info("Dropping a queued message: the bot is in a voice call in that guild")
            continue
        handler = MessageHandler(message, client)

        # Every queued message already passed should_handle_message() in
        # on_message, so handle it directly.
        guild_id = message.guild.id if message.guild else 0
        logger.info("Picking up message from queue")
        # None when the message was not enqueued through on_message: then
        # there is nothing to measure, and no zero is recorded in its place.
        enqueued_at = pop_enqueued(message.id)
        if enqueued_at is not None:
            observe_queue_wait(guild_id, time.monotonic() - enqueued_at)
        outcome = "exception"
        try:
            # The per-channel lock is SCOPED inside handle_message to the two
            # fast phases (build + send) — see classes/message_queue.py — so
            # the worker loop itself never serializes channels or
            # same-channel messages around the LLM run. The typing indicator
            # spans the whole handle, so the channel keeps showing "typing"
            # during the (unlocked) LLM/tool phase.
            async with message.channel.typing():
                await handler.handle_message()
            outcome = getattr(handler, "outcome", "replied")
            # Only after a real reply: not after the ❌ of a failed run.
            if outcome == "replied":
                await maybe_announce_whats_new(message)
            inc_messages_processed(guild_id)
            message_queue.task_done()
            logger.info("Done with message from queue")
        except Exception as e:
            logger.warning("Error handling message: " + str(e))
            message_queue.task_done()
            logger.info("Done with message from queue")
        if enqueued_at is not None:
            observe_reply_latency(guild_id, outcome, time.monotonic() - enqueued_at)


async def schedule_forever():
    while True:
        try:
            await automation_runner.poll_schedules(message_queue)
            set_message_queue_size(message_queue.qsize())
        except Exception:
            logger.exception("Schedule poll failed")
        await asyncio.sleep(15)


if __name__ == "__main__":
    # Guarded so the module can be imported (by the tests, and by anything
    # that just wants to inspect it) without connecting to Discord.
    automation_settings()
    client.run(os.environ['DISCORD_TOKEN'])
