"""Queue items and execution of schedules and matched rule groups."""
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import discord

from classes.automation_policy import matches, settings, validate_schedule
from classes.automation_store import AutomationStore, PREFIX
from classes.message_queue import get_channel_lock
from classes.response_filter import filter_response as clean_response, chunk_for_discord, format_thinking_for_discord
from classes.text_llm_handler import TextLLMHandler
from classes.metrics import inc_automation, observe_automation_wait, observe_automation_execution

logger = logging.getLogger(__name__)


class AccessLost(PermissionError):
    """The entry can no longer run as configured; it is suspended, not retried."""


@dataclass
class AutomationJob:
    records: list
    claims: list
    message: object = None
    enqueued_at: float = 0
    renewal_task: object = None


async def _renew(job):
    store = AutomationStore()
    while True:
        await asyncio.sleep(120)
        for claim in job.claims:
            try:
                if not await store.renew(claim):
                    logger.warning("Automation claim expired before completion")
                    return
            except Exception:
                logger.exception("Automation claim renewal failed")


async def match_message(message, queue):
    if (not settings()["enabled"] or not message.guild or not message.content.strip()
            or message.author.bot or getattr(message, "webhook_id", None)
            or message.content.lower() == "!reset_history"):
        return False
    try:
        store = AutomationStore()
        rows = await store.list(message.guild.id, "rule")
        rows = [r for r in rows if r["status"] == "enabled" and r["channel_id"] == message.channel.id
                and matches(r["pattern"], message.content, r["match_mode"])]
        if not rows:
            return False
        if queue.full():
            inc_automation("rule", "skipped")
            return False
        admitted = []
        claims = []
        for row in rows:
            claim = await store.claim_rule(row, message.id)
            if claim:
                admitted.append(row)
                claims.append(claim)
        if not admitted:
            return False
        job = AutomationJob(admitted, claims, message, time.monotonic())
        try:
            queue.put_nowait(job)
        except asyncio.QueueFull:
            for claim in claims:
                await store.cancel_claim(claim, rule=True)
            inc_automation("rule", "skipped")
            return False
        job.renewal_task = asyncio.create_task(_renew(job))
        inc_automation("rule", "admitted")
        logger.info("Admitted rule job message=%s rules=%s", message.id, [r["id"] for r in admitted])
        return True
    except Exception:
        logger.exception("Rule admission failed; ordinary chat may continue")
        return False


async def poll_schedules(queue):
    if not settings()["enabled"]:
        return
    store = AutomationStore()
    await store.purge_terminal()
    for member in await store.due():
        if queue.full():
            inc_automation("schedule", "skipped")
            return
        guild, ident = member.split(":", 1)
        row = await store.get(guild, ident)
        if not row or row["status"] != "enabled":
            await store.redis.zrem(f"{PREFIX}:due", member)
            continue
        if row["timing"]["type"] != "once":
            try:
                validate_schedule(row["timing"], row["timezone"], settings()["min_hours"], datetime.now(timezone.utc))
            except ValueError as exc:
                await store.update(guild, ident, row["editor_id"], row["revision"], status="suspended")
                await store.finish(row, "suspended", str(exc))
                continue
        occurrence = str(row["next_run"])
        if await store.redis.exists(f"{PREFIX}:busy:{guild}:{ident}"):
            continue
        marker = f"{PREFIX}:started:{guild}:{ident}:{occurrence}"
        if await store.redis.exists(marker):
            await store.finish(row, "interrupted", "Process stopped after this occurrence started")
            inc_automation("schedule", "interrupted")
            continue
        claim = await store.claim(row, occurrence)
        if claim:
            job = AutomationJob([row], [claim], None, time.monotonic())
            try:
                queue.put_nowait(job)
            except asyncio.QueueFull:
                await store.cancel_claim(claim)
                inc_automation("schedule", "skipped")
                return
            job.renewal_task = asyncio.create_task(_renew(job))
            inc_automation("schedule", "admitted")
            logger.info("Admitted schedule=%s occurrence=%s", row["id"], occurrence)


async def execute(job, client):
    store = AutomationStore()
    kind = job.records[0]["kind"]
    started = time.monotonic()
    started_at = datetime.now(timezone.utc)
    observe_automation_wait(kind, started - job.enqueued_at)
    outcome = "skipped"
    rows = []
    channel = None
    send_started = False
    try:
        for row in job.records:
            current = await store.get(row["guild_id"], row["id"])
            if current and current["revision"] == row["revision"] and current["status"] == "enabled":
                rows.append(row)
        if not rows:
            return
        # NotFound/Forbidden here mean the channel or the last editor is
        # gone; that will not fix itself, so the entry is suspended rather
        # than failing (and posting a notice) on every occurrence.
        try:
            channel = client.get_channel(rows[0]["channel_id"])
            if channel is None:
                channel = await client.fetch_channel(rows[0]["channel_id"])
            guild = client.get_guild(rows[0]["guild_id"])
            actor = guild.get_member(rows[0]["editor_id"]) if guild else None
            if actor is None and guild:
                actor = await guild.fetch_member(rows[0]["editor_id"])
        except discord.NotFound as exc:
            channel = None
            raise AccessLost("The channel or the last editor is no longer in the server") from exc
        except discord.Forbidden as exc:
            channel = None
            raise AccessLost("The bot can no longer see this channel") from exc
        if actor is None:
            raise AccessLost("The last editor is no longer in the server")
        perms = channel.permissions_for(actor)
        if not perms.view_channel or not perms.send_messages:
            raise AccessLost("The last editor no longer has access to this channel")
        bot = getattr(guild, "me", None)
        if bot is not None:
            bot_perms = channel.permissions_for(bot)
            if not bot_perms.view_channel or not bot_perms.send_messages:
                raise AccessLost("The bot can no longer send in this channel")
        for row in rows:
            if row["kind"] == "schedule":
                marker = f"{PREFIX}:started:{row['guild_id']}:{row['id']}:{row['next_run']}"
                if not await store.redis.set(marker, "1", ex=604800, nx=True):
                    return
        now = datetime.now(timezone.utc)
        if job.message is None:
            local = now.astimezone(ZoneInfo(rows[0]["timezone"]))
            prompt = "Execute this scheduled action now. Current local date and time: " + local.isoformat()
            prompt += f" ({rows[0]['timezone']}). Current UTC time: {now.isoformat()}. Action: {rows[0]['action']}"
        else:
            prompt = "A message matched these rules. Execute their actions together in order and send one response. "
            prompt += f"Current UTC time: {now.isoformat()}. Message by {job.message.author.display_name}: {job.message.content}\n"
            prompt += "\n".join(f"{i}. {row['action']}" for i, row in enumerate(rows, 1))
        handler = TextLLMHandler([{"role": "user", "content": prompt}], rows[0]["guild_id"], None,
                                 client=client, actor_id=rows[0]["editor_id"], channel=channel, automatic=True)
        result = await handler.generate()
        if result == "Error":
            raise RuntimeError("Model run failed")
        output = clean_response(result, mention=str(client.user.id))
        async with get_channel_lock(channel.id):
            send_started = True
            for chunk in chunk_for_discord(output):
                await channel.send(chunk)
            if handler.sandbox_thread:
                await channel.send(f"Sandbox work: {handler.sandbox_thread.mention}")
            if handler.reasoning and os.getenv("SHOW_THINKING", "0").lower() in ("1", "true"):
                for chunk in format_thinking_for_discord(handler.reasoning):
                    await channel.send(chunk)
        for row in rows:
            await store.finish(row, "completed", started_at=started_at)
        outcome = "completed"
        inc_automation(kind, outcome)
        logger.info("Completed automation kind=%s ids=%s", kind, [r["id"] for r in rows])
    except Exception as exc:
        outcome = ("suspended" if isinstance(exc, AccessLost)
                   else "uncertain_delivery" if send_started else "failed")
        inc_automation(kind, outcome)
        logger.exception("Automation execution failed: %s", [r["id"] for r in job.records])
        for row in rows:
            try:
                if isinstance(exc, AccessLost):
                    await store.update(row["guild_id"], row["id"], row["editor_id"], row["revision"], status="suspended")
                await store.finish(row, outcome, str(exc), started_at=started_at)
            except Exception:
                logger.exception("Could not record automation failure")
        if channel and not send_started:
            try:
                await channel.send(f"⏸️ An automatic {kind} was suspended: {exc}. Resume it with /{kind} resume."
                                   if isinstance(exc, AccessLost) else
                                   "❌ An automatic run failed. Check its status with /schedule or /rule.")
            except Exception:
                logger.warning("Could not send the failure notice for %s", [r["id"] for r in rows], exc_info=True)
    finally:
        if job.renewal_task:
            job.renewal_task.cancel()
            await asyncio.gather(job.renewal_task, return_exceptions=True)
        observe_automation_execution(kind, outcome, time.monotonic() - started)
        for claim in job.claims:
            try:
                await store.release(claim)
            except Exception:
                logger.exception("Could not release automation claim")
