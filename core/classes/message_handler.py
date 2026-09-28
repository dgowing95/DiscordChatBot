import logging
import asyncio, re, json, time, os, io, base64
import contextlib
import random
import aiohttp
import discord
import pillow_heif
from PIL import Image

# Teaches Pillow to open HEIC/HEIF (and AVIF); their output is re-encoded to PNG
# further down (Ollama cannot decode HEIF), so format detection just needs to work.
pillow_heif.register_heif_opener()
from classes.text_llm_handler import TextLLMHandler, slot_context_tokens
from classes import attachment_cache, history_policy, reply_policy
from classes.history_policy import (
    ANCHORED,
    KEEP,
    REFRESH,
    REASON_TOKEN_BUDGET,
    Decision,
    estimate_tokens,
    history_budget,
    history_limit,
    history_mode,
    initial_decision,
    max_window,
    refresh_messages,
    reserve_tokens,
    state_after_refresh,
    trim_count,
    window_decision,
)
from classes.response_filter import (
    chunk_for_discord,
    filter_response as clean_response,
    format_thinking_for_discord,
)
from classes.metrics import (
    inc_attachment_cache_lookup,
    inc_attachment_download_failure,
    inc_history_refresh,
    inc_history_trimmed,
    observe_response_generation,
    observe_stage,
    set_attachment_cache_bytes,
)
from classes.message_queue import get_channel_lock, in_flight_hint

logger = logging.getLogger(__name__)

# Max image attachments forwarded to the LLM per message (keeps prompts a sane size).
MAX_IMAGES_PER_MESSAGE = 3
# Image downloads running at once, across every build in the process.
DOWNLOAD_CONCURRENCY = 4
DOWNLOAD_TIMEOUT_SECONDS = 30
# Stands in for an image while estimating tokens (estimate_tokens counts
# every image the same, whatever its data).
_PLACEHOLDER_IMAGE = {"type": "input_image", "image_url": ""}

# 1/0: send the model's <think> reasoning to Discord, collapsed behind a
# spoiler-hidden code block (default: off — the reasoning is dropped).
# Set to 1/true to send it as spoiler-hidden follow-up message(s).
SHOW_THINKING = os.environ.get("SHOW_THINKING", "0").lower() not in ("0", "false")

# Formats Ollama's image decoder can actually handle (Go's image/* + a few extras).
# Anything else (e.g. WebP) is re-encoded to PNG before being sent.
_OLLAMA_FRIENDLY = {"JPEG": "image/jpeg", "PNG": "image/png", "GIF": "image/gif", "BMP": "image/bmp"}


def encode_image_for_llm(data: bytes):
    """Validate and normalize one downloaded image.

    Returns (bytes, mime_type) ready for a base64 data URL, or None when the
    payload is not a decodable image. Ollama rejects formats it cannot decode
    (e.g. WebP: 'Failed to load image or audio file'), so we re-encode to a
    supported format and derive the real MIME type from the pixels instead of
    trusting the CDN's Content-Type header."""
    try:
        img = Image.open(io.BytesIO(data))
        fmt = (img.format or "").upper()
    except Exception:
        return None
    if fmt in _OLLAMA_FRIENDLY:
        # Already decodable by Ollama as-is; re-encoding through PIL would
        # just burn CPU for no behavioural difference.
        return data, _OLLAMA_FRIENDLY[fmt]
    out = io.BytesIO()
    img.convert("RGBA").save(out, format="PNG")
    return out.getvalue(), "image/png"


def _encode_data_url(data: bytes):
    """encode_image_for_llm plus base64, as one call for a worker thread."""
    try:
        encoded = encode_image_for_llm(data)
    except Exception:
        return None
    if encoded is None:
        return None
    image_data, ctype = encoded
    return f"data:{ctype};base64,{base64.b64encode(image_data).decode()}"


def image_targets(attachments) -> list:
    """(attachment id, url, content type) for the attachments sent to the
    LLM: images only, at most MAX_IMAGES_PER_MESSAGE."""
    targets = []
    for attachment in attachments or []:
        if len(targets) >= MAX_IMAGES_PER_MESSAGE:
            break
        content_type = (attachment.content_type or "").lower()
        if not content_type.startswith("image/"):
            continue
        url = attachment.proxy_url or attachment.url
        if not url:
            continue
        targets.append((attachment.id, url, content_type))
    return targets


# One HTTP session and one download limit shared by every build, each tied
# to the event loop it was made on (the tests run a loop per test).
_session = None
_session_loop = None
_download_slots = None
_download_slots_loop = None


def _http_session():
    global _session, _session_loop
    loop = asyncio.get_running_loop()
    if _session is None or _session.closed or _session_loop is not loop:
        _session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT_SECONDS))
        _session_loop = loop
    return _session


def _download_semaphore():
    global _download_slots, _download_slots_loop
    loop = asyncio.get_running_loop()
    if _download_slots is None or _download_slots_loop is not loop:
        _download_slots = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)
        _download_slots_loop = loop
    return _download_slots


async def close_http_session():
    """Close the shared download session (bot shutdown)."""
    global _session
    if _session is not None and not _session.closed:
        await _session.close()
    _session = None


async def _download(url):
    try:
        async with _download_semaphore():
            async with _http_session().get(url) as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to download image {url}: HTTP {resp.status}")
                    inc_attachment_download_failure("http")
                    return None
                return await resp.read()
    except Exception as e:
        logger.warning(f"Failed to download image {url}: {e}")
        inc_attachment_download_failure("error")
        return None


async def load_image_data_url(url, content_type):
    """Download one image and return it as a base64 data URL, or None."""
    data = await _download(url)
    if not data:
        return None
    data_url = await asyncio.to_thread(_encode_data_url, data)
    if data_url is None:
        logger.warning(f"Skipping image {url}: not a decodable image (format={content_type!r})")
        inc_attachment_download_failure("undecodable")
    return data_url


def is_reset(message) -> bool:
    """A `!reset_history` message: history before it (and it) is left out."""
    return (message.content or "").lower() == "!reset_history"


class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.text_response = ""
        self.discord_message_object = None
        # How the handle ended, for the reply-latency metric (main.py):
        # "replied", or "llm_error" when the run failed and got the ❌.
        self.outcome = "replied"
        self._history = []
        self._generation = 0

    def _guild_id(self):
        guild = getattr(self.message, "guild", None)
        return guild.id if guild else 0

    def _observe_stage(self, stage, started):
        observe_stage(stage, self._guild_id(), time.monotonic() - started)

    @contextlib.asynccontextmanager
    async def _channel_lock(self):
        """The channel's scoped lock, with the wait for it measured."""
        started = time.monotonic()
        async with get_channel_lock(self.message.channel.id):
            self._observe_stage("lock_wait", started)
            yield

    async def build_messages(self):
        """Build self.messages: channel history, then the triggering message.

        Two halves, so handle_message can hold the channel lock for the first
        only: selecting reads Discord and picks the window, preparing
        downloads and encodes the images."""
        await self.select_messages()
        await self.prepare_messages()

    async def select_messages(self):
        """Choose the history messages for this prompt (run under the lock).

        History is read from Discord BEFORE the trigger's id rather than as
        "the newest N minus the first": a message that waited in the queue
        has newer messages above it, and dropping the newest one used to drop
        someone else's message while the trigger appeared twice.

        Which history goes in is the history policy's call (sliding or
        anchored - see classes/history_policy.py). Token estimates use a
        placeholder per image: estimate_tokens counts every image the same,
        so the estimate matches the prepared prompt (or is a little high if a
        download fails). Nothing is downloaded here.
        """
        self._generation = attachment_cache.cache().generation()
        limit = history_limit()
        budget = history_budget(slot_context_tokens(), reserve_tokens())
        fixed = self._estimate(self.message, always=True)
        if history_mode() == ANCHORED:
            groups = await self._anchored_groups(limit, budget, fixed)
        else:
            recent = await self._fetch_recent(limit - 1)
            groups = self._trim([(m, self._estimate(m)) for m in recent], fixed, budget)
        self._history = [m for m, _ in groups]

    async def prepare_messages(self):
        """Download/encode the selected messages' images and build
        self.messages (run outside the lock).

        Every message, the trigger included, goes through the same
        _format_group, so this turn's trigger serializes exactly as it will
        as history next turn; that is what lets the LLM server reuse its
        cached prefix. A history message deleted since select_messages (its
        delete event reached the attachment cache in between) is left out.
        """
        history = self._history
        wanted = [m for m in [*history, self.message] if image_targets(m.attachments)]
        started = time.monotonic()
        parts = await asyncio.gather(*(self.image_parts(m) for m in wanted))
        if wanted:
            observe_stage("attachments", self._guild_id(), time.monotonic() - started)
        parts = {m.id: p for m, p in zip(wanted, parts)}
        deleted = attachment_cache.cache().deleted_since(
            self._generation, self.message.channel.id, [m.id for m in history])
        messages = []
        for m in history:
            if m.id not in deleted:
                messages.extend(self._format_group(m, parts.get(m.id, [])))
        messages.extend(self._format_group(
            self.message, parts.get(self.message.id, []), role="user", always=True))
        self.messages = messages

    def _estimate(self, message, always=False):
        placeholders = [_PLACEHOLDER_IMAGE] * len(image_targets(message.attachments))
        return estimate_tokens(self._format_group(message, placeholders, always=always))

    async def _anchored_groups(self, limit, budget, fixed):
        """(message, tokens) history pairs for anchored mode; commits the
        anchor on a refresh."""
        store = history_policy.store()
        key = (self._guild_id(), self.message.channel.id)
        trigger_id = self.message.id
        refresh_every = refresh_messages()
        state = store.get(key)
        decision = initial_decision(state, trigger_id)
        groups = None
        if decision is None:
            fetched = await self._fetch_after(state.after_id, max_window(limit, refresh_every) + 1)
            decision = window_decision(
                state, trigger_id, [(m.id, is_reset(m)) for m in fetched], limit, refresh_every)
            if decision.kind == KEEP:
                groups = [(m, self._estimate(m)) for m in fetched]
                tokens = fixed + sum(t for _, t in groups)
                if budget is not None and tokens > budget:
                    decision = Decision(REFRESH, REASON_TOKEN_BUDGET)
                    groups = None
        if groups is None:
            groups = [(m, self._estimate(m)) for m in await self._fetch_recent(limit - 1)]
        groups = self._trim(groups, fixed, budget)
        if decision.kind != KEEP:
            inc_history_refresh(decision.reason)
            logger.info(f"History {decision.kind} ({decision.reason}) in channel "
                        f"{self.message.channel.id} for message {trigger_id}")
        if decision.kind == REFRESH:
            store.commit(key, state_after_refresh([m.id for m, _ in groups], trigger_id))
        return groups

    def _trim(self, groups, fixed, budget):
        """Drop the oldest whole messages until the prompt fits the budget."""
        drop = trim_count([tokens for _, tokens in groups], fixed, budget)
        if drop:
            inc_history_trimmed(drop)
            logger.info(f"History trimmed {drop} oldest message(s) to fit the "
                        f"{budget}-token budget (message {self.message.id})")
        return groups[drop:]

    async def _fetch_recent(self, count):
        """The newest `count` messages before the trigger, oldest first,
        stopping at (and leaving out) a `!reset_history`."""
        if count <= 0:
            return []
        started = time.monotonic()
        found = []
        async for m in self.message.channel.history(limit=count, before=self.message, oldest_first=False):
            if is_reset(m):
                break
            found.append(m)
        self._observe_stage("history_fetch", started)
        found.reverse()
        return found

    async def _fetch_after(self, after_id, cap):
        """Up to `cap` messages after `after_id` (exclusive) and before the
        trigger, oldest first. Resets are kept: the policy acts on them."""
        started = time.monotonic()
        found = [m async for m in self.message.channel.history(
            limit=cap, after=discord.Object(id=after_id), before=self.message, oldest_first=True)]
        self._observe_stage("history_fetch", started)
        return found

    def _format_group(self, message, image_parts, role=None, always=False):
        """One Discord message as prompt entries: its text (with any images)
        first, then its embeds in order - how Discord shows it.

        Never mutates the message. `always` keeps an entry for a message with
        no text or images (the trigger must always be in the prompt).
        """
        if role is None:
            role = "assistant" if message.author.id == self.client.user.id else "user"
        entries = []
        if message.content or image_parts or always:
            text = f"Message from '{message.author.name}': {self.clean_message_content(message)}"
            # With images the content becomes a list of parts (base64 images + text);
            # the agents SDK forwards them as multimodal chat-completions input.
            entries.append({
                'role': role,
                'content': [*image_parts, {"type": "text", "text": text}] if image_parts else text,
            })
        for embed in message.embeds:
            embed_dict = embed.to_dict()
            embed_dict.pop('fields', None)
            entries.append({
                'role': role,
                'content': f"Discord Embed from '{message.author.name}' converted to JSON: {json.dumps(embed_dict)}",
            })
        return entries


    async def image_parts(self, message) -> list:
        """A message's image attachments as chat-content image parts.

        Each part is {'type': 'input_image', 'image_url': '<base64 data URL>'}
        which the agents SDK converts to the OpenAI 'image_url' wire format
        for the LLM. Encoded images come from the attachment cache; the rest
        are downloaded (at most DOWNLOAD_CONCURRENCY at once, process-wide).
        Non-image attachments and failed downloads are skipped, and a failure
        is not cached, so the next build retries it."""
        targets = image_targets(message.attachments)
        if not targets:
            return []
        cache = attachment_cache.cache()
        channel_id = self.message.channel.id

        async def _one(attachment_id, url, content_type):
            value, result = await cache.get_or_load(
                (channel_id, message.id, attachment_id),
                lambda: load_image_data_url(url, content_type))
            inc_attachment_cache_lookup(result)
            return value

        urls = await asyncio.gather(*(_one(*t) for t in targets))
        set_attachment_cache_bytes(cache.bytes)
        return [{"type": "input_image", "image_url": u} for u in urls if u]


    def clean_message_content(self, message):
        return (message.content or "").replace(f'<@{self.client.user.id}>', '').strip()
    


    def filter_response(self, text_response):
        # Filter logic lives in classes.response_filter (pure, unit-tested);
        # we only inject the guild-specific bits (bot mention id).
        return clean_response(text_response, mention=str(self.client.user.id))


    async def handle_message_send(self, message_content, channel=None):
        channel = channel or self.message.channel
        # chunk_for_discord, not textwrap.wrap directly: wrap alone can emit a
        # chunk over Discord's 2000-char limit when the text contains a long
        # whitespace-free run, and the rejected send costs the whole reply.
        chunks = chunk_for_discord(message_content)
        last = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            await channel.send(chunk)
            if i < last:
                await asyncio.sleep(1)


    async def handle_thinking_send(self, thinking, channel=None):
        # Sent as follow-up message(s) after the answer, each a spoiler-hidden
        # code block (closed by default - click to reveal), instead of being
        # discarded like the rest of the <think> block.
        channel = channel or self.message.channel
        chunks = format_thinking_for_discord(thinking)
        if not chunks:
            return
        await channel.send("-# Reasoning (click to expand):")
        last = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            await channel.send(chunk)
            if i < last:
                await asyncio.sleep(1)




    async def handle_message(self):
        logger.info(f'Handling message: {self.message.content}')
        # Histogram covers the whole handling: prompt build + LLM run + send.
        # Observed on both outcomes (the ❌ path is still a timed attempt;
        # its failures are separately counted in llm_errors).
        start = time.monotonic()
        # SCOPED per-channel lock (see classes/message_queue.py): the lock
        # guards only the two FAST phases — build and send — never the LLM
        # run or tool calls. A free worker can therefore answer a NEW message
        # in this same channel while this one is stuck in a slow tool, the
        # chunked replies never interleave, and every history snapshot is
        # consistent. The lock is never held across an LLM/tool await, so no
        # deadlock is possible. (channel.typing() is held by
        # main.process_messages for the whole handle, so the channel keeps
        # showing "typing" during the unlocked LLM phase.)
        async with self._channel_lock():
            # 1) Select the history under the lock: the channel.history()
            #    read is a consistent snapshot (no half-sent replies from a
            #    concurrent same-channel handle).
            await self.select_messages()
            # If an earlier message's slow tool is still running (or just
            # finished) in this channel, tell the model — it can then answer
            # follow-ups honestly ("it's still running") instead of guessing
            # that the previous request went unanswered.
            hint = in_flight_hint(self.message.channel.id)
            allow_double_reply = reply_policy.policy.eligible(self.message.channel.id)
        # 2) Images are downloaded/encoded (or taken from the attachment
        #    cache) after the lock is released, so a slow CDN does not hold
        #    up other handles in this channel.
        await self.prepare_messages()
        if hint:
            self.messages.append({"role": "user", "content": hint})
        # 3) LLM run + tool calls UNLOCKED (the slow phase; other messages —
        #    same channel or not — can build/generate concurrently).
        ollama = TextLLMHandler(
            self.messages, self.message.guild.id, self.message,
            client=self.client,
        )
        ollama.allow_double_reply = allow_double_reply
        response = await ollama.generate()

        # generate() collects the reasoning itself: our llama.cpp server
        # returns it in reasoning_content, not as think tags inside the
        # answer, so it is gone from `response` by the time we get here.
        # Read the attribute directly (no getattr default) so dropping it
        # from TextLLMHandler breaks a test instead of silently killing the
        # feature again.
        thinking = ollama.reasoning if SHOW_THINKING else ""
        # If run_code_sandbox created/reused a thread this run, the sandbox's
        # own output already lives there — send the outer agent's reply
        # there too instead of the original channel, so the conversation
        # doesn't end up split across two places.
        target_channel = getattr(ollama, "sandbox_thread", None) or self.message.channel

        if response == "Error":
            # The run broke part-way (e.g. it ran out of turns chaining tool
            # calls). Its tools have already posted their embeds and files,
            # so a bare ❌ reads as "the tool worked but the bot went quiet" —
            # send the reasoning it did produce so the failure is legible.
            self.outcome = "llm_error"
            await self.message.add_reaction('❌')
            if thinking:
                async with self._channel_lock():
                    started = time.monotonic()
                    await self.handle_thinking_send(thinking, channel=target_channel)
                    self._observe_stage("send", started)
        else:
            response = self.filter_response(response)
            # 4) Send under the lock: serializes the chunked replies of
            #    concurrent same-channel handles (no interleaved chunks).
            async with self._channel_lock():
                started = time.monotonic()
                parts = reply_policy.reply_parts(
                    response,
                    allowed=(allow_double_reply
                             and not getattr(ollama, "sandbox_thread", None)
                             and reply_policy.policy.available(target_channel.id)),
                )
                if len(parts) == 2:
                    await self.handle_message_send(parts[0], channel=target_channel)
                    reply_policy.policy.started_pair(target_channel.id)
                    await asyncio.sleep(random.uniform(0.8, 1.5))
                    await self.handle_message_send(parts[1], channel=target_channel)
                    logger.debug("Double reply delivered in channel %s", target_channel.id)
                else:
                    if reply_policy.MARKER in response:
                        logger.debug("Double reply fallback (eligibility, cooldown or validation) in channel %s",
                                     target_channel.id)
                    await self.handle_message_send(parts[0], channel=target_channel)
                    if parts[0]:
                        reply_policy.policy.sent_single(target_channel.id)
                if thinking:
                    await self.handle_thinking_send(thinking, channel=target_channel)
                self._observe_stage("send", started)
        observe_response_generation(self.message.guild.id, time.monotonic() - start)
