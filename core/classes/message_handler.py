import logging
import asyncio, re, json, time, os, io, base64
import aiohttp
import pillow_heif
from PIL import Image

# Teaches Pillow to open HEIC/HEIF (and AVIF); their output is re-encoded to PNG
# further down (Ollama cannot decode HEIF), so format detection just needs to work.
pillow_heif.register_heif_opener()
from classes.text_llm_handler import TextLLMHandler
from classes.response_filter import (
    chunk_for_discord,
    filter_response as clean_response,
    format_thinking_for_discord,
)
from classes.metrics import observe_response_generation
from classes.message_queue import get_channel_lock, in_flight_hint

logger = logging.getLogger(__name__)

# Max image attachments forwarded to the LLM per message (keeps prompts a sane size).
MAX_IMAGES_PER_MESSAGE = 3

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

class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.text_response = ""
        self.discord_message_object = None


    async def build_messages(self):
       self.history = [message async for message in self.message.channel.history(limit=int(os.environ.get("MSG_HISTORY_LIMIT", 5)))]
       self.history.pop(0) # Remove current message

       formatted_history = []
       # One shared session for every attachment download in this build
       # (across all history messages + the current message), instead of a
       # fresh aiohttp.ClientSession per attachment.
       async with aiohttp.ClientSession() as session:
           for message in self.history:

              if message.content.lower() == "!reset_history":
                  break


              for embed in message.embeds:
                  embed_dict = embed.to_dict()
                  embed_dict.pop('fields', None)
                  content = json.dumps(embed_dict)
                  formatted_history.append({
                      'role': "assistant" if message.author.id == self.client.user.id else "user",
                      'content': f"Discord Embed from '{message.author.name}' converted to JSON: {content}"
                  })

              image_parts = await self.download_image_parts(message.attachments, session=session)
              if len(message.content) == 0 and not image_parts:
                 continue

              text = f"Message from '{message.author.name}': {message.content.replace(f'<@{self.client.user.id}>', '').strip()}"
              # With images the content becomes a list of parts (base64 images + text);
              # the agents SDK forwards them as multimodal chat-completions input.
              formatted_history.append({
                 'role': "assistant" if message.author.id == self.client.user.id else "user",
                 'content': [*image_parts, {"type": "text", "text": text}] if image_parts else text
              })
           formatted_history.reverse()  # Reverse the history to have the oldest message first

           self.message.content = self.clean_message_content(self.message)
           image_parts = await self.download_image_parts(self.message.attachments, session=session)
       user_text = f"Message from '{self.message.author.name}': {self.message.content}"
       formatted_history.append({
            'role': 'user',
            'content': [*image_parts, {"type": "text", "text": user_text}] if image_parts else user_text
       })
       self.messages = formatted_history


    async def download_image_parts(self, attachments, session=None) -> list:
        """Download image attachments and return them as chat-content image parts.

        Each part is {'type': 'input_image', 'image_url': '<base64 data URL>'} which
        the agents SDK converts to the OpenAI 'image_url' wire format for the LLM.
        Non-image attachments and failed downloads are skipped.

        `session` lets a caller (build_messages) share one aiohttp session
        across every attachment of every message in a build instead of
        opening a fresh one per attachment; when omitted, one is opened
        just for this call. Downloads for this call's attachments run
        concurrently."""
        targets = []  # (url, content_type)
        for attachment in attachments or []:
            if len(targets) >= MAX_IMAGES_PER_MESSAGE:
                break
            content_type = (attachment.content_type or "").lower()
            if not content_type.startswith("image/"):
                continue
            url = attachment.proxy_url or attachment.url
            if not url:
                continue
            targets.append((url, content_type))

        if not targets:
            return []

        async def _download(sess, url):
            try:
                async with sess.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Failed to download image {url}: HTTP {resp.status}")
                        return None
                    return await resp.read()
            except Exception as e:
                logger.warning(f"Failed to download image {url}: {e}")
                return None

        if session is not None:
            downloads = await asyncio.gather(*(_download(session, url) for url, _ in targets))
        else:
            async with aiohttp.ClientSession() as own_session:
                downloads = await asyncio.gather(*(_download(own_session, url) for url, _ in targets))

        parts = []
        for (url, content_type), data in zip(targets, downloads):
            if not data:
                continue
            encoded = await asyncio.to_thread(encode_image_for_llm, data)
            if encoded is None:
                logger.warning(f"Skipping image {url}: not a decodable image (format={content_type!r})")
                continue
            image_data, ctype = encoded
            parts.append({
                "type": "input_image",
                "image_url": f"data:{ctype};base64,{base64.b64encode(image_data).decode()}",
            })
        return parts


    def clean_message_content(self, message):
        return message.content.replace(f'<@{self.client.user.id}>', '').strip()
    


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
        async with get_channel_lock(self.message.channel.id):
            # 1) Build the prompt under the lock: the channel.history() read
            #    is a consistent snapshot (no half-sent replies from a
            #    concurrent same-channel handle).
            await self.build_messages()
            # If an earlier message's slow tool is still running (or just
            # finished) in this channel, tell the model — it can then answer
            # follow-ups honestly ("it's still running") instead of guessing
            # that the previous request went unanswered.
            hint = in_flight_hint(self.message.channel.id)
            if hint:
                self.messages.append({"role": "user", "content": hint})
        # 2) LLM run + tool calls UNLOCKED (the slow phase; other messages —
        #    same channel or not — can build/generate concurrently).
        ollama = TextLLMHandler(
            self.messages, self.message.guild.id, self.message,
            client=self.client,
        )
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
            await self.message.add_reaction('❌')
            if thinking:
                async with get_channel_lock(self.message.channel.id):
                    await self.handle_thinking_send(thinking, channel=target_channel)
        else:
            response = self.filter_response(response)
            # 3) Send under the lock: serializes the chunked replies of
            #    concurrent same-channel handles (no interleaved chunks).
            async with get_channel_lock(self.message.channel.id):
                await self.handle_message_send(response, channel=target_channel)
                if thinking:
                    await self.handle_thinking_send(thinking, channel=target_channel)
        observe_response_generation(self.message.guild.id, time.monotonic() - start)
