import random, asyncio, re, json, time, os, io, base64
import aiohttp
import pillow_heif
from PIL import Image

# Teaches Pillow to open HEIC/HEIF (and AVIF); their output is re-encoded to PNG
# further down (Ollama cannot decode HEIF), so format detection just needs to work.
pillow_heif.register_heif_opener()
from classes.text_llm_handler import TextLLMHandler
from classes.config_manager import configManager
from classes.response_filter import filter_response as clean_response
from classes.metrics import observe_response_generation

# Max image attachments forwarded to the LLM per message (keeps prompts a sane size).
MAX_IMAGES_PER_MESSAGE = 3

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
        out = io.BytesIO()
        img.save(out, format=fmt)
        return out.getvalue(), _OLLAMA_FRIENDLY[fmt]
    out = io.BytesIO()
    img.convert("RGBA").save(out, format="PNG")
    return out.getvalue(), "image/png"

class MessageHandler:

    def __init__(self, message, client):
        self.message = message
        self.client = client
        self.text_response = ""
        self.discord_message_object = None
        self.config = configManager()


    async def build_messages(self):
       self.history = [message async for message in self.message.channel.history(limit=int(os.environ.get("MSG_HISTORY_LIMIT", 5)))]
       self.history.pop(0) # Remove current message

       formatted_history = []
       self.attachment_refs = []
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
            
          image_parts = await self.download_image_parts(message.attachments)
          if len(message.content) == 0 and not image_parts:
             continue        
          
          text = f"Message from '{message.author.name}': {message.content.replace(f'<@{self.client.user.id}>', '').strip()}"
          # With images the content becomes a list of parts (base64 images + text);
          # the agents SDK forwards them as multimodal chat-completions input.
          text = self._with_image_labels(text, message, message.attachments)
          formatted_history.append({
             'role': "assistant" if message.author.id == self.client.user.id else "user",
             'content': [*image_parts, {"type": "text", "text": text}] if image_parts else text
          })
       formatted_history.reverse()  # Reverse the history to have the oldest message first
       self.attachment_refs.reverse()  # keep ref order aligned with the history

       self.message.content = self.clean_message_content(self.message)
       image_parts = await self.download_image_parts(self.message.attachments)
       user_text = f"Message from '{self.message.author.name}': {self.message.content}"
       user_text = self._with_image_labels(user_text, self.message, self.message.attachments)
       formatted_history.append({
            'role': 'user',
            'content': [*image_parts, {"type": "text", "text": user_text}] if image_parts else user_text
       })
       self.messages = formatted_history


    def _collect_attachment_refs(self, message, attachments) -> list:
        """Register this message's image attachments for the edit_image tool.

        Returns the short labels assigned (e.g. ['[1]', '[2]']). The signed
        CDN URLs are deliberately never shown to the LLM — long hex
        signatures get corrupted when the model copies them into tool args,
        which 404s on fetch. The model references an image by its short
        label; the edit_image tool resolves the label to the real URL via
        the attachment_refs context (same cap/filter/order as
        download_image_parts, so labels match the images the LLM sees)."""
        labels = []
        for attachment in attachments or []:
            if len(labels) >= MAX_IMAGES_PER_MESSAGE:
                break
            content_type = (attachment.content_type or "").lower()
            if not content_type.startswith("image/"):
                continue
            url = attachment.proxy_url or attachment.url
            if not url:
                continue
            self.attachment_refs.append({
                "ref": str(len(self.attachment_refs) + 1),
                "author": message.author.name,
                "filename": attachment.filename or "image",
                "url": url,
            })
            labels.append(f"[{self.attachment_refs[-1]['ref']}]")
        return labels

    def _with_image_labels(self, text: str, message, attachments) -> str:
        labels = self._collect_attachment_refs(message, attachments)
        if not labels:
            return text
        entries = self.attachment_refs[-len(labels):]
        listing = ", ".join(
            f"{lab} {e['filename']} (from {e['author']})"
            for lab, e in zip(labels, entries)
        )
        return (f"{text}\nAttached images — to modify one, call edit_image with "
                f"its label as image_ref (e.g. image_ref=\"1\"): {listing}")

    async def download_image_parts(self, attachments) -> list:
        """Download image attachments and return them as chat-content image parts.

        Each part is {'type': 'input_image', 'image_url': '<base64 data URL>'} which
        the agents SDK converts to the OpenAI 'image_url' wire format for the LLM.
        Non-image attachments and failed downloads are skipped."""
        parts = []
        for attachment in attachments or []:
            if len(parts) >= MAX_IMAGES_PER_MESSAGE:
                break
            content_type = (attachment.content_type or "").lower()
            if not content_type.startswith("image/"):
                continue
            url = attachment.proxy_url or attachment.url
            if not url:
                continue
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            print(f"Failed to download image {url}: HTTP {resp.status}")
                            continue
                        data = await resp.read()
            except Exception as e:
                print(f"Failed to download image {url}: {e}")
                continue
            if not data:
                continue
            encoded = encode_image_for_llm(data)
            if encoded is None:
                print(f"Skipping image {url}: not a decodable image (format={content_type!r})")
                continue
            image_data, ctype = encoded
            parts.append({
                "type": "input_image",
                "image_url": f"data:{ctype};base64,{base64.b64encode(image_data).decode()}",
            })
        return parts


    def should_process_message(self):
        if len(self.message.content) == 0 and len(self.message.embeds) == 0 and not self.message.attachments:
            return False
        if self.message.author == self.client.user:
            return False
        if self.message.content.lower() == "!reset_history":
            return False
        if self.client.user in self.message.mentions:
            return True
        return self.random_chance_reply()
    
    def random_chance_reply(self):
        guild_id = self.message.guild.id if self.message.guild else 0
        chance = self.config.get_setting("response_chance", guild_id) or 5
        try:
            chance = min(max(float(chance), 0), 50)
        except (ValueError, TypeError):
            chance = 5
        return random.uniform(0, 100) < chance
    
    
    def clean_message_content(self, message):
        return message.content.replace(f'<@{self.client.user.id}>', '').strip()
    


    def filter_response(self, text_response):
        # Filter logic lives in classes.response_filter (pure, unit-tested);
        # we only inject the guild-specific bits (bot mention id).
        return clean_response(text_response, mention=str(self.client.user.id))


    async def handle_message_send(self, message_content):
        from textwrap import wrap
        chunks = wrap(message_content, 2000, break_long_words=False, replace_whitespace=False)
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            await self.message.channel.send(chunk)
            await asyncio.sleep(1)
        


    async def handle_message(self):
        print(f'Handling message: {self.message.content}')
        # Histogram covers the whole handling: prompt build + LLM run + send.
        # Observed on both outcomes (the ❌ path is still a timed attempt;
        # its failures are separately counted in llm_errors).
        start = time.monotonic()
        await self.build_messages()
        ollama = TextLLMHandler(self.messages, self.message.guild.id, self.message, self.attachment_refs)
        response = await ollama.generate()

        if response == "Error":
            await self.message.add_reaction('❌')
        else:
            response = self.filter_response(response)
            await self.handle_message_send(response)
        observe_response_generation(self.message.guild.id, time.monotonic() - start)
