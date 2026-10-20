"""End-to-end checks: the PR's bot, answering real Discord messages.

e2e/start_stack.sh starts the bot (core) from the checked-out source, with
its voice sidecar and speech service, against a real LLM and image service.
This script then talks to it the way a person would -- through a webhook in
a private test server -- and checks what comes back through Discord's REST
API. It is what lets a dependency bump be merged without trying it by hand.

What is checked is STRUCTURE, never wording: the model may phrase a reply
however it likes, but a reply must arrive, a remembered fact must be in
Redis, a picture must be a PNG attachment, a poll must be a poll, the call
must be joined and spoken in. Each case reads the bot's own Prometheus
metrics to know when the message has been fully handled, and how it ended
(reply_latency is counted once per message, by outcome), so no case waits
on a guess.

Every run gets its own thread, so runs never read each other's replies and
history from an earlier run never reaches the prompt. A failed reply is the
❌ reaction the bot puts on the trigger (message_handler.py).

Environment:
  E2E_DISCORD_TOKEN     the TEST bot's token (the same one core runs with)
  E2E_WEBHOOK_URL       webhook in the test server's text channel; its guild
                        and channel are read from the webhook itself
  E2E_VOICE_CHANNEL_ID  voice channel in the test server; empty skips voice
  E2E_LOG_DIR           where start_stack.sh writes the logs (e2e/logs)
  E2E_METRICS_URL       core's /metrics (default http://127.0.0.1:9464/metrics)
  E2E_SPEECH_URL        the speech service (default http://127.0.0.1:8002)
  E2E_RUN_NAME          name for the run's thread (default: a timestamp)

Exits 1 if any case fails, after writing a table to $GITHUB_STEP_SUMMARY.
"""
import array
import asyncio
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
import redis.asyncio as redis

API = "https://discord.com/api/v10"
WAKE_PHRASE = "hey tester"
# Generous: the LLM is shared with real traffic, and an image is a generation
# plus a vision check of it.
REPLY_TIMEOUT = 300
IMAGE_TIMEOUT = 420
READY_TIMEOUT = 600

LOG_DIR = Path(os.environ.get("E2E_LOG_DIR", Path(__file__).parent / "logs"))
METRICS_URL = os.environ.get("E2E_METRICS_URL", "http://127.0.0.1:9464/metrics")
SPEECH_URL = os.environ.get("E2E_SPEECH_URL", "http://127.0.0.1:8002").rstrip("/")


class CheckFailed(Exception):
    pass


# --------------------------------------------------------------- Discord REST


class Discord:
    """The few REST calls the checks need, with 429 handling."""

    def __init__(self, session: aiohttp.ClientSession, token: str):
        self.session = session
        self.headers = {"Authorization": f"Bot {token}"}

    async def request(self, method: str, url: str, *, auth: bool = True, ok_404: bool = False, **kwargs):
        if not url.startswith("http"):
            url = API + url
        for _ in range(10):
            async with self.session.request(method, url, headers=self.headers if auth else None,
                                            **kwargs) as response:
                if response.status == 429:
                    body = await response.json()
                    await asyncio.sleep(float(body.get("retry_after", 1)) + 0.1)
                    continue
                if response.status == 404 and ok_404:
                    return None
                if response.status >= 400:
                    # Never echo the URL: a webhook URL is its own credential.
                    raise CheckFailed(f"Discord {method} returned {response.status}: "
                                      f"{(await response.text())[:300]}")
                if response.status == 204:
                    return None
                return await response.json()
        raise CheckFailed(f"Discord {method} still rate limited after 10 tries")


# -------------------------------------------------------------------- metrics


_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)')
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


async def read_metrics(session) -> list[tuple[str, dict, float]]:
    async with session.get(METRICS_URL) as response:
        text = await response.text()
    samples = []
    for line in text.splitlines():
        match = _SAMPLE.match(line)
        if match and not line.startswith("#"):
            name, labels, value = match.groups()
            samples.append((name, dict(_LABEL.findall(labels or "")), float(value)))
    return samples


async def metric(session, name: str, **labels) -> float:
    """The sum of every sample of `name` whose labels include `labels`."""
    return sum(value for sample, have, value in await read_metrics(session)
               if sample == name and all(have.get(k) == v for k, v in labels.items()))


async def wait_until(predicate, timeout: float, what: str, interval: float = 2.0):
    """Polls `predicate` (plain or async) until it returns something truthy."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return result
        await asyncio.sleep(interval)
    raise CheckFailed(f"timed out after {timeout:.0f}s waiting for {what}")


def core_log() -> str:
    try:
        return (LOG_DIR / "core.log").read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


# ----------------------------------------------------------------------- run


class Run:
    def __init__(self, session, discord: Discord, webhook_url: str, voice_channel_id: str):
        self.session = session
        self.discord = discord
        self.webhook_url = webhook_url
        self.voice_channel_id = voice_channel_id
        self.results: list[tuple[str, bool, str]] = []
        self.bot_id = ""
        self.guild_id = ""
        self.channel_id = ""
        self.webhook_id = ""
        self.thread_id = ""
        self.redis = redis.Redis(host="127.0.0.1", decode_responses=True)

    async def setup(self) -> None:
        def logged_in() -> bool:
            log = core_log()
            if "Logged in as" in log:
                return True
            if "Traceback (most recent call last)" in log:  # e.g. a rejected token
                raise CheckFailed("core crashed before logging in:\n" + log[-1500:])
            return False
        await wait_until(logged_in, READY_TIMEOUT, "core to log in to Discord")
        me = await self.discord.request("GET", "/users/@me")
        self.bot_id = me["id"]
        hook = await self.discord.request("GET", self.webhook_url, auth=False)
        self.webhook_id, self.guild_id, self.channel_id = hook["id"], hook["guild_id"], hook["channel_id"]
        # A fixed wake phrase, so the voice case does not depend on the
        # test bot's display name.
        await self.redis.set(f"dcb:{self.guild_id}:voice_wake_word", WAKE_PHRASE)
        name = os.environ.get("E2E_RUN_NAME") or time.strftime("%Y%m%d-%H%M%S")
        thread = await self.discord.request("POST", f"/channels/{self.channel_id}/threads", json={
            "name": f"e2e {name}"[:100], "type": 11, "auto_archive_duration": 60})
        self.thread_id = thread["id"]
        print(f"Bot {self.bot_id}; testing in thread {self.thread_id}")

    async def teardown(self) -> None:
        if self.thread_id:
            try:
                await self.discord.request("PATCH", f"/channels/{self.thread_id}", json={"archived": True})
            except CheckFailed as e:
                print(f"Could not archive the thread: {e}")
        await self.redis.aclose()

    async def post(self, content: str) -> dict:
        return await self.discord.request(
            "POST", self.webhook_url, auth=False,
            params={"wait": "true", "thread_id": self.thread_id},
            json={"content": content, "username": "e2e"})

    async def ask(self, content: str, timeout: float = REPLY_TIMEOUT) -> list[dict]:
        """Mention the bot, wait until it has handled the message, and return
        its messages that came after the trigger (oldest first)."""
        # reply_latency is observed once per handled message whatever the
        # outcome ("replied", "llm_error", "exception"), unlike
        # messages_processed, which a raised handler never reaches.
        async def outcomes() -> dict:
            return {have.get("outcome"): value for name, have, value in await read_metrics(self.session)
                    if name == "discord_bot_reply_latency_seconds_count"
                    and have.get("guild_id") == self.guild_id}
        before = await outcomes()
        trigger = await self.post(f"<@{self.bot_id}> {content}")

        async def handled():
            now = await outcomes()
            return {k: v - before.get(k, 0) for k, v in now.items() if v > before.get(k, 0)}
        changed = await wait_until(handled, timeout, "the bot to handle the message")
        if "replied" not in changed:
            raise CheckFailed(f"the reply failed (outcome: {', '.join(changed)}; see core.log)")
        message = await self.discord.request("GET", f"/channels/{self.thread_id}/messages/{trigger['id']}")
        if any(r["emoji"].get("name") == "❌" for r in message.get("reactions", [])):
            raise CheckFailed("the bot marked the message ❌ (the reply failed; see core.log)")
        messages = await self.discord.request(
            "GET", f"/channels/{self.thread_id}/messages", params={"after": trigger["id"], "limit": 50})
        mine = [m for m in messages if m["author"]["id"] == self.bot_id]
        mine.sort(key=lambda m: int(m["id"]))
        if not mine:
            raise CheckFailed("the bot handled the message but posted nothing")
        return mine

    async def case(self, name: str, check) -> None:
        started = time.monotonic()
        try:
            detail = await check() or ""
            self.results.append((name, True, detail))
            print(f"PASS {name} ({time.monotonic() - started:.0f}s) {detail}")
        except Exception as e:  # one broken case must not hide the others
            self.results.append((name, False, str(e)))
            print(f"FAIL {name} ({time.monotonic() - started:.0f}s) {e}")

    # -- cases ----------------------------------------------------------------

    async def chat(self):
        messages = await self.ask("This is an automated test. Reply with one short friendly sentence.")
        text = " ".join(m["content"] for m in messages).strip()
        if not text:
            raise CheckFailed("the reply has no text")
        return _quote(text)

    async def memory(self):
        await self.ask("Please remember this about me for later: my favourite colour is teal.")
        stored = await self.redis.get(f"guild:{self.guild_id}:user:{self.webhook_id}")
        facts = json.loads(stored) if stored else []
        if not any("teal" in str(fact).lower() for fact in facts):
            raise CheckFailed(f"no stored memory mentions teal (stored: {facts})")
        return f"stored: {facts}"

    async def image(self):
        messages = await self.ask("Generate an image of a red apple on a wooden table.", IMAGE_TIMEOUT)
        for message in messages:
            for attachment in message.get("attachments", []):
                if (attachment["filename"] == "generated-image.png"
                        and attachment.get("content_type", "").startswith("image/png")
                        and attachment.get("size", 0) > 0):
                    return f"{attachment['size']} bytes"
        raise CheckFailed("no generated-image.png attachment in the reply")

    async def poll(self):
        messages = await self.ask("Create a poll asking 'Tea or coffee?' with the answers Tea and Coffee.")
        polls = [m["poll"] for m in messages if m.get("poll")]
        if not polls:
            raise CheckFailed("the bot posted no poll")
        answers = [a["poll_media"].get("text", "") for a in polls[0].get("answers", [])]
        if len(answers) < 2:
            raise CheckFailed(f"the poll has fewer than two answers: {answers}")
        return f"{polls[0]['question'].get('text', '')!r} {answers}"

    async def slash_commands(self):
        # From this run's log, not GET /applications/.../commands: the
        # registered set outlives the process, so an earlier run's sync
        # would hide this one failing.
        synced = await wait_until(
            lambda: set(re.findall(r"Command '([^']+)' synced", core_log())),
            120, "the slash commands to sync")
        expected = {"help", "whats_new", "system", "generate_image"}
        if self.voice_channel_id:
            expected.add("voice")
        missing = expected - synced
        if missing:
            raise CheckFailed(f"not synced: {sorted(missing)} (synced: {sorted(synced)})")
        return f"{len(synced)} synced"

    async def speech_roundtrip(self):
        """The speech service on its own: say a sentence, hear it back."""
        async with self.session.post(f"{SPEECH_URL}/speak", json={
                "text": "The quick brown fox jumps over the lazy dog."}) as response:
            if response.status != 200:
                raise CheckFailed(f"/speak returned {response.status}")
            stereo = array.array("h", await response.read())
        # /speak answers 48 kHz stereo; /transcribe takes 48 kHz mono.
        mono = array.array("h", ((stereo[i] + stereo[i + 1]) // 2 for i in range(0, len(stereo) - 1, 2)))
        async with self.session.post(f"{SPEECH_URL}/transcribe", data=mono.tobytes()) as response:
            if response.status != 200:
                raise CheckFailed(f"/transcribe returned {response.status}")
            text = (await response.json()).get("text", "")
        if not {"fox", "dog"} & set(re.findall(r"[a-z]+", text.lower())):
            raise CheckFailed(f"transcript does not match what was said: {text!r}")
        return _quote(text)

    async def voice(self):
        """Join, greet, answer one spoken request, leave -- driven from text
        by VOICE_DEV_INJECT (`!voice_join` / `!voice_say`), so no one has to
        speak. Last: while the bot is in a call it does not answer text."""
        tts = lambda: metric(self.session, "discord_bot_voice_speech_seconds_count", kind="tts")
        active = lambda: metric(self.session, "discord_bot_voice_sessions_active")
        state = lambda: self.discord.request(
            "GET", f"/guilds/{self.guild_id}/voice-states/{self.bot_id}", ok_404=True)
        spoken = await tts()

        await self.post(f"!voice_join {self.voice_channel_id}")
        await wait_until(lambda: _gt(active(), 0), 60, "the call to start")
        await wait_until(lambda: _in_channel(state(), self.voice_channel_id), 60,
                         "Discord to show the bot in the voice channel")
        # The greeting is two lines: the model's hello and the wake hint.
        await wait_until(lambda: _gt(tts(), spoken + 1), 180, "the greeting to be spoken")

        answered = await metric(self.session, "discord_bot_voice_turns_total", outcome="answered")
        spoken = await tts()
        await self.post(f"!voice_say {WAKE_PHRASE}, what is two plus two?")
        await wait_until(lambda: _gt(metric(self.session, "discord_bot_voice_turns_total",
                                            outcome="answered"), answered),
                         REPLY_TIMEOUT, "the spoken request to be answered")
        await wait_until(lambda: _gt(tts(), spoken), 60, "the answer to be spoken")

        await self.post(f"!voice_say {WAKE_PHRASE}, leave")
        await wait_until(lambda: _eq(active(), 0), 60, "the call to end")
        await wait_until(lambda: _not_in_voice(state()), 60, "Discord to show the bot has left")
        if "Voice: speech failed" in core_log():
            raise CheckFailed("core logged 'Voice: speech failed'")
        return f"{await tts():.0f} sentences spoken"

    async def clean_log(self):
        log = core_log()
        if "Traceback (most recent call last)" in log:
            start = log.index("Traceback (most recent call last)")
            raise CheckFailed("core.log has a traceback:\n" + log[max(0, start - 300):start + 1200])
        return ""


async def _gt(value, than) -> bool:
    return (await value) > than


async def _eq(value, to) -> bool:
    return (await value) == to


async def _in_channel(state, channel_id) -> bool:
    state = await state
    return bool(state) and state.get("channel_id") == channel_id


async def _not_in_voice(state) -> bool:
    state = await state
    return not state or not state.get("channel_id")


def _quote(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    return repr(text if len(text) <= limit else text[:limit] + "...")


def write_summary(results) -> None:
    lines = ["## End-to-end checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for name, ok, detail in results:
        cell = detail.replace("|", "\\|").replace("\n", " ")[:300]
        lines.append(f"| {name} | {'✅' if ok else '❌'} | {cell} |")
    lines += ["", "Logs are attached to the run as the `e2e-logs` artifact."]
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


async def main() -> int:
    token = os.environ.get("E2E_DISCORD_TOKEN", "")
    webhook_url = os.environ.get("E2E_WEBHOOK_URL", "")
    voice_channel_id = os.environ.get("E2E_VOICE_CHANNEL_ID", "").strip()
    if not token or not webhook_url:
        print("E2E_DISCORD_TOKEN and E2E_WEBHOOK_URL are required", file=sys.stderr)
        return 1
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        run = Run(session, Discord(session, token), webhook_url, voice_channel_id)
        try:
            await run.setup()
        except Exception as e:
            run.results.append(("setup", False, str(e)))
            print(f"FAIL setup: {e}")
        else:
            await run.case("speech service", run.speech_roundtrip)
            await run.case("slash commands", run.slash_commands)
            await run.case("chat reply", run.chat)
            await run.case("memory tool", run.memory)
            await run.case("image", run.image)
            await run.case("poll", run.poll)
            if voice_channel_id:
                await run.case("voice call", run.voice)
            await run.case("no tracebacks", run.clean_log)
        finally:
            await run.teardown()
    write_summary(run.results)
    return 0 if all(ok for _, ok, _ in run.results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
