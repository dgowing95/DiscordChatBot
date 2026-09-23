import logging
import asyncio
import io

from agents import FunctionTool, function_tool,RunContextWrapper
from classes.common import Common
from classes.content_guard import check_web_request
import discord, aiohttp
from ddgs import DDGS
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

async def add_emoji_to_message(message: discord.Message, emoji: str) -> None:
    try:
        await message.add_reaction(emoji)
        logger.info(f"Added emoji {emoji} to message {message.id}")
    except Exception as e:
        logger.warning(f"Failed to add emoji {emoji} to message {message.id}: {e}")
    
@function_tool
async def web_search(wrapper: RunContextWrapper[dict], search_request: str) -> str:
    """Searches the internet for a given query.

    Args:
        search_request: The query to search for.
    """
    logger.info(f"Searching the web for: {search_request}")

    allowed, reason = await check_web_request(search_request)
    if not allowed:
        logger.warning(f"Web search blocked by content guard: {reason}")
        return ("I can't perform that search — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful query.")

    await Common.send_tool_discord_embed(
        wrapper.context.get("original_message").channel,
        f"Searching the web for: {search_request}",
    )
    try:
        results = await asyncio.to_thread(DDGS().text, search_request, max_results=5)
    except Exception as e:
        logger.warning(f"An error occurred while searching: {e}")
        return "Error fetching search results."
    return results
    
@function_tool
async def fetch_url(wrapper: RunContextWrapper[dict], url: str) -> str:
    """Fetches the content of a URL. Returns the text content of the page.

    Args:
        url: The URL to fetch.
    """
    logger.info(f"Fetching content from URL: {url}")

    allowed, reason = await check_web_request(url)
    if not allowed:
        logger.warning(f"URL fetch blocked by content guard: {reason}")
        return ("I can't fetch that URL — it was blocked by the safety "
                "guard.")

    await Common.send_tool_discord_embed(
        wrapper.context.get("original_message").channel,
        f"Visiting URL: {url}",
    )
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": "dis-ai-bot"}) as response:
                html = await response.text()
    except Exception as e:
        logger.warning(f"An error occurred while fetching the URL: {e}")
        return "Error fetching URL content."

    text = await asyncio.to_thread(_extract_page_text, html)
    logger.info(f"Fetched content from {url} successfully.")
    return text


def _extract_page_text(html: str) -> str:
    """CPU-bound HTML->text extraction, run off the event loop via asyncio.to_thread."""
    soup = BeautifulSoup(html, features='html.parser')
    for script in soup(["script", "style"]):
        script.extract()  # remove all javascript and stylesheet code

    text = soup.body.get_text()
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return '\n'.join(chunk for chunk in chunks if chunk)

async def get_current_datetime() -> str:
    """Returns the current date and time."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    now = datetime.now(ZoneInfo("Europe/London"))
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Current date and time: {now_formatted}")
    return now_formatted

@function_tool
async def store_memory(wrapper: RunContextWrapper[dict], data: str) -> str:
    """Stores a lasting fact about the user — a preference, a personal
    detail, anything worth recalling in a later conversation. Not for
    one-off context that only matters in this thread.

    Args:
        data: The fact to store, e.g. "prefers metric units".
    """

    # The model sometimes repeats a tool call within one reply; only the first
    # is performed. Say so honestly rather than reporting a success that did not
    # happen - the model reads this back when it writes the reply.
    times_called = wrapper.context.get("redis_save_tool_calls")
    if times_called > 0:
        logger.warning(f"store_memory already called {times_called} time(s) this run; skipping.")
        return ("A memory was already stored for this request, so this duplicate "
                "call was skipped. Nothing further is needed.")
    wrapper.context["redis_save_tool_calls"] += 1
    

    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")

    response_message = ""
    try:
        logger.info(f"Storing data for user {user_id} in guild {guild_id}: {data}")
        user_memory = UserMemory(user_id, guild_id)
        await user_memory.append(data)
        await add_emoji_to_message(wrapper.context.get("original_message"), "💾")
        await Common.send_tool_discord_embed(
            wrapper.context.get("original_message").channel,
            f"Stored data: {data}",
        )
        
        response_message = "Data stored successfully."
    except Exception as e:
        response_message = f"An error occurred while storing user data: {e}"
        logger.info(response_message)

    return response_message

@function_tool
async def remove_memory(wrapper: RunContextWrapper[dict], data: str) -> str:
    """Removes one stored memory, matched on its exact stored text.

    Args:
        data: The specific memory to remove.
    """
    
    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")
    try:
        user_memory = UserMemory(user_id, guild_id)
        removed = await user_memory.remove(data)
        if removed:
            await add_emoji_to_message(wrapper.context.get("original_message"), "🗑️")
            return f"Removed memory: {data}"
        else:
            return "Memory not found."
    except Exception as e:
        logger.warning(f"An error occurred while removing user memory: {e}")
        return "Error removing user memory."

@function_tool
async def clear_memories(wrapper: RunContextWrapper[dict]) -> str:
    """Deletes EVERY stored memory for this user. Only when they ask to be
    forgotten — it cannot be undone. To drop just one, use remove_memory."""
    
    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")
    try:
        user_memory = UserMemory(user_id, guild_id)
        await user_memory.clear()
        await add_emoji_to_message(wrapper.context.get("original_message"), "🧹")
        return "All memories cleared."
    except Exception as e:
        logger.warning(f"An error occurred while clearing user memories: {e}")
        return "Error clearing user memories."


@function_tool
async def generate_image(wrapper: RunContextWrapper[dict], prompt: str) -> str:
    """Generates an image from a text description and sends it to the channel.
    Use it when the user asks for art, illustrations, pictures or drawings.
    The image is sent automatically; never try to send it yourself.
    Args:
        prompt: A plain-language description of the picture you want, in
            ordinary sentences. Include everything that matters: the main
            subject and what it is doing, where it is, how it is positioned,
            what else is in shot, the lighting, the mood, and the style or
            medium. Say what the user implied but did not spell out. A
            separate step rewrites this into the image model's own prompt
            format, so do NOT write comma-separated tag lists, quality
            boilerplate like "high resolution", or weighting syntax - and do
            not leave detail out to keep it short. Anything that must NOT
            appear can simply be written as a normal sentence ("no people in
            the shot"); it is moved to a negative prompt for you.
    """
    from classes.image_generation import generate_image_from_api
    from classes.image_prompt import build_image_prompt

    message = wrapper.context.get("original_message")
    logger.info(f"Generating image for prompt: {prompt}")
    await add_emoji_to_message(message, "🎨")
    # The rewritten prompt, not the requested one, is what the embed shows:
    # what the image model was actually given is the thing worth seeing when
    # the result does not match what was asked for.
    image_prompt, negative_prompt = await build_image_prompt(prompt)
    await Common.send_tool_discord_embed(
        message.channel,
        f"Generating image: {image_prompt}",
    )
    try:
        image_bytes = await generate_image_from_api(image_prompt, negative_prompt)
    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
        return ("Image generation failed. Tell the user the image service is "
                "unavailable right now and do not retry.")
    try:
        await message.channel.send(
            file=discord.File(io.BytesIO(image_bytes), filename="generated-image.png")
        )
    except Exception as e:
        logger.warning(f"Image generated but failed to send to Discord: {e}")
        return "The image was generated but could not be sent to the channel."
    return ("Image generated and sent to the channel. The user can already see "
            "it; do not send the image again or describe it as if pending.")


async def _send_sandbox_closing_note(
    channel, snapshot_id, in_thread: bool, outcome: str = "",
) -> int | None:
    """Posts the sandbox's closing embed: how the run ended, and (in a thread)
    how much longer it can be resumed from here.

    Returns the remaining resume window in seconds, or None when there is no
    resumable workspace — the caller uses it to keep what it tells the outer
    model in step with what this embed just told the user, so the two can
    never disagree about whether a follow-up here picks the work back up.

    Without it the only signal that a run has ended is the ABSENCE of a
    delivery reaction on your next message, which you cannot see until after
    you've sent one — so people can't tell whether they're steering a live
    sandbox or talking to the bot again.

    Exception-safe and posted on every path where a "Running in sandbox"
    embed already went out: a failure to post this must never change what
    the tool returns to the model.
    """
    from classes.sandbox_agent import (
        sandbox_closing_note,
        sandbox_snapshot_remaining_seconds,
    )

    remaining = None
    try:
        remaining = await sandbox_snapshot_remaining_seconds(snapshot_id)
        await Common.send_tool_discord_embed(
            channel,
            sandbox_closing_note(remaining, in_thread, outcome),
            color=0x99AAB5,  # muted grey: this run is over, unlike the cyan "running" embed
            title="Sandbox closed",
        )
    except Exception as e:
        logger.warning(f"Sandbox: failed to post the closing note: {e}")
    return remaining


@function_tool
async def run_code_sandbox(wrapper: RunContextWrapper[dict], task: str) -> str:
    """Hands a request to a code-sandbox agent: a Linux container (Python +
    shell) with its own model, which writes and actually runs code. Use it
    when the answer depends on running something rather than reasoning about
    it — writing or debugging a program, computing a value, processing data,
    generating or converting files. Not for questions you can answer
    directly.

    The sandbox agent designs it, not you — it can run code and look at the
    output, you can't. Pass the user's request in their own words, plus only
    what the sandbox cannot see for itself: results or filenames from earlier
    runs in this thread, attachment contents, and constraints the user
    actually stated. Nothing else. For "can you make me a gif of a cow doing
    a backflip", the whole task is `Generate a gif of a cow doing a backflip.`

    Usually runs in a Discord thread and sends any files it produces there
    itself; a call from inside an existing sandbox thread resumes that
    thread's workspace, so follow-ups build on earlier work.

    This call is fully synchronous: by the time it returns, the sandbox run
    is already completely finished (whether it succeeded, partially
    succeeded, or failed) and any file it produced has already been sent to
    the thread — there is no follow-up after this. Never tell the user to
    wait, that a file is still being generated, or that it will "pop up" or
    "arrive shortly".

    Args:
        task: The user's request in their own words, plus context from this
            conversation the sandbox cannot see. Short, and with no
            implementation details of your own.
    """
    logger.info(f"Running sandbox task: {task}")

    allowed, reason = await check_web_request(task)
    if not allowed:
        logger.warning(f"Sandbox task blocked by content guard: {reason}")
        return ("I can't run that in the sandbox — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful task.")

    from classes.sandbox_agent import ensure_sandbox_thread
    from classes import sandbox_thread_inbox

    original_message = wrapper.context.get("original_message")
    requesting_user_id = wrapper.context.get("user_id")
    # A sandbox thread this outer turn already resolved. The context dict is
    # one object for the whole Runner.run (text_llm_handler builds it once),
    # so a second call in the same turn — the outer model retrying a run that
    # was stopped — finds the thread the first one opened. Without this it
    # would ask ensure_sandbox_thread for a thread off a message that already
    # has one, be refused by Discord, and run in the PARENT CHANNEL instead:
    # away from the work it meant to continue, and unsnapshotted, so the
    # partial workspace the stopped run saved is silently abandoned.
    previous = wrapper.context.get("sandbox_thread")
    if isinstance(previous, discord.Thread):
        channel, thread_created = previous, False
    else:
        channel, thread_created = await ensure_sandbox_thread(original_message, task)
    # The outer agent's own final reply is sent by MessageHandler after this
    # tool returns, to self.message.channel by default — which would post it
    # outside the thread the sandbox's output actually lives in. Recording
    # the resolved channel back onto the shared run context (read by
    # TextLLMHandler.generate() after Runner.run returns) lets the caller
    # redirect that final reply into the same thread.
    wrapper.context["sandbox_thread"] = channel
    in_thread = isinstance(channel, discord.Thread)
    # Claimed immediately after the thread is resolved and BEFORE any other
    # await: the old check-then-register had several awaits in between
    # (snapshot lookup, config read, embeds), so two runs could both pass the
    # check. Two paths reach this with a run already claimed: queue lag (two
    # mentions land before either worker starts) and the outer model emitting
    # two run_code_sandbox calls in a single turn. Either way two containers
    # would race to persist to dcb:sandbox_snapshot:{thread_id} on teardown
    # and the last to finish would clobber the other — so the task is
    # forwarded into the run already in flight instead of starting a second.
    ledger = None
    if in_thread:
        ledger = sandbox_thread_inbox.claim_run(channel.id, requesting_user_id)
        if ledger is None:
            return _forward_to_running_sandbox(channel, original_message, task)
    try:
        return await _run_claimed_sandbox(
            wrapper, task, channel, thread_created, in_thread, ledger)
    finally:
        # Held through artifact delivery and the closing note, not just the
        # run: the claim used to end before delivery, so a message posted
        # while files were still uploading went to the outer LLM instead and
        # was answered out of context. release_run only drops THIS ledger,
        # so a stale caller cannot end somebody else's run.
        if ledger is not None:
            from classes.sandbox_conversation import record_exit_metrics
            record_exit_metrics(ledger)
            sandbox_thread_inbox.release_run(channel.id, ledger)


def _forward_to_running_sandbox(channel, original_message, task: str) -> str:
    """Hands a second run_code_sandbox call to the run already claimed in
    this thread, and tells the outer model honestly whether that worked."""
    from classes import sandbox_thread_inbox

    author = getattr(original_message, "author", None)
    outcome = sandbox_thread_inbox.deliver(
        channel.id,
        getattr(original_message, "id", None),
        getattr(author, "id", None),
        getattr(author, "display_name", "the user"),
        task,
    )
    if outcome in (sandbox_thread_inbox.ACCEPTED, sandbox_thread_inbox.DUPLICATE):
        return ("A sandbox is already running in this thread, so this request was "
                "handed to the run in progress instead of starting a second one. "
                "Tell the user it was passed along to the sandbox that's already "
                "working, and do not retry.")
    if outcome == sandbox_thread_inbox.FINISHING:
        return ("A sandbox run in this thread is finishing, so this request could not "
                "be applied to it; it has been noted for the next run here. Tell the "
                "user to @mention you in this thread once it has closed to carry on "
                "with it, and do not retry now.")
    return ("A sandbox is already running in this thread and could not take this "
            "request (it is too long, or the run already has too many unread "
            "messages). Tell the user to ask again in the thread once it has "
            "finished, and do not retry now.")


async def _run_claimed_sandbox(wrapper, task, channel, thread_created, in_thread, ledger) -> str:
    """run_code_sandbox after the thread has been resolved and (in a thread)
    claimed: announce, run, deliver, close. Split out only so the claim's
    release can sit in one `finally` around all of it."""
    from classes.config_manager import configManager
    from classes.sandbox_agent import (
        run_sandbox_task,
        sandbox_snapshot_exists,
        sandbox_snapshot_id_for,
        sandbox_outcome_note,
        sandbox_tool_result,
        sandbox_unresolved_note,
        sandbox_workspace_note,
    )
    from classes.sandbox_conversation_store import SandboxConversationStore, resume_preamble
    from classes.sandbox_progress import (
        DESCRIPTION_CHARS,
        SandboxProgressHooks,
        sandbox_progress_updates_enabled,
    )

    original_message = wrapper.context.get("original_message")
    discord_client = wrapper.context.get("discord_client")
    requesting_user_id = wrapper.context.get("user_id")
    if thread_created:
        try:
            await Common.send_tool_discord_embed(
                original_message.channel,
                f"🧵 Started a sandbox thread: {channel.mention}",
            )
        except Exception as e:
            logger.warning(f"Sandbox: failed to notify the original channel of the new thread: {e}")
    # Whether this run will resume the thread's saved workspace or start
    # empty. Asked before the run so the embed below can say which — the SDK
    # exposes no way to find out afterwards (see sandbox_snapshot_exists).
    snapshot_id = sandbox_snapshot_id_for(channel)
    resumed = await sandbox_snapshot_exists(snapshot_id)
    workspace_note = sandbox_workspace_note(resumed)
    # Loose ends the previous run in this thread left behind (messages that
    # arrived too late, an unanswered question). Offered to this run as
    # context — a follow-up in the thread IS the explicit resume — and
    # replaced by this run's own record at the end. Independent of the
    # workspace snapshot: either can be missing without the other.
    run_task = task
    if snapshot_id is not None:
        try:
            run_task = resume_preamble(await SandboxConversationStore().load(snapshot_id)) + task
        except Exception as e:
            logger.warning(f"Sandbox: could not load this thread's conversation record: {e}")

    progress = None
    # Per-guild toggle (/sandbox_progress_updates, default off). A config
    # read failure falls back to off — progress is a nice-to-have, not a
    # dependency of the run itself.
    try:
        raw = await configManager().get_setting(
            "sandbox_progress_updates", wrapper.context.get("guild_id"))
    except Exception as e:
        logger.warning(f"Could not read sandbox_progress_updates setting: {e}")
        raw = False
    if sandbox_progress_updates_enabled(raw):
        # Live progress: one Discord message, edited in place, showing each
        # command the sandbox runs and its output (throttled for Discord's
        # 5-edits/minute limit). start() is exception-safe (swallows send
        # failures), so a progress problem never blocks the run.
        progress = SandboxProgressHooks(channel, task, workspace_note=workspace_note)
        await progress.start()
    else:
        # The task is capped here the same way the live-progress embed caps
        # it, so a verbose task degrades to a truncated line rather than a
        # screenful of embed.
        shown = task if len(task) <= DESCRIPTION_CHARS else task[: DESCRIPTION_CHARS - 1] + "…"
        await Common.send_tool_discord_embed(
            channel,
            f"{workspace_note}\nRunning in sandbox: {shown}",
        )
    # Sent whatever the progress setting: conversational replies are not
    # part of the optional progress embed, so a user needs to know they can
    # talk to the run either way. Only in a real thread, where a run is
    # claimed and main.py routes messages to it; sent in the parent channel
    # (the thread-creation-failed fallback) it would promise a conversation
    # nothing would pick up.
    if in_thread:
        await Common.send_tool_discord_embed(
            channel,
            "📨 Reply here while it runs — no @mention needed — to change the task "
            "or ask a question. It will answer before its next step.",
            0xB0F400,
            "Thread Linked to Sandbox",
        )

    try:
        result = await run_sandbox_task(
            run_task,
            progress,
            thread=channel,
            client=discord_client,
            requesting_user_id=requesting_user_id,
            resumed=resumed,
            ledger=ledger,
        )
    except Exception as e:
        logger.warning(f"Sandbox task failed: {e}")
        if progress is not None:
            await progress.finalize("❌ Stopped: the sandbox itself failed.")
        # The "Running in sandbox" embed already went out, so this path needs
        # a closing note too or the thread is left looking mid-run forever.
        outcomes = ledger.outcomes() if ledger is not None else []
        await _send_sandbox_closing_note(
            channel, snapshot_id, in_thread,
            "❌ Stopped: the sandbox itself failed." + sandbox_unresolved_note(outcomes, in_thread))
        await _save_conversation_record(snapshot_id, outcomes, "infra_error", ledger)
        return ("The sandbox task failed (the code sandbox may be unavailable). "
                "Tell the user the sandbox is not working right now and do "
                "not retry the same task.")

    # Ground truth from the run itself. It only disagrees with the badge
    # already posted above when the saved workspace turned out not to be
    # restorable and was dropped (see _create_sandbox_session), so correct
    # the record rather than leaving a wrong "Resumed" standing.
    resume_correction = ""
    if resumed and not result.resumed:
        resume_correction = (
            "\n\n(This thread's saved workspace could not be restored, so the "
            "sandbox started empty despite what the status message said. Tell "
            "the user their earlier work in this thread was lost; this run's "
            "work has been saved, so further follow-ups will resume normally.)"
        )
        try:
            await channel.send(
                "⚠️ This thread's saved workspace couldn't be restored, so the "
                "sandbox started fresh. It's been reset — the next run in this "
                "thread will pick up from this one."
            )
        except Exception as e:
            logger.warning(f"Sandbox: failed to post the resume correction: {e}")

    if progress is not None:
        # give the live message its final state (it would otherwise sit on
        # the last "still running" / thinking snapshot)
        note = "✅ Done." if result.ok else sandbox_outcome_note(result.error)
        await progress.finalize(note)

    # Acknowledgements the sandbox wrote while Discord was refusing sends.
    # Posted now, before the result, so the thread reads in order.
    for reply in getattr(result, "unsent_replies", None) or []:
        try:
            await channel.send(reply[:1900])
        except Exception as e:
            logger.warning(f"Sandbox: failed to post a delayed reply: {e}")

    # The sandbox agent's own closing message, written FOR the user (see
    # SANDBOX_INSTRUCTIONS' final bullet). It rides on the first file so the
    # user gets one message — summary plus image — the way a preview already
    # arrives, and because the agent is the only party that knows what it
    # actually did: the outer model, which used to be the sole author of this
    # message, cannot see the run. Clamped to Discord's per-message limit
    # rather than chunked — this is a short note beside a file, and a report
    # long enough to need chunking would bury the file it belongs to.
    lead = result.text.strip()[:1900] if (result.ok and result.text) else ""
    sent_names = []
    for artifact in result.artifacts:
        content = lead or (artifact.caption.strip()[:1900] or None)
        try:
            await channel.send(
                content=content,
                file=discord.File(io.BytesIO(artifact.data), filename=artifact.name),
            )
            logger.info(f"Sandbox artifact sent to channel: {artifact.name} ({len(artifact.data)} bytes)")
            sent_names.append(artifact.name)
            # Only now: a failed send carried the message with it, so keeping
            # `lead` set is what lets the fallback below still deliver it.
            lead = ""
        except Exception as e:
            logger.warning(f"Sandbox artifact {artifact.name} generated but failed to send: {e}")
    if lead:
        # No files, or every send failed: the message still has to reach the
        # user, or the run's own account of itself is lost and only the outer
        # model's second-hand version survives.
        try:
            await channel.send(lead)
        except Exception as e:
            logger.warning(f"Sandbox: failed to post the agent's closing message: {e}")

    # What happened to each thread message. Read from the live ledger rather
    # than the result, which was built before delivery: anything posted while
    # the files were uploading was filed as a follow-up in the meantime.
    outcomes = ledger.outcomes() if ledger is not None else list(getattr(result, "conversation", None) or [])

    # After the artifacts, before the return: the thread's "here is how the run
    # ended, and how long you can pick it back up" marker. The outcome line only
    # goes out when the run did NOT finish normally. Posted on every path where
    # a "Running in sandbox" embed already went out, and never on the early
    # returns (content guard, forwarded-to-a-running-run) where nothing opened.
    outcome = "" if result.ok else sandbox_outcome_note(result.error)
    remaining = await _send_sandbox_closing_note(
        channel, snapshot_id, in_thread, outcome + sandbox_unresolved_note(outcomes, in_thread))
    await _save_conversation_record(
        snapshot_id, outcomes, "ok" if result.ok else (result.error or "failed"), ledger)

    # Everything the outer model is told is built by sandbox_tool_result
    # (sandbox_agent.py) — pure, and unit-tested without a Discord channel.
    # `resumable` comes from the closing embed's own live-TTL answer, so the
    # model and the user can never disagree about whether a follow-up in this
    # thread picks the work back up.
    return sandbox_tool_result(
        result,
        sent_names=sent_names,
        in_thread=in_thread,
        resumable=remaining is not None,
        steering=outcomes,
        resume_correction=resume_correction,
    )


_CONVERSATION_SAVE_ATTEMPTS = 5


async def _save_conversation_record(snapshot_id, outcomes, outcome: str, ledger) -> None:
    """Persists this run's loose ends for the next run in the thread (see
    classes/sandbox_conversation_store.py). Best-effort and independent of
    the workspace snapshot; a failure is logged, never raised."""
    if snapshot_id is None:
        return
    from classes.sandbox_conversation_store import SandboxConversationStore, build_record

    def _current():
        if ledger is None:
            return outcomes, None
        return ledger.outcomes(), (ledger.question.text if ledger.question is not None else None)

    # Read at save time, and again after each save: the claim is held through
    # the closing note, so a message can land in ledger.follow_ups after the
    # caller took its snapshot OR while the save itself is awaiting Redis —
    # and it was promised it would be kept. Follow-ups are capped
    # (MAX_FOLLOW_UPS), so this settles; the attempt cap is only a backstop.
    saved = None
    try:
        for _ in range(_CONVERSATION_SAVE_ATTEMPTS):
            current = _current()
            if current == saved:
                break
            rows, question = current
            await SandboxConversationStore().save(
                snapshot_id, build_record(rows, outcome=outcome, open_question=question))
            saved = current
    except Exception as e:
        logger.warning(f"Sandbox: could not save this thread's conversation record: {e}")


@function_tool
async def change_personality(wrapper: RunContextWrapper[dict], personality: str) -> bool:
    """Changes the personality of the bot. Returns True if successful, False otherwise.
    
    Args:
        personality: The new personality to set.
    """

    # As in store_memory: the model sometimes repeats a call within one reply.
    times_called = wrapper.context.get("personality_tool_calls")
    if times_called > 0:
        logger.warning(f"change_personality already called {times_called} time(s) this run; skipping.")
        return True
    wrapper.context["personality_tool_calls"] += 1

    from classes.config_manager import configManager
    logger.info(f"Changing personality to: {personality}")
    try:
        configmanager = configManager()
        await configmanager.update_setting("system", personality, wrapper.context.get("guild_id"))
        logger.info(f"Changed personality to: {personality}")

        embed = discord.Embed(title="Personality Updated",
                      description=personality)
        await wrapper.context.get("original_message").channel.send(embed=embed)
        return True
    except Exception as e:
        logger.warning(f"An error occurred while changing personality: {e}")
        return False
