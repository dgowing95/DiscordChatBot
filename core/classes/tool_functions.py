import asyncio
import io

from agents import FunctionTool, function_tool,RunContextWrapper
from classes.common import Common
from classes.content_guard import check_web_request
import discord, aiohttp
from ddgs import DDGS
from bs4 import BeautifulSoup

async def add_emoji_to_message(message: discord.Message, emoji: str) -> None:
    try:
        await message.add_reaction(emoji)
        print(f"Added emoji {emoji} to message {message.id}")
    except Exception as e:
        print(f"Failed to add emoji {emoji} to message {message.id}: {e}")
    
@function_tool
async def web_search(wrapper: RunContextWrapper[dict], search_request: str) -> str:
    """Searches the internet for a given query.

    Args:
        search_request: The query to search for.
    """
    print(f"Searching the web for: {search_request}")

    allowed, reason = await check_web_request(search_request)
    if not allowed:
        print(f"Web search blocked by content guard: {reason}")
        return ("I can't perform that search — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful query.")

    await Common.send_tool_discord_embed(
        wrapper.context.get("original_message").channel,
        f"Searching the web for: {search_request}",
    )
    try:
        results = await asyncio.to_thread(DDGS().text, search_request, max_results=5)
    except Exception as e:
        print(f"An error occurred while searching: {e}")
        return "Error fetching search results."
    return results
    
@function_tool
async def fetch_url(wrapper: RunContextWrapper[dict], url: str) -> str:
    """Fetches the content of a URL. Returns the text content of the page.

    Args:
        url: The URL to fetch.
    """
    print(f"Fetching content from URL: {url}")

    allowed, reason = await check_web_request(url)
    if not allowed:
        print(f"URL fetch blocked by content guard: {reason}")
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
        print(f"An error occurred while fetching the URL: {e}")
        return "Error fetching URL content."

    text = await asyncio.to_thread(_extract_page_text, html)
    print(f"Fetched content from {url} successfully.")
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
    print(f"Current date and time: {now_formatted}")
    return now_formatted

@function_tool
async def store_memory(wrapper: RunContextWrapper[dict], data: str) -> str:
    """Stores a lasting fact about the user — a preference, a personal
    detail, anything worth recalling in a later conversation. Not for
    one-off context that only matters in this thread.

    Args:
        data: The fact to store, e.g. "prefers metric units".
    """

    # Sometimes OpenAI repeats a tool call.
    times_called = wrapper.context.get("redis_save_tool_calls")
    if times_called > 0:
        err = f"Tool call limit reached: {times_called}. Not storing data."
        print(err)
        return "Data stored successfully."
    wrapper.context["redis_save_tool_calls"] += 1
    

    from classes.user_memory import UserMemory
    user_id = wrapper.context.get("user_id")
    guild_id = wrapper.context.get("guild_id")

    response_message = ""
    try:
        print(f"Storing data for user {user_id} in guild {guild_id}: {data}")
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
        print(response_message)

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
        print(f"An error occurred while removing user memory: {e}")
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
        print(f"An error occurred while clearing user memories: {e}")
        return "Error clearing user memories."


@function_tool
async def generate_image(wrapper: RunContextWrapper[dict], prompt: str) -> str:
    """Generates an image from a text description and sends it to the channel.
    Use it when the user asks for art, illustrations, pictures or drawings.
    The image is sent automatically; never try to send it yourself.
    Args:
        prompt: The text description of the image to generate. The image
            model (Juggernaut XI) responds best to precise, specific
            prompts: put the main subject in the first sentence, then add
            the setting, the subject's action, secondary objects, colors,
            lighting, style/medium, mood and camera angle; name textures
            and materials explicitly (e.g. "coarse fur", "polished
            steel"); for people, describe their clothing and emphasize
            the emotion they should show; keep it tight - a couple of
            short sentences with no filler, since long prompts reduce
            adherence; append "high resolution"; for any text that must
            appear in the image, use a short phrase in quotes near the
            start (long text is often misspelled).
    """
    from classes.image_generation import generate_image_from_api

    message = wrapper.context.get("original_message")
    print(f"Generating image for prompt: {prompt}")
    await add_emoji_to_message(message, "🎨")
    await Common.send_tool_discord_embed(
        message.channel,
        f"Generating image: {prompt}",
    )
    try:
        image_bytes = await generate_image_from_api(prompt)
    except Exception as e:
        print(f"Image generation failed: {e}")
        return ("Image generation failed. Tell the user the image service is "
                "unavailable right now and do not retry.")
    try:
        await message.channel.send(
            file=discord.File(io.BytesIO(image_bytes), filename="generated-image.png")
        )
    except Exception as e:
        print(f"Image generated but failed to send to Discord: {e}")
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
        print(f"Sandbox: failed to post the closing note: {e}")
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
    print(f"Running sandbox task: {task}")

    allowed, reason = await check_web_request(task)
    if not allowed:
        print(f"Sandbox task blocked by content guard: {reason}")
        return ("I can't run that in the sandbox — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful task.")

    from classes.config_manager import configManager
    from classes.sandbox_agent import (
        ensure_sandbox_thread,
        run_sandbox_task,
        sandbox_max_turns,
        sandbox_closing_note,
        sandbox_snapshot_exists,
        sandbox_snapshot_id_for,
        sandbox_snapshot_remaining_seconds,
        sandbox_timeout,
        sandbox_workspace_note,
        MAX_ARTIFACT_BYTES,
        MAX_ARTIFACT_FILES,
    )
    from classes.sandbox_progress import (
        DESCRIPTION_CHARS,
        SandboxProgressHooks,
        sandbox_progress_updates_enabled,
    )
    from classes import sandbox_thread_inbox

    original_message = wrapper.context.get("original_message")
    discord_client = wrapper.context.get("discord_client")
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
    # Backstop for the concurrency guard in main.py's on_message. Two paths
    # get past that one: queue lag (two mentions land before either worker
    # starts, so both see no active run) and the outer model emitting two
    # run_code_sandbox calls in a single turn. Either way two containers
    # would race to persist to dcb:sandbox_snapshot:{thread_id} on teardown
    # and the last to finish would clobber the other — so forward the task
    # into the run already in flight instead of starting a second one.
    if in_thread and sandbox_thread_inbox.is_run_active(channel.id):
        sandbox_thread_inbox.deliver(
            channel.id, getattr(original_message, "id", 0),
            getattr(getattr(original_message, "author", None), "display_name", "the user"),
            task,
        )
        return ("A sandbox is already running in this thread, so this request was "
                "handed to the run in progress instead of starting a second one. "
                "Tell the user it was passed along to the sandbox that's already "
                "working, and do not retry.")
    if thread_created:
        try:
            await Common.send_tool_discord_embed(
                original_message.channel,
                f"🧵 Started a sandbox thread: {channel.mention}",
            )
        except Exception as e:
            print(f"Sandbox: failed to notify the original channel of the new thread: {e}")
    # Whether this run will resume the thread's saved workspace or start
    # empty. Asked before the run so the embed below can say which — the SDK
    # exposes no way to find out afterwards (see sandbox_snapshot_exists).
    snapshot_id = sandbox_snapshot_id_for(channel)
    resumed = await sandbox_snapshot_exists(snapshot_id)
    workspace_note = sandbox_workspace_note(resumed)

    progress = None
    # Per-guild toggle (/sandbox_progress_updates, default off). A config
    # read failure falls back to off — progress is a nice-to-have, not a
    # dependency of the run itself.
    try:
        raw = await configManager().get_setting(
            "sandbox_progress_updates", wrapper.context.get("guild_id"))
    except Exception as e:
        print(f"Could not read sandbox_progress_updates setting: {e}")
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
        await Common.send_tool_discord_embed(
            channel,
            f"📨 Messages you send here will be recived by the sandbox AI while the sandbox is running. The AI may or may not respond.",
            0xB0F400,
            "Thread Linked to Sandbox"
        )

    if in_thread:
        sandbox_thread_inbox.begin_run(channel.id)
    # What people said to the run while it happened. main.py routes those
    # messages to the sandbox instead of enqueuing them, and the outer
    # model's history was built before the run started, so without this it
    # answers a request it never saw change — observed: a mid-run "make the
    # milk red" produced a red image the outer model called a mistake.
    steering = ""
    try:
        result = await run_sandbox_task(
            task,
            progress,
            thread=channel,
            client=discord_client,
            requesting_user_id=requesting_user_id,
            resumed=resumed,
        )
    except Exception as e:
        print(f"Sandbox task failed: {e}")
        if progress is not None:
            await progress.finalize("❌ Stopped: the sandbox itself failed.")
        # The "Running in sandbox" embed already went out, so this path needs
        # a closing note too or the thread is left looking mid-run forever.
        await _send_sandbox_closing_note(
            channel, snapshot_id, in_thread, "❌ Stopped: the sandbox itself failed.")
        return ("The sandbox task failed (the code sandbox may be unavailable). "
                "Tell the user the sandbox is not working right now and do "
                "not retry the same task.")
    finally:
        # Must be in a finally: leaving the thread registered would silently
        # swallow every later message posted there (main.py routes them to a
        # run that no longer exists) instead of answering them. history()
        # first — end_run drops the transcript along with the queue.
        #
        # The `except` path above cannot see this: its return value is already
        # built by the time a finally runs. Accepted — that path means the
        # sandbox itself died, so there is no result for steering to explain.
        if in_thread:
            steering = sandbox_thread_inbox.history(channel.id)
            sandbox_thread_inbox.end_run(channel.id)

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
            print(f"Sandbox: failed to post the resume correction: {e}")

    # Per-failure-reason wording so the caller (the outer LLM) knows what,
    # if anything, to change before retrying — a bare "it failed" gives it
    # nothing to act on. See SandboxResult.error for the reason codes.
    finalize_notes = {
        "timeout": "⏱ Stopped: the task timed out.",
        "max_turns": "🔁 Stopped: the task ran out of turns.",
        "model_error": "⚠️ Stopped: the sandbox's model misbehaved.",
    }
    # Deliberately none of these tells the model to retry. A retry starts the
    # whole task over in a new container, so on a run stopped part way it
    # throws away the partial workspace teardown just saved — and in practice
    # the outer model answers "retry, more focused" by writing the
    # implementation spec this tool's docstring exists to prevent (observed:
    # a timed-out cow GIF retried with an invented canvas size, frame count
    # and library choice). The resumable follow-up below is strictly better
    # and costs nothing but asking.
    no_artifact_messages = {
        "timeout": (
            f"The sandbox task ran out of time and was stopped at the "
            f"{sandbox_timeout()}s limit before it finished. Tell the user "
            "it timed out."
        ),
        "max_turns": (
            f"The sandbox task ran out of turns ({sandbox_max_turns()} max) "
            "before finishing. Tell the user it did not finish in the number "
            "of steps it had."
        ),
        "model_error": (
            "The sandbox's own model produced an invalid action mid-task "
            "and the run was stopped before finishing. Tell the user the "
            "sandbox could not complete this task."
        ),
    }
    failure_reason = {
        "timeout": "it timed out",
        "max_turns": "it ran out of turns",
        "model_error": "the sandbox's model misbehaved",
    }

    if progress is not None:
        # give the live message its final state (it would otherwise sit on
        # the last "still running" / thinking snapshot)
        note = "✅ Done." if result.ok else finalize_notes.get(result.error, "❌ Stopped.")
        await progress.finalize(note)

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
            print(f"Sandbox artifact sent to channel: {artifact.name} ({len(artifact.data)} bytes)")
            sent_names.append(artifact.name)
            # Only now: a failed send carried the message with it, so keeping
            # `lead` set is what lets the fallback below still deliver it.
            lead = ""
        except Exception as e:
            print(f"Sandbox artifact {artifact.name} generated but failed to send: {e}")
    if lead:
        # No files, or every send failed: the message still has to reach the
        # user, or the run's own account of itself is lost and only the outer
        # model's second-hand version survives.
        try:
            await channel.send(lead)
        except Exception as e:
            print(f"Sandbox: failed to post the agent's closing message: {e}")

    steering_note = ""
    if steering:
        steering_note = (
            "\n\nThe user changed the request while the sandbox worked, in the "
            f"thread:\n{steering}\nThe sandbox received these and adapted, so the "
            "result reflects them and not the original wording. Treat them as "
            "part of what was asked: do not call the result a mistake, do not "
            "say it went wrong, and do not offer to undo it."
        )

    skip_note = ""
    if result.skipped_artifacts:
        skip_note = (
            f"\n\n{len(result.skipped_artifacts)} output file(s) were NOT sent "
            f"(over the {MAX_ARTIFACT_FILES}-file / {MAX_ARTIFACT_BYTES}-byte "
            f"limit): {', '.join(result.skipped_artifacts)}. Tell the user "
            "which file(s) could not be delivered and why; if it matters, "
            "retry producing a smaller or fewer files."
        )

    # After the artifacts, before every return below: the thread's "here is
    # how the run ended, and how long you can pick it back up" marker. The
    # outcome line only goes out when the run did NOT finish normally.
    outcome = "" if result.ok else finalize_notes.get(result.error, "❌ Stopped.")
    remaining = await _send_sandbox_closing_note(
        channel, snapshot_id, in_thread, outcome)

    if not result.ok:
        # What the model should do about it, kept in step with the closing
        # embed: only offer a resume when there is genuinely a saved workspace
        # to resume (persisting is best-effort — _persist_sandbox_snapshot).
        if in_thread and remaining is not None:
            unfinished_note = (
                " Its workspace WAS saved, so do NOT retry: a retry starts the "
                "whole task again from scratch in an empty sandbox, while the "
                "user simply asking again in THIS thread carries on from where "
                "it stopped. Offer them that, and do not add build instructions "
                "of your own when you do."
            )
        else:
            unfinished_note = (
                " Do not retry the same task unchanged; ask the user how they "
                "want to proceed."
            )
        if sent_names:
            return (
                f"The sandbox task did not finish ({failure_reason.get(result.error, 'it failed')}), "
                f"but before it was stopped it had already produced and verified "
                f"{len(sent_names)} file(s), which were recovered and sent to the "
                f"channel: {', '.join(sent_names)}. Tell the user the file(s) were "
                "delivered despite the task not finishing, and do not claim they "
                "are pending." + unfinished_note + skip_note + steering_note
                + resume_correction
            )
        return no_artifact_messages.get(
            result.error,
            "The sandbox task was stopped before finishing. Tell the user the "
            "task failed.",
        ) + unfinished_note + skip_note + steering_note + resume_correction

    if not sent_names:
        # Only warn when nothing was produced at all — an artifact that WAS
        # found but failed to send (Discord error) is a different case and
        # already correctly described by the exception log above; claiming
        # "no files were found" there would be false.
        no_file_note = (
            "\n\n(The sandbox attached no file for this run. If one was "
            "expected, it was not produced — tell the user that plainly. Do "
            "not claim a file exists, was sent, or is still being generated.)"
        ) if not result.artifacts else ""
        return (result.text + no_file_note + skip_note + steering_note
                + resume_correction)
    return (
        f"{result.text}\n\n"
        f"The sandbox chose {len(sent_names)} file(s) to deliver and they are "
        f"already in the thread: {', '.join(sent_names)}. The text above is its "
        "own message to the user, posted there beside them — so the user has "
        "already read it. Add at most one short sentence of your own: do not "
        "repeat it, re-describe the files, or second-guess the result."
        + skip_note + steering_note + resume_correction
    )


@function_tool
async def change_personality(wrapper: RunContextWrapper[dict], personality: str) -> bool:
    """Changes the personality of the bot. Returns True if successful, False otherwise.
    
    Args:
        personality: The new personality to set.
    """

    # Sometimes OpenAI repeats a tool call.
    times_called = wrapper.context.get("personality_tool_calls")
    if times_called > 0:
        err = f"Tool call limit reached: {times_called}. Not storing data."
        print(err)
        return True
    wrapper.context["personality_tool_calls"] += 1

    from classes.config_manager import configManager
    print(f"Changing personality to: {personality}")
    try:
        configmanager = configManager()
        await configmanager.update_setting("system", personality, wrapper.context.get("guild_id"))
        print(f"Changed personality to: {personality}")

        embed = discord.Embed(title="Personality Updated",
                      description=personality)
        await wrapper.context.get("original_message").channel.send(embed=embed)
        return True
    except Exception as e:
        print(f"An error occurred while changing personality: {e}")
        return False
