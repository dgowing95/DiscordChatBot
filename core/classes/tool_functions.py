import logging
import asyncio
import io

from agents import FunctionTool, function_tool,RunContextWrapper
from classes.common import Common
from classes.content_guard import check_web_request
from classes import tool_embeds
import discord

logger = logging.getLogger(__name__)

async def add_emoji_to_message(message: discord.Message, emoji: str) -> None:
    # No message to react to in a voice turn or an automatic run.
    if message is None:
        return
    try:
        await message.add_reaction(emoji)
        logger.info(f"Added emoji {emoji} to message {message.id}")
    except Exception as e:
        logger.warning(f"Failed to add emoji {emoji} to message {message.id}: {e}")
    
async def _tool_notice(wrapper: RunContextWrapper[dict], title: str, text: str,
                       color: int = tool_embeds.COLOR_WEB) -> None:
    """The embed a lookup posts where it was asked. Skipped when the run has
    no channel: a brain call may have no thread (its tests, and any caller
    that passes none), and a lookup must still work there rather than fail
    on None.send."""
    channel = wrapper.context.get("channel")
    if channel is not None:
        await Common.send_tool_discord_embed(channel, text, color, title)


@function_tool
async def web_search(wrapper: RunContextWrapper[dict], search_request: str) -> str:
    """Searches the web, reads the top results and returns a short brief of
    what they say, with numbered source links. Use it for anything current or
    factual you are not sure of, and before making something (a picture, a
    file) that depends on facts you don't know: for "draw the new Mars
    rover", look up what it looks like first, then describe that. To read
    one particular link, use fetch_url instead.

    Args:
        search_request: What you want to find out, as a question or search
            terms.
    """
    logger.info(f"Searching the web for: {search_request}")

    allowed, reason = await check_web_request(search_request)
    if not allowed:
        logger.warning(f"Web search blocked by content guard: {reason}")
        return ("I can't perform that search — it was blocked by the safety "
                "guard. Please rephrase with a safe, non-harmful query.")

    await _tool_notice(wrapper, tool_embeds.WEB_SEARCH, search_request)
    from classes import web_research
    try:
        return await web_research.research(search_request, await get_current_datetime())
    except Exception as e:
        logger.warning(f"An error occurred while searching: {e}")
        return "Error fetching search results."
    
@function_tool
async def fetch_url(wrapper: RunContextWrapper[dict], url: str, part: int = 1) -> str:
    """Reads a web page and returns its text. Also reads Reddit threads,
    Steam store pages and YouTube videos (from the video's captions). A long
    page comes in parts; the result says how to read the next one.

    Args:
        url: The URL to read.
        part: Which part of a long page to read, starting at 1.
    """
    from classes import web_reader
    from classes.web_policy import paginate, settings
    logger.info(f"Fetching content from URL: {url} (part {part})")

    allowed, reason = await check_web_request(url)
    if not allowed:
        logger.warning(f"URL fetch blocked by content guard: {reason}")
        return ("I can't fetch that URL — it was blocked by the safety "
                "guard.")

    await _tool_notice(wrapper, tool_embeds.reading_page_title(part), tool_embeds.page_link(url))
    try:
        page = await web_reader.read(url)
    except web_reader.FetchError as e:
        logger.info(f"Could not read {url}: {e.reason}")
        return e.message
    except Exception:
        logger.exception(f"An error occurred while fetching {url}")
        return "Error fetching URL content."

    logger.info(f"Fetched content from {url} ({page.source}, {len(page.text)} chars).")
    return paginate(page.text, part, settings()["part_chars"])

async def get_current_datetime() -> str:
    """Returns the current date and time, with its offset, in the server
    timezone (AUTOMATIONS_TIMEZONE) the time tools read naive times in.
    Without the offset the model guessed one: a follow-up meant for "3
    minutes from now" was saved as London time marked +00:00, an hour late."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from classes.automation_policy import server_timezone
    zone = server_timezone()
    now = datetime.now(ZoneInfo(zone))
    now_formatted = f"{now.isoformat(timespec='seconds')} ({zone})"
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
            wrapper.context["channel"], data, tool_embeds.COLOR_MEMORY, tool_embeds.REMEMBERED)
        
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


def _bot_user_id(context):
    return getattr(getattr(context.get("discord_client"), "user", None), "id", None)


@function_tool
async def generate_image(wrapper: RunContextWrapper[dict], prompt: str,
                         edit_previous: bool = False) -> str:
    """Generates an image from a text description and sends it to the channel.
    Use it when the user asks for art, illustrations, pictures or drawings,
    or wants a picture you made earlier changed.
    The image is sent automatically; never try to send it yourself.
    Args:
        prompt: A plain-language description of the picture you want, in
            ordinary sentences. Include everything that matters: the main
            subject and what it is doing, where it is, how it is positioned,
            what else is in shot, the lighting, the mood, and the style or
            medium. Say what the user implied but did not spell out. Any
            reformatting the image model needs is done for you, so do NOT
            write comma-separated tag lists, quality boilerplate like "high
            resolution", or weighting syntax - and do not leave detail out to
            keep it short. Anything that must NOT appear can simply be
            written as a normal sentence ("no people in the shot").
        edit_previous: True to change a picture you already made instead of
            starting a new one. Then prompt says only what changes and what
            stays the same ("make her much fatter; keep the armour, pose and
            street").
    """
    from classes.image_generation import (
        GENERATED_IMAGE_FILENAME,
        MAX_IMAGE_CALLS,
        create_image,
        find_edit_source,
        service_capabilities,
        too_many_images_text,
        tool_result_text,
    )

    context = wrapper.context
    context["image_calls"] = context.get("image_calls", 0) + 1
    # A workflow sets its own, higher cap (workflow_policy.MAX_IMAGES).
    cap = context.get("image_cap", MAX_IMAGE_CALLS)
    if context["image_calls"] > cap:
        logger.info("generate_image: limit reached; not generating")
        return too_many_images_text(cap, "workflow" if "image_cap" in context else "reply")
    channel = context["channel"]
    message = context.get("original_message")
    logger.info(f"{'Editing' if edit_previous else 'Generating'} image for prompt: {prompt}")
    if message is not None:
        await add_emoji_to_message(message, "🎨")

    source = None
    if edit_previous:
        # Checked up front because an edit prompt only describes the CHANGE;
        # drawn from scratch, "make her fatter" would be a picture of nobody.
        if not (await service_capabilities()).edits:
            return ("The current image model cannot edit pictures, so nothing was "
                    "made. Call generate_image again with edit_previous false and a "
                    "description of the whole picture, the change included.")
        source = await find_edit_source(channel, message, _bot_user_id(context))
        if source is None:
            return ("There is no earlier picture of mine in this channel to edit, so "
                    "nothing was made. Call generate_image again with edit_previous "
                    "false and a description of the whole picture.")
    title = tool_embeds.EDITING if source is not None else tool_embeds.DRAWING

    # The prompt the image model is given, not the requested one, is what the
    # embed shows: it is the thing worth seeing when the result does not match
    # what was asked for.
    embed = None

    async def announce(first_prompt):
        nonlocal embed
        try:
            embed = await Common.send_tool_discord_embed(
                channel, first_prompt, tool_embeds.COLOR_IMAGE, title)
        except Exception as e:
            logger.warning(f"Could not post the image tool embed: {e}")

    # What was asked, for the check of the finished picture: the message, or
    # in a voice call (no message) what the speaker said.
    user_request = ((getattr(message, "clean_content", None) or getattr(message, "content", None)
                     or "") if message is not None else context.get("request_text") or "")
    try:
        result = await create_image(prompt, user_request, reference=source, on_first_prompt=announce)
    except Exception as e:
        logger.warning(f"Image generation failed: {e}")
        return ("Image generation failed. Tell the user the image service is "
                "unavailable right now and do not retry.")
    if embed is not None and result.attempts > 1:
        # Only the final image is posted, so the embed must describe that one.
        await Common.edit_tool_discord_embed(
            embed, result.prompt, tool_embeds.COLOR_IMAGE, title)
    try:
        await channel.send(
            file=discord.File(io.BytesIO(result.png), filename=GENERATED_IMAGE_FILENAME)
        )
    except Exception as e:
        logger.warning(f"Image generated but failed to send to Discord: {e}")
        return "The image was generated but could not be sent to the channel."
    # Counted only once sent: the brain's report lists it as already posted.
    context["images_posted"] = context.get("images_posted", 0) + 1
    kept = ""
    if context.get("workflow_id"):
        from classes import workflows
        kept = await workflows.keep_image(context, result.png)
    return tool_result_text(result) + kept


@function_tool
async def ask_brain(wrapper: RunContextWrapper[dict], request: str,
                    files: list[str] | None = None) -> str:
    """Hands a job to your brain, which has the tools: the web and links,
    saving and forgetting what is known about people, your own personality,
    pictures, polls, events, reminders, this server's messages, voice calls,
    and a computer for code, files and charts (it keeps its earlier work
    and this server's saved skills).
    You cannot open links, see its files, change yourself or act later, so
    use it for anything beyond talking. For example:
    "what's the weather in Oslo?" -> request="Look up today's weather in Oslo";
    "summarise that" (a link) -> request="Summarise https://...";
    "in an hour, check the score" -> request="In an hour, check the score";
    "remember I'm vegan" -> request="Remember that they are vegan";
    "be a pirate from now on" -> request="Change the personality to a pirate";
    "send the script for that chart" -> request="Send the script that made the last chart";
    "do a deep dive on printers" -> request="Start a workflow: a deep dive on printers";
    "use the weekly-chart skill" -> request="Use the weekly-chart skill".
    Not for chat, opinions or jokes, or what the chat or what is saved
    about the person (shown below it) already answers.
    Its result lists facts to keep exact and what it already posted.

    Args:
        request: What the user wants, in their words (timing too), plus
            anything from the chat it needs (links, names, quoted text).
        files: Chat files to work on, by the name shown in [Attached: ...].
            A picture you made is "generated-image.png".
    """
    from classes.brain import run_brain
    from classes.brain_result import tool_result_for_persona

    logger.info(f"Asking the brain: {request}")
    context = wrapper.context
    # One brain call at a time per reply: the SDK runs one response's tool
    # calls concurrently, and every brain call shares (and cleans up) the
    # persona's context dict, so two at once would trample each other's
    # session, files and report.
    async with context.setdefault("brain_lock", asyncio.Lock()):
        report = await run_brain(context, request, files)
    # Kept for the check after the reply (TextLLMHandler: must-keep facts the
    # reply left out are logged and counted).
    context.setdefault("brain_reports", []).append(report)
    return tool_result_for_persona(report)


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

        await Common.send_tool_discord_embed(
            wrapper.context["channel"], personality, tool_embeds.COLOR_PERSONA,
            tool_embeds.PERSONALITY)
        return True
    except Exception as e:
        logger.warning(f"An error occurred while changing personality: {e}")
        return False


@function_tool
async def create_poll(wrapper: RunContextWrapper[dict], question: str, answers: list[str],
                      duration_hours: int = 24, allow_multiple: bool = False) -> str:
    """Posts a Discord poll in the channel for people to vote on. Use it when
    someone asks for a poll or a vote, or a group needs to pick between
    options. You write the question and the options. The poll is posted
    automatically, so do not list the options again in your reply; its votes
    show up in the chat history as they come in.

    Args:
        question: The question to vote on, up to 300 characters.
        answers: 2 to 10 short options, each 55 characters or fewer.
        duration_hours: How long voting stays open, 1 to 768 hours.
        allow_multiple: True to let people pick more than one option.
    """
    from datetime import datetime, timedelta, timezone
    from classes.poll_format import validate

    question, answers, hours, error = validate(question, answers, duration_hours)
    if error:
        return error

    # As in store_memory: the model sometimes repeats a call within one reply.
    context = wrapper.context
    if context.get("poll_tool_calls", 0) > 0:
        logger.warning("create_poll already called this run; skipping.")
        return ("A poll was already posted for this request, so this duplicate call "
                "was skipped. Nothing further is needed.")

    poll = discord.Poll(question=question, duration=timedelta(hours=hours),
                        multiple=allow_multiple)
    for answer in answers:
        poll.add_answer(text=answer)
    logger.info(f"Posting poll: {question} {answers} ({hours}h)")
    try:
        await context["channel"].send(poll=poll)
    except discord.Forbidden:
        logger.warning("Not allowed to post a poll in this channel")
        return ("No poll was posted: I'm not allowed to post polls in this channel "
                "(it needs the Create Polls permission). Tell the user.")
    except Exception as e:
        logger.warning(f"Failed to post a poll: {e}")
        return "No poll was posted: Discord rejected it. Tell the user and do not retry."
    context["poll_tool_calls"] = context.get("poll_tool_calls", 0) + 1
    await add_emoji_to_message(context.get("original_message"), "📊")
    ends = datetime.now(timezone.utc) + timedelta(hours=hours)
    return (f"Poll posted: \"{question}\" with {len(answers)} options, open for {hours} "
            f"hours (voting ends {ends.strftime('%Y-%m-%d %H:%M')} UTC, <t:{int(ends.timestamp())}:R>). "
            "It is already in the channel, so do not repeat the options. Votes "
            "will appear in the chat history.")


# Polls check_polls reports at most, newest first.
CHECK_POLLS_LIMIT = 5


@function_tool
async def check_polls(wrapper: RunContextWrapper[dict]) -> str:
    """Looks up the polls posted in this channel over the last week, with
    their current votes and who voted. Use it when someone asks about a poll
    that is not in the messages you can see."""
    from classes import poll_store
    from classes.polls import describe_poll, message_poll, poll_voters

    channel = wrapper.context["channel"]
    try:
        ids = await poll_store.recent(channel.id, CHECK_POLLS_LIMIT)
    except Exception as e:
        logger.warning(f"Could not read this channel's polls: {e}")
        return "Polls could not be looked up right now. Tell the user."

    async def _one(message_id):
        try:
            message = await channel.fetch_message(message_id)
        except discord.NotFound:
            await poll_store.forget(channel.id, message_id)
            return None
        except Exception as e:
            logger.warning(f"Could not fetch poll {message_id}: {e}")
            return None
        if message_poll(message) is None:
            return None
        return describe_poll(message, await poll_voters(message))

    found = [text for text in await asyncio.gather(*(_one(i) for i in ids)) if text]
    if not found:
        return "There have been no polls in this channel in the last week."
    return "Polls in this channel, newest first:\n\n" + "\n\n".join(found)


@function_tool
async def join_voice_channel(wrapper: RunContextWrapper[dict]) -> str:
    """Joins the voice channel the person asking is in, so you can talk with
    them out loud. Use it when someone asks you to join voice, come into the
    call, or talk to them. While you are in the call you only answer there,
    not in text."""
    from classes import voice_session

    message = wrapper.context.get("original_message")
    member = getattr(message, "author", None)
    voice_state = getattr(member, "voice", None)
    channel = getattr(voice_state, "channel", None)
    if channel is None:
        return ("The person asking is not in a voice channel, so you could not join. Tell "
                "them to join one first and ask again, or use /voice join.")
    try:
        session = await voice_session.start(wrapper.context["discord_client"], channel,
                                            wrapper.context["channel"], member)
    except voice_session.VoiceError as e:
        return f"You could not join the voice channel: {e} Tell the user that."
    return (f"You joined the voice channel {channel.name}. Say so in one short sentence and "
            f"tell them to say \"{session.wake_phrase}\" to talk to you there.")


@function_tool
async def leave_voice_channel(wrapper: RunContextWrapper[dict]) -> str:
    """Leaves the voice call. Use it only when someone asks you to leave,
    hang up or go away."""
    from classes import voice_gate

    session = voice_gate.session(wrapper.context.get("guild_id"))
    if session is None:
        return "You are not in a voice call."
    session.leave_after_turn = True
    return "You will leave the call as soon as you finish speaking. Say a short goodbye."
