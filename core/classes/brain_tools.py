"""The brain's tools: the one home of the list.

Since v3 the persona has a single tool, ask_brain, and every other tool
belongs to the brain (classes/brain.py). This replaces both the persona's old
list (text_llm_handler.agent_tools) and the sandbox's allowlist of "shared"
tools with its gated copies: every tool is ONE object, used by the brain
only.

Which tools a call gets depends on what kind of run it is:

| Tools                                                   | Chat | Automatic | Voice |
|---------------------------------------------------------|------|-----------|-------|
| web, message search/read, polls, events, personality    | yes  | yes       | yes   |
| memory, schedules / rules / follow-ups                  | yes  | no        | yes   |
| join voice (VOICE_ENABLED)                              | yes  | no        | no    |
| leave voice (VOICE_ENABLED)                             | no   | no        | yes   |
| generate_image (IMAGE_GEN_ENABLED)                      | yes  | yes       | yes   |
| workspace: attach_file, save_messages_to_file, ask_user, report (+ Shell) | yes | yes | yes |

An automatic run (a schedule, a rule or a follow-up) has no personal memory
and cannot set up more automations, as before. Its usual job is "read the poll,
make the event", so the event tools stay. The event tools check the editor's
permission in a schedule or a rule, but not in a follow-up
(event_tools._checks_requester).

The order is fixed, so the brain's tool schemas (part of its cached prompt
prefix) are the same on every call of the same kind.

Imports are inside the function: the tool modules import each other, and
tool_functions imports nothing from here.
"""


def brain_tools(*, automatic: bool = False, voice: bool = False) -> list:
    from classes import voice_policy
    from classes.automation_tools import automation_tools
    from classes.brain import ask_user, report
    from classes.event_tools import event_tools
    from classes.image_generation import image_generation_enabled
    from classes.message_search_tools import message_search_tools, save_messages_to_file
    from classes.tool_functions import (
        change_personality,
        check_polls,
        clear_memories,
        create_poll,
        fetch_url,
        generate_image,
        join_voice_channel,
        leave_voice_channel,
        remove_memory,
        store_memory,
        web_search,
    )
    from classes.workspace_tools import attach_file

    tools = [web_search, fetch_url, *message_search_tools(), check_polls, create_poll,
             *event_tools(), change_personality]
    if not automatic:
        tools.extend((store_memory, remove_memory, clear_memories))
        tools.extend(automation_tools())
        if voice_policy.settings()["enabled"]:
            tools.append(leave_voice_channel if voice else join_voice_channel)
    if image_generation_enabled():
        tools.append(generate_image)
    tools.extend((attach_file, save_messages_to_file, ask_user, report))
    return tools
