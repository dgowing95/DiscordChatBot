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
| skills: use_skill, save_skill, search_skills (SKILLS_ENABLED)           | yes  | yes       | yes   |
| workspace: attach_file, save_messages_to_file, ask_user (+ Shell) | yes | yes | yes |
| start_workflow (when Tier 3 is allowed: brain_policy.effective_tier) | yes | yes* | no |
| report                                                  | yes  | yes       | yes   |

(*) only with WORKFLOW_FROM_AUTOMATIONS. Never in a reply nobody asked for.

An automatic run (a schedule, a rule or a follow-up) has no personal memory
and cannot set up more automations, as before. Its usual job is "read the poll,
make the event", so the event tools stay. The event tools check the editor's
permission in a schedule or a rule, but not in a follow-up
(event_tools._checks_requester).

A Tier 3 workflow's agents (classes/workflows.py) get their own `mode`, from
the same objects:

| mode        | Tools                                                         |
|-------------|---------------------------------------------------------------|
| manager     | search_skills, submit_plan, and the `rounds` still allowed (discover, ask_requester) |
| scout       | web, fetch, message search/read, save_messages_to_file, report (+ Shell) |
| review      | submit_review (the thread's messages between steps)          |
| finish      | report                                                        |
| researcher  | web, fetch, message search/read, attach_file, save_messages_to_file, report (+ Shell) |
| implementer | fetch, generate_image, attach_file, report (+ Shell)          |

No workflow agent can save memories, change the personality, post polls or
events, set automations, ask inline questions or start another workflow
(depth <= 2): those belong to a conversation, and a workflow's agents work
alone in its thread. The manager has no web tools: it looks at a job through
scouts, whose prompt keeps them to its shape (classes/workflow_planning.py).

The order is fixed, so the brain's tool schemas (part of its cached prompt
prefix) are the same on every call of the same kind.

Imports are inside the function: the tool modules import each other, and
tool_functions imports nothing from here.
"""
BRAIN = "brain"
MANAGER = "manager"
SCOUT = "scout"
REVIEW = "review"
FINISH = "finish"
RESEARCHER = "researcher"
IMPLEMENTER = "implementer"
MODES = (BRAIN, MANAGER, SCOUT, REVIEW, FINISH, RESEARCHER, IMPLEMENTER)


def brain_tools(*, automatic: bool = False, voice: bool = False, workflows: bool = False,
                mode: str = BRAIN, rounds=()) -> list:
    from classes import skill_format, voice_policy
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
    from classes.skill_tools import search_skills, skill_tools
    from classes.workflow_tools import (
        ask_requester,
        discover,
        start_workflow,
        submit_plan,
        submit_review,
    )
    from classes.workspace_tools import attach_file

    if mode not in MODES:
        raise ValueError(f"unknown brain_tools mode {mode!r}")
    if mode == MANAGER:
        tools = []
        if skill_format.enabled():
            # search_skills only: use_skill posts a "Using a skill" embed, and a
            # planner opened unrelated skills just to read them (live, twice).
            tools.append(search_skills)
        tools.extend(tool for tool in (discover, ask_requester) if tool.name in rounds)
        tools.append(submit_plan)
        return tools
    if mode == SCOUT:
        return [web_search, fetch_url, *message_search_tools(), save_messages_to_file, report]
    if mode == REVIEW:
        return [submit_review]
    if mode == FINISH:
        return [report]
    if mode == RESEARCHER:
        return [web_search, fetch_url, *message_search_tools(), attach_file,
                save_messages_to_file, report]
    if mode == IMPLEMENTER:
        tools = [fetch_url]
        if image_generation_enabled():
            tools.append(generate_image)
        return tools + [attach_file, report]

    tools = [web_search, fetch_url, *message_search_tools(), check_polls, create_poll,
             *event_tools(), change_personality]
    if not automatic:
        tools.extend((store_memory, remove_memory, clear_memories))
        tools.extend(automation_tools())
        if voice_policy.settings()["enabled"]:
            tools.append(leave_voice_channel if voice else join_voice_channel)
    if image_generation_enabled():
        tools.append(generate_image)
    tools.extend((attach_file, save_messages_to_file, ask_user))
    if skill_format.enabled():
        tools.extend(skill_tools())
    if workflows and not voice:
        tools.append(start_workflow)
    tools.append(report)
    return tools
