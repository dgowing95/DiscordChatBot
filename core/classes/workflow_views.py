"""The buttons on a workflow's plan card: Approve / Edit / Cancel, then Cancel
alone while it runs (classes/workflows.py, classes/card.py).

Each button is a discord.ui.DynamicItem whose custom_id carries the
workflow's id and the action ("workflow:<id>:approve"). The class is
registered once (client.add_dynamic_items in main.on_ready), so a press is
handled even on a message posted before a restart, with no view kept per
workflow. Resume after a restart is out of scope (every live workflow is
INTERRUPTED at startup), so such a press only answers "This workflow has
stopped" instead of Discord's "This interaction failed".

What a press does, and who may press it, is in workflows.on_button.
"""
import discord

TEMPLATE = r"workflow:(?P<id>[0-9a-f]{1,32}):(?P<action>approve|edit|cancel)"

_BUTTONS = {
    "approve": ("Approve", discord.ButtonStyle.success),
    "edit": ("Edit", discord.ButtonStyle.secondary),
    "cancel": ("Cancel", discord.ButtonStyle.danger),
}


def custom_id(workflow_id: str, action: str) -> str:
    return f"workflow:{workflow_id}:{action}"


class WorkflowButton(discord.ui.DynamicItem[discord.ui.Button], template=TEMPLATE):
    def __init__(self, workflow_id: str, action: str):
        label, style = _BUTTONS[action]
        super().__init__(discord.ui.Button(label=label, style=style,
                                           custom_id=custom_id(workflow_id, action)))
        self.workflow_id = workflow_id
        self.action = action

    @classmethod
    async def from_custom_id(cls, interaction, item, match):
        return cls(match["id"], match["action"])

    async def callback(self, interaction: discord.Interaction) -> None:
        from classes import workflows
        await workflows.on_button(interaction, self.workflow_id, self.action)


def approval_buttons(workflow_id: str) -> list:
    return [WorkflowButton(workflow_id, action) for action in ("approve", "edit", "cancel")]


def running_buttons(workflow_id: str) -> list:
    return [WorkflowButton(workflow_id, "cancel")]
