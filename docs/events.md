# Server events

The bot can make Discord server events (the ones people click **Interested** on), list what is coming up with who is interested, and move or cancel events it made. Ask it in chat, for example:

- "Make an event called Game night on Saturday at 8pm in the Games voice channel"
- "What events are coming up, and who's going?"
- "Move game night to 9pm" / "Cancel game night"

## Finding a time that suits everyone

Ask the bot to find a time, e.g. "find a time for game night this week". It posts a poll of candidate times and sets a [follow-up](automations.md#follow-ups) for when the poll ends. The follow-up reads the result, makes the event at the winning time and mentions you. You can also do the steps yourself: ask for a poll, then "when the poll ends, make the event at the winning time".

## Permissions

- The bot's role needs the **Create Events** permission.
- The person asking needs **Create Events** (or **Manage Events**) too. Otherwise the bot would let anyone make events the server does not let them make. A follow-up checks this again, as the person who set it up, when it runs.
- The bot only moves or cancels events it made.

## Details

- Times without a timezone are read in `AUTOMATIONS_TIMEZONE` (default Europe/London). The bot writes times back as Discord timestamps, so everyone sees them in their own timezone.
- An event is held in a voice or stage channel, or somewhere else with a location. One with a location ends 2 hours after it starts unless you give an end time (Discord requires one).
- Events are visible to server members only.
- Events do not repeat. Asked for a daily or weekly event, the bot makes the next one and says so.
- `EVENTS_ENABLED=0` (Helm: `events.enabled: false`) removes the event tools.
