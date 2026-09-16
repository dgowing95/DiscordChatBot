# Contributing

Thanks for helping improve DiscordChatBot.

## Before opening a change

- Use a GitHub issue for bugs and feature proposals. Report security problems
  privately as described in [SECURITY.md](SECURITY.md).
- Keep credentials out of commits, logs, screenshots, fixtures, and issue text.
- For substantial design changes, open an issue first so the approach can be
  agreed before implementation.

## Development setup

Use Python 3.13 or newer, create a virtual environment, and install the test
dependencies:

```bash
python -m venv .venv
python -m pip install -r core/requirements-dev.txt
pytest
```

The Docker Compose stack is available for integration testing. Copy
`.env.example` to `.env`, add local credentials, and never commit `.env`.

## Pull requests

1. Fork the repository and work on a focused branch.
2. Add or update tests for behavior changes.
3. Run `pytest` and `helm lint charts/dis-ai-bot` when the chart is affected.
4. Explain the problem, the solution, and how you tested it in the pull request.

Production and tests import application modules as `classes.X`, not
`core.classes.X`. See [AGENTS.md](AGENTS.md) for architecture and repository
conventions.

By participating, you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
