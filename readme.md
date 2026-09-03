# DiscordChatBot

A Discord bot that answers messages with a **self-hosted** LLM. Nothing is sent to a
hosted model provider by default: a llama.cpp server serves a GGUF model on your own
GPU, and Redis stores per-guild settings and per-user memories.

Beyond chat it can search the web, read a URL, generate images, and hand a task to a
**code sandbox** — a throwaway Docker container with its own agent that writes and
actually runs code, in a Discord thread you can talk to while it works.

It is designed to run on Kubernetes via the Helm chart in `charts/dis-ai-bot`, with a
docker-compose stack for local development.

## Components

| Component | What it is |
|---|---|
| `core/` | the bot itself: Discord client, message queue, agent + tools, Prometheus metrics |
| `llamacpp` | the LLM server (OpenAI-compatible API), serving the GGUF named by `model` |
| `redis` | per-guild settings, per-user memories, and sandbox workspace snapshots |
| `diffusionservice/` | optional image generation (text-to-image); disable with `diffusion.enabled=false` |

## Requirements

- A Kubernetes cluster (microk8s, k3s and minikube all work).
- Helm.
- An NVIDIA GPU with the [NVIDIA Container Runtime Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- A registered Discord bot and its token.
- For the code sandbox: a Docker daemon on the node, whose socket is mounted into the
  core pod. Disable with `sandbox.enabled=false`.

## Installing

1. Write a `values.yaml` with at least `discordToken`. Every option and its default is
   documented in [`charts/dis-ai-bot/values.yaml`](charts/dis-ai-bot/values.yaml).
2. Install the packaged chart from a release (see
   [Releases](https://github.com/dgowing95/DiscordChatBot/releases) for the latest tag):

   ```bash
   helm upgrade --install dchatbot \
     https://github.com/dgowing95/DiscordChatBot/releases/download/v2.26/dchatbot-v2.26.tgz \
     -f values.yaml --namespace dchatbot --create-namespace
   ```

The llama.cpp pod downloads the GGUF on first boot, so the first start is slow; the same
applies to the diffusion pod's model. Both cache onto a volume and survive redeploys.

## Local development

```bash
cp .env.example .env        # fill in DISCORD_TOKEN at minimum; never commit .env
docker compose up -d
docker compose logs core -f
```

`./core` is bind-mounted into the container, so `docker compose restart core` picks up
code edits without a rebuild. Changing `MODEL` needs `docker compose up -d` (a plain
`restart` does not re-read `.env`).

## Tests

```bash
pip install -r core/requirements-dev.txt
pytest
```

No arguments and no `PYTHONPATH` — configuration lives in `pyproject.toml`, and this is
exactly what CI runs.

## More

[`AGENTS.md`](AGENTS.md) covers the runtime architecture, the full environment-variable
reference, the prompt surface, and the conventions and gotchas worth knowing before
changing anything. It is written for both humans and AI coding agents.
