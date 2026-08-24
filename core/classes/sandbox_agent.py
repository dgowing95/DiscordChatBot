"""Docker-backed code sandbox for the run_code_sandbox tool.

Uses the OpenAI Agents SDK sandbox feature (beta, `agents.sandbox`): a nested
SandboxAgent does the work inside a throwaway Docker container (shell +
filesystem capabilities, python image) and returns its final answer.

Lifecycle: every call creates a FRESH container (empty workspace), runs the
nested agent loop, then the SDK stops and deletes the container. Nothing
persists between calls.

Requirements (see docker-compose.yaml / charts/dis-ai-bot):
  - a reachable Docker daemon (docker.sock mounted into the core container;
    `docker` + `websocket-client` Python packages in requirements.txt)
  - the sandbox image is pulled once onto the daemon, then cached

The heavy imports (docker SDK, agents.sandbox) stay inside the builder
functions so the pure helpers below are testable without them.
"""
import asyncio
import os

from agents import AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE

DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_SECONDS = 600

# Instructions for the nested sandbox agent. The workspace is always empty,
# so the task text is its only input — it must behave self-sufficiently.
# It MUST explicitly name the shell tools: the SDK's default sandbox prompt
# (suppressed — see build_sandbox_agent) tells the model to use an
# apply_patch tool we do not expose, and a model that calls it aborts the
# whole run with ModelBehaviorError.
SANDBOX_INSTRUCTIONS = """You work inside a fresh, isolated Linux sandbox (a minimal
Python container) with shell tools. You receive one self-contained task.

- Your ONLY tools are the shell tools: `exec_command` (run a command) and
  `write_stdin` (interact with a running process). There is NO file-editing,
  patching or editor tool — create and modify files through the shell
  (heredocs: `cat > file.py << 'EOF' ... EOF`, or python one-liners), then
  run and verify them.
- The workspace starts completely empty. Never assume files, packages or
  context exist unless the task provides them; install what you need
  (e.g. `pip install ...`).
- Do the work: write files, run commands, read the output, iterate until the
  task is done. If a step fails, read the error, fix it and retry.
- If the task includes code or data, create it exactly as given.
- Do not ask questions: make reasonable assumptions and state them.
- When done, end with a concise final report: what you did, the key results
  (exact numbers/outputs where they matter), and any failures or assumptions.
  The final report is all the caller sees, so make it self-contained."""


def sandbox_enabled() -> bool:
    """True when the run_code_sandbox tool should be offered to the LLM.

    Controlled by SANDBOX_ENABLED (set from the helm chart's sandbox.enabled,
    or .env locally); defaults to enabled.
    """
    raw = os.environ.get("SANDBOX_ENABLED", "1")
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def sandbox_image() -> str:
    """Container image for the sandbox workspace (pulled once, then cached)."""
    return os.environ.get("SANDBOX_IMAGE", DEFAULT_PYTHON_SANDBOX_IMAGE)


def _env_or(primary: str, fallback: str, default: str) -> str:
    """First non-empty of: env var `primary`, env var `fallback`, `default`."""
    for name in (primary, fallback):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


def sandbox_model() -> str:
    """Model id for the nested sandbox agent (SANDBOX_MODEL; default: the
    main bot's MODEL — i.e. the local llama.cpp model unless SANDBOX_MODEL
    points the sandbox at a different OpenAI-compatible API, e.g.
    OpenRouter)."""
    return _env_or("SANDBOX_MODEL", "MODEL", "qwen3:4b")


def sandbox_llm_host() -> str:
    """Base URL of the LLM the sandbox agent talks to (SANDBOX_LLM_HOST;
    default: the main bot's LLM_HOST). The core appends /v1 itself, so this
    must not include it (OpenRouter: https://openrouter.ai/api)."""
    return _env_or("SANDBOX_LLM_HOST", "LLM_HOST", "http://llamacpp:8080")


def sandbox_llm_api_key() -> str:
    """API key for the sandbox agent's LLM (SANDBOX_LLM_API_KEY; default:
    the main bot's LLM_PASS placeholder)."""
    return _env_or("SANDBOX_LLM_API_KEY", "LLM_PASS", "llamacpp")


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(str(raw).strip())
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def sandbox_max_turns() -> int:
    """Max model turns for one sandbox task (SANDBOX_MAX_TURNS, default 10)."""
    return _positive_int(os.environ.get("SANDBOX_MAX_TURNS"), DEFAULT_MAX_TURNS)


def sandbox_timeout() -> int:
    """Wall-clock seconds before a sandbox task is stopped (SANDBOX_TIMEOUT,
    default 600)."""
    return _positive_int(os.environ.get("SANDBOX_TIMEOUT"), DEFAULT_TIMEOUT_SECONDS)


def build_sandbox_agent() -> "object":
    """The nested SandboxAgent that does the work inside the sandbox.

    Uses the same LLM as the main agent by default (MODEL / LLM_HOST /
    LLM_PASS); SANDBOX_MODEL / SANDBOX_LLM_HOST / SANDBOX_LLM_API_KEY point
    it at a different OpenAI-compatible API instead (e.g. an OpenRouter
    model — see .env.example / the chart's sandbox.model/llmHost/apiKey).
    No default manifest, so the workspace starts empty.

    Capabilities: Shell ONLY. The Filesystem capability adds `apply_patch`, a
    grammar (hosted) tool that the ChatCompletions API — what llama.cpp serves
    — does not support (the converter raises "Hosted tools are not supported").
    exec_command already gives full filesystem access (heredocs, cat, pip, …),
    which is the pattern the SDK's own docker_runner.py example uses.

    base_instructions is an empty string ON PURPOSE: with the default (None)
    the SDK injects its bundled computer-use prompt, which repeatedly tells
    the model to use the apply_patch tool — which we do not expose. A model
    that obeys it aborts the run with ModelBehaviorError ("Model produced
    apply_patch call without an apply_patch tool"). Empty string suppresses
    the default entirely; our instructions + the Shell capability's own
    instructions are all the model needs.
    """
    from agents.sandbox import SandboxAgent
    from agents.sandbox.capabilities import Shell

    return SandboxAgent(
        name="Code Sandbox",
        model=OpenAIChatCompletionsModel(
            model=sandbox_model(),
            openai_client=AsyncOpenAI(
                base_url=sandbox_llm_host() + "/v1",
                api_key=sandbox_llm_api_key(),
            ),
        ),
        instructions=SANDBOX_INSTRUCTIONS,
        base_instructions="",
        capabilities=[Shell()],
        # Slightly cooler than the chat agent: code tasks want determinism.
        model_settings=ModelSettings(temperature=0.5),
    )


def build_sandbox_run_config() -> RunConfig:
    """Run config that creates a fresh Docker container for the run.

    The SDK owns the session lifecycle here (no live session is passed in):
    it creates the container from the agent's manifest, runs the agent, then
    stops and deletes the container — even when the run is cancelled.
    """
    from docker import from_env as docker_from_env

    from agents.sandbox import SandboxRunConfig
    from agents.sandbox.sandboxes.docker import DockerSandboxClient, DockerSandboxClientOptions

    return RunConfig(
        sandbox=SandboxRunConfig(
            client=DockerSandboxClient(docker_from_env()),
            options=DockerSandboxClientOptions(image=sandbox_image()),
        )
    )


async def run_sandbox_task(task: str, progress_hooks=None) -> str:
    """Run one self-contained task in a fresh Docker sandbox.

    Returns the sandbox agent's final report (string). progress_hooks is
    an optional agents.RunHooks instance (e.g. classes.sandbox_progress.
    SandboxProgressHooks) attached to the nested run so every tool call
    (exec_command/write_stdin) and its output can be observed — e.g. to
    stream them to Discord. Raises asyncio.TimeoutError if the task
    outlives sandbox_timeout() seconds (the SDK still tears the container
    down when the run is cancelled); other errors propagate to the caller
    (the run_code_sandbox tool).
    """
    result = await asyncio.wait_for(
        Runner.run(
            build_sandbox_agent(),
            task,
            max_turns=sandbox_max_turns(),
            run_config=build_sandbox_run_config(),
            hooks=progress_hooks,
        ),
        timeout=sandbox_timeout(),
    )
    final_output = result.final_output
    return final_output if isinstance(final_output, str) else str(final_output)
