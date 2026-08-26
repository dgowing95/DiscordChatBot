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
import posixpath
from dataclasses import dataclass, field

from agents import AsyncOpenAI, ModelSettings, OpenAIChatCompletionsModel, Runner, RunConfig
from agents.sandbox.config import DEFAULT_PYTHON_SANDBOX_IMAGE

DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_SECONDS = 600
SANDBOX_DELETE_TIMEOUT_SECONDS = 30  # bound on tearing down our own container
SANDBOX_RECOVERY_TIMEOUT_SECONDS = 30  # bound on best-effort artifact recovery after a timeout

# Directory (relative to the sandbox's own shell cwd) where the sandbox
# agent must save anything it wants returned as a file. Everything else in
# the workspace is discarded with the container.
#
# This is intentionally relative, not the SDK's nominal "/workspace" root:
# with the Shell-only capability we use (see build_sandbox_agent), that
# root is never actually materialized in the container — only the
# Filesystem capability sets it up, and we deliberately don't use it. The
# model's exec_command calls run with whatever cwd the image itself
# defaults to (verified empirically to be "/" for the default sandbox
# image), so we resolve `out/` the same way: by asking the live session
# for its own cwd (_sandbox_output_dir). We now do this resolution BEFORE
# Runner.run starts (not just post-hoc to collect artifacts afterward), so
# we can mkdir -p it before the model's first turn — but we deliberately
# never tell the model the resolved ABSOLUTE path (see SANDBOX_INSTRUCTIONS
# below): exec_command's own `workdir` argument is validated by the SDK's
# manifest system, which unconditionally rejects absolute paths
# (InvalidManifestPathError: "manifest path must be relative: ...") even
# when the path genuinely exists — confirmed in production, where a model
# told the absolute path reused it as `workdir` and the tool call failed
# outright. The model only ever gets the relative name; we resolve/track
# the absolute form ourselves for our own raw session.exec() calls, which
# bypass that manifest check entirely (see _collect_artifacts).
SANDBOX_OUTPUT_DIRNAME = "out"
MAX_ARTIFACT_FILES = 10
# Discord's own non-boosted upload cap is ~25MB; enforcing it here fails
# fast instead of reading a huge file into memory only for Discord to
# reject the upload.
MAX_ARTIFACT_BYTES = 25_000_000

# Instructions for the nested sandbox agent. The workspace is always empty,
# so the task text is its only input — it must behave self-sufficiently.
# It MUST explicitly name the shell tools: the SDK's default sandbox prompt
# (suppressed — see build_sandbox_agent) tells the model to use an
# apply_patch tool we do not expose, and a model that calls it aborts the
# whole run with ModelBehaviorError.
#
# {output_bullet} is filled in by build_sandbox_agent(out_dir): both
# variants explicitly rule out "/workspace" (left unguided, models default
# to that nonexistent path) while only ever naming the RELATIVE `out/`
# dirname — never the resolved absolute path (see SANDBOX_OUTPUT_DIRNAME
# comment above for why: exec_command's `workdir` argument rejects
# absolute paths outright).
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
- {output_bullet}
- Once you've run a check and confirmed a result is correct, stop — do not
  repeat the same or similar verification again on output you've already
  confirmed is correct.
- When done, end with a concise final report: what you did, the key results
  (exact numbers/outputs where they matter), and any failures or assumptions.
  The final report is all the caller sees, so make it self-contained."""

_OUTPUT_BULLET_RESOLVED = (
    "Do NOT assume `/workspace` exists — it does not in this sandbox. To "
    "return a FILE (a plot, a converted document, generated data, etc.), "
    "save it under the relative path `out/` from wherever you already are — "
    "that directory has already been created for you, no `cd` or `mkdir` "
    "needed. Never pass an absolute path (e.g. as exec_command's `workdir` "
    "argument) — only relative paths like `out/` are accepted; an absolute "
    "one will be rejected even though the directory is real. Anything saved "
    "under `out/` is sent back to the user automatically. Do not print file "
    "contents to stdout — only files under `out/` are returned; nothing "
    "else in the workspace persists."
)
_OUTPUT_BULLET_FALLBACK = (
    "Do NOT assume `/workspace` exists — it does not in this sandbox. To "
    "return a FILE (a plot, a converted document, generated data, etc.), "
    "save it under the `out/` directory relative to your working directory "
    "(create it first: `mkdir -p out`). Anything saved there is sent back to "
    "the user automatically. Do not print file contents to stdout — only "
    "files under `out/` are returned; nothing else in the workspace persists."
)


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


def build_sandbox_agent(out_dir: str | None) -> "object":
    """The nested SandboxAgent that does the work inside the sandbox.

    out_dir: whether the sandbox's real output path was already resolved
    and mkdir -p'd by the caller (see run_sandbox_task) — only used to pick
    between wording variants (both say "out/", never the absolute value:
    see SANDBOX_OUTPUT_DIRNAME comment for why). None means resolution
    failed and out/ hasn't been created yet, so the model is told to
    create it itself.

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

    output_bullet = _OUTPUT_BULLET_RESOLVED if out_dir is not None else _OUTPUT_BULLET_FALLBACK

    return SandboxAgent(
        name="Code Sandbox",
        model=OpenAIChatCompletionsModel(
            model=sandbox_model(),
            openai_client=AsyncOpenAI(
                base_url=sandbox_llm_host() + "/v1",
                api_key=sandbox_llm_api_key(),
            ),
        ),
        instructions=SANDBOX_INSTRUCTIONS.format(output_bullet=output_bullet),
        base_instructions="",
        capabilities=[Shell()],
        # Slightly cooler than the chat agent: code tasks want determinism.
        model_settings=ModelSettings(temperature=0.5),
    )


@dataclass
class SandboxArtifact:
    """One file the sandbox agent saved under SANDBOX_OUTPUT_DIR."""
    name: str
    data: bytes


@dataclass
class SandboxResult:
    """What run_sandbox_task returns: the agent's text report plus any
    files it saved under out/ (empty when it returned none).

    ok is False only when the run timed out; artifacts may still be
    populated in that case (best-effort recovery — see run_sandbox_task),
    and text is empty since Runner.run never returned a final report.
    Every other failure mode still raises instead of returning ok=False."""
    text: str
    artifacts: list[SandboxArtifact] = field(default_factory=list)
    ok: bool = True
    error: str | None = None


def build_sandbox_client() -> "object":
    """A fresh Docker sandbox client bound to the local daemon."""
    from docker import from_env as docker_from_env
    from agents.sandbox.sandboxes.docker import DockerSandboxClient

    return DockerSandboxClient(docker_from_env())


async def _create_sandbox_session(client) -> "object":
    """Creates (and starts) one fresh, empty-workspace container.

    We create the session ourselves — rather than letting RunConfig create
    one automatically — so we retain a live handle to it after Runner.run
    returns and can read files out of it before it is torn down. See
    build_sandbox_run_config for the other half of this.
    """
    from agents.sandbox.sandboxes.docker import DockerSandboxClientOptions

    return await client.create(options=DockerSandboxClientOptions(image=sandbox_image()))


def build_sandbox_run_config(client, session) -> RunConfig:
    """Run config that reuses an already-created container for the run.

    Passing a live `session` (rather than just `client`/`options`) makes
    the SDK treat it as caller-owned: RunConfig(sandbox=...) makes the SDK
    run the agent inside it, but it will NOT stop or delete the container
    afterward — that becomes our responsibility (see _delete_sandbox_session),
    which is what gives us the window to read artifacts out of it first.
    """
    from agents.sandbox import SandboxRunConfig

    return RunConfig(sandbox=SandboxRunConfig(client=client, session=session))


async def _delete_sandbox_session(client, session) -> None:
    """Tears down a container we own, tolerating our own cancellation.

    Wrapped in asyncio.shield so that if the caller of run_sandbox_task is
    itself cancelled (e.g. process shutdown), this delete keeps running in
    the background instead of being cancelled mid-teardown and orphaning
    the container; wait_for still bounds how long we wait for it here.
    """
    try:
        await asyncio.wait_for(
            asyncio.shield(client.delete(session)),
            timeout=SANDBOX_DELETE_TIMEOUT_SECONDS,
        )
    except Exception as e:
        print(f"Sandbox: failed to delete session/container: {e}")


def _parse_find_output(raw: str) -> list[tuple[str, int]]:
    """Parses `find OUT_DIR -maxdepth 3 -type f -printf '%s %p\\n'` output
    into (container_path, size_bytes) pairs. Pure: no Docker access."""
    entries: list[tuple[str, int]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        size_str, _, path = line.partition(" ")
        if not path:
            continue
        try:
            entries.append((path, int(size_str)))
        except ValueError:
            continue
    return entries


def _relative_artifact_name(container_path: str, out_dir: str) -> str:
    """The Discord-safe filename for a file found under out_dir: its path
    relative to out_dir, with subdirectory separators flattened (a Discord
    attachment name is a single path segment). Pure."""
    if container_path.startswith(out_dir):
        rel = container_path[len(out_dir):].lstrip("/")
    else:
        rel = container_path.lstrip("/")
    rel = rel.replace("/", "_")
    return rel or "artifact"


def _select_artifacts(
    listed: list[tuple[str, int]],
    *,
    max_files: int = MAX_ARTIFACT_FILES,
    max_bytes: int = MAX_ARTIFACT_BYTES,
) -> tuple[list[str], list[str]]:
    """Applies the file-count/size caps to files found under out/.

    listed: (container_path, size_bytes) pairs, in the order `find`
    reported them. Returns (paths_to_fetch, skip_notes). Pure: runs on
    sizes already known from `find`, never touches Docker — lets the caps
    be unit-tested without a real session.
    """
    to_fetch: list[str] = []
    skipped: list[str] = []
    total = 0
    for path, size in listed:
        if len(to_fetch) >= max_files:
            skipped.append(f"{path} (skipped: more than {max_files} output files)")
            continue
        if size > max_bytes:
            skipped.append(f"{path} (skipped: {size} bytes over the {max_bytes}-byte limit)")
            continue
        if total + size > max_bytes:
            skipped.append(f"{path} (skipped: would exceed the {max_bytes}-byte total limit)")
            continue
        to_fetch.append(path)
        total += size
    return to_fetch, skipped


async def _sandbox_output_dir(session) -> str | None:
    """The absolute path to SANDBOX_OUTPUT_DIRNAME under the sandbox's own
    current working directory — queried live via `pwd` rather than assumed,
    since the same session.exec() call is what exec_command uses under the
    hood, so this always matches wherever the model actually is. Returns
    None if the cwd can't be determined (treated as "no artifacts").
    """
    res = await session.exec("pwd", shell=False)
    if not res.ok():
        return None
    cwd = res.stdout.decode("utf-8", errors="replace").strip()
    return posixpath.join(cwd, SANDBOX_OUTPUT_DIRNAME) if cwd else None


async def _collect_artifacts(session, out_dir: str | None) -> list[SandboxArtifact]:
    """Reads back whatever the sandbox agent saved under out_dir.

    out_dir is resolved once by the caller (run_sandbox_task, via
    _sandbox_output_dir) up front — before Runner.run starts — rather than
    re-queried here, so the same path used to tell the model where to save
    files is also the one we look in afterward.

    Uses `find`+`exec("cat", ...)` directly — not session.read(), and not
    the exec_command/write_stdin tool path the model's own turns go
    through. session.read() is out because it validates paths against the
    SDK's nominal manifest root ("/workspace"), which the Shell-only
    capability we use never actually creates (see SANDBOX_OUTPUT_DIRNAME) —
    it rejects our real, cwd-relative paths as "not relative to the
    manifest". exec_command is out because it truncates large output to a
    token budget and would burn the (often small, local) sandbox model's
    own context on raw file bytes for no reason. Plain exec("cat", ...) is
    the same primitive session.read() uses internally, just without the
    manifest-relative check that doesn't apply to our setup.
    """
    if out_dir is None:
        return []
    res = await session.exec(
        "find", out_dir, "-maxdepth", "3", "-type", "f", "-printf", "%s %p\n",
        shell=False,
    )
    if not res.ok():
        return []  # a missing/empty out/ dir is not an error
    listed = _parse_find_output(res.stdout.decode("utf-8", errors="replace"))
    to_fetch, skipped = _select_artifacts(listed)
    for note in skipped:
        print(f"Sandbox: {note}")

    artifacts: list[SandboxArtifact] = []
    for path in to_fetch:
        try:
            read_result = await session.exec("cat", "--", path, shell=False)
        except Exception as e:
            print(f"Sandbox: failed to read output file {path}: {e}")
            continue
        if not read_result.ok():
            print(f"Sandbox: failed to read output file {path}: exit {read_result.exit_code}")
            continue
        artifacts.append(
            SandboxArtifact(name=_relative_artifact_name(path, out_dir), data=read_result.stdout)
        )
    return artifacts


async def run_sandbox_task(task: str, progress_hooks=None) -> SandboxResult:
    """Run one self-contained task in a fresh Docker sandbox.

    Returns a SandboxResult: the agent's final report (text) plus any
    files it saved under the output dir (artifacts, possibly empty).
    progress_hooks is an optional agents.RunHooks instance (e.g.
    classes.sandbox_progress.SandboxProgressHooks) attached to the nested
    run so every tool call (exec_command/write_stdin) and its output can be
    observed — e.g. to stream them to Discord.

    We create and own the container ourselves (rather than letting
    RunConfig(sandbox=...) manage it) so we can read artifacts out of it
    after the run finishes but before it is destroyed; the container is
    always deleted in `finally`, including on timeout or any other error.

    On timeout (the task outliving sandbox_timeout() seconds), we do NOT
    raise: the container is still alive at that point, and a task that ran
    out of time may already have produced and verified a good file, so we
    make a bounded, best-effort attempt to recover whatever is under the
    output dir before teardown and return SandboxResult(ok=False,
    error="timeout", artifacts=<recovered>) instead — recovery failures are
    swallowed, since it's strictly better-than-nothing on top of today's
    "discard everything" behavior. Every other error (e.g.
    ModelBehaviorError, MaxTurnsExceeded, a sandbox that's unavailable)
    still propagates to the caller unchanged: those are run/configuration
    failures with no verified-good artifact behind them.
    """
    client = build_sandbox_client()
    session = await _create_sandbox_session(client)
    try:
        out_dir = await _sandbox_output_dir(session)
        if out_dir is not None:
            await session.exec("mkdir", "-p", out_dir, shell=False)
        agent = build_sandbox_agent(out_dir)
        try:
            run_result = await asyncio.wait_for(
                Runner.run(
                    agent,
                    task,
                    max_turns=sandbox_max_turns(),
                    run_config=build_sandbox_run_config(client, session),
                    hooks=progress_hooks,
                ),
                timeout=sandbox_timeout(),
            )
        except asyncio.TimeoutError:
            print("Sandbox: task timed out, attempting best-effort artifact recovery")
            artifacts: list[SandboxArtifact] = []
            try:
                artifacts = await asyncio.wait_for(
                    _collect_artifacts(session, out_dir),
                    timeout=SANDBOX_RECOVERY_TIMEOUT_SECONDS,
                )
            except Exception as e:
                print(f"Sandbox: artifact recovery after timeout failed: {e}")
            return SandboxResult(text="", artifacts=artifacts, ok=False, error="timeout")
        final_output = run_result.final_output
        text = final_output if isinstance(final_output, str) else str(final_output)
        artifacts = await _collect_artifacts(session, out_dir)
        return SandboxResult(text=text, artifacts=artifacts)
    finally:
        await _delete_sandbox_session(client, session)
