"""The server workspace's settings and lifecycle rules (PURE: stdlib only).

Each server (guild) has one long-lived Docker container, its workspace, which
workspace_manager.py starts, snapshots and stops. This module is the part of
that with no I/O: the WORKSPACE_* settings, and a per-guild state machine
that says what the manager must do next.

    COLD --acquire--> STARTING --started--> WARM <-- release (refcount 0: CLEANING, then idle)
      ^                  | start_failed       | idle deadline / LRU eviction / reset
      |                  v                    v
      +---------------- COLD <----------- STOPPING

Rules worth knowing before changing them:

  * Many calls in one server use the workspace at once (a refcount), each in
    its own jobs/<id>/ folder. Nothing stops or resets a workspace while a
    call is using it: eviction picks an idle guild or makes the new acquire
    wait, and a reset asked for while busy runs when the last call ends.
  * When the last call ends the manager kills every process left in the
    container (CLEANING) before anyone else may use it, so a call never has
    its own commands killed by the cleanup of the one before.
  * A timed snapshot may run while calls are busy: persist_workspace() reads
    the workspace with tar while commands run (checked live, spike check 4).
    The manager serialises it with the cleanup and the stop.
  * `dirty` is set by every release and cleared by a snapshot, so an idle
    workspace nobody changed is never snapshotted twice.
"""
import os
from dataclasses import dataclass

# --- settings ---------------------------------------------------------------

DEFAULT_IMAGE = "python:3.14-slim"
DEFAULT_IDLE_SECONDS = 900
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 1800
DEFAULT_MAX_WARM = 4
DEFAULT_JOB_RETENTION_HOURS = 72
DEFAULT_PIDS_LIMIT = 512
DEFAULT_SNAPSHOT_TIMEOUT_SECONDS = 120
DEFAULT_INSTANCE = "dcb-dev"
TICK_SECONDS = 60

# The layout made on every start (idempotent). Paths are relative to the
# workspace root, which is how the model and exec_command's workdir see them.
SHARED_DIR = "shared"
JOBS_DIR = "jobs"
SKILLS_DIR = "skills"
LAYOUT = (SHARED_DIR, JOBS_DIR, SKILLS_DIR)


def _env_str(name: str, env=None) -> str:
    """A setting, stripped ("" when unset). env None = the process environment."""
    return str((os.environ if env is None else env).get(name, "")).strip()


def _env_int(name: str, default: int, minimum: int = 0, env=None) -> int:
    try:
        value = int(_env_str(name, env))
    except ValueError:
        return default
    return value if value >= minimum else default


def workspace_image(env=None) -> str:
    return _env_str("WORKSPACE_IMAGE", env) or DEFAULT_IMAGE


def idle_seconds(env=None) -> int:
    """Seconds a workspace stays warm after its last call (WORKSPACE_IDLE_SECONDS)."""
    return _env_int("WORKSPACE_IDLE_SECONDS", DEFAULT_IDLE_SECONDS, 1, env)


def snapshot_interval_seconds(env=None) -> int:
    """Timed snapshot of a warm, changed workspace (WORKSPACE_SNAPSHOT_INTERVAL_SECONDS);
    0 = only on idle stop."""
    return _env_int("WORKSPACE_SNAPSHOT_INTERVAL_SECONDS", DEFAULT_SNAPSHOT_INTERVAL_SECONDS, 0, env)


def max_warm(env=None) -> int:
    """Warm workspaces per core process (WORKSPACE_MAX_WARM)."""
    return _env_int("WORKSPACE_MAX_WARM", DEFAULT_MAX_WARM, 1, env)


def job_retention_hours(env=None) -> int:
    """jobs/<id>/ folders older than this are pruned (WORKSPACE_JOB_RETENTION_HOURS)."""
    return _env_int("WORKSPACE_JOB_RETENTION_HOURS", DEFAULT_JOB_RETENTION_HOURS, 1, env)


def pids_limit(env=None) -> int:
    """Processes per container (WORKSPACE_PIDS_LIMIT); 0 = no limit."""
    return _env_int("WORKSPACE_PIDS_LIMIT", DEFAULT_PIDS_LIMIT, 0, env)


def cpu_limit(env=None) -> float | None:
    """CPUs per container (WORKSPACE_CPU_LIMIT, e.g. 2 or 0.5); empty = no limit."""
    raw = _env_str("WORKSPACE_CPU_LIMIT", env)
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def snapshot_timeout_seconds(env=None) -> int:
    """Bound on one snapshot, and on all of them at shutdown (WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS)."""
    return _env_int("WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS", DEFAULT_SNAPSHOT_TIMEOUT_SECONDS, 1, env)


def instance(env=None) -> str:
    """Label value that keeps two bots on one Docker daemon from reaping each
    other's containers (WORKSPACE_INSTANCE: the chart sets the release name)."""
    return _env_str("WORKSPACE_INSTANCE", env) or DEFAULT_INSTANCE


def prewarm(env=None) -> bool:
    """WORKSPACE_PREWARM: start a server's workspace when it gets a message
    (off by default; phase 3 wires the trigger)."""
    return _env_str("WORKSPACE_PREWARM", env).lower() in {"1", "true", "yes", "on"}


def job_dir(job_id: str) -> str:
    """A call's folder, relative to the workspace root."""
    return f"{JOBS_DIR}/{job_id}"


def labels(guild_id, instance_name: str) -> dict[str, str]:
    return {"dcb.role": "workspace", "dcb.guild": str(guild_id), "dcb.instance": instance_name}


def snapshot_id(guild_id) -> str:
    return f"guild-{guild_id}"


# --- the state machine --------------------------------------------------------

COLD = "cold"
STARTING = "starting"
WARM = "warm"
STOPPING = "stopping"

# acquire() answers
USE = "use"        # the workspace is warm: the refcount is taken
START = "start"    # the caller starts it (the refcount is taken); then started() or start_failed()
WAIT = "wait"      # someone else is starting, cleaning, stopping or resetting it, or no room
EVICT = "evict"    # stop Decision.victim first (it is marked STOPPING already), then ask again

# due() actions
SNAPSHOT = "snapshot"
STOP = "stop"


@dataclass
class Decision:
    action: str
    victim: str | None = None


@dataclass
class GuildState:
    phase: str = COLD
    refcount: int = 0
    last_used: float = 0.0
    idle_since: float | None = None
    started_at: float | None = None
    last_snapshot: float | None = None
    dirty: bool = False
    cleaning: bool = False
    reset_pending: bool = False


class Lifecycle:
    """Per-guild state for one core process. Every method is synchronous and
    returns at once, so the manager can call it on the event loop without a
    lock: nothing here awaits."""

    def __init__(self, *, idle: float, interval: float, max_warm: int):
        self.idle = idle
        self.interval = interval
        self.max_warm = max_warm
        self.guilds: dict[str, GuildState] = {}

    def state(self, guild: str) -> GuildState:
        return self.guilds.setdefault(str(guild), GuildState())

    def warm_count(self) -> int:
        return sum(1 for s in self.guilds.values() if s.phase != COLD)

    def is_warm(self, guild: str) -> bool:
        s = self.guilds.get(str(guild))
        return s is not None and s.phase == WARM

    def _lru_idle(self, exclude: str) -> str | None:
        idle = [(s.last_used, g) for g, s in self.guilds.items()
                if g != exclude and s.phase == WARM and s.refcount == 0
                and not s.cleaning and not s.reset_pending]
        return min(idle)[1] if idle else None

    def acquire(self, guild: str, now: float) -> Decision:
        guild = str(guild)
        s = self.state(guild)
        if s.cleaning or s.reset_pending or s.phase in (STARTING, STOPPING):
            return Decision(WAIT)
        if s.phase == WARM:
            s.refcount += 1
            s.last_used = now
            s.idle_since = None
            return Decision(USE)
        if self.warm_count() >= self.max_warm:
            victim = self._lru_idle(exclude=guild)
            if victim is None:
                return Decision(WAIT)
            self.guilds[victim].phase = STOPPING
            return Decision(EVICT, victim)
        s.phase = STARTING
        s.refcount = 1
        s.last_used = now
        s.idle_since = None
        return Decision(START)

    def started(self, guild: str, now: float) -> None:
        s = self.state(guild)
        s.phase = WARM
        s.started_at = now

    def start_failed(self, guild: str) -> None:
        self.guilds[str(guild)] = GuildState()

    def release(self, guild: str, now: float) -> bool:
        """One call is done. True when it was the last one: the manager then
        cleans up leftover processes and calls cleaned()."""
        s = self.state(guild)
        if s.refcount <= 0:
            return False
        s.refcount -= 1
        s.dirty = True
        s.last_used = now
        if s.refcount == 0:
            s.cleaning = True
            return True
        return False

    def cleaned(self, guild: str, now: float) -> bool:
        """Cleanup done. True when a reset was waiting for it: the manager
        runs it now (the guild is marked STOPPING)."""
        s = self.state(guild)
        s.cleaning = False
        s.idle_since = now
        if s.reset_pending:
            s.reset_pending = False
            s.phase = STOPPING
            return True
        return False

    def request_reset(self, guild: str) -> str:
        """"now" (the guild is idle or cold; it is marked STOPPING) or "later"
        (calls are running; it runs when the last one ends)."""
        s = self.state(guild)
        if s.refcount > 0 or s.cleaning or s.phase == STARTING:
            s.reset_pending = True
            return "later"
        s.phase = STOPPING
        return "now"

    def begin_stop(self, guild: str) -> bool:
        s = self.state(guild)
        if s.phase != WARM or s.refcount or s.cleaning or s.reset_pending:
            return False
        s.phase = STOPPING
        return True

    def claim_cold(self, guild: str) -> bool:
        """Mark a COLD guild STOPPING so its leftover container (from an
        earlier process) can be snapshotted and removed without a call
        starting it meanwhile. False when it is in use here."""
        s = self.state(guild)
        if s.phase != COLD or s.refcount or s.reset_pending:
            return False
        s.phase = STOPPING
        return True

    def stopped(self, guild: str) -> None:
        self.guilds[str(guild)] = GuildState()

    def snapshotted(self, guild: str, now: float, ok: bool) -> None:
        s = self.state(guild)
        s.last_snapshot = now
        if ok:
            s.dirty = False

    def due(self, now: float) -> list[tuple[str, str]]:
        """(guild, SNAPSHOT | STOP) for every warm guild whose timer is up."""
        out = []
        for g, s in self.guilds.items():
            if s.phase != WARM or s.cleaning or s.reset_pending:
                continue
            if s.refcount == 0 and s.idle_since is not None and now - s.idle_since >= self.idle:
                out.append((g, STOP))
            elif self.interval > 0 and s.dirty:
                since = s.last_snapshot if s.last_snapshot is not None else s.started_at
                if since is not None and now - since >= self.interval:
                    out.append((g, SNAPSHOT))
        return out

    def warm_guilds(self) -> list[str]:
        return [g for g, s in self.guilds.items() if s.phase == WARM]
