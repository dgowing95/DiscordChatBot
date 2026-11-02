"""One long-lived Docker container per server: the server's workspace.

It replaces the throwaway container every run_code_sandbox call used to get.
A server's workspace survives between calls and core restarts, is snapshotted
to Redis on a timer and before it stops, stops after WORKSPACE_IDLE_SECONDS
unused, and gives each call its own folder (jobs/<id>/). The pure rules
(settings, the per-guild state machine) are in workspace_lifecycle.py; this
module does the Docker and Redis work they call for.

Getting a workspace (acquire), in order:

  1. warm in this process: use it ("cached");
  2. a record in Redis (dcb:workspace:v1:guild:{id}) with the SDK's
     serialised session state: DockerSandboxClient.resume() re-attaches to
     the container if this daemon still has it ("resumed"; a stopped one is
     started again), else makes a new one;
  3. otherwise a new container.

A NEW container (2's fallback, or 3) is filled from the guild's snapshot when
there is one ("snapshot"), else starts empty ("fresh"). A re-attached one never
is: its files are newer than any snapshot. That is why the session has no SDK
RemoteSnapshot (see workspace_snapshot_store.py for what one does on resume).

Checked live against the dev Docker daemon before this was written (phase 2
spike): resume re-attaches across a process restart; persist_workspace() works
while commands run; two Runner.run calls on one session in their own folders
do not mix; containers made by resume() go through the same
docker_client.containers.create the limits wrap; `kill -9 -1` in the container
kills everything but its init and leaves it usable.

Docker is required (main.py exits at startup without it, check_docker). A
daemon lost later is fail-soft: acquire raises WorkspaceUnavailable and the
caller tells the model the workspace is unavailable.

Labels on every container: dcb.role=workspace, dcb.guild, dcb.instance
(WORKSPACE_INSTANCE), so the reaper only ever removes its own containers and
two bots on one daemon never remove each other's.
"""
import asyncio
import contextlib
import json
import logging
import os
import posixpath
import time
from dataclasses import dataclass, field

from classes import metrics
from classes import workspace_lifecycle as wl
from classes.workspace_snapshot_store import SnapshotTooLargeError, WorkspaceSnapshotStore

logger = logging.getLogger(__name__)

RECORD_KEY = "dcb:workspace:v1:guild:{}"
DEFAULT_ROOT = "/workspace"
EXEC_TIMEOUT = 30
DELETE_TIMEOUT = 30
WAIT_SLICE = 5.0  # a missed wake-up only ever delays a waiting acquire this long
DOCKER_WAIT_SECONDS = 60


class WorkspaceUnavailable(Exception):
    """The server's workspace could not be got: Docker is down, or every
    warm workspace stayed busy for the caller's whole timeout."""


# --- Docker client and limits -------------------------------------------------

def memory_limit(env=None) -> int | None:
    """Memory cap in bytes for each workspace container (WORKSPACE_MEMORY_LIMIT,
    Docker's notation: "4g", "4gb", "512m" or plain bytes). Empty = no cap.

    The containers run on the node's Docker daemon, beside Kubernetes rather
    than inside it, so no pod limit covers them; one heavy task could take the
    RAM an LLM server on the same node was planned around. An unparseable
    value is logged and ignored (the chart refuses one at render time)."""
    from docker.utils import parse_bytes

    raw = wl._env_str("WORKSPACE_MEMORY_LIMIT", env)
    if not raw:
        return None
    try:
        limit = int(parse_bytes(raw))
    except Exception:
        limit = 0
    if limit <= 0:
        logger.error("WORKSPACE_MEMORY_LIMIT=%r is not a size like 4g or 512m; "
                     "workspaces run WITHOUT a memory cap", raw)
        return None
    return limit


def container_limits(env=None) -> dict:
    """The create() arguments for the WORKSPACE_* limits that are set."""
    limits = {}
    memory = memory_limit(env)
    if memory:
        limits["mem_limit"] = memory
        # The same figure for memory + swap, so swapping cannot dodge the cap.
        limits["memswap_limit"] = memory
    cpus = wl.cpu_limit(env)
    if cpus:
        limits["nano_cpus"] = int(cpus * 1_000_000_000)
    pids = wl.pids_limit(env)
    if pids:
        limits["pids_limit"] = pids
    return limits


class _LimitedContainers:
    """`docker_client.containers`, with the limits added to create()."""

    def __init__(self, containers, limits: dict):
        self._containers = containers
        self._limits = limits

    def create(self, *args, **kwargs):
        for key, value in self._limits.items():
            kwargs.setdefault(key, value)
        return self._containers.create(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._containers, name)


class _LimitedDockerClient:
    """A docker client that applies the limits to every container it creates.

    The SDK has no option for them: DockerSandboxClientOptions carries only
    image, ports, network mode and labels. Wrapping the client it is handed is
    the one seam that does not depend on SDK internals, and resume() creates
    its replacement containers through the same containers.create (spike
    check 7). Everything else passes straight through."""

    def __init__(self, client, limits: dict):
        self._client = client
        self.containers = _LimitedContainers(client.containers, limits)

    def __getattr__(self, name):
        return getattr(self._client, name)


def build_docker_client(env=None):
    from docker import from_env

    client = from_env()
    limits = container_limits(env)
    return _LimitedDockerClient(client, limits) if limits else client


def check_docker(*, wait_seconds: float = DOCKER_WAIT_SECONDS, from_env=None,
                 sleep=time.sleep, clock=time.monotonic, image: str | None = None) -> str | None:
    """None when the Docker daemon answers and the workspace image is there
    (pulled if missing), else the error to log before exiting. Retries the
    daemon for up to wait_seconds (one still starting); a missing socket
    fails instantly, so this does not stall a misconfigured start. Blocking:
    main.py calls it before the event loop starts."""
    if from_env is None:
        from docker import from_env
    image = image or wl.workspace_image()
    deadline = clock() + wait_seconds
    while True:
        try:
            client = from_env()
            client.ping()
            break
        except Exception as e:
            if clock() >= deadline:
                return (f"The server workspace needs Docker, and the Docker daemon could not be "
                        f"reached ({e}). Mount its socket into core (docker compose and the chart "
                        f"do) and make sure the daemon is running.")
            sleep(2)
    try:
        if not client.images.list(name=image):
            from docker.utils import parse_repository_tag
            logger.info(f"Pulling the workspace image {image}")
            repo, tag = parse_repository_tag(image)
            client.images.pull(repo, tag=tag or None)
    except Exception as e:
        return f"The workspace image {image} is missing and could not be pulled ({e})."
    return None


# --- one guild's workspace ----------------------------------------------------

@dataclass
class GuildWorkspace:
    """A warm workspace. Shared by every call in the guild: per-call details
    are on the Lease."""
    guild_id: str
    session: object
    sandbox_client: object
    root: str = DEFAULT_ROOT
    started_wall: float = field(default_factory=time.time)
    # Snapshot, process cleanup and stop never overlap (each runs commands
    # in the container that the others would break).
    ops: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def container_id(self) -> str:
        return self.session.state.container_id


@dataclass
class Lease:
    """What one call gets from acquire(): the shared workspace plus how it
    was made ready for this call."""
    workspace: GuildWorkspace
    kind: str
    restore_failed: bool = False

    @property
    def session(self):
        return self.workspace.session

    @property
    def sandbox_client(self):
        return self.workspace.sandbox_client

    @property
    def root(self) -> str:
        return self.workspace.root

    @property
    def guild_id(self) -> str:
        return self.workspace.guild_id

    def job_dir(self, job_id: str) -> str:
        """The call's folder relative to the workspace root (what the model
        and exec_command's workdir see)."""
        return wl.job_dir(job_id)

    def job_path(self, job_id: str) -> str:
        return posixpath.join(self.root, wl.job_dir(job_id))


class Workspaces:
    """Every server's workspace for this core process. One instance
    (workspaces()); tests build their own with fakes."""

    def __init__(self, *, sandbox_client=None, docker_client=None, store=None, redis=None,
                 clock=time.monotonic, env=None):
        self._env = env
        self._sandbox_client = sandbox_client
        self._docker = docker_client
        self._store = store
        self._redis = redis
        self._clock = clock
        self.lifecycle = wl.Lifecycle(
            idle=wl.idle_seconds(env), interval=wl.snapshot_interval_seconds(env),
            max_warm=wl.max_warm(env))
        self._live: dict[str, GuildWorkspace] = {}
        self._changed = asyncio.Condition()
        self._snapshot_failed: dict[str, bool] = {}

    # --- wiring -------------------------------------------------------------

    @property
    def docker(self):
        if self._docker is None:
            self._docker = build_docker_client(self._env)
        return self._docker

    @property
    def sandbox_client(self):
        if self._sandbox_client is None:
            from agents.sandbox.sandboxes.docker import DockerSandboxClient
            self._sandbox_client = DockerSandboxClient(self.docker)
        return self._sandbox_client

    @property
    def store(self):
        if self._store is None:
            self._store = WorkspaceSnapshotStore()
        return self._store

    @property
    def redis(self):
        if self._redis is None:
            from classes.redis_client import text_client
            self._redis = text_client()
        return self._redis

    def _instance(self) -> str:
        return wl.instance(self._env)

    def _gauge(self) -> None:
        metrics.set_workspaces_warm(self.lifecycle.warm_count())

    async def _notify(self) -> None:
        async with self._changed:
            self._changed.notify_all()

    # --- records ------------------------------------------------------------

    async def _load_record(self, guild: str) -> dict | None:
        try:
            raw = await self.redis.get(RECORD_KEY.format(guild))
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.warning(f"Workspace: could not read guild {guild}'s record: {e}")
            return None

    async def _save_record(self, guild: str, **changes) -> None:
        """Merges changes into the guild's record (best-effort)."""
        try:
            record = await self._load_record(guild) or {}
            record.update(changes, instance=self._instance(), updated_at=time.time())
            await self.redis.set(RECORD_KEY.format(guild), json.dumps(record))
        except Exception as e:
            logger.warning(f"Workspace: could not save guild {guild}'s record: {e}")

    async def _delete_record(self, guild: str) -> None:
        try:
            await self.redis.delete(RECORD_KEY.format(guild))
        except Exception as e:
            logger.warning(f"Workspace: could not delete guild {guild}'s record: {e}")

    async def _save_session(self, guild: str, session) -> None:
        await self._save_record(
            guild, container_id=session.state.container_id, image=wl.workspace_image(self._env),
            state=self.sandbox_client.serialize_session_state(session.state))

    # --- acquire / release ----------------------------------------------------

    def is_warm(self, guild_id) -> bool:
        return self.lifecycle.is_warm(str(guild_id))

    @contextlib.asynccontextmanager
    async def acquire(self, guild_id, *, timeout: float | None = None):
        """The guild's workspace for one call, as a Lease. Waits (up to
        timeout) while it starts, stops or resets, or while every warm slot
        is busy. Raises WorkspaceUnavailable."""
        lease = await self._acquire(str(guild_id), timeout)
        try:
            yield lease
        finally:
            await self._release(str(guild_id))

    async def _acquire(self, guild: str, timeout: float | None) -> Lease:
        # Real time, not self._clock: this bounds an actual wait, while the
        # injected clock drives the lifecycle's timers (frozen in tests).
        deadline = None if timeout is None else time.monotonic() + timeout
        started = self._clock()
        while True:
            decision = self.lifecycle.acquire(guild, self._clock())
            if decision.action == wl.USE:
                metrics.observe_workspace_start("cached", self._clock() - started)
                return Lease(self._live[guild], "cached")
            if decision.action == wl.START:
                try:
                    workspace, kind, restore_failed = await self._start(guild)
                except BaseException as e:
                    self.lifecycle.start_failed(guild)
                    await self._notify()
                    if isinstance(e, Exception):
                        logger.warning(f"Workspace: could not start guild {guild}'s workspace: {e}")
                        raise WorkspaceUnavailable(str(e)) from e
                    raise
                self._live[guild] = workspace
                self.lifecycle.started(guild, self._clock())
                self._gauge()
                await self._notify()
                metrics.observe_workspace_start(kind, self._clock() - started)
                logger.info(f"Workspace: guild {guild} ready ({kind})")
                return Lease(workspace, kind, restore_failed)
            if decision.action == wl.EVICT:
                logger.info(f"Workspace: stopping guild {decision.victim}'s idle workspace "
                            f"to make room for guild {guild}")
                await self._stop(decision.victim)
                continue
            # WAIT
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise WorkspaceUnavailable("every workspace stayed busy")
            wait = WAIT_SLICE if remaining is None else min(WAIT_SLICE, remaining)
            async with self._changed:
                try:
                    await asyncio.wait_for(self._changed.wait(), wait)
                except asyncio.TimeoutError:
                    pass

    async def _release(self, guild: str) -> None:
        if not self.lifecycle.release(guild, self._clock()):
            return
        workspace = self._live.get(guild)
        try:
            if workspace is not None:
                async with workspace.ops:
                    await self._kill_leftovers(workspace)
        finally:
            reset = self.lifecycle.cleaned(guild, self._clock())
            await self._notify()
        await self._save_record(guild)  # updated_at = last used
        if reset:
            await self._wipe(guild)

    async def _kill_leftovers(self, workspace: GuildWorkspace) -> None:
        """Ends everything a call left running (a server, a sleep loop), so
        it does not hold CPU or memory until the idle stop. The container's
        init (`tail -f /dev/null`, PID 1) is never signalled by kill -1."""
        try:
            with contextlib.suppress(Exception):
                await workspace.session.pty_terminate_all()
            await workspace.session.exec("kill -9 -1 2>/dev/null; true", timeout=EXEC_TIMEOUT)
        except Exception as e:
            logger.warning(f"Workspace: could not stop leftover processes in guild "
                           f"{workspace.guild_id}'s workspace: {e}")

    # --- starting ------------------------------------------------------------

    async def _start(self, guild: str) -> tuple[GuildWorkspace, str, bool]:
        client = self.sandbox_client
        image = wl.workspace_image(self._env)
        record = await self._load_record(guild)
        session, kind = None, None
        if record and record.get("instance") == self._instance() and record.get("state"):
            if record.get("image") == image:
                try:
                    session = await client.resume(client.deserialize_session_state(record["state"]))
                    await session.start()
                    if session.state.container_id == record.get("container_id"):
                        kind = "resumed"
                except Exception as e:
                    logger.warning(f"Workspace: could not resume guild {guild}'s container, "
                                   f"making a new one: {e}")
                    session = None
            else:
                # WORKSPACE_IMAGE changed: save the old container's files,
                # then move them to a container of the new image.
                await self._retire(guild, record)
        if session is None:
            session = await self._create(guild)
        restore_failed = False
        if kind is None:
            kind, session, restore_failed = await self._fill(guild, session)
        root = await self._root(session)
        workspace = GuildWorkspace(guild, session, client, root)
        await self._exec(session, "mkdir -p " + " ".join(wl.LAYOUT), root)
        await self._save_session(guild, session)
        return workspace, kind, restore_failed

    async def _create(self, guild: str):
        from agents.sandbox.sandboxes.docker import DockerSandboxClientOptions

        options = DockerSandboxClientOptions(
            image=wl.workspace_image(self._env), labels=wl.labels(guild, self._instance()))
        session = await self.sandbox_client.create(options=options, snapshot=None)
        await session.start()
        return session

    async def _fill(self, guild: str, session):
        """A NEW container: the guild's snapshot goes in, if it has one. A
        snapshot that cannot be restored is dropped (it would otherwise fail
        every start until its TTL ran out) and the guild starts empty in a
        clean container, since a half-unpacked tar may be left behind."""
        sid = wl.snapshot_id(guild)
        try:
            if not await self.store.exists(sid):
                return "fresh", session, False
        except Exception as e:
            logger.warning(f"Workspace: could not check guild {guild}'s snapshot: {e}")
            return "fresh", session, False
        try:
            await session.hydrate_workspace(await self.store.download(sid))
            return "snapshot", session, False
        except Exception as e:
            logger.warning(f"Workspace: guild {guild}'s snapshot could not be restored, "
                           f"starting empty: {e}")
        with contextlib.suppress(Exception):
            await self.store.delete(sid)
        await self._delete_container(session)
        return "fresh", await self._create(guild), True

    async def _root(self, session) -> str:
        try:
            result = await session.exec("pwd", shell=False, timeout=EXEC_TIMEOUT)
            root = result.stdout.decode("utf-8", errors="replace").strip()
            if result.ok() and root.startswith("/"):
                return root
        except Exception:
            pass
        return DEFAULT_ROOT

    async def _exec(self, session, command: str, root: str):
        return await session.exec(f"cd {root} && {command}", timeout=EXEC_TIMEOUT)

    # --- snapshot / stop / reset ------------------------------------------

    async def snapshot(self, guild_id) -> bool:
        """Saves a warm guild's workspace now. True when it was saved."""
        guild = str(guild_id)
        workspace = self._live.get(guild)
        if workspace is None:
            return False
        async with workspace.ops:
            return await self._snapshot_locked(workspace)

    async def _snapshot_locked(self, workspace: GuildWorkspace) -> bool:
        guild = workspace.guild_id
        started = self._clock()
        ok, reason = False, None
        try:
            ok = await asyncio.wait_for(self._save_snapshot(workspace),
                                        wl.snapshot_timeout_seconds(self._env))
        except SnapshotTooLargeError as e:
            reason = "too_large"
            logger.warning(f"Workspace: {e}; keeping the previous snapshot")
            await self._save_record(guild, snapshot_too_large=True)
        except asyncio.TimeoutError:
            reason = "timeout"
            logger.warning(f"Workspace: snapshot of guild {guild} timed out")
        except Exception as e:
            reason = "error"
            logger.warning(f"Workspace: snapshot of guild {guild} failed: {e}")
        if reason:
            metrics.inc_workspace_snapshot_failure(reason)
        self.lifecycle.snapshotted(guild, self._clock(), ok)
        logger.info(f"Workspace: guild {guild} snapshot {'saved' if ok else 'not saved'} "
                    f"in {self._clock() - started:.1f}s")
        return ok

    async def _save_snapshot(self, workspace: GuildWorkspace) -> bool:
        # Pruned in the container first: persist_workspace() cannot leave
        # paths out. Old job folders go for good (WORKSPACE_JOB_RETENTION_HOURS),
        # caches are rebuilt on demand.
        minutes = wl.job_retention_hours(self._env) * 60
        await self._exec(
            workspace.session,
            f"find {wl.JOBS_DIR} -mindepth 1 -maxdepth 1 -mmin +{minutes} -exec rm -rf {{}} + ; "
            "find . -name __pycache__ -type d -prune -exec rm -rf {} + ; "
            "rm -rf .cache ; true",
            workspace.root)
        started = self._clock()
        data = await workspace.session.persist_workspace()
        size = await self.store.upload(wl.snapshot_id(workspace.guild_id), data)
        metrics.observe_workspace_snapshot(self._clock() - started, size)
        await self._save_record(workspace.guild_id, last_snapshot_at=time.time(),
                                snapshot_bytes=size, snapshot_too_large=False)
        return True

    async def stop(self, guild_id, *, snapshot: bool = True) -> bool:
        """Snapshot (when changed) and stop an idle guild's workspace. False
        when it is busy or not warm."""
        guild = str(guild_id)
        if not self.lifecycle.begin_stop(guild):
            return False
        await self._stop(guild, snapshot=snapshot)
        return True

    async def _stop(self, guild: str, *, snapshot: bool = True) -> None:
        """The guild is already STOPPING in the lifecycle."""
        workspace = self._live.pop(guild, None)
        try:
            if workspace is not None:
                async with workspace.ops:
                    state = self.lifecycle.state(guild)
                    saved = True
                    if snapshot and state.dirty:
                        saved = await self._snapshot_locked(workspace)
                    await self._put_away(guild, workspace.session, saved)
        finally:
            self.lifecycle.stopped(guild)
            self._gauge()
            await self._notify()
        logger.info(f"Workspace: guild {guild} stopped")

    async def _put_away(self, guild: str, session, saved: bool) -> None:
        """Removes a container whose files are safely in the snapshot. One
        whose snapshot failed (too large, say) is only stopped and its record
        kept: removing it would lose everything since the last snapshot. The
        next call resumes it, files and all; the reaper leaves stopped
        containers alone, and /workspace reset removes it."""
        if saved:
            await self._delete_container(session)
            await self._save_record(guild, container_id=None, state=None)
            return
        logger.warning(f"Workspace: guild {guild}'s workspace could not be saved, so its "
                       f"container is stopped and kept rather than removed")

        def _stop():
            self.docker.containers.get(session.state.container_id).stop(timeout=5)
        try:
            await asyncio.to_thread(_stop)
        except Exception as e:
            logger.warning(f"Workspace: could not stop guild {guild}'s container: {e}")

    async def _delete_container(self, session) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(self.sandbox_client.delete(session)),
                                   DELETE_TIMEOUT)
        except Exception as e:
            logger.warning(f"Workspace: could not remove a container: {e}")

    async def reset(self, guild_id) -> str:
        """Wipes the guild's workspace: container, record and snapshot.
        "now", or "later" when calls are running (it happens when the last
        one ends; new calls wait for it)."""
        guild = str(guild_id)
        when = self.lifecycle.request_reset(guild)
        if when == "now":
            await self._wipe(guild)
        return when

    async def _wipe(self, guild: str) -> None:
        """The guild is STOPPING in the lifecycle."""
        try:
            workspace = self._live.pop(guild, None)
            if workspace is not None:
                async with workspace.ops:
                    await self._delete_container(workspace.session)
            record = await self._load_record(guild)
            if record and record.get("container_id"):
                await self._remove_container_id(record["container_id"])
            await self._delete_record(guild)
            with contextlib.suppress(Exception):
                await self.store.delete(wl.snapshot_id(guild))
        finally:
            self.lifecycle.stopped(guild)
            self._gauge()
            await self._notify()
        logger.info(f"Workspace: guild {guild} reset")

    async def _remove_container_id(self, container_id: str) -> bool:
        def _remove():
            try:
                self.docker.containers.get(container_id).remove(force=True)
                return True
            except Exception:
                return False
        return await asyncio.to_thread(_remove)

    async def _retire(self, guild: str, record: dict) -> None:
        """Snapshot and remove a container this process is not using (left
        by an earlier process, or on the old image). Best-effort."""
        try:
            client = self.sandbox_client
            session = await client.resume(client.deserialize_session_state(record["state"]))
            await session.start()
            if session.state.container_id != record.get("container_id"):
                # Gone already: resume made a new, empty one. Nothing to save.
                await self._delete_container(session)
                await self._save_record(guild, container_id=None, state=None)
                return
            workspace = GuildWorkspace(guild, session, client, await self._root(session))
            async with workspace.ops:
                saved = await self._snapshot_locked(workspace)
                await self._put_away(guild, session, saved)
        except Exception as e:
            logger.warning(f"Workspace: could not retire guild {guild}'s old container: {e}")

    # --- background --------------------------------------------------------

    async def reap(self) -> int:
        """Removes our labelled containers that no record names (a stale
        process's leftovers), and retires ones whose record says they have
        been unused past WORKSPACE_IDLE_SECONDS while nothing here holds them
        (core restarted and nobody came back). Returns how many it removed."""
        instance = self._instance()

        def _list():
            return self.docker.containers.list(
                all=True, filters={"label": ["dcb.role=workspace", f"dcb.instance={instance}"]})

        try:
            containers = await asyncio.to_thread(_list)
        except Exception as e:
            logger.warning(f"Workspace: could not list containers to reap: {e}")
            return 0
        removed = 0
        for container in containers:
            guild = (getattr(container, "labels", None) or {}).get("dcb.guild")
            live = self._live.get(guild) if guild else None
            if live is not None and live.container_id == container.id:
                continue
            record = await self._load_record(guild) if guild else None
            if not record or record.get("container_id") != container.id:
                if await self._remove_container_id(container.id):
                    removed += 1
                continue
            idle_for = time.time() - float(record.get("updated_at") or 0)
            # Running ones only: a stopped one is a workspace whose snapshot
            # failed (_put_away), kept for its next call.
            running = getattr(container, "status", "running") == "running"
            if (live is None and running and idle_for > self.lifecycle.idle
                    and self.lifecycle.claim_cold(guild)):
                try:
                    await self._retire(guild, record)
                    removed += 1
                finally:
                    self.lifecycle.stopped(guild)
                    await self._notify()
        if removed:
            metrics.inc_workspace_reaped(removed)
            logger.info(f"Workspace: removed {removed} leftover container(s)")
        return removed

    async def tick(self) -> None:
        """Timed snapshots and idle stops that are due, then the reaper."""
        for guild, action in self.lifecycle.due(self._clock()):
            try:
                if action == wl.STOP:
                    await self.stop(guild)
                else:
                    await self.snapshot(guild)
            except Exception as e:
                logger.warning(f"Workspace: {action} of guild {guild} failed: {e}")
        await self.reap()

    async def run_forever(self) -> None:
        """The 60 s loop main.py starts once the bot is ready."""
        while True:
            try:
                await self.tick()
            except Exception as e:
                logger.warning(f"Workspace: background tick failed: {e}")
            await asyncio.sleep(wl.TICK_SECONDS)

    async def shutdown(self) -> None:
        """Core is closing: snapshot every changed warm workspace, all within
        WORKSPACE_SNAPSHOT_TIMEOUT_SECONDS, and leave the containers running
        for the next process to resume (the reaper cleans up if none does)."""
        async def _all():
            for guild in list(self.lifecycle.warm_guilds()):
                if self.lifecycle.state(guild).dirty:
                    await self.snapshot(guild)
        try:
            await asyncio.wait_for(_all(), wl.snapshot_timeout_seconds(self._env))
        except Exception as e:
            logger.warning(f"Workspace: snapshots at shutdown did not all finish: {e}")

    # --- /workspace status -------------------------------------------------

    async def status(self, guild_id) -> dict:
        guild = str(guild_id)
        record = await self._load_record(guild) or {}
        workspace = self._live.get(guild)
        state = self.lifecycle.state(guild)
        jobs = None
        if workspace is not None:
            try:
                result = await self._exec(workspace.session,
                                          f"ls -1 {wl.JOBS_DIR} 2>/dev/null | wc -l", workspace.root)
                jobs = int(result.stdout.decode().strip() or 0)
            except Exception:
                jobs = None
        try:
            size = await self.store.size(wl.snapshot_id(guild))
        except Exception:
            size = None
        return {
            "warm": workspace is not None,
            "busy": state.refcount,
            "started_at": workspace.started_wall if workspace is not None else None,
            "last_snapshot_at": record.get("last_snapshot_at"),
            "snapshot_bytes": size,
            "snapshot_too_large": bool(record.get("snapshot_too_large")),
            "jobs": jobs,
        }


_WORKSPACES: Workspaces | None = None


def workspaces() -> Workspaces:
    """The process's one Workspaces."""
    global _WORKSPACES
    if _WORKSPACES is None:
        _WORKSPACES = Workspaces()
    return _WORKSPACES
