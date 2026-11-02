"""The server workspace manager (classes/workspace_manager.py), on fakes of the
SDK's Docker client, the Docker daemon, Redis and the snapshot store.

What the fakes stand for was checked live (the phase 2 container spike):
resume() re-attaches to a container the daemon still has, else makes a new
empty one; a stopped container is started again by resume + start.

Run from the repo root: `pytest core/tests/workspace_manager_tests.py`
"""
import asyncio
import io
import json
import itertools
from types import SimpleNamespace

import pytest

from classes import workspace_manager as wm
from classes.workspace_manager import Workspaces, WorkspaceUnavailable
from classes.workspace_snapshot_store import SnapshotTooLargeError

_ids = itertools.count(1)


class FakeResult:
    def __init__(self, stdout=b"", code=0):
        self.stdout, self.exit_code = stdout, code

    def ok(self):
        return self.exit_code == 0


class FakeState:
    def __init__(self, container_id, labels=None, image="python:3.14-slim"):
        self.container_id, self.labels, self.image = container_id, labels or {}, image


class FakeSession:
    def __init__(self, daemon, state):
        self.daemon, self.state = daemon, state
        self.commands, self.hydrated, self.started = [], [], False

    async def start(self):
        self.started = True
        self.daemon.containers[self.state.container_id]["status"] = "running"

    async def exec(self, *command, shell=True, timeout=None):
        cmd = " ".join(str(c) for c in command)
        self.commands.append(cmd)
        self.daemon.log.append((self.state.container_id, cmd))
        if cmd == "pwd":
            return FakeResult(b"/workspace\n")
        if "wc -l" in cmd:
            return FakeResult(b"3\n")
        return FakeResult()

    async def pty_terminate_all(self):
        pass

    async def persist_workspace(self):
        if self.daemon.persist_error:
            raise self.daemon.persist_error
        return io.BytesIO(f"tar-of-{self.state.container_id}".encode())

    async def hydrate_workspace(self, data):
        raw = data.read()
        if raw == b"corrupt":
            raise RuntimeError("bad tar")
        self.hydrated.append(raw)


class FakeSandboxClient:
    """DockerSandboxClient's surface the manager uses."""

    def __init__(self, daemon):
        self.daemon = daemon
        self.created, self.resumed, self.deleted = [], [], []

    def _new(self, labels, image):
        cid = f"c{next(_ids)}"
        self.daemon.containers[cid] = {"labels": dict(labels), "status": "running", "image": image}
        self.created.append(cid)
        return FakeSession(self.daemon, FakeState(cid, labels, image))

    async def create(self, *, options, snapshot=None):
        assert snapshot is None  # the manager owns restore (see workspace_snapshot_store.py)
        if self.daemon.down:
            raise RuntimeError("docker is down")
        return self._new(options.labels, options.image)

    async def resume(self, state):
        self.resumed.append(state.container_id)
        if state.container_id in self.daemon.containers:
            return FakeSession(self.daemon, state)
        return self._new(state.labels, state.image)

    async def delete(self, session):
        self.deleted.append(session.state.container_id)
        self.daemon.containers.pop(session.state.container_id, None)

    def serialize_session_state(self, state):
        return {"container_id": state.container_id, "labels": state.labels, "image": state.image}

    def deserialize_session_state(self, payload):
        return FakeState(payload["container_id"], payload["labels"], payload["image"])


class FakeDaemon:
    """The docker SDK client: containers.list/get, as the reaper and stop use them."""

    def __init__(self):
        self.containers, self.log = {}, []
        self.down = False
        self.persist_error = None
        self.list_filters = None
        self.containers_api = SimpleNamespace(list=self._list, get=self._get)

    @property
    def docker(self):
        return SimpleNamespace(containers=self.containers_api)

    def _list(self, filters=None, **_):
        self.list_filters = filters
        wanted = dict(f.split("=", 1) for f in filters["label"])
        out = []
        for cid, c in self.containers.items():
            if all(c["labels"].get(k) == v for k, v in wanted.items()) if False else \
                    all(c["labels"].get(k) == v for k, v in wanted.items()):
                out.append(SimpleNamespace(id=cid, labels=c["labels"], status=c["status"]))
        return out

    def _get(self, cid):
        if cid not in self.containers:
            raise LookupError(cid)
        daemon = self

        def remove(force=False):
            daemon.containers.pop(cid, None)

        def stop(timeout=None):
            daemon.containers[cid]["status"] = "exited"
        return SimpleNamespace(id=cid, remove=remove, stop=stop,
                               status=daemon.containers[cid]["status"])


class FakeStore:
    def __init__(self):
        self.data, self.too_large = {}, False

    async def upload(self, sid, data):
        raw = data.read()
        if self.too_large:
            raise SnapshotTooLargeError("too big")
        self.data[sid] = raw
        return len(raw)

    async def download(self, sid):
        return io.BytesIO(self.data.get(sid, b""))

    async def exists(self, sid):
        return sid in self.data

    async def delete(self, sid):
        self.data.pop(sid, None)

    async def size(self, sid):
        return len(self.data[sid]) if sid in self.data else None


class FakeRedis:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value):
        self.data[key] = value

    async def delete(self, key):
        self.data.pop(key, None)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class Env:
    def __init__(self, **env):
        self.daemon = FakeDaemon()
        self.client = FakeSandboxClient(self.daemon)
        self.store = FakeStore()
        self.redis = FakeRedis()
        self.clock = Clock()
        self.env = {"WORKSPACE_IDLE_SECONDS": "100", "WORKSPACE_SNAPSHOT_INTERVAL_SECONDS": "0",
                    "WORKSPACE_MAX_WARM": "2", "WORKSPACE_INSTANCE": "test", **env}

    def manager(self):
        """A new Workspaces sharing this daemon, Redis and store: a new core process."""
        return Workspaces(sandbox_client=self.client, docker_client=self.daemon.docker,
                          store=self.store, redis=self.redis, clock=self.clock, env=self.env)

    def record(self, guild="1"):
        raw = self.redis.data.get(f"dcb:workspace:v1:guild:{guild}")
        return json.loads(raw) if raw else None


async def _use(manager, guild="1", **kw):
    async with manager.acquire(guild, **kw) as lease:
        return lease


# ---------------------- getting a workspace ----------------------

@pytest.mark.asyncio
async def test_first_call_makes_a_fresh_container_with_the_layout_and_a_record():
    env = Env()
    lease = await _use(env.manager())
    assert lease.kind == "fresh"
    assert lease.root == "/workspace"
    assert lease.job_dir("t9") == "jobs/t9" and lease.job_path("t9") == "/workspace/jobs/t9"
    assert any("mkdir -p shared jobs skills" in c for c in lease.session.commands)
    record = env.record()
    assert record["container_id"] == lease.session.state.container_id
    assert record["instance"] == "test" and record["image"] == "python:3.14-slim"
    labels = env.daemon.containers[record["container_id"]]["labels"]
    assert labels == {"dcb.role": "workspace", "dcb.guild": "1", "dcb.instance": "test"}


@pytest.mark.asyncio
async def test_a_second_call_uses_the_warm_one():
    env = Env()
    manager = env.manager()
    first = await _use(manager)
    second = await _use(manager)
    assert second.kind == "cached"
    assert second.session is first.session
    assert len(env.client.created) == 1


@pytest.mark.asyncio
async def test_two_calls_at_once_share_one_container():
    env = Env()
    manager = env.manager()
    async with manager.acquire("1") as a, manager.acquire("1") as b:
        assert a.session is b.session
        assert manager.lifecycle.state("1").refcount == 2
    assert len(env.client.created) == 1


@pytest.mark.asyncio
async def test_a_new_process_re_attaches_to_the_running_container():
    env = Env()
    first = await _use(env.manager())
    lease = await _use(env.manager())
    assert lease.kind == "resumed"
    assert lease.session.state.container_id == first.session.state.container_id
    assert lease.session.hydrated == []  # its files are newer than any snapshot


@pytest.mark.asyncio
async def test_a_gone_container_is_replaced_and_filled_from_the_snapshot():
    env = Env()
    first = await _use(env.manager())
    env.store.data["guild-1"] = b"saved"
    env.daemon.containers.pop(first.session.state.container_id)
    lease = await _use(env.manager())
    assert lease.kind == "snapshot"
    assert lease.session.hydrated == [b"saved"]
    assert env.record()["container_id"] == lease.session.state.container_id


@pytest.mark.asyncio
async def test_no_record_but_a_snapshot_fills_a_new_container():
    env = Env()
    env.store.data["guild-1"] = b"saved"
    lease = await _use(env.manager())
    assert lease.kind == "snapshot" and lease.session.hydrated == [b"saved"]


@pytest.mark.asyncio
async def test_a_corrupt_snapshot_is_dropped_and_the_guild_starts_clean():
    env = Env()
    env.store.data["guild-1"] = b"corrupt"
    lease = await _use(env.manager())
    assert lease.kind == "fresh" and lease.restore_failed is True
    assert "guild-1" not in env.store.data
    assert len(env.client.created) == 2  # the half-filled container was replaced
    assert env.client.deleted == [env.client.created[0]]


@pytest.mark.asyncio
async def test_a_record_from_another_instance_is_not_resumed():
    env = Env()
    await _use(env.manager())
    env.env["WORKSPACE_INSTANCE"] = "other"
    lease = await _use(env.manager())
    assert lease.kind == "fresh"


@pytest.mark.asyncio
async def test_a_changed_image_moves_the_files_to_a_new_container():
    env = Env()
    first = await _use(env.manager())
    old = first.session.state.container_id
    env.env["WORKSPACE_IMAGE"] = "python:3.15-slim"
    lease = await _use(env.manager())
    assert old not in env.daemon.containers
    assert lease.kind == "snapshot"
    assert lease.session.hydrated == [f"tar-of-{old}".encode()]
    assert env.daemon.containers[lease.session.state.container_id]["image"] == "python:3.15-slim"


@pytest.mark.asyncio
async def test_docker_down_is_workspace_unavailable_and_the_next_call_retries():
    env = Env()
    manager = env.manager()
    env.daemon.down = True
    with pytest.raises(WorkspaceUnavailable):
        await _use(manager)
    assert manager.lifecycle.state("1").phase == "cold"
    env.daemon.down = False
    assert (await _use(manager)).kind == "fresh"


# ---------------------- release, idle stop, snapshots ----------------------

@pytest.mark.asyncio
async def test_the_last_call_ending_kills_leftover_processes():
    env = Env()
    manager = env.manager()
    async with manager.acquire("1") as a:
        async with manager.acquire("1"):
            pass
        assert not any("kill -9 -1" in c for c in a.session.commands)
    assert any("kill -9 -1" in c for c in a.session.commands)


@pytest.mark.asyncio
async def test_idle_stop_snapshots_then_removes_the_container():
    env = Env()
    manager = env.manager()
    lease = await _use(manager)
    cid = lease.session.state.container_id
    env.clock.now += 50
    await manager.tick()
    assert cid in env.daemon.containers
    env.clock.now += 100
    await manager.tick()
    assert cid not in env.daemon.containers
    assert env.store.data["guild-1"] == f"tar-of-{cid}".encode()
    assert env.record()["container_id"] is None and env.record()["snapshot_bytes"] > 0
    assert not manager.is_warm("1")
    # The next call starts a new container from the snapshot.
    lease = await _use(manager)
    assert lease.kind == "snapshot"


@pytest.mark.asyncio
async def test_a_snapshot_prunes_old_job_folders_and_caches_first():
    env = Env(WORKSPACE_JOB_RETENTION_HOURS="2")
    manager = env.manager()
    lease = await _use(manager)
    assert await manager.snapshot("1") is True
    prune = [c for c in lease.session.commands if "find jobs" in c]
    assert prune and "-mmin +120" in prune[0] and "__pycache__" in prune[0]


@pytest.mark.asyncio
async def test_a_too_large_snapshot_keeps_the_container_stopped_not_removed():
    env = Env()
    manager = env.manager()
    lease = await _use(manager)
    cid = lease.session.state.container_id
    env.store.too_large = True
    env.clock.now += 200
    await manager.tick()
    assert env.daemon.containers[cid]["status"] == "exited"
    assert env.record()["container_id"] == cid
    assert env.record()["snapshot_too_large"] is True
    # The reaper leaves a stopped container its record names alone...
    env.clock.now += 10_000
    await manager.tick()
    assert cid in env.daemon.containers
    # ...and the next call resumes it, files and all.
    lease = await _use(manager)
    assert lease.kind == "resumed" and lease.session.state.container_id == cid


@pytest.mark.asyncio
async def test_timed_snapshot_of_a_changed_warm_workspace():
    env = Env(WORKSPACE_SNAPSHOT_INTERVAL_SECONDS="30", WORKSPACE_IDLE_SECONDS="10000")
    manager = env.manager()
    await _use(manager)
    env.clock.now += 31
    await manager.tick()
    assert "guild-1" in env.store.data
    assert manager.is_warm("1")


@pytest.mark.asyncio
async def test_a_failed_snapshot_is_counted_and_does_not_raise():
    env = Env()
    manager = env.manager()
    await _use(manager)
    env.daemon.persist_error = RuntimeError("tar failed")
    assert await manager.snapshot("1") is False


# ---------------------- eviction ----------------------

@pytest.mark.asyncio
async def test_a_new_guild_stops_the_least_recently_used_idle_one():
    env = Env(WORKSPACE_MAX_WARM="1")
    manager = env.manager()
    a = await _use(manager, "a")
    lease = await _use(manager, "b")
    assert lease.kind == "fresh"
    assert a.session.state.container_id not in env.daemon.containers
    assert "guild-a" in env.store.data
    assert not manager.is_warm("a") and manager.is_warm("b")


@pytest.mark.asyncio
async def test_eviction_never_stops_a_busy_workspace_and_the_new_call_times_out():
    env = Env(WORKSPACE_MAX_WARM="1")
    manager = env.manager()
    async with manager.acquire("a") as a:
        with pytest.raises(WorkspaceUnavailable):
            await _use(manager, "b", timeout=0.01)
        assert a.session.state.container_id in env.daemon.containers


@pytest.mark.asyncio
async def test_a_waiting_call_gets_the_slot_when_the_busy_one_ends():
    env = Env(WORKSPACE_MAX_WARM="1")
    manager = env.manager()
    release = asyncio.Event()

    async def busy():
        async with manager.acquire("a"):
            await release.wait()

    task = asyncio.create_task(busy())
    await asyncio.sleep(0)
    waiter = asyncio.create_task(_use(manager, "b", timeout=5))
    await asyncio.sleep(0.01)
    assert not waiter.done()
    release.set()
    lease = await waiter
    await task
    assert lease.kind == "fresh"


# ---------------------- reset ----------------------

@pytest.mark.asyncio
async def test_reset_of_an_idle_workspace_wipes_container_record_and_snapshot():
    env = Env()
    manager = env.manager()
    lease = await _use(manager)
    await manager.snapshot("1")
    assert await manager.reset("1") == "now"
    assert lease.session.state.container_id not in env.daemon.containers
    assert env.record() is None and "guild-1" not in env.store.data
    assert (await _use(manager)).kind == "fresh"


@pytest.mark.asyncio
async def test_reset_of_a_cold_workspace_removes_its_left_container():
    env = Env()
    lease = await _use(env.manager())
    manager = env.manager()  # a new process: nothing warm here
    assert await manager.reset("1") == "now"
    assert lease.session.state.container_id not in env.daemon.containers
    assert env.record() is None


@pytest.mark.asyncio
async def test_reset_while_busy_runs_when_the_last_call_ends():
    env = Env()
    manager = env.manager()
    async with manager.acquire("1") as lease:
        assert await manager.reset("1") == "later"
        assert lease.session.state.container_id in env.daemon.containers
    assert lease.session.state.container_id not in env.daemon.containers
    assert env.record() is None


# ---------------------- reaper, shutdown, status ----------------------

@pytest.mark.asyncio
async def test_reaper_removes_only_our_orphans():
    env = Env()
    manager = env.manager()
    lease = await _use(manager)
    env.daemon.containers["orphan"] = {"labels": {"dcb.role": "workspace", "dcb.guild": "2",
                                                  "dcb.instance": "test"}, "status": "running"}
    env.daemon.containers["theirs"] = {"labels": {"dcb.role": "workspace", "dcb.guild": "2",
                                                  "dcb.instance": "prod"}, "status": "running"}
    assert await manager.reap() == 1
    assert "orphan" not in env.daemon.containers
    assert "theirs" in env.daemon.containers
    assert lease.session.state.container_id in env.daemon.containers
    assert set(env.daemon.list_filters["label"]) == {"dcb.role=workspace", "dcb.instance=test"}


@pytest.mark.asyncio
async def test_reaper_retires_a_container_left_idle_by_an_earlier_process():
    env = Env()
    lease = await _use(env.manager())
    cid = lease.session.state.container_id
    record = env.record()
    record["updated_at"] -= 1000
    env.redis.data["dcb:workspace:v1:guild:1"] = json.dumps(record)
    manager = env.manager()
    assert await manager.reap() == 1
    assert cid not in env.daemon.containers
    assert env.store.data["guild-1"] == f"tar-of-{cid}".encode()
    assert manager.lifecycle.state("1").phase == "cold"


@pytest.mark.asyncio
async def test_reaper_leaves_a_recently_used_container_for_its_next_call():
    env = Env()
    lease = await _use(env.manager())
    assert await env.manager().reap() == 0
    assert lease.session.state.container_id in env.daemon.containers


@pytest.mark.asyncio
async def test_shutdown_snapshots_changed_workspaces_and_leaves_them_running():
    env = Env()
    manager = env.manager()
    a = await _use(manager, "a")
    b = await _use(manager, "b")
    await manager.snapshot("b")  # b is clean now
    env.store.data.pop("guild-b")
    await manager.shutdown()
    assert "guild-a" in env.store.data and "guild-b" not in env.store.data
    assert a.session.state.container_id in env.daemon.containers
    assert b.session.state.container_id in env.daemon.containers


@pytest.mark.asyncio
async def test_status():
    env = Env()
    manager = env.manager()
    cold = await manager.status("1")
    assert cold["warm"] is False and cold["snapshot_bytes"] is None
    await _use(manager)
    await manager.snapshot("1")
    warm = await manager.status("1")
    assert warm["warm"] is True and warm["jobs"] == 3 and warm["snapshot_bytes"] > 0
    assert warm["last_snapshot_at"] is not None and warm["snapshot_too_large"] is False
    assert warm["waiting"] is False


@pytest.mark.asyncio
async def test_status_after_a_restart_sees_the_container_still_running():
    env = Env()
    first = env.manager()
    await _use(first)
    record = await first._load_record("1")
    env.daemon.containers[record["container_id"]] = {"labels": {}, "status": "running"}
    later = env.manager()  # a new core process: nothing warm in it yet
    status = await later.status("1")
    assert status["warm"] is False and status["waiting"] is True
    env.daemon.containers[record["container_id"]]["status"] = "exited"
    assert (await later.status("1"))["waiting"] is False


# ---------------------- Docker check and limits ----------------------

class _Images:
    def __init__(self, present):
        self.present, self.pulled = present, []

    def list(self, name=None):
        return [name] if self.present else []

    def pull(self, repo, tag=None):
        self.pulled.append((repo, tag))


def test_check_docker_passes_when_the_daemon_answers():
    images = _Images(present=True)
    client = SimpleNamespace(ping=lambda: True, images=images)
    assert wm.check_docker(from_env=lambda: client, sleep=lambda s: None) is None
    assert images.pulled == []


def test_check_docker_pulls_a_missing_image():
    images = _Images(present=False)
    client = SimpleNamespace(ping=lambda: True, images=images)
    assert wm.check_docker(from_env=lambda: client, sleep=lambda s: None,
                           image="python:3.14-slim") is None
    assert images.pulled == [("python", "3.14-slim")]


def test_check_docker_retries_then_gives_the_error():
    calls, now = [], [0.0]

    def from_env():
        calls.append(1)
        raise ConnectionError("no socket")

    def sleep(s):
        now[0] += s

    error = wm.check_docker(from_env=from_env, sleep=sleep, clock=lambda: now[0], wait_seconds=10)
    assert "needs Docker" in error and "no socket" in error
    assert len(calls) == 6  # 0, 2, ... 10 seconds


def test_check_docker_reports_an_image_that_cannot_be_pulled():
    images = _Images(present=False)
    images.pull = lambda repo, tag=None: (_ for _ in ()).throw(RuntimeError("offline"))
    client = SimpleNamespace(ping=lambda: True, images=images)
    error = wm.check_docker(from_env=lambda: client, sleep=lambda s: None)
    assert "could not be pulled" in error


@pytest.mark.parametrize("raw, expected", [
    ("4g", 4 * 1024 ** 3), ("4gb", 4 * 1024 ** 3), ("512m", 512 * 1024 ** 2), ("", None),
    ("lots", None), ("0", None),
])
def test_memory_limit_parsing(raw, expected):
    assert wm.memory_limit({"WORKSPACE_MEMORY_LIMIT": raw}) == expected


def test_container_limits():
    assert wm.container_limits({"WORKSPACE_PIDS_LIMIT": "0"}) == {}
    limits = wm.container_limits({"WORKSPACE_MEMORY_LIMIT": "1g", "WORKSPACE_CPU_LIMIT": "1.5"})
    assert limits == {"mem_limit": 1024 ** 3, "memswap_limit": 1024 ** 3,
                      "nano_cpus": 1_500_000_000, "pids_limit": 512}


def test_limited_client_adds_limits_to_create_and_passes_the_rest_through():
    created = []
    containers = SimpleNamespace(create=lambda **kw: created.append(kw) or "made",
                                 get=lambda cid: f"got {cid}")
    inner = SimpleNamespace(containers=containers, ping=lambda: "pong")
    client = wm._LimitedDockerClient(inner, {"pids_limit": 512, "mem_limit": 10})
    assert client.containers.create(image="x", mem_limit=5) == "made"
    assert created == [{"image": "x", "mem_limit": 5, "pids_limit": 512}]  # the caller's own wins
    assert client.containers.get("c1") == "got c1"
    assert client.ping() == "pong"


@pytest.mark.asyncio
async def test_reset_during_an_idle_stop_wipes_after_the_stop_saves():
    """A wipe racing the stop's save would let the save put the files back."""
    env = Env()
    manager = env.manager()
    lease = await _use(manager)
    assert manager.lifecycle.begin_stop("1") is True
    assert await manager.reset("1") == "later"
    await manager._stop("1")
    assert lease.session.state.container_id not in env.daemon.containers
    assert env.record() is None and "guild-1" not in env.store.data
    assert (await _use(manager)).kind == "fresh"


@pytest.mark.asyncio
async def test_reaper_leaves_a_container_that_is_still_starting():
    env = Env()
    manager = env.manager()
    assert manager.lifecycle.acquire("2", env.clock.now).action == "start"
    env.daemon.containers["new"] = {"labels": {"dcb.role": "workspace", "dcb.guild": "2",
                                               "dcb.instance": "test"}, "status": "running"}
    assert await manager.reap() == 0
    assert "new" in env.daemon.containers


@pytest.mark.asyncio
async def test_record_writes_for_one_guild_do_not_drop_each_others_fields():
    env = Env()
    manager = env.manager()
    real_get = env.redis.get

    async def slow_get(key):
        value = await real_get(key)
        await asyncio.sleep(0)  # let the other write interleave
        return value
    env.redis.get = slow_get
    await asyncio.gather(manager._save_record("1", a=1), manager._save_record("1", b=2))
    assert env.record()["a"] == 1 and env.record()["b"] == 2
