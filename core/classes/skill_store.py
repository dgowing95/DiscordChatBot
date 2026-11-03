"""Skills in Redis: one record per skill, every version kept.

Redis is the source of truth, outside the workspace container, so a reset, a
lost container or a restart never loses a skill. Everyone in a server sees
and manages its skills; every change is a new version (so a bad edit is a
/skills revert away) and a delete is soft (restore brings it back).

Layout, per guild:
  dcb:skills:v1:{guild}:names        set of every skill name ever saved
  dcb:skills:v1:{guild}:live         set of the names not deleted (the quota)
  dcb:skills:v1:{guild}:skill:{name} JSON: counters + `versions`, oldest first;
                                     the current version is the LAST one
A revert appends a copy of the old version, so history is never rewritten.

Every change is one WATCH/MULTI on the skill's record and the two sets, retried
when something else wrote meanwhile, so two saves, or a save and a use, never
lose each other (the pattern of automation_store).
"""
import asyncio
import io
import json
import logging
import random
import time
import zipfile

from redis.exceptions import WatchError

from classes import skill_format
from classes.redis_client import text_client

logger = logging.getLogger(__name__)

PREFIX = "dcb:skills:v1"
RETRIES = 25
SOURCES = ("brain", "reflection", "command", "curator", "revert")

CREATED, UPDATED, UNCHANGED = "created", "updated", "unchanged"


class SkillError(ValueError):
    """A change the store refuses; the text is written for the person or the
    model who asked."""


class QuotaError(SkillError):
    pass


class VersionError(SkillError):
    """The skill is no longer at the version the caller expected."""


def _names(guild):
    return f"{PREFIX}:{guild}:names"


def _live(guild):
    return f"{PREFIX}:{guild}:live"


def _key(guild, name):
    return f"{PREFIX}:{guild}:skill:{name}"


def _check_version(doc, name, expected) -> None:
    if expected is None:
        return
    if not doc or doc.get("deleted") or doc["versions"][-1]["v"] != expected:
        raise VersionError(f"{name} changed since v{expected}; left as it is.")


def _summary(doc: dict) -> dict:
    """A record without its history: what lists and the index need."""
    current = doc["versions"][-1]
    return {
        "name": doc["name"],
        "description": current["description"],
        "body": current["body"],
        "files": current.get("files") or {},
        "version": current["v"],
        "versions": len(doc["versions"]),
        "deleted": bool(doc.get("deleted")),
        "uses": doc.get("uses", 0),
        "successes": doc.get("successes", 0),
        "last_used_at": doc.get("last_used_at"),
        "last_channel_id": doc.get("last_channel_id"),
        "updated_by": current.get("author_id"),
        "updated_at": current["created_at"],
        "source": current.get("source"),
    }


def _count_save(source: str, result: str) -> None:
    try:
        from classes import metrics
        metrics.inc_skill_saved(source, result)
    except Exception:
        pass


class SkillStore:
    def __init__(self, redis=None, clock=time.time):
        self.redis = redis or text_client()
        self.clock = clock

    # -- reading ---------------------------------------------------------

    async def _doc(self, guild, name):
        raw = await self.redis.get(_key(guild, name))
        return json.loads(raw) if raw else None

    async def list(self, guild, include_deleted: bool = False) -> list[dict]:
        names = await self.redis.smembers(_names(guild) if include_deleted else _live(guild))
        names = sorted(names)
        # One round trip for them all: the brain reads this on every call.
        raws = await self.redis.mget([_key(guild, n) for n in names]) if names else []
        rows = []
        for raw in raws:
            doc = json.loads(raw) if raw else None
            if doc and (include_deleted or not doc.get("deleted")):
                rows.append(_summary(doc))
        return rows

    async def get(self, guild, name, version: int | None = None) -> dict | None:
        """A live skill at its current version, or at `version`. None when it
        is unknown or deleted (history() still shows a deleted one)."""
        doc = await self._doc(guild, name)
        if not doc or doc.get("deleted"):
            return None
        return self._at(doc, version)

    async def get_any(self, guild, name, version: int | None = None) -> dict | None:
        """Like get, deleted skills included (for restore and history)."""
        doc = await self._doc(guild, name)
        return self._at(doc, version) if doc else None

    @staticmethod
    def _at(doc, version):
        summary = _summary(doc)
        if version is None:
            return summary
        for old in doc["versions"]:
            if old["v"] == version:
                summary.update(description=old["description"], body=old["body"],
                               files=old.get("files") or {}, version=old["v"],
                               updated_by=old.get("author_id"), updated_at=old["created_at"],
                               source=old.get("source"))
                return summary
        return None

    async def history(self, guild, name) -> list[dict]:
        """Versions, newest first: who, when, from where, and the note."""
        doc = await self._doc(guild, name)
        if not doc:
            return []
        return [{"v": v["v"], "author_id": v.get("author_id"), "source": v.get("source"),
                 "created_at": v["created_at"], "note": v.get("note", ""),
                 "job_id": v.get("job_id", ""), "current": v is doc["versions"][-1]}
                for v in reversed(doc["versions"])]

    async def count_live(self, guild) -> int:
        return int(await self.redis.scard(_live(guild)))

    # -- changing --------------------------------------------------------

    async def _change(self, guild, name, build):
        """Runs build(doc_or_None, live_count) -> (doc_or_None_to_leave, result)
        inside WATCH/MULTI. build may raise SkillError."""
        for _ in range(RETRIES):
            async with self.redis.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(_key(guild, name), _live(guild))
                    raw = await pipe.get(_key(guild, name))
                    live = int(await pipe.scard(_live(guild)))
                    doc = json.loads(raw) if raw else None
                    new, result = build(doc, live)
                    if new is None:
                        await pipe.unwatch()
                        return result
                    pipe.multi()
                    pipe.set(_key(guild, name), json.dumps(new))
                    pipe.sadd(_names(guild), name)
                    if new.get("deleted"):
                        pipe.srem(_live(guild), name)
                    else:
                        pipe.sadd(_live(guild), name)
                    await pipe.execute()
                    return result
                except WatchError:
                    pass
            await asyncio.sleep(random.random() * 0.01)
        raise SkillError("Skills are busy right now, please try again")

    def _append(self, doc, name, version: dict) -> dict:
        cfg = skill_format.settings()
        versions = list(doc["versions"]) if doc else []
        version["v"] = (versions[-1]["v"] + 1) if versions else 1
        versions.append(version)
        while len(versions) > cfg["max_versions"]:
            dropped = versions.pop(0)
            logger.info(f"Skills: dropped v{dropped['v']} of {name} (version cap)")
        out = dict(doc or {"name": name, "uses": 0, "successes": 0})
        out["versions"] = versions
        out["deleted"] = False
        return out

    async def save(self, guild, name, description, body, files=None, *, author_id,
                   source="brain", job_id="", note="", channel_id=None,
                   expect_version: int | None = None) -> str:
        """Creates or updates a skill. Returns CREATED, UPDATED or UNCHANGED
        (the same text again makes no new version). Raises SkillError with the
        problems, or QuotaError when the server is full.

        channel_id: where it was saved; kept as the skill's channel until a
        use sets one, so a skill never used still has somewhere for the
        curator's embeds. expect_version: refuse unless the live skill is
        still at this version (the curator's changes lose to a person's edit
        made while it was thinking)."""
        name = skill_format.clean_name(name)
        description = skill_format.one_line(description, 10_000)
        files = dict(files or {})
        problems = skill_format.validate(name, description, body, files)
        if problems:
            raise SkillError("Not saved: " + "; ".join(problems))
        if source not in SOURCES:
            raise SkillError(f"unknown source {source!r}")
        cfg = skill_format.settings()
        body = str(body).strip()
        note = skill_format.one_line(note, skill_format.NOTE_CHARS)

        def build(doc, live):
            was_live = bool(doc) and not doc.get("deleted")
            _check_version(doc, name, expect_version)
            if doc:
                current = doc["versions"][-1]
                if (was_live and current["description"] == description and current["body"] == body
                        and (current.get("files") or {}) == files):
                    return None, UNCHANGED
            if not was_live and live >= cfg["max_per_guild"]:
                raise QuotaError(f"This server already has {live} skills (the limit is "
                                 f"{cfg['max_per_guild']}). Delete one first.")
            version = {"description": description, "body": body, "files": files,
                       "author_id": author_id, "source": source, "job_id": job_id,
                       "created_at": self.clock(), "note": note}
            out = self._append(doc, name, version)
            if channel_id is not None and not out.get("last_channel_id"):
                out["last_channel_id"] = channel_id
            return out, (UPDATED if doc else CREATED)
        result = await self._change(guild, name, build)
        _count_save(source, result)
        return result

    async def delete(self, guild, name, actor, expect_version: int | None = None) -> None:
        def build(doc, live):
            if not doc or doc.get("deleted"):
                raise SkillError(f"No skill named {name} in this server.")
            _check_version(doc, name, expect_version)
            out = dict(doc)
            out["deleted"] = True
            out["deleted_by"] = actor
            return out, None
        await self._change(guild, name, build)

    async def restore(self, guild, name, actor) -> None:
        cfg = skill_format.settings()

        def build(doc, live):
            if not doc or not doc.get("deleted"):
                raise SkillError(f"No deleted skill named {name} in this server.")
            if live >= cfg["max_per_guild"]:
                raise QuotaError(f"This server already has {live} skills; delete one first.")
            out = dict(doc)
            out["deleted"] = False
            return out, None
        await self._change(guild, name, build)

    async def revert(self, guild, name, version: int, actor) -> int:
        """Appends a copy of an old version as the newest. Returns its number."""
        def build(doc, live):
            if not doc or doc.get("deleted"):
                raise SkillError(f"No skill named {name} in this server.")
            old = next((v for v in doc["versions"] if v["v"] == version), None)
            if old is None:
                raise SkillError(f"{name} has no version {version}.")
            if old is doc["versions"][-1]:
                raise SkillError(f"Version {version} is already the current one.")
            new = {"description": old["description"], "body": old["body"],
                   "files": old.get("files") or {}, "author_id": actor, "source": "revert",
                   "job_id": "", "created_at": self.clock(), "note": f"back to v{version}"}
            out = self._append(doc, name, new)
            return out, out["versions"][-1]["v"]
        return await self._change(guild, name, build)

    async def record_use(self, guild, name, channel_id=None) -> None:
        def build(doc, live):
            if not doc or doc.get("deleted"):
                return None, None
            out = dict(doc)
            out["uses"] = doc.get("uses", 0) + 1
            out["last_used_at"] = self.clock()
            if channel_id is not None:
                out["last_channel_id"] = channel_id
            return out, None
        await self._change(guild, name, build)

    async def record_success(self, guild, name) -> None:
        def build(doc, live):
            if not doc or doc.get("deleted"):
                return None, None
            out = dict(doc)
            out["successes"] = doc.get("successes", 0) + 1
            return out, None
        await self._change(guild, name, build)

    async def export(self, guild) -> bytes:
        """A zip: every live skill's current version as <name>/SKILL.md plus
        its files, and history.json with every version of every skill."""
        buf = io.BytesIO()
        history = {}
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(await self.redis.smembers(_names(guild))):
                doc = await self._doc(guild, name)
                if not doc:
                    continue
                history[name] = {"deleted": bool(doc.get("deleted")), "versions": doc["versions"]}
                if doc.get("deleted"):
                    continue
                current = doc["versions"][-1]
                archive.writestr(f"{name}/SKILL.md", skill_format.render_skill_md(
                    name, current["description"], current["body"]))
                for path, text in (current.get("files") or {}).items():
                    archive.writestr(f"{name}/{path}", text)
            archive.writestr("history.json", json.dumps(history, indent=2))
        return buf.getvalue()
