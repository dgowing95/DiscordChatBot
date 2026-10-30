"""Reads a link for the fetch_url tool: cache, site readers, plain fetch and
the browser fallback.

The order, for one URL:

  1. route it (web_policy.route): a YouTube video, a Reddit thread or
     subreddit, a Steam app, or any other page; its canonical URL is the
     cache key, so every spelling of the same link is one entry.
  2. the Redis cache. Tool results are not kept between replies (history is
     rebuilt from Discord), so without it a link still in the channel was
     fetched again on every reply that called the tool. A failure is cached
     too, briefly, so a dead link is not retried on every reply.
  3. one load per URL at a time: a second caller waits on the first's task
     (through asyncio.shield, so a caller that is cancelled never cancels the
     load others wait on; the attachment_cache pattern).
  4. the site's own reader (site_format.py says why each is read the way it
     is), else a plain GET whose text trafilatura extracts, else, when that
     text is too thin to be the page, the browser service (BROWSER_URL), which
     runs the page's scripts first.

Every outside request goes through one session whose resolver only returns
public addresses, and every redirect hop is checked for a private IP
literal (aiohttp skips the resolver for those). See web_policy for why.
"""
import asyncio
import json
import logging
import socket
import time
from collections import namedtuple
from urllib.parse import quote, urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

from classes import metrics, site_format
from classes.redis_client import text_client
from classes.web_policy import (
    PAGE, REDDIT_LISTING, REDDIT_SHARE, REDDIT_THREAD, STEAM_APP, USER_AGENT, YOUTUBE,
    cache_key, is_public_ip, needs_browser, private_ip_literal, route, settings,
)

logger = logging.getLogger(__name__)

# source: plain / browser / youtube / reddit / steam (also the metric label)
Page = namedtuple("Page", "text title source url")

_REDIRECTS = {301, 302, 303, 307, 308}
# Statuses bot checks answer with (Cloudflare's challenge is a 403 or 503).
_BOT_CHECK_STATUSES = {401, 403, 429, 503}
MAX_REDIRECTS = 5
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_TEXT_TYPES = ("text/plain", "text/markdown", "text/csv", "application/json", "text/xml",
               "application/xml", "application/rss+xml", "application/atom+xml")
# Steam's age check would otherwise stand in front of some store pages.
_STEAM_COOKIES = "birthtime=0; lastagecheckage=1-0-1990; wants_mature_content=1"


class FetchError(Exception):
    """A link that could not be read. `message` is what the model is told
    (it reads it when writing the reply, so it says what happened in plain
    words); `reason` is the metric label; `ttl` overrides how long the
    failure is cached (Reddit says when it will answer again)."""

    def __init__(self, reason: str, message: str, ttl: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.ttl = ttl


class BlockedAddress(OSError):
    pass


class PublicResolver(AbstractResolver):
    """The default resolver, minus every address that is not public. A name
    with only private addresses fails to resolve, so the connection is
    never made: the check is on the addresses actually connected to."""

    def __init__(self):
        self._inner = DefaultResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        results = [r for r in await self._inner.resolve(host, port, family)
                   if is_public_ip(r["host"])]
        if not results:
            raise BlockedAddress(f"{host} is not on the public internet")
        return results

    async def close(self):
        await self._inner.close()


_session = None
_internal_session = None
_inflight = {}
_reddit_retry_at = 0.0


def _get_session() -> aiohttp.ClientSession:
    """The shared session for outside requests (public addresses only).
    Built on first use, inside the running loop, like the other shared
    clients; never closed, since it lives as long as the bot."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(resolver=PublicResolver(), limit=20),
            headers={"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"},
        )
    return _session


def _get_internal_session() -> aiohttp.ClientSession:
    """For the browser service only: it is inside the cluster, which the
    public session would (rightly) refuse to reach."""
    global _internal_session
    if _internal_session is None or _internal_session.closed:
        _internal_session = aiohttp.ClientSession()
    return _internal_session


Response = namedtuple("Response", "status url content_type body headers")


async def _get(url: str, headers: dict | None = None, follow: bool = True) -> Response:
    """GET `url` from the internet, following up to MAX_REDIRECTS redirects
    when `follow`, and reading at most WEB_FETCH_MAX_BYTES of the body."""
    config = settings()
    timeout = aiohttp.ClientTimeout(total=config["timeout"])
    for _ in range(MAX_REDIRECTS + 1):
        if urlsplit(url).scheme not in ("http", "https"):
            raise FetchError("invalid", "That link redirects somewhere that isn't a web page.")
        if private_ip_literal(url):
            raise FetchError("blocked", "That address is inside a private network, so it "
                                        "can't be read.")
        try:
            async with _get_session().get(url, headers=headers, allow_redirects=False,
                                          timeout=timeout) as response:
                location = response.headers.get("Location")
                if follow and response.status in _REDIRECTS and location:
                    url = urljoin(url, location)
                    continue
                body = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    body += chunk
                    if len(body) >= config["max_bytes"]:
                        del body[config["max_bytes"]:]
                        break
                return Response(response.status, url, response.content_type or "",
                                bytes(body), response.headers)
        except FetchError:
            raise
        except asyncio.TimeoutError:
            raise FetchError("timeout", "The site took too long to answer.") from None
        except aiohttp.ClientConnectorError as e:
            if isinstance(getattr(e, "os_error", None), BlockedAddress):
                raise FetchError("blocked", "That address is inside a private network, so it "
                                            "can't be read.") from None
            raise FetchError("network", "The site could not be reached.") from None
        except (aiohttp.ClientError, OSError, ValueError):
            raise FetchError("network", "The site could not be reached.") from None
    raise FetchError("network", "The link redirects too many times.")


def _decode(response: Response) -> str:
    charset = "utf-8"
    for part in response.headers.get("Content-Type", "").split(";"):
        if part.strip().lower().startswith("charset="):
            charset = part.split("=", 1)[1].strip().strip('"') or charset
    try:
        return response.body.decode(charset, errors="replace")
    except LookupError:
        return response.body.decode("utf-8", errors="replace")


def _extract_html(html, url: str) -> tuple:
    """(title, text) of an HTML page. trafilatura keeps the main text and
    drops menus, footers and cookie banners; for pages it finds no article
    in (a listing, a store front), the whole visible text is used instead.
    CPU-bound: run off the event loop."""
    import trafilatura
    from bs4 import BeautifulSoup
    text = trafilatura.extract(html, url=url, favor_recall=True, include_comments=True,
                               include_tables=True, deduplicate=True) or ""
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
        title = (meta.title or "") if meta else ""
    except Exception:
        title = ""
    if not text.strip():
        soup = BeautifulSoup(html, features="html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.extract()
        root = soup.body or soup
        lines = (line.strip() for line in root.get_text("\n").splitlines())
        text = "\n".join(line for line in lines if line)
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
    return title.strip(), text.strip()


def _page(title: str, url: str, text: str, source: str) -> Page:
    heading = f"{title}\n{url}" if title else url
    return Page(f"{heading}\n\n{text}".strip(), title, source, url)


async def _render(url: str):
    """(title, text, final url) from the browser service, or None when it is
    off, the render failed, or the page it got was itself an error (the
    caller keeps what it already had)."""
    config = settings()
    if not config["browser_url"]:
        return None
    try:
        async with _get_internal_session().post(
                f"{config['browser_url']}/render", json={"url": url},
                timeout=aiohttp.ClientTimeout(total=config["browser_wait"])) as response:
            if response.status != 200:
                logger.info(f"browser render of {url} answered {response.status}")
                return None
            data = await response.json()
    except Exception as e:
        logger.warning(f"browser render of {url} failed: {e}")
        return None
    if int(data.get("status") or 0) >= 400:
        logger.info(f"browser render of {url} got HTTP {data.get('status')}")
        return None
    title, text = await asyncio.to_thread(_extract_html, data.get("html") or "", data.get("url") or url)
    return (title or data.get("title") or "").strip(), text, data.get("url") or url


async def _read_page(url: str) -> Page:
    config = settings()
    try:
        response = await _get(url)
    except FetchError as e:
        if e.reason in ("network", "timeout"):
            rendered = await _render(url)
            if rendered and rendered[1]:
                return _page(rendered[0], rendered[2], rendered[1], "browser")
        raise
    kind = response.content_type.lower()
    if kind in _TEXT_TYPES:
        return _page("", response.url, _decode(response), "plain")
    if kind not in _HTML_TYPES and kind:
        what = "a PDF" if kind == "application/pdf" else (
            "an image" if kind.startswith("image/") else (
                "a video" if kind.startswith("video/") else f"a {kind} file"))
        raise FetchError("unsupported", f"That link is {what}, which can't be read as text.")
    title, text = await asyncio.to_thread(_extract_html, response.body, response.url)
    thin = needs_browser(text, config["browser_min_chars"])
    # An error page is only worth rendering when the error may be a bot check
    # that a real browser passes; a 404 or a 500 renders as the same error.
    if thin and (response.status < 400 or response.status in _BOT_CHECK_STATUSES):
        rendered = await _render(response.url)
        if rendered and len(rendered[1]) > len(text):
            return _page(rendered[0], rendered[2], rendered[1], "browser")
    if response.status >= 400 and thin:
        raise FetchError("http", f"The site answered with an error (HTTP {response.status}), "
                                 "so there is nothing to read.")
    if not text:
        raise FetchError("empty", "The page has no readable text.")
    return _page(title, response.url, text, "plain")


def _fetch_transcript(video_id: str):
    """(language, generated, snippets) of the best caption track: English
    written by a person, else English auto-captions, else any written by a
    person, else any. None when the video has no captions."""
    from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, YouTubeTranscriptApi
    try:
        tracks = list(YouTubeTranscriptApi().list(video_id))
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    if not tracks:
        return None
    english = [t for t in tracks if t.language_code.lower().startswith("en")]
    ranked = ([t for t in english if not t.is_generated] + [t for t in english if t.is_generated]
              + [t for t in tracks if not t.is_generated] + tracks)
    best = ranked[0]
    snippets = [{"text": s.text, "start": s.start} for s in best.fetch()]
    return best.language, best.is_generated, snippets


async def _read_youtube(video_id: str, url: str) -> Page:
    import youtube_transcript_api as yta
    title = channel = ""
    try:
        response = await _get(f"https://www.youtube.com/oembed?format=json&url={quote(url, safe='')}")
        if response.status == 200:
            meta = json.loads(response.body)
            title, channel = meta.get("title", ""), meta.get("author_name", "")
    except (FetchError, ValueError):
        pass  # the title is a nicety; the captions are what matter
    try:
        found = await asyncio.wait_for(asyncio.to_thread(_fetch_transcript, video_id),
                                       settings()["timeout"] * 2)
    except asyncio.TimeoutError:
        raise FetchError("timeout", "YouTube took too long to send the video's captions.") from None
    except (yta.VideoUnavailable, yta.VideoUnplayable, yta.InvalidVideoId):
        raise FetchError("unavailable", "That YouTube video is unavailable (private, removed "
                                        "or blocked in this region).") from None
    except yta.AgeRestricted:
        raise FetchError("unavailable", "That YouTube video is age-restricted, so its captions "
                                        "can't be read.") from None
    except (yta.RequestBlocked, yta.IpBlocked, yta.PoTokenRequired):
        raise FetchError("blocked_by_site", "YouTube refused to send the captions (it is "
                                            "blocking this server for now). Try again later.") from None
    except yta.YouTubeTranscriptApiException as e:
        logger.warning(f"captions for {video_id} failed: {e}")
        raise FetchError("youtube", "The video's captions could not be read.") from None
    if found is None:
        # Not a failure: kept for the page TTL, not the video one, since
        # YouTube can add auto-captions some hours after an upload.
        return Page(site_format.format_no_captions(title, channel), title, "youtube_nocaptions", url)
    language, generated, snippets = found
    return Page(site_format.format_transcript(title, channel, language, generated, snippets),
                title, "youtube", url)


async def _reddit_feed(path: str) -> str:
    """The RSS feed at reddit.com/<path>/.rss. Reddit allows about one
    unauthenticated request a minute per address and says so in its
    x-ratelimit headers; while that is spent, no request is made at all."""
    global _reddit_retry_at
    wait = int(_reddit_retry_at - time.monotonic())
    if wait > 0:
        raise FetchError("rate_limited", f"Reddit only allows a few reads a minute; try again "
                                         f"in about {wait} seconds.", ttl=wait)
    response = await _get(f"https://www.reddit.com/{path.strip('/')}/.rss")
    try:
        reset = int(float(response.headers.get("x-ratelimit-reset", "0")))
        if float(response.headers.get("x-ratelimit-remaining", "1")) <= 0 and reset > 0:
            _reddit_retry_at = time.monotonic() + reset
    except ValueError:
        reset = 0
    if response.status == 429:
        wait = max(reset, 30)
        _reddit_retry_at = time.monotonic() + wait
        raise FetchError("rate_limited", f"Reddit only allows a few reads a minute; try again "
                                         f"in about {wait} seconds.", ttl=wait)
    if response.status == 404:
        raise FetchError("not_found", "That Reddit page doesn't exist (or was removed).")
    if response.status != 200:
        raise FetchError("blocked_by_site", f"Reddit refused the request (HTTP {response.status}).")
    return _decode(response)


async def _read_reddit(kind: str, ident: str, url: str) -> Page:
    if kind == REDDIT_SHARE:
        # Only Reddit knows which thread a share link means: ask, without
        # following, and read the thread it names.
        response = await _get(ident, follow=False)
        target = route(urljoin(ident, response.headers.get("Location", "")))
        if response.status not in _REDIRECTS or target is None or target.kind != REDDIT_THREAD:
            raise FetchError("not_found", "That Reddit share link doesn't lead to a thread.")
        kind, ident, url = target
    feed = await _reddit_feed(ident)
    formatter = (site_format.format_reddit_thread if kind == REDDIT_THREAD
                 else site_format.format_reddit_listing)
    try:
        title, text = await asyncio.to_thread(formatter, feed)
    except Exception as e:
        logger.warning(f"could not format the Reddit feed for {ident}: {e!r}")
        raise FetchError("not_found", "That Reddit page has nothing to read.") from e
    return Page(f"{text}\n\n{url}", title, "reddit", url)


async def _read_steam(app: str, url: str) -> Page:
    details, store = await asyncio.gather(
        _get(f"https://store.steampowered.com/api/appdetails?appids={app}&l=english"),
        _get(url + "?l=english", headers={"Cookie": _STEAM_COOKIES}),
        return_exceptions=True)
    data = None
    if isinstance(details, Response) and details.status == 200:
        try:
            entry = json.loads(details.body).get(app) or {}
            data = entry.get("data") if entry.get("success") else None
        except (ValueError, AttributeError):
            data = None
    if data is None:
        if isinstance(details, FetchError) and details.reason == "blocked":
            raise details
        # No app data (a removed or region-locked app): read the page as any other.
        return await _read_page(url)
    store_html = _decode(store) if isinstance(store, Response) and store.status == 200 else ""
    title, text = site_format.format_steam_app(data, store_html)
    return Page(f"{text}\n\n{url}", title, "steam", url)


async def _load(target) -> Page:
    kind, ident, url = target
    if kind == YOUTUBE:
        return await _read_youtube(ident, url)
    if kind in (REDDIT_THREAD, REDDIT_LISTING, REDDIT_SHARE):
        return await _read_reddit(kind, ident, url)
    if kind == STEAM_APP:
        return await _read_steam(ident, url)
    return await _read_page(url)


async def _cache_get(key: str):
    try:
        raw = await text_client().get(key)
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"web cache read failed, fetching instead: {e}")
        return None


async def _cache_set(key: str, value: dict, ttl: int) -> None:
    try:
        await text_client().set(key, json.dumps(value), ex=max(1, int(ttl)))
    except Exception as e:
        logger.warning(f"web cache write failed: {e}")


async def _load_and_store(target, key: str) -> Page:
    config = settings()
    try:
        page = await _load(target)
    except FetchError as e:
        metrics.inc_web_fetch_failure(e.reason)
        await _cache_set(key, {"error": e.message, "reason": e.reason},
                         e.ttl or config["failure_ttl"])
        raise
    page = page._replace(text=page.text[:config["max_chars"]])
    ttl = config["video_ttl"] if page.source == "youtube" else config["page_ttl"]
    await _cache_set(key, page._asdict(), ttl)
    metrics.inc_web_fetch(page.source, "miss")
    return page


def _forget(key: str, task) -> None:
    if _inflight.get(key) is task:
        del _inflight[key]
    if not task.cancelled():
        task.exception()  # retrieved, so an unawaited failure is not logged as lost


async def read(url: str) -> Page:
    """The readable text of `url`. Raises FetchError with what to tell the
    user when it cannot be read."""
    target = route(url)
    if target is None:
        raise FetchError("invalid", "That isn't a web address that can be read (it needs to "
                                    "start with http or https).")
    key = cache_key(target.url)
    task = _inflight.get(key)
    if task is None:
        cached = await _cache_get(key)
        if cached and "error" in cached:
            metrics.inc_web_fetch("none", "failure_hit")
            raise FetchError(cached.get("reason", "cached"), cached["error"])
        if cached:
            metrics.inc_web_fetch(cached.get("source", "plain"), "hit")
            return Page(cached["text"], cached.get("title", ""), cached.get("source", "plain"),
                        cached.get("url", target.url))
        task = _inflight.get(key)  # another caller may have started it meanwhile
    if task is None:
        task = asyncio.create_task(_load_and_store(target, key))
        _inflight[key] = task
        task.add_done_callback(lambda done: _forget(key, done))
    else:
        metrics.inc_web_fetch("none", "coalesced")
    return await asyncio.shield(task)
