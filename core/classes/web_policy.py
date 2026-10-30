"""Rules for reading links (PURE, stdlib only): settings, URL clean-up, which
reader a URL gets, which addresses may be fetched, and paging long text.

web_reader.py does the fetching; this module decides everything that can be
decided from strings, so it is unit tested without aiohttp or Redis
(core/tests/web_policy_tests.py).

Why URLs are normalised
-----------------------
The cache key is the cleaned URL. The same video shared as youtu.be/X,
youtube.com/watch?v=X&si=... and youtube.com/shorts/X must be one entry, as
must a Reddit thread reached through old., new. or m.reddit.com, or a page
whose link carries utm_ tracking. Otherwise each spelling would be fetched
(and transcribed, or rendered) again.

Why private addresses are refused
---------------------------------
fetch_url runs inside the cluster, next to Redis, the LLM server and the
image service. A link to http://redis:6379, or to a public name that resolves
to 10.x, would otherwise let anyone in the server read them through the bot.
is_public_ip is applied to the addresses actually connected to (web_reader's
resolver), so a redirect or a DNS answer that changes cannot get round it.
"""
import hashlib
import ipaddress
import math
import os
import re
from collections import namedtuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Sent by the plain fetch. A desktop Chrome, because many sites turn away or
# strip a page for an unknown agent (the old "dis-ai-bot" got bot-blocked
# pages). The browser service builds the same shape from its real version.
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36")

CACHE_PREFIX = "dcb:webcache:v1:"

PAGE = "page"
YOUTUBE = "youtube"
REDDIT_THREAD = "reddit_thread"
REDDIT_LISTING = "reddit_listing"
REDDIT_SHARE = "reddit_share"
STEAM_APP = "steam_app"

# kind: one of the names above; ident: the id the reader needs (video id,
# Reddit path, Steam app id); url: the canonical URL (also the cache key).
Route = namedtuple("Route", "kind ident url")


def _positive(name, default, cast=int):
    try:
        value = cast(str(os.environ.get(name, "")).strip())
    except ValueError:
        return default
    return value if value > 0 and math.isfinite(value) else default


def settings() -> dict:
    """Every WEB_* / BROWSER_* setting core reads, parsed. Read on each call
    (a few env lookups), like the other policy modules, so tests can
    monkeypatch them."""
    browser_timeout = _positive("BROWSER_TIMEOUT", 30.0, float)
    browser_settle = _positive("BROWSER_SETTLE_SECONDS", 5.0, float)
    return {
        "timeout": _positive("WEB_FETCH_TIMEOUT", 15.0, float),
        "max_bytes": _positive("WEB_FETCH_MAX_BYTES", 5_000_000),
        "part_chars": _positive("WEB_FETCH_PART_CHARS", 10_000),
        "max_chars": _positive("WEB_FETCH_MAX_CHARS", 200_000),
        "page_ttl": _positive("WEB_CACHE_TTL_SECONDS", 6 * 3600),
        "video_ttl": _positive("WEB_CACHE_VIDEO_TTL_SECONDS", 30 * 24 * 3600),
        "failure_ttl": _positive("WEB_CACHE_FAILURE_TTL_SECONDS", 300),
        "browser_url": (os.environ.get("BROWSER_URL") or "").strip().rstrip("/"),
        "browser_min_chars": _positive("BROWSER_MIN_CHARS", 400),
        # How long core waits on one render: the service's own budget for it
        # (the same configmap values), plus room for the queue and the reply.
        "browser_wait": browser_timeout + browser_settle + 15,
    }


# Query parameters that only track who shared a link. Dropped from every URL
# for the cache key and the fetch alike: no site needs them to serve a page.
_TRACKING = {"fbclid", "gclid", "dclid", "gbraid", "wbraid", "msclkid", "yclid", "igshid",
             "mc_cid", "mc_eid", "_hsenc", "_hsmi", "ref_src", "ref_url", "share_id",
             "rdt", "spm"}
_YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
                  "youtube-nocookie.com", "www.youtube-nocookie.com"}
_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _clean_query(query: str) -> str:
    pairs = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING]
    return urlencode(sorted(pairs))


def youtube_video_id(url: str) -> str | None:
    """The 11-character video id of any YouTube video link, else None."""
    try:
        parts = urlsplit(url if "://" in url else "https://" + url)
    except ValueError:
        return None
    host = (parts.hostname or "").lower()
    segments = [s for s in parts.path.split("/") if s]
    candidate = None
    if host == "youtu.be" and segments:
        candidate = segments[0]
    elif host in _YOUTUBE_HOSTS:
        if segments[:1] == ["watch"]:
            candidate = dict(parse_qsl(parts.query)).get("v")
        elif len(segments) >= 2 and segments[0] in ("shorts", "live", "embed", "v"):
            candidate = segments[1]
    return candidate if candidate and _VIDEO_ID.match(candidate) else None


def normalize_url(url: str) -> str | None:
    """The URL with tracking removed and spelling fixed, or None when it is
    not an http(s) URL with a host. A bare "example.com/page" is read as
    https."""
    url = (url or "").strip()
    if not url:
        return None
    if "://" not in url:
        url = "https://" + url
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    if scheme not in ("http", "https") or not host:
        return None
    netloc = host if ":" not in host else f"[{host}]"
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc += f":{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", _clean_query(parts.query), ""))


def _reddit_route(host: str, path: str):
    segments = [s for s in path.split("/") if s]
    if host == "redd.it" and segments:
        ident = f"comments/{segments[0]}"
        return Route(REDDIT_THREAD, ident, f"https://www.reddit.com/{ident}/")
    if not (host == "reddit.com" or host.endswith(".reddit.com")):
        return None
    if segments[:1] == ["comments"] and len(segments) >= 2:
        ident = f"comments/{segments[1]}"
        return Route(REDDIT_THREAD, ident, f"https://www.reddit.com/{ident}/")
    if segments[:1] != ["r"] or len(segments) < 2:
        return None
    sub = segments[1]
    rest = segments[2:]
    if rest[:1] == ["comments"] and len(rest) >= 2:
        # The title slug after the id is decoration; the id alone finds it.
        ident = f"r/{sub}/comments/{rest[1]}"
        return Route(REDDIT_THREAD, ident, f"https://www.reddit.com/{ident}/")
    if rest[:1] == ["s"] and len(rest) >= 2:
        # A share link: only Reddit knows which thread it points at.
        url = f"https://www.reddit.com/r/{sub}/s/{rest[1]}"
        return Route(REDDIT_SHARE, url, url)
    if not rest or (len(rest) == 1 and rest[0] in ("hot", "new", "top", "rising")):
        ident = "/".join(["r", sub] + rest)
        return Route(REDDIT_LISTING, ident, f"https://www.reddit.com/{ident}/")
    return None


def route(url: str) -> Route | None:
    """Which reader a URL gets, with its canonical URL. None for a URL that
    cannot be fetched at all (not http(s), no host)."""
    normalized = normalize_url(url)
    if normalized is None:
        return None
    parts = urlsplit(normalized)
    host = parts.hostname or ""
    video = youtube_video_id(normalized)
    if video:
        return Route(YOUTUBE, video, f"https://www.youtube.com/watch?v={video}")
    reddit = _reddit_route(host, parts.path)
    if reddit:
        return reddit
    if host == "store.steampowered.com":
        match = re.match(r"^/app/(\d+)", parts.path)
        if match:
            app = match.group(1)
            return Route(STEAM_APP, app, f"https://store.steampowered.com/app/{app}/")
    return Route(PAGE, normalized, normalized)


def cache_key(canonical_url: str) -> str:
    return CACHE_PREFIX + hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()


def is_public_ip(address: str) -> bool:
    """True only for an address on the public internet: not private,
    loopback, link-local (cloud metadata), shared/CGNAT, reserved or
    multicast. An IPv4 address wrapped in IPv6 (::ffff:10.0.0.1) is judged as
    the IPv4 address it is."""
    try:
        ip = ipaddress.ip_address(str(address).split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def private_ip_literal(url: str) -> bool:
    """Whether `url`'s host is an IP address that is not public. aiohttp
    connects to an IP literal without asking the resolver, so these are
    checked before each request instead."""
    try:
        host = urlsplit(url).hostname or ""
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return not is_public_ip(host)


_JS_MARKERS = ("enable javascript", "javascript is required", "javascript is disabled",
               "turn on javascript", "requires javascript", "javascript to run this app",
               "you need to enable javascript")


def needs_browser(text: str, min_chars: int) -> bool:
    """Whether a plain fetch's text is too thin to be the page: almost none
    of it, or a short page that only says scripts are needed."""
    text = (text or "").strip()
    if len(text) < min_chars:
        return True
    lowered = text.lower()
    return len(text) < 2000 and any(marker in lowered for marker in _JS_MARKERS)


def split_parts(text: str, size: int) -> list:
    """`text` in pieces of at most `size` characters, each cut at the last
    line break (else space) in its second half, so words and lines are not
    broken. Deterministic, so part N is the same piece on every call."""
    text = text or ""
    if len(text) <= size:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = window.rfind("\n", size // 2)
            if cut < 0:
                cut = window.rfind(" ", size // 2)
            if cut > 0:
                end = start + cut + 1
        parts.append(text[start:end])
        start = end
    return parts


def paginate(text: str, part: int, size: int) -> str:
    """Part `part` (1-based) of `text`, ending with where it is in the
    whole. The model reads that line, so it says how to read on."""
    parts = split_parts(text, size)
    total = len(parts)
    if part < 1 or part > total:
        return (f"There is no part {part}: this page has {total} part"
                f"{'s' if total != 1 else ''}. Use part=1 to {total}.")
    body = parts[part - 1].rstrip()
    if total == 1:
        return body
    if part < total:
        return (f"{body}\n\n[Part {part} of {total}. Call fetch_url again with "
                f"part={part + 1} to read on.]")
    return f"{body}\n\n[Part {total} of {total}: the end.]"
