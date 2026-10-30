"""How a Reddit feed, a Steam app or a YouTube transcript reads in a prompt
(PURE, stdlib only).

web_reader.py fetches these; this module turns what came back into the
plain text fetch_url returns. Every shape here was taken from a real
response (the fixtures in core/tests/fixtures/web/), not from memory.

Reddit is read through its RSS feeds. In 2026 Reddit refuses unauthenticated
.json (403 "blocked by network security"), old.reddit.com asks for a login,
and a headless browser gets a "prove your humanity" page. The feeds still
answer, at about one request a minute per address (x-ratelimit headers),
which the cache in web_reader makes enough for a chat bot. A thread's feed
is the post followed by its comments, flat: no scores, no nesting.
"""
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from html.parser import HTMLParser

_ATOM = {"a": "http://www.w3.org/2005/Atom"}

# A Reddit comment is cut at this, and at most this many are kept: enough to
# see what a thread says without one essay crowding out the rest.
COMMENT_CHARS = 1500
MAX_COMMENTS = 40
LISTING_SNIPPET_CHARS = 300
STEAM_ABOUT_CHARS = 4000

_BLOCK_TAGS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "tr",
               "blockquote", "pre", "table", "hr"}


class _TextExtractor(HTMLParser):
    """Visible text of an HTML fragment, with line breaks at block tags and
    a "- " before list items. Also notes where a "[link]" anchor points,
    which is how Reddit's feed gives a link post's target."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.link_target = None
        self._href = None
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag in _BLOCK_TAGS:
            self.out.append("\n")
        if tag == "li":
            self.out.append("- ")
        if tag == "a":
            self._href = dict(attrs).get("href")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag in _BLOCK_TAGS and tag != "li":
            self.out.append("\n")
        if tag == "a":
            self._href = None

    def handle_data(self, data):
        if self._skip:
            return
        if data.strip() == "[link]" and self._href:
            self.link_target = self._href
        self.out.append(data)

    def text(self) -> str:
        joined = "".join(self.out)
        lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in joined.splitlines())
        return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def html_to_text(fragment: str) -> str:
    parser = _TextExtractor()
    parser.feed(fragment or "")
    parser.close()
    return parser.text()


def _reddit_body(fragment: str):
    """(text, link target) of one feed entry's content. The feed appends
    "submitted by /u/x [link] [comments]" to every post; that is dropped,
    after noting where [link] points."""
    parser = _TextExtractor()
    parser.feed(fragment or "")
    parser.close()
    text = re.sub(r"\s*submitted by\s+/u/\S+[\s\S]*$", "", parser.text()).strip()
    link = parser.link_target
    if link and "/comments/" in link and "reddit.com" in link:
        link = None  # a text post's [link] is the thread itself
    return text, link


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def _date(entry) -> str:
    stamp = entry.findtext("a:published", default="", namespaces=_ATOM) or \
        entry.findtext("a:updated", default="", namespaces=_ATOM)
    try:
        return datetime.fromisoformat(stamp).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def _author(entry) -> str:
    name = entry.findtext("a:author/a:name", default="", namespaces=_ATOM).strip()
    return name[3:] if name.startswith("/u/") else (name or "[deleted]")


def _subreddit(feed, entry=None) -> str:
    node = (entry if entry is not None else feed).find("a:category", _ATOM)
    if node is None and entry is not None:
        node = feed.find("a:category", _ATOM)
    return node.get("label", "") if node is not None else ""


def format_reddit_thread(feed_xml: str) -> tuple:
    """(title, text) of a thread's feed. Raises ValueError on a feed that
    is not a thread (no entries, or not Atom)."""
    feed = ET.fromstring(feed_xml)
    entries = feed.findall("a:entry", _ATOM)
    if not entries:
        raise ValueError("feed has no entries")
    post, comments = entries[0], entries[1:]
    title = post.findtext("a:title", default="", namespaces=_ATOM).strip()
    body, link = _reddit_body(post.findtext("a:content", default="", namespaces=_ATOM))
    sub = _subreddit(feed, post)
    lines = [f"Reddit thread{' in ' + sub if sub else ''}: {title}",
             f"Posted by u/{_author(post)}{' on ' + _date(post) if _date(post) else ''}"]
    if link:
        lines.append(f"Links to: {link}")
    if body:
        lines += ["", body]
    shown = comments[:MAX_COMMENTS]
    if shown:
        count = f"{len(shown)} shown" + (f" of {len(comments)}" if len(comments) > len(shown) else "")
        lines += ["", f"Comments ({count}):"]
        for comment in shown:
            text, _ = _reddit_body(comment.findtext("a:content", default="", namespaces=_ATOM))
            text = _clip(text, COMMENT_CHARS).replace("\n", "\n  ")
            lines.append(f"- u/{_author(comment)}: {text}")
    else:
        lines += ["", "No comments yet."]
    return title, "\n".join(lines)


def format_reddit_listing(feed_xml: str) -> tuple:
    """(title, text) of a subreddit's feed: each post's title, author, date,
    link and the start of its text."""
    feed = ET.fromstring(feed_xml)
    entries = feed.findall("a:entry", _ATOM)
    sub = _subreddit(feed) or feed.findtext("a:title", default="", namespaces=_ATOM).strip()
    title = f"Reddit: {sub}" if sub else "Reddit"
    lines = [f"{title} ({len(entries)} posts)"]
    for entry in entries:
        name = entry.findtext("a:title", default="", namespaces=_ATOM).strip()
        node = entry.find("a:link", _ATOM)
        href = node.get("href", "") if node is not None else ""
        body, _ = _reddit_body(entry.findtext("a:content", default="", namespaces=_ATOM))
        date = _date(entry)
        lines.append(f"\n- {name} (u/{_author(entry)}{', ' + date if date else ''})\n  {href}")
        if body:
            lines.append("  " + _clip(body, LISTING_SNIPPET_CHARS).replace("\n", " "))
    return title, "\n".join(lines)


_REVIEW_TOOLTIP = re.compile(r'data-tooltip-html="([^"]*user reviews[^"]*)"')


def steam_review_summaries(store_html: str) -> list:
    """The review lines from a Steam store page ("98% of the 2,174 user
    reviews in the last 30 days are positive."), each once, in page order.
    Steam's own review API now redirects to the store front, so the page is
    where these are read."""
    seen = []
    for match in _REVIEW_TOOLTIP.finditer(store_html or ""):
        text = html_to_text(html.unescape(match.group(1)))
        if text and text not in seen:
            seen.append(text)
    return seen


def _steam_label(store_html: str) -> str:
    match = re.search(r'game_review_summary[^"]*"[^>]*>([^<]+)<', store_html or "")
    return match.group(1).strip() if match else ""


def format_steam_app(data: dict, store_html: str = "") -> tuple:
    """(title, text) of a Steam appdetails `data` object, with the review
    summary from the store page when that was fetched."""
    name = data.get("name") or "Unknown game"
    lines = [f"Steam: {name}"]
    price = data.get("price_overview") or {}
    if data.get("is_free"):
        lines.append("Price: free to play")
    elif price.get("final_formatted"):
        sale = ""
        if price.get("discount_percent"):
            sale = f" ({price['discount_percent']}% off, was {price.get('initial_formatted') or '?'})"
        lines.append(f"Price: {price['final_formatted']}{sale}")
    release = data.get("release_date") or {}
    if release.get("coming_soon"):
        lines.append(f"Release date: not out yet ({release.get('date') or 'date to be announced'})")
    elif release.get("date"):
        lines.append(f"Release date: {release['date']}")
    for key, label in (("developers", "Developer"), ("publishers", "Publisher")):
        if data.get(key):
            lines.append(f"{label}: {', '.join(data[key])}")
    genres = [g.get("description") for g in data.get("genres") or [] if g.get("description")]
    if genres:
        lines.append(f"Genres: {', '.join(genres)}")
    platforms = [p.capitalize() for p, on in (data.get("platforms") or {}).items() if on]
    if platforms:
        lines.append(f"Platforms: {', '.join(platforms)}")
    label, summaries = _steam_label(store_html), steam_review_summaries(store_html)
    if label or summaries:
        detail = "; ".join(s.rstrip(".") for s in summaries)
        lines.append(f"User reviews: {label}" + (f" ({detail})" if detail else ""))
    if (data.get("metacritic") or {}).get("score"):
        lines.append(f"Metacritic: {data['metacritic']['score']}")
    if (data.get("recommendations") or {}).get("total"):
        lines.append(f"Recommendations: {data['recommendations']['total']:,}")
    if data.get("short_description"):
        lines += ["", html_to_text(data["short_description"])]
    about = html_to_text(data.get("about_the_game") or data.get("detailed_description") or "")
    if about:
        lines += ["", "About the game:", _clip(about, STEAM_ABOUT_CHARS)]
    return name, "\n".join(lines)


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _youtube_header(title: str, channel: str) -> list:
    lines = [f"YouTube video: {title or 'unknown title'}"]
    if channel:
        lines.append(f"Channel: {channel}")
    return lines


def format_transcript(title: str, channel: str, language: str, generated: bool,
                      snippets: list, block_seconds: float = 30.0) -> str:
    """A transcript as text, one `[m:ss]` line per ~30 seconds, so the
    model can say where in the video something is said. `snippets` are
    dicts with text and start (youtube-transcript-api's own fields)."""
    lines = _youtube_header(title, channel)
    lines.append(f"Captions: {language}{' (auto-generated)' if generated else ''}")
    lines.append("")
    block, block_start = [], None
    for snippet in snippets:
        text = " ".join(str(snippet.get("text", "")).split())
        if not text:
            continue
        start = float(snippet.get("start") or 0)
        if block_start is None:
            block_start = start
        elif start - block_start >= block_seconds:
            lines.append(f"[{_clock(block_start)}] {' '.join(block)}")
            block, block_start = [], start
        block.append(text)
    if block:
        lines.append(f"[{_clock(block_start)}] {' '.join(block)}")
    return "\n".join(lines)


def format_no_captions(title: str, channel: str) -> str:
    lines = _youtube_header(title, channel)
    lines += ["", "This video has no captions, so what is said in it can't be read. "
                  "(Transcribing a video's audio isn't supported.) Only the title and channel "
                  "above are known."]
    return "\n".join(lines)
