"""Tests for reading links (core/classes/web_reader.py): the cache, one load
per URL, the private-address guard, the browser fallback and the site
readers. Redis and the network are faked; nothing here leaves the machine.

To run this pytest file from the command line, use:
    pytest core/tests/web_reader_tests.py
"""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from multidict import CIMultiDict

from classes import web_reader as wr
from classes.web_policy import cache_key

FIXTURES = Path(__file__).parent / "fixtures" / "web"
ARTICLE = ("<html><head><title>An article</title></head><body><nav>Home | About</nav><article>"
           "<h1>An article</h1>" + "<p>" + "A real sentence about the topic. " * 40 + "</p>"
           "</article><footer>Copyright</footer></body></html>").encode()


class FakeRedis:
    def __init__(self, broken=False):
        self.data, self.ttls, self.broken = {}, {}, broken

    async def get(self, key):
        if self.broken:
            raise ConnectionError("redis down")
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        if self.broken:
            raise ConnectionError("redis down")
        self.data[key], self.ttls[key] = value, ex


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(wr, "text_client", lambda: redis)
    monkeypatch.setattr(wr, "_inflight", {})
    monkeypatch.setattr(wr, "_reddit_retry_at", 0.0)
    monkeypatch.setattr(wr, "_session", None)
    monkeypatch.setattr(wr, "_internal_session", None)
    for name in ("BROWSER_URL", "WEB_CACHE_TTL_SECONDS", "WEB_CACHE_VIDEO_TTL_SECONDS",
                 "WEB_CACHE_FAILURE_TTL_SECONDS", "WEB_FETCH_MAX_BYTES"):
        monkeypatch.delenv(name, raising=False)
    return redis


def _response(body=b"", status=200, kind="text/html", url="https://example.com/a", headers=None):
    return wr.Response(status, url, kind, body, CIMultiDict(headers or {}))


def _fake_get(monkeypatch, answers):
    """Patch wr._get: `answers` maps a URL prefix to a Response or an
    exception; every call is recorded."""
    calls = []

    async def fake(url, headers=None, follow=True):
        calls.append(url)
        for prefix, answer in answers.items():
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(f"unexpected fetch {url}")
    monkeypatch.setattr(wr, "_get", fake)
    return calls


@pytest.mark.asyncio
async def test_page_is_extracted_cached_and_reused(monkeypatch, fresh):
    calls = _fake_get(monkeypatch, {"https://example.com/a": _response(ARTICLE)})
    page = await wr.read("https://example.com/a?utm_source=chat")
    assert page.source == "plain"
    assert page.text.startswith("An article\nhttps://example.com/a\n\n")
    assert "A real sentence about the topic." in page.text
    assert "Home | About" not in page.text and "Copyright" not in page.text
    key = cache_key("https://example.com/a")
    assert fresh.ttls[key] == 6 * 3600
    again = await wr.read("https://example.com/a")
    assert again == page
    assert calls == ["https://example.com/a"]


@pytest.mark.asyncio
async def test_failures_are_cached_briefly(monkeypatch, fresh):
    calls = _fake_get(monkeypatch, {"https://example.com/": wr.FetchError("timeout", "too slow")})
    for _ in range(2):
        with pytest.raises(wr.FetchError) as caught:
            await wr.read("https://example.com/")
        assert caught.value.message == "too slow"
    assert len(calls) == 1
    assert fresh.ttls[cache_key("https://example.com/")] == 300


@pytest.mark.asyncio
async def test_concurrent_reads_share_one_load(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def slow(url, headers=None, follow=True):
        calls.append(url)
        started.set()
        await release.wait()
        return _response(ARTICLE)
    monkeypatch.setattr(wr, "_get", slow)
    first = asyncio.create_task(wr.read("https://example.com/a"))
    await started.wait()
    second = asyncio.create_task(wr.read("https://example.com/a"))
    await asyncio.sleep(0)
    release.set()
    assert (await first) == (await second)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_a_cancelled_reader_does_not_cancel_the_load(monkeypatch):
    release = asyncio.Event()

    async def slow(url, headers=None, follow=True):
        await release.wait()
        return _response(ARTICLE)
    monkeypatch.setattr(wr, "_get", slow)
    first = asyncio.create_task(wr.read("https://example.com/a"))
    await asyncio.sleep(0.01)
    second = asyncio.create_task(wr.read("https://example.com/a"))
    await asyncio.sleep(0.01)
    first.cancel()
    release.set()
    assert "A real sentence" in (await second).text


@pytest.mark.asyncio
async def test_redis_down_still_reads(monkeypatch):
    monkeypatch.setattr(wr, "text_client", lambda: FakeRedis(broken=True))
    _fake_get(monkeypatch, {"https://example.com/a": _response(ARTICLE)})
    assert (await wr.read("https://example.com/a")).source == "plain"


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["http://127.0.0.1:6379/", "http://[::1]/", "http://169.254.169.254/x",
                                 "http://localhost:1/"])
async def test_private_addresses_are_refused(url):
    # Real _get and the real resolver: localhost resolves, then is dropped.
    with pytest.raises(wr.FetchError) as caught:
        await wr._get(url)
    assert caught.value.reason == "blocked"


class _FakeContent:
    def __init__(self, chunks):
        self.chunks = chunks

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, status=200, headers=None, chunks=(), content_type="text/html"):
        self.status = status
        self.headers = CIMultiDict(headers or {})
        self.content = _FakeContent(chunks)
        self.content_type = content_type

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_a_redirect_to_a_private_address_is_refused(monkeypatch):
    session = _FakeSession([_FakeResponse(302, {"Location": "http://10.0.0.7/admin"})])
    monkeypatch.setattr(wr, "_get_session", lambda: session)
    with pytest.raises(wr.FetchError) as caught:
        await wr._get("https://example.com/go")
    assert caught.value.reason == "blocked"
    assert session.urls == ["https://example.com/go"]


@pytest.mark.asyncio
async def test_redirects_are_followed_and_the_body_capped(monkeypatch):
    monkeypatch.setenv("WEB_FETCH_MAX_BYTES", "1000")
    session = _FakeSession([_FakeResponse(301, {"Location": "/b"}),
                            _FakeResponse(200, chunks=[b"x" * 600, b"y" * 600, b"z" * 600])])
    monkeypatch.setattr(wr, "_get_session", lambda: session)
    response = await wr._get("https://example.com/a")
    assert session.urls == ["https://example.com/a", "https://example.com/b"]
    assert response.url == "https://example.com/b"
    assert len(response.body) == 1000


@pytest.mark.asyncio
async def test_too_many_redirects(monkeypatch):
    session = _FakeSession([_FakeResponse(302, {"Location": "/loop"}) for _ in range(10)])
    monkeypatch.setattr(wr, "_get_session", lambda: session)
    with pytest.raises(wr.FetchError):
        await wr._get("https://example.com/loop")
    assert len(session.urls) == wr.MAX_REDIRECTS + 1


THIN = b"<html><head><title>App</title></head><body><div id=root></div><noscript>You need to enable JavaScript to run this app.</noscript></body></html>"


@pytest.mark.asyncio
async def test_thin_pages_go_to_the_browser(monkeypatch):
    monkeypatch.setenv("BROWSER_URL", "http://browser:8000")
    _fake_get(monkeypatch, {"https://app.example.com/": _response(THIN, url="https://app.example.com/")})
    rendered = []

    async def render(url):
        rendered.append(url)
        return "App", "Rendered text " * 50, url
    monkeypatch.setattr(wr, "_render", render)
    page = await wr.read("https://app.example.com/")
    assert page.source == "browser"
    assert rendered == ["https://app.example.com/"]


@pytest.mark.asyncio
async def test_without_a_browser_thin_pages_are_returned_as_they_are(monkeypatch):
    _fake_get(monkeypatch, {"https://app.example.com/": _response(THIN, url="https://app.example.com/")})
    page = await wr.read("https://app.example.com/")
    assert page.source == "plain"
    assert "enable JavaScript" in page.text


@pytest.mark.asyncio
async def test_render_is_skipped_when_no_browser_is_configured():
    assert await wr._render("https://example.com/") is None


@pytest.mark.asyncio
async def test_unreachable_sites_try_the_browser(monkeypatch):
    monkeypatch.setenv("BROWSER_URL", "http://browser:8000")
    _fake_get(monkeypatch, {"https://slow.example.com/": wr.FetchError("network", "unreachable")})

    async def render(url):
        return "", "Rendered " * 100, url
    monkeypatch.setattr(wr, "_render", render)
    assert (await wr.read("https://slow.example.com/")).source == "browser"


@pytest.mark.asyncio
async def test_error_pages_and_files(monkeypatch):
    _fake_get(monkeypatch, {
        "https://example.com/missing": _response(b"<html><body>Not found</body></html>", status=404),
        "https://example.com/doc.pdf": _response(b"%PDF", kind="application/pdf"),
        "https://example.com/data.json": _response(b'{"a": 1}', kind="application/json"),
    })
    with pytest.raises(wr.FetchError) as caught:
        await wr.read("https://example.com/missing")
    assert "HTTP 404" in caught.value.message
    with pytest.raises(wr.FetchError) as caught:
        await wr.read("https://example.com/doc.pdf")
    assert "a PDF" in caught.value.message
    assert (await wr.read("https://example.com/data.json")).text.endswith('{"a": 1}')


@pytest.mark.asyncio
async def test_not_a_web_address():
    with pytest.raises(wr.FetchError) as caught:
        await wr.read("file:///etc/passwd")
    assert caught.value.reason == "invalid"


@pytest.mark.asyncio
async def test_reddit_thread_reads_the_feed(monkeypatch):
    feed = (FIXTURES / "reddit_thread.rss").read_bytes()
    calls = _fake_get(monkeypatch, {"https://www.reddit.com/r/examplesub/comments/1abc234/.rss":
                                    _response(feed, kind="application/atom+xml",
                                              headers={"x-ratelimit-remaining": "5"})})
    page = await wr.read("https://old.reddit.com/r/examplesub/comments/1abc234/a_tool_i_made/")
    assert page.source == "reddit"
    assert page.title == "A tool I made"
    assert "u/example_user: Nice work" in page.text
    assert calls == ["https://www.reddit.com/r/examplesub/comments/1abc234/.rss"]


@pytest.mark.asyncio
async def test_reddit_rate_limit_is_respected(monkeypatch, fresh):
    feed = (FIXTURES / "reddit_listing.rss").read_bytes()
    calls = _fake_get(monkeypatch, {"https://www.reddit.com/r/examplesub/.rss":
                                    _response(feed, kind="application/atom+xml",
                                              headers={"x-ratelimit-remaining": "0.0",
                                                       "x-ratelimit-reset": "40"})})
    assert (await wr.read("https://www.reddit.com/r/examplesub/")).source == "reddit"
    with pytest.raises(wr.FetchError) as caught:
        await wr.read("https://www.reddit.com/r/othersub/")
    assert caught.value.reason == "rate_limited"
    assert len(calls) == 1  # the second read never asked Reddit
    assert fresh.ttls[cache_key("https://www.reddit.com/r/othersub/")] <= 40


@pytest.mark.asyncio
async def test_reddit_429(monkeypatch):
    _fake_get(monkeypatch, {"https://www.reddit.com/": _response(b"", status=429,
                                                                headers={"x-ratelimit-reset": "12"})})
    with pytest.raises(wr.FetchError) as caught:
        await wr.read("https://www.reddit.com/r/examplesub/")
    assert caught.value.reason == "rate_limited"
    assert wr._reddit_retry_at > 0


@pytest.mark.asyncio
async def test_reddit_share_link_is_followed_to_its_thread(monkeypatch):
    feed = (FIXTURES / "reddit_thread.rss").read_bytes()
    calls = _fake_get(monkeypatch, {
        "https://www.reddit.com/r/examplesub/s/AbC": _response(status=301, headers={
            "Location": "https://www.reddit.com/r/examplesub/comments/1abc234/a_tool_i_made/?share_id=x"}),
        "https://www.reddit.com/r/examplesub/comments/1abc234/.rss": _response(feed, kind="application/atom+xml"),
    })
    page = await wr.read("https://www.reddit.com/r/examplesub/s/AbC")
    assert page.title == "A tool I made"
    assert len(calls) == 2


def _youtube(monkeypatch, found):
    _fake_get(monkeypatch, {"https://www.youtube.com/oembed": _response(
        json.dumps({"title": "A video", "author_name": "A channel"}).encode(), kind="application/json")})
    monkeypatch.setattr(wr, "_fetch_transcript", lambda video_id: found)


@pytest.mark.asyncio
async def test_youtube_captions_are_cached_for_long(monkeypatch, fresh):
    _youtube(monkeypatch, ("English", True, [{"text": "hello there", "start": 0.0}]))
    page = await wr.read("https://youtu.be/dQw4w9WgXcQ?si=share")
    assert page.source == "youtube"
    assert "YouTube video: A video" in page.text and "[0:00] hello there" in page.text
    assert fresh.ttls[cache_key("https://www.youtube.com/watch?v=dQw4w9WgXcQ")] == 30 * 24 * 3600


@pytest.mark.asyncio
async def test_youtube_without_captions(monkeypatch, fresh):
    _youtube(monkeypatch, None)
    page = await wr.read("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert "no captions" in page.text
    assert fresh.ttls[cache_key("https://www.youtube.com/watch?v=dQw4w9WgXcQ")] == 6 * 3600


@pytest.mark.asyncio
async def test_youtube_blocked(monkeypatch):
    import youtube_transcript_api as yta
    _fake_get(monkeypatch, {"https://www.youtube.com/oembed": wr.FetchError("http", "x")})

    def blocked(video_id):
        raise yta.RequestBlocked(video_id)
    monkeypatch.setattr(wr, "_fetch_transcript", blocked)
    with pytest.raises(wr.FetchError) as caught:
        await wr.read("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert caught.value.reason == "blocked_by_site"


def test_fetch_transcript_prefers_english_written_by_a_person(monkeypatch):
    import youtube_transcript_api as yta

    def track(code, generated):
        return SimpleNamespace(language_code=code, language=code, is_generated=generated,
                               fetch=lambda: [SimpleNamespace(text=code, start=0.0)])

    class FakeApi:
        def list(self, video_id):
            return [track("de", False), track("en", True), track("en-GB", False)]
    monkeypatch.setattr(yta, "YouTubeTranscriptApi", FakeApi)
    assert wr._fetch_transcript("x") == ("en-GB", False, [{"text": "en-GB", "start": 0.0}])


@pytest.mark.asyncio
async def test_steam_app(monkeypatch):
    _fake_get(monkeypatch, {
        "https://store.steampowered.com/api/appdetails": _response(
            (FIXTURES / "steam_appdetails.json").read_bytes(), kind="application/json"),
        "https://store.steampowered.com/app/620/": _response(
            (FIXTURES / "steam_store_reviews.html").read_bytes()),
    })
    page = await wr.read("https://store.steampowered.com/app/620/Portal_2/")
    assert page.source == "steam"
    assert "User reviews: Overwhelmingly Positive" in page.text


@pytest.mark.asyncio
async def test_steam_app_without_details_is_read_as_a_page(monkeypatch):
    _fake_get(monkeypatch, {
        "https://store.steampowered.com/api/appdetails": _response(b'{"999": {"success": false}}',
                                                                   kind="application/json"),
        "https://store.steampowered.com/app/999/": _response(ARTICLE),
    })
    assert (await wr.read("https://store.steampowered.com/app/999/")).source == "plain"
