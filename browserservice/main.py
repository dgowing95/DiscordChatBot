"""Standalone browser service: renders a web page in headless Chromium and
returns its HTML, for pages that only fill in their content with JavaScript.

Small FastAPI app run as its own pod/container. core's fetch_url tool
(core/classes/web_reader.py) calls it only when its plain fetch came back
with almost no text, and extracts the text from the returned HTML itself:

    POST /render   {"url": "https://..."}
                   ->  {"url", "status", "title", "html"}
    GET  /health   ->  200 once the browser is running, 503 before

It loads pages from anywhere on the internet, and a page decides for itself
what else to request. So Chromium's only way out is egress_proxy.py, run in
this process: every connection it makes (the page, its scripts, every hop
of a redirect, WebSockets) goes through it, and it connects to public
addresses only (its docstring says why page.route alone was not enough).
page.route still drops images, media and fonts, which a reader never needs,
and service workers are blocked. Each render gets a fresh browser context,
so no cookies or storage carry from one page to the next. The chart's
NetworkPolicy blocks private ranges as well, where the cluster enforces it.

Configuration (all env vars optional):
  BROWSER_CONCURRENCY     pages rendered at once (default: 2)
  BROWSER_QUEUE_SIZE      renders allowed to wait, then 503 (default: 8)
  BROWSER_TIMEOUT         seconds for one whole render (default: 30)
  BROWSER_SETTLE_SECONDS  seconds to wait for the network to go quiet after
                          the page loads, so scripts can fill it in (default: 5)
  BROWSER_MAX_HTML_BYTES  most HTML sent back (default: 5000000)
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeout
from playwright.async_api import async_playwright
from pydantic import BaseModel

from egress_proxy import EgressProxy
from render_policy import (
    BLOCKED_RESOURCE_TYPES,
    cap_html,
    env_positive_float,
    env_positive_int,
    is_public_ip,
    literal_ip,
    user_agent,
    web_host,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONCURRENCY = env_positive_int("BROWSER_CONCURRENCY", 2)
QUEUE_SIZE = env_positive_int("BROWSER_QUEUE_SIZE", 8)
TIMEOUT = env_positive_float("BROWSER_TIMEOUT", 30.0)
SETTLE_SECONDS = env_positive_float("BROWSER_SETTLE_SECONDS", 5.0)
MAX_HTML_BYTES = env_positive_int("BROWSER_MAX_HTML_BYTES", 5_000_000)

_playwright = None
_proxy = None
_proxy_port = None
_browser = None
_launch_lock = asyncio.Lock()
_render_gate = asyncio.Semaphore(CONCURRENCY)
_waiting = 0
# Renders in a row that ran out of time. A browser that is still connected
# but no longer gets anything done (a renderer crash loop, memory pressure)
# shows up as this, and is thrown away so the next render starts a new one.
_timeouts_in_a_row = 0
MAX_TIMEOUTS_IN_A_ROW = 3


async def _get_browser():
    """The running browser, launched again if it has crashed or exited."""
    global _browser
    async with _launch_lock:
        if _browser is None or not _browser.is_connected():
            # "<-loopback>" sends loopback through the proxy too; Chromium
            # otherwise connects to localhost directly.
            _browser = await _playwright.chromium.launch(
                args=["--disable-gpu"],
                proxy={"server": f"http://127.0.0.1:{_proxy_port}", "bypass": "<-loopback>"})
            logger.info(f"browser started: Chromium {_browser.version}")
        return _browser


async def _discard(browser) -> None:
    """Close `browser` (if it is still the current one) so the next render
    launches a fresh one."""
    global _browser
    async with _launch_lock:
        if _browser is browser:
            _browser = None
    try:
        await asyncio.wait_for(browser.close(), 10)
    except Exception:
        pass
    logger.warning("browser discarded; the next render starts a new one")


async def _resolves_public(host: str) -> bool:
    """Every address `host` resolves to is public. Only an early, clear
    refusal for the URL asked for; the proxy enforces it for everything."""
    if literal_ip(host):
        return is_public_ip(host)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError:
        return False
    return bool(infos) and all(is_public_ip(info[4][0]) for info in infos)


async def _drop_unneeded(route):
    """Images, media, fonts and non-web URLs are never fetched. Where a
    request may go is the proxy's decision, not this one's: route handlers
    do not see redirects."""
    request = route.request
    if request.resource_type in BLOCKED_RESOURCE_TYPES or web_host(request.url) is None:
        await route.abort()
        return
    await route.continue_()


async def _render(url: str) -> dict:
    browser = await _get_browser()
    try:
        context = await asyncio.wait_for(browser.new_context(
            user_agent=user_agent(browser.version),
            locale="en-US",
            viewport={"width": 1280, "height": 900},
            # Service-worker requests bypass page.route, so they are not allowed.
            service_workers="block",
        ), 15)
    except Exception:
        # Opening a context does not depend on any website: if it fails, the
        # browser itself is broken.
        await _discard(browser)
        raise
    try:
        await context.route("**/*", _drop_unneeded)
        page = await context.new_page()
        response = await page.goto(url, wait_until="domcontentloaded",
                                   timeout=TIMEOUT * 1000)
        try:
            await page.wait_for_load_state("networkidle", timeout=SETTLE_SECONDS * 1000)
        except PlaywrightTimeout:
            pass  # a page that never goes quiet (ads, polling) is read as it is
        return {
            "url": page.url,
            "status": response.status if response else 0,
            "title": await page.title(),
            "html": cap_html(await page.content(), MAX_HTML_BYTES),
        }
    finally:
        await context.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _playwright, _proxy, _proxy_port
    _proxy = EgressProxy(is_public_ip)
    _proxy_port = await _proxy.start()
    _playwright = await async_playwright().start()
    try:
        await _get_browser()
    except Exception:
        # /health stays 503 and the next render tries again.
        logger.exception("browser failed to start")
    yield
    if _browser is not None:
        await _browser.close()
    await _playwright.stop()
    await _proxy.close()


app = FastAPI(lifespan=lifespan)


class RenderRequest(BaseModel):
    url: str


@app.get("/health")
async def health():
    """200 while a browser is running. A browser that has exited is started
    again here: once readiness fails, the Service stops sending renders, so
    a render could not be what restarts it."""
    try:
        browser = await asyncio.wait_for(_get_browser(), 30)
    except Exception as e:
        logger.warning(f"browser could not be started: {e}")
        return JSONResponse({"status": "starting"}, status_code=503)
    return {"status": "ok", "browser": browser.version}


@app.post("/render")
async def render(body: RenderRequest):
    global _waiting, _timeouts_in_a_row
    host = web_host(body.url)
    if host is None:
        raise HTTPException(400, "only http and https URLs can be rendered")
    if not await _resolves_public(host):
        raise HTTPException(403, "that address is not on the public internet")
    if _waiting >= QUEUE_SIZE:
        raise HTTPException(503, "busy")
    _waiting += 1
    try:
        await _render_gate.acquire()
    finally:
        _waiting -= 1
    try:
        result = await asyncio.wait_for(_render(body.url), TIMEOUT + SETTLE_SECONDS + 5)
        _timeouts_in_a_row = 0
        return result
    except (asyncio.TimeoutError, PlaywrightTimeout):
        _timeouts_in_a_row += 1
        if _timeouts_in_a_row >= MAX_TIMEOUTS_IN_A_ROW and _browser is not None:
            _timeouts_in_a_row = 0
            await _discard(_browser)
        raise HTTPException(504, "the page took too long to load") from None
    except PlaywrightError as e:
        logger.warning(f"render failed for {body.url}: {e}")
        raise HTTPException(502, "the page could not be loaded") from None
    finally:
        _render_gate.release()
