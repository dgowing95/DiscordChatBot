"""Browser-service policy that needs no browser: env parsing, which addresses
and requests a page may reach, and the size of what is sent back.

Pure module (stdlib only), the speechservice/speech_params.py pattern: CI
installs no Playwright or Chromium, so everything that can be decided from
strings lives here and is tested in browserservice/tests/. main.py keeps
everything that touches the browser.
"""
import ipaddress
import os
from urllib.parse import urlsplit

# Requests a page makes that are never needed to read its text. Blocking them
# saves most of the bandwidth and much of the memory a render costs.
BLOCKED_RESOURCE_TYPES = frozenset({"image", "media", "font"})

def user_agent(browser_version: str) -> str:
    """A desktop Chrome user agent for the bundled browser's own version.

    Chromium's headless shell otherwise announces itself as
    "HeadlessChrome", which many sites turn away on sight. Built from the
    real version so it never goes stale against the browser Playwright
    installed."""
    major = (browser_version or "").split(".", 1)[0]
    major = major if major.isdigit() else "150"
    return ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            f"(KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36")


def env_positive_int(name: str, default: int) -> int:
    try:
        value = int(str(os.environ.get(name, "")).strip())
    except ValueError:
        return default
    return value if value > 0 else default


def env_positive_float(name: str, default: float) -> float:
    try:
        value = float(str(os.environ.get(name, "")).strip())
    except ValueError:
        return default
    return value if value > 0 else default


def env_str(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


def is_public_ip(address: str) -> bool:
    """True only for an address on the public internet.

    The browser renders whatever URL the model was given, and a page can
    point its own requests anywhere. Without this check a link could make
    it read Redis, the LLM server or the cloud metadata address from inside
    the cluster. `is_global` already excludes private, loopback, link-local,
    shared (CGNAT) and reserved ranges; multicast is excluded on top, and an
    IPv4 address wrapped in IPv6 (::ffff:10.0.0.1) is judged as the IPv4
    address it is."""
    try:
        ip = ipaddress.ip_address(str(address).split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def web_host(url: str) -> str | None:
    """The host of an http(s) URL, or None for anything else (file:, data:,
    chrome:, a URL with no host)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return parts.hostname


def literal_ip(host: str) -> str | None:
    """`host` itself when it is an IP address (brackets already stripped by
    urlsplit), else None."""
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None
    return host


def cap_html(html: str, max_bytes: int) -> str:
    """At most `max_bytes` of UTF-8, cut on a character boundary."""
    data = html.encode("utf-8", errors="replace")
    if len(data) <= max_bytes:
        return html
    return data[:max_bytes].decode("utf-8", errors="ignore")
