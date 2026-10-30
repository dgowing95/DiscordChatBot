"""Tests for the browser service's stdlib-only policy (browserservice/render_policy.py).

To run this pytest file from the command line, use:
    pytest browserservice/tests/render_policy_tests.py
"""
import pytest

import render_policy as rp


@pytest.mark.parametrize("address, public", [
    ("93.184.215.14", True),
    ("2606:4700:4700::1111", True),
    ("10.1.2.3", False),
    ("192.168.0.10", False),
    ("127.0.0.1", False),
    ("169.254.169.254", False),
    ("100.100.1.1", False),
    ("::1", False),
    ("::ffff:192.168.0.1", False),
    ("fe80::1%eth0", False),
    ("239.1.1.1", False),
    ("nonsense", False),
])
def test_is_public_ip(address, public):
    assert rp.is_public_ip(address) is public


@pytest.mark.parametrize("url, host", [
    ("https://example.com/a", "example.com"),
    ("http://[::1]:8000/", "::1"),
    ("file:///etc/passwd", None),
    ("data:text/html,hi", None),
    ("chrome://settings", None),
    ("https://", None),
])
def test_web_host(url, host):
    assert rp.web_host(url) == host


def test_literal_ip():
    assert rp.literal_ip("10.0.0.1") == "10.0.0.1"
    assert rp.literal_ip("::1") == "::1"
    assert rp.literal_ip("example.com") is None


def test_blocked_resource_types():
    assert {"image", "media", "font"} <= rp.BLOCKED_RESOURCE_TYPES
    assert "document" not in rp.BLOCKED_RESOURCE_TYPES and "script" not in rp.BLOCKED_RESOURCE_TYPES


def test_cap_html_cuts_on_a_character_boundary():
    assert rp.cap_html("short", 100) == "short"
    capped = rp.cap_html("é" * 10, 5)  # 2 bytes each
    assert capped == "éé"


def test_user_agent_follows_the_browser_version():
    assert "Chrome/156.0.0.0" in rp.user_agent("156.0.8078.4")
    assert "HeadlessChrome" not in rp.user_agent("156.0.8078.4")
    assert "Chrome/150.0.0.0" in rp.user_agent("")


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("X_INT", "0")
    monkeypatch.setenv("X_FLOAT", "2.5")
    assert rp.env_positive_int("X_INT", 3) == 3
    assert rp.env_positive_float("X_FLOAT", 1.0) == 2.5
    assert rp.env_str("X_MISSING", "d") == "d"
