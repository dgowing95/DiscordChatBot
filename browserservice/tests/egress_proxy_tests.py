"""Tests for the browser's egress proxy (browserservice/egress_proxy.py), with
real sockets on loopback. "Allowed" is 127.0.0.1 here, standing in for a
public address; anything else stands in for a private one.

To run this pytest file from the command line, use:
    pytest browserservice/tests/egress_proxy_tests.py
"""
import asyncio

import pytest

from egress_proxy import EgressProxy, split_host_port


def _allowed(address):
    return address == "127.0.0.1"


async def _upstream(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _ask(proxy_port, raw: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(raw)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(65536), 5)
    writer.close()
    return data


@pytest.mark.parametrize("target, expected", [
    ("example.com:443", ("example.com", 443)),
    ("example.com", ("example.com", 443)),
    ("[2606:4700::1]:8443", ("2606:4700::1", 8443)),
    ("2606:4700::1", ("2606:4700::1", 443)),
])
def test_split_host_port(target, expected):
    assert split_host_port(target, 443) == expected


@pytest.mark.asyncio
async def test_connect_tunnels_to_an_allowed_address():
    async def echo(reader, writer):
        writer.write(b"upstream:" + await reader.read(100))
        await writer.drain()
        writer.close()
    server, port = await _upstream(echo)
    proxy = EgressProxy(_allowed)
    proxy_port = await proxy.start()
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
    writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    await writer.drain()
    assert (await reader.readuntil(b"\r\n\r\n")).startswith(b"HTTP/1.1 200")
    writer.write(b"hello")
    await writer.drain()
    assert await asyncio.wait_for(reader.read(100), 5) == b"upstream:hello"
    writer.close()
    await proxy.close()
    server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_line", [
    b"CONNECT 127.0.0.2:443 HTTP/1.1",
    b"GET http://127.0.0.2/secret HTTP/1.1",
    b"GET http://[::1]/secret HTTP/1.1",
    b"GET ftp://127.0.0.1/x HTTP/1.1",
    b"GET /relative HTTP/1.1",
])
async def test_private_and_odd_targets_are_refused(request_line):
    proxy = EgressProxy(_allowed)
    port = await proxy.start()
    assert (await _ask(port, request_line + b"\r\nHost: x\r\n\r\n")).startswith(b"HTTP/1.1 403")
    await proxy.close()


@pytest.mark.asyncio
async def test_a_name_with_any_private_address_is_refused():
    async def resolve(host, port):
        return ["127.0.0.1", "10.0.0.5"]
    proxy = EgressProxy(_allowed, resolve=resolve)
    port = await proxy.start()
    assert (await _ask(port, b"CONNECT mixed.example:443 HTTP/1.1\r\n\r\n")).startswith(b"HTTP/1.1 403")
    await proxy.close()


@pytest.mark.asyncio
async def test_names_are_resolved_by_the_proxy_and_connected_by_address():
    seen = []

    async def handler(reader, writer):
        seen.append(await reader.readuntil(b"\r\n\r\n"))
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()
    server, upstream_port = await _upstream(handler)

    async def resolve(host, port):
        assert host == "site.example"
        return ["127.0.0.1"]
    proxy = EgressProxy(_allowed, resolve=resolve)
    port = await proxy.start()
    answer = await _ask(port, (f"GET http://site.example:{upstream_port}/a/b?c=1 HTTP/1.1\r\n"
                               "Host: site.example\r\nProxy-Connection: keep-alive\r\n"
                               "Connection: keep-alive\r\nUser-Agent: t\r\n\r\n").encode())
    assert answer.endswith(b"ok")
    request = seen[0].decode()
    assert request.startswith("GET /a/b?c=1 HTTP/1.1\r\n")
    assert "Host: site.example" in request and "User-Agent: t" in request
    assert "Proxy-Connection" not in request and "keep-alive" not in request
    assert "Connection: close" in request
    await proxy.close()
    server.close()


@pytest.mark.asyncio
async def test_garbage_gets_a_502():
    proxy = EgressProxy(_allowed)
    port = await proxy.start()
    assert (await _ask(port, b"nonsense\r\n\r\n")).startswith(b"HTTP/1.1 502")
    await proxy.close()
