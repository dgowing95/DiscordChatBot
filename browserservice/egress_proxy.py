"""The browser's only way out: a small HTTP proxy that connects to public
addresses and nothing else (stdlib only).

Chromium is launched with this as its proxy, loopback included, so every
connection it makes passes through here: the page, its scripts' requests,
WebSockets, and every hop of a redirect. Playwright's page.route was tried
first and is not enough: its handler sees only the first URL of a redirect,
so a public link answering 302 to http://10.x/ or http://redis:6379/ was
followed by Chromium unchecked, and a page script could fetch() such a link
and write the internal response into the page (both reproduced, then
fixed by this).

The proxy resolves each host itself and connects to the checked address,
so a DNS answer that changes between check and connect cannot get round it.
HTTPS arrives as CONNECT and is tunnelled as is (end to end, never
decrypted). Plain HTTP is forwarded one upstream connection per client
connection, with "Connection: close", so a kept-alive connection cannot be
reused for a different host.
"""
import asyncio
import ipaddress
import logging
import socket
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

HEAD_TIMEOUT = 15
CONNECT_TIMEOUT = 10
_HOP_HEADERS = {"proxy-connection", "proxy-authorization", "connection", "keep-alive"}


class Refused(Exception):
    pass


def split_host_port(target: str, default_port: int):
    """("example.com", 443) from "example.com:443"; IPv6 in brackets."""
    if target.startswith("["):
        host, _, rest = target[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif target.count(":") == 1:
        host, _, port = target.partition(":")
    else:
        host, port = target, ""
    return host, int(port) if port.isdigit() else default_port


class EgressProxy:
    def __init__(self, is_allowed, resolve=None):
        self._is_allowed = is_allowed
        self._resolve = resolve
        self._server = None

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> int:
        self._server = await asyncio.start_server(self._handle, host, port)
        return self._server.sockets[0].getsockname()[1]

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _addresses(self, host: str, port: int) -> list:
        try:
            ipaddress.ip_address(host)
            return [host]
        except ValueError:
            pass
        if self._resolve is not None:
            return await self._resolve(host, port)
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
        return list(dict.fromkeys(info[4][0] for info in infos))

    async def _open(self, host: str, port: int):
        """A connection to `host`, to an address that was checked. Refused
        unless EVERY address the name has is allowed: a name with one
        private answer among public ones is not to be trusted."""
        try:
            addresses = await self._addresses(host, port)
        except OSError:
            raise Refused(f"{host} does not resolve") from None
        if not addresses or not all(self._is_allowed(a) for a in addresses):
            raise Refused(f"{host} is not on the public internet")
        error = None
        for address in addresses:
            try:
                return await asyncio.wait_for(asyncio.open_connection(address, port), CONNECT_TIMEOUT)
            except (OSError, asyncio.TimeoutError) as e:
                error = e
        raise OSError(f"could not connect to {host}: {error}")

    @staticmethod
    async def _pipe(reader, writer):
        try:
            while True:
                data = await reader.read(65536)
                if not data:
                    break
                writer.write(data)
                await writer.drain()
        except (OSError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()

    async def _handle(self, reader, writer):
        upstream = None
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT)
            request_line, *header_lines = head.decode("latin-1").split("\r\n")
            method, target, version = request_line.split(" ", 2)
            if method.upper() == "CONNECT":
                host, port = split_host_port(target, 443)
                upstream = await self._open(host, port)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                url = urlsplit(target)
                if url.scheme != "http" or not url.hostname:
                    raise Refused(f"not a proxied http URL: {target[:100]}")
                upstream = await self._open(url.hostname, url.port or 80)
                path = (url.path or "/") + (f"?{url.query}" if url.query else "")
                headers = [h for h in header_lines
                           if h and h.split(":", 1)[0].strip().lower() not in _HOP_HEADERS]
                lines = [f"{method} {path} {version}", *headers, "Connection: close", "", ""]
                upstream[1].write("\r\n".join(lines).encode("latin-1"))
            await asyncio.gather(self._pipe(reader, upstream[1]), self._pipe(upstream[0], writer))
        except Refused as e:
            logger.info(f"refused: {e}")
            await self._answer(writer, b"403 Forbidden")
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                ValueError, OSError):
            await self._answer(writer, b"502 Bad Gateway")
        finally:
            if upstream is not None:
                upstream[1].close()
            writer.close()

    @staticmethod
    async def _answer(writer, status: bytes):
        try:
            writer.write(b"HTTP/1.1 " + status + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        except OSError:
            pass
