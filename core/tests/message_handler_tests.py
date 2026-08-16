import os
import sys

# message_handler pulls in sibling modules via the production "classes.*" import
# style (the app runs with cwd=core/), so make core/ importable for tests too.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.classes.message_handler import MessageHandler, MAX_IMAGES_PER_MESSAGE


# To run this pytest file from the command line, use:
# PYTHONPATH=$(pwd) pytest core/tests/message_handler_tests.py


def _handler():
    # Skip __init__ (it needs REDIS_HOST); download_image_parts does not use self state.
    return MessageHandler.__new__(MessageHandler)


def _attachment(url="http://cdn.example/x.png", content_type="image/png"):
    att = MagicMock()
    att.url = url
    att.proxy_url = url
    att.content_type = content_type
    return att


def _mock_client_session(payload=b"\x89PNG", status=200,
                         content_type_header="image/png", raise_exc=None,
                         content_type_headers=None):
    if content_type_headers is None:
        content_type_headers = [content_type_header]
    response = MagicMock()
    response.status = status
    response.read = AsyncMock(return_value=payload)
    if raise_exc:
        response.read.side_effect = raise_exc
    response.__aenter__ = AsyncMock(return_value=response)
    response.__aexit__ = AsyncMock(return_value=False)

    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    def get(url, **kwargs):
        # One Content-Type header per call; the last one repeats.
        # call_count is 1-based while side_effect is running.
        index = session.get.call_count - 1
        response.headers = {"Content-Type": content_type_headers[min(index, len(content_type_headers) - 1)]}
        return response

    session.get = MagicMock(side_effect=get)
    return session


@pytest.mark.asyncio
async def test_image_attachment_becomes_base64_data_url_part():
    payload = b"\x89PNG\r\nfake"
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(payload=payload)):
        parts = await handler.download_image_parts([_attachment()])

    assert len(parts) == 1
    expected = "data:image/png;base64," + base64.b64encode(payload).decode()
    assert parts[0] == {"type": "input_image", "image_url": expected}


@pytest.mark.asyncio
async def test_non_image_attachments_are_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session()):
        parts = await handler.download_image_parts([
            _attachment(url="http://cdn.example/v.mp4", content_type="video/mp4"),
            _attachment(url="http://cdn.example/f.pdf", content_type="application/pdf"),
            _attachment(content_type=None),
        ])

    assert parts == []


@pytest.mark.asyncio
async def test_download_failure_is_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(raise_exc=ConnectionError("down"))):
        parts = await handler.download_image_parts([_attachment()])
    assert parts == []


@pytest.mark.asyncio
async def test_non_200_response_is_skipped():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(status=404)):
        parts = await handler.download_image_parts([_attachment()])
    assert parts == []


@pytest.mark.asyncio
async def test_max_images_per_message_cap():
    handler = _handler()
    attachments = [_attachment(url=f"http://cdn.example/{i}.png")
                   for i in range(MAX_IMAGES_PER_MESSAGE + 3)]
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session()):
        parts = await handler.download_image_parts(attachments)
    assert len(parts) == MAX_IMAGES_PER_MESSAGE


@pytest.mark.asyncio
async def test_mixed_good_and_bad_attachments():
    handler = _handler()
    with patch("core.classes.message_handler.aiohttp.ClientSession",
               return_value=_mock_client_session(content_type_headers=["image/png", "image/jpeg"])):
        parts = await handler.download_image_parts([
            _attachment(url="http://cdn.example/a.png"),
            _attachment(url="http://cdn.example/b.jpg", content_type="image/jpeg"),
            _attachment(url="http://cdn.example/c.txt", content_type="text/plain"),
        ])
    assert [p["image_url"].split(";")[0] for p in parts] == ["data:image/png", "data:image/jpeg"]
