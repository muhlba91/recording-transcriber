"""Shared fixtures for the backend tests."""

import httpx
import pytest


class Recorder:
    """An httpx mock transport handler that answers with fixed JSON and keeps the
    requests."""

    def __init__(self, data, status=200):
        """Initialize the recorder.

        data: the JSON answer
        status: the HTTP status (default: 200)"""
        self.data = data
        self.status = status
        self.requests = []

    def __call__(self, request):
        """Record the request and answer."""
        self.requests.append(request)
        return httpx.Response(self.status, json=self.data)

    def client(self) -> httpx.Client:
        """Return a client that sends every request to this recorder."""
        return httpx.Client(transport=httpx.MockTransport(self))

    def body(self) -> str:
        """Return the last request body as text."""
        return self.requests[-1].content.decode()


def text_client(status: int, text: str) -> httpx.Client:
    """Return a client that answers every request with a plain text body.

    status: the HTTP status
    text: the body"""
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, text=text))
    )


@pytest.fixture
def clip(tmp_path):
    """A fake mp3 clip."""
    path = tmp_path / "clip.mp3"
    path.write_bytes(b"mp3")
    return path
