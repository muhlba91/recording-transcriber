"""Tests for the OpenAI-compatible chat backend."""

import json

import httpx
import pytest

from recording_transcriber.backends.chat_api import ChatApiBackend
from recording_transcriber.models.usage import Usage
from tests.backends.conftest import Recorder

_SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}}


def _answer(content='{"a": "b"}', finish="stop", usage=None):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": usage
        or {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 4},
        },
    }


def _backend(recorder) -> ChatApiBackend:
    return ChatApiBackend(
        "litellm", "https://llm.example/v1/", "secret", "own logs", recorder.client()
    )


def test_complete():
    recorder = Recorder(_answer())
    response = _backend(recorder).complete("Write.", _SCHEMA, "big")
    assert response.text == '{"a": "b"}'
    assert response.status == "completed"
    assert response.usage.input_text_tokens == 20
    assert (response.usage.output_tokens, response.usage.thought_tokens) == (6, 4)
    request = recorder.requests[-1]
    assert str(request.url) == "https://llm.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(recorder.body())
    assert body["model"] == "big"
    assert body["messages"] == [{"role": "user", "content": "Write."}]
    assert body["response_format"]["json_schema"]["schema"] == _SCHEMA


def test_complete_truncated_and_unknown_status():
    assert (
        _backend(Recorder(_answer(finish="length"))).complete("p", _SCHEMA, "m").status
        == "incomplete"
    )
    assert (
        _backend(Recorder(_answer(finish=None, usage={})))
        .complete("p", _SCHEMA, "m")
        .status
        == "unknown"
    )


def test_describe():
    assert _backend(Recorder({})).describe() == {
        "provider": "litellm",
        "endpoint": "https://llm.example/v1/chat/completions",
        "logging": "own logs",
    }


def test_complete_http_error():
    recorder = Recorder({"error": "overloaded"}, status=503)
    assert _backend(recorder).complete("p", _SCHEMA, "m") is None


def test_complete_connection_error():
    def refuse(request):
        raise httpx.ConnectError("refused")

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    backend = ChatApiBackend("litellm", "https://x", "k", "p", client)
    assert backend.complete("p", _SCHEMA, "m") is None


@pytest.mark.parametrize("data", [{}, {"choices": []}, _answer(content=None)])
def test_complete_invalid_answer(data):
    assert _backend(Recorder(data)).complete("p", _SCHEMA, "m") is None


def test_parse_missing_counts():
    answer = _answer(usage={"completion_tokens_details": {"reasoning_tokens": None}})
    response = _backend(Recorder(answer)).complete("p", _SCHEMA, "m")
    assert response.usage == Usage()


def test_complete_not_json():
    client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<"))
    )
    assert (
        ChatApiBackend("h", "https://x", "k", "p", client).complete("p", _SCHEMA, "m")
        is None
    )


def test_default_client():
    backend = ChatApiBackend("openai", "https://x", "k", "p")
    assert backend.describe()["endpoint"] == "https://x/chat/completions"


def test_check():
    recorder = Recorder({"data": [{"id": "big"}, {"id": "small"}]})
    assert _backend(recorder).check("big")
    request = recorder.requests[-1]
    assert str(request.url) == "https://llm.example/v1/models"
    assert request.headers["authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    "entry",
    [
        {"id": "big-2", "aliases": ["big"]},
        {"id": "models/big"},
        {"id": "big", "aliases": None},
    ],
)
def test_check_aliases(entry):
    assert _backend(Recorder({"data": [entry]})).check("big")


@pytest.mark.parametrize(
    ("data", "status"),
    [({"data": [{"id": "other"}]}, 200), ({"error": "no"}, 401)],
)
def test_check_fails(data, status):
    assert not _backend(Recorder(data, status)).check("big")


@pytest.mark.parametrize(
    ("data", "status"),
    [({"error": "scope"}, 403), ({"error": "none"}, 404), ({"models": []}, 200)],
)
def test_check_unlisted_passes(data, status):
    assert _backend(Recorder(data, status)).check("big")


def test_check_unreachable():
    def refuse(request):
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(refuse))
    backend = ChatApiBackend("litellm", "https://llm.example/v1", "secret", "", client)
    assert not backend.check("big")


def test_close():
    client = Recorder({}).client()
    ChatApiBackend("litellm", "https://llm.example/v1", "secret", "", client).close()
    assert client.is_closed
