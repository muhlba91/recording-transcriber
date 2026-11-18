"""Tests for the shared speech-to-text API base."""

import httpx

from recording_transcriber.backends.mistral_speech import MistralSpeechBackend
from recording_transcriber.backends.openai_speech import OpenAiSpeechBackend
from tests.backends.conftest import Recorder, text_client


def test_transcribe_speech_http_status_error(clip):
    backend = OpenAiSpeechBackend("key", client=Recorder({"e": 1}, 400).client())
    assert backend.transcribe_speech(clip, "m", []) is None


def test_transcribe_speech_request(clip):
    recorder = Recorder({"text": "x"})
    backend = OpenAiSpeechBackend("key", client=recorder.client())
    backend.transcribe_speech(clip, "whisper-x", [])
    request = recorder.requests[0]
    assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer key"
    assert 'name="file"; filename="clip.mp3"' in recorder.body()
    assert "whisper-x" in recorder.body()


def test_transcribe_speech_transport_error(clip):
    def handler(request):
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    assert (
        OpenAiSpeechBackend("key", client=client).transcribe_speech(clip, "m", [])
        is None
    )


def test_transcribe_speech_unexpected_shape(clip):
    backend = MistralSpeechBackend("key", client=Recorder({"text": "x"}).client())
    assert backend.transcribe_speech(clip, "m", []) is None


def test_transcribe_speech_not_json(clip):
    backend = OpenAiSpeechBackend("key", client=text_client(200, "no"))
    assert backend.transcribe_speech(clip, "m", []) is None


def test_language_primary_subtag(clip):
    recorder = Recorder({"text": "x"})
    backend = OpenAiSpeechBackend("key", "de-AT", client=recorder.client())
    backend.transcribe_speech(clip, "m", [])
    assert backend.language == "de"
    assert 'name="language"\r\n\r\nde\r\n' in recorder.body()


def test_check():
    assert MistralSpeechBackend("key", client=Recorder({}).client()).check("voxtral")
    assert not OpenAiSpeechBackend("", client=Recorder({}).client()).check("m")


def test_describe():
    backend = MistralSpeechBackend("key")
    description = backend.describe()
    assert set(description) == {"provider", "endpoint", "logging"}
    assert description["provider"] == "mistral"
    assert description["endpoint"] == (
        "https://api.eu.mistral.ai/v1/audio/transcriptions"
    )
    backend.close()
