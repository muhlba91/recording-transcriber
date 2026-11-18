"""Tests for the OpenAI speech-to-text backend."""

from recording_transcriber.backends.openai_speech import OpenAiSpeechBackend
from tests.backends.conftest import Recorder


def test_transcribe_speech_returns_untimed_text(clip):
    recorder = Recorder({"text": " Guten Morgen. ", "usage": {"seconds": 3}})
    backend = OpenAiSpeechBackend("key", client=recorder.client())
    result = backend.transcribe_speech(clip, "gpt-transcribe", [])
    assert [(s.speaker, s.start_s, s.text) for s in result.segments] == [
        (None, None, "Guten Morgen.")
    ]


def test_transcribe_speech_sends_key_and_vocabulary(clip):
    recorder = Recorder({"text": "x"})
    backend = OpenAiSpeechBackend("key", client=recorder.client())
    backend.transcribe_speech(clip, "gpt-transcribe", ["Finerenon"])
    assert recorder.requests[0].headers["authorization"] == "Bearer key"
    assert 'name="prompt"' in recorder.body() and "Finerenon" in recorder.body()


def test_transcribe_speech_without_vocabulary(clip):
    recorder = Recorder({"text": "x"})
    OpenAiSpeechBackend("key", client=recorder.client()).transcribe_speech(
        clip, "m", []
    )
    assert 'name="prompt"' not in recorder.body()


def test_transcribe_speech_empty_text(clip):
    backend = OpenAiSpeechBackend("key", client=Recorder({"text": " "}).client())
    assert backend.transcribe_speech(clip, "m", []).segments == []
