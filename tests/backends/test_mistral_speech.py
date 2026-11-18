"""Tests for the Mistral speech-to-text backend."""

from recording_transcriber.backends.mistral_speech import MistralSpeechBackend
from tests.backends.conftest import Recorder

_SEGMENTS = [
    {"text": " Hallo.", "start": 0.5, "speaker_id": "speaker_1"},
    {"text": " ", "start": 1.0, "speaker_id": "speaker_1"},
    {"text": "Danke.", "start": 2, "speaker_id": "speaker_2"},
]


def test_transcribe_speech_returns_diarized_segments(clip):
    recorder = Recorder({"segments": _SEGMENTS})
    backend = MistralSpeechBackend("key", client=recorder.client())
    result = backend.transcribe_speech(clip, "voxtral", [])
    assert [(s.speaker, s.start_s, s.text) for s in result.segments] == [
        ("speaker_1", 0.5, "Hallo."),
        ("speaker_2", 2.0, "Danke."),
    ]


def test_transcribe_speech_uses_eu_endpoint_and_diarization(clip):
    recorder = Recorder({"segments": []})
    MistralSpeechBackend("key", client=recorder.client()).transcribe_speech(
        clip, "voxtral", []
    )
    assert str(recorder.requests[0].url).startswith("https://api.eu.mistral.ai/")
    assert 'name="diarize"' in recorder.body()
    assert 'name="context_bias"' not in recorder.body()


def test_transcribe_speech_sends_vocabulary_as_repeated_fields(clip):
    recorder = Recorder({"segments": []})
    MistralSpeechBackend("key", client=recorder.client()).transcribe_speech(
        clip, "voxtral", ["A", "B"]
    )
    assert recorder.body().count('name="context_bias"') == 2
