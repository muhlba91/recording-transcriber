"""Tests for the Gemini speech-to-text backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from google.genai import errors

from recording_transcriber.backends.gemini_speech import (
    GeminiSpeechBackend,
    _offset,
    _word_segments,
)


def _word(text, speaker, start):
    return {
        "type": "word_info",
        "text": text,
        "speaker": speaker,
        "start_offset": start,
    }


def _response(data, status="completed"):
    usage = SimpleNamespace(
        input_tokens_by_modality=[SimpleNamespace(modality="audio", tokens=75)],
        total_output_tokens=10,
        total_thought_tokens=0,
    )
    return SimpleNamespace(status=status, usage=usage, model_dump=lambda: data)


class TestGeminiSpeechBackend:
    @pytest.fixture
    def client(self):
        client = MagicMock()
        client.files.upload.return_value = SimpleNamespace(
            name="files/1", uri="https://f/1", state="FileState.ACTIVE"
        )
        yield client

    def test_diarized(self, client, tmp_path):
        words = [
            _word("Guten", "spk:0", "0.100s"),
            _word("Morgen.", "spk:0", "0.400s"),
            _word("Wie", "spk:0", "1s"),
            _word("Danke", "spk:1", "2.5s"),
        ]
        client.interactions.create.return_value = _response(
            {"steps": [{"content": [{"annotations": words}]}]}
        )
        backend = GeminiSpeechBackend(client=client)
        result = backend.transcribe_speech(tmp_path / "a.mp3", "m", ["Finerenon"])
        assert [(s.speaker, s.start_s, s.text) for s in result.segments] == [
            ("spk:0", 0.1, "Guten Morgen."),
            ("spk:0", 1.0, "Wie"),
            ("spk:1", 2.5, "Danke"),
        ]
        assert result.usage.input_audio_tokens == 75
        kwargs = client.interactions.create.call_args.kwargs
        mode = kwargs["generation_config"]["transcription_config"]["mode"]
        assert mode["diarization_mode"] == "speaker"
        assert kwargs["model"] == "m"
        assert kwargs["input"][0]["uri"] == "https://f/1"
        client.files.delete.assert_called_once_with(name="files/1")

    def test_smart(self, client, tmp_path):
        client.interactions.create.return_value = _response({"output_text": " Text. "})
        backend = GeminiSpeechBackend(client=client, smart=True)
        result = backend.transcribe_speech(tmp_path / "a.mp3", "m", ["Finerenon"])
        assert [(s.speaker, s.start_s, s.text) for s in result.segments] == [
            (None, None, "Text.")
        ]
        config = client.interactions.create.call_args.kwargs["generation_config"]
        assert config["transcription_config"]["custom_vocabulary"] == ["Finerenon"]

    def test_smart_empty(self, client, tmp_path):
        client.interactions.create.return_value = _response({"output_text": None})
        backend = GeminiSpeechBackend(client=client, smart=True)
        assert backend.transcribe_speech(tmp_path / "a.mp3", "m", []).segments == []
        config = client.interactions.create.call_args.kwargs["generation_config"]
        assert "custom_vocabulary" not in config["transcription_config"]

    def test_not_completed(self, client, tmp_path):
        client.interactions.create.return_value = _response({}, status="failed")
        backend = GeminiSpeechBackend(client=client)
        assert backend.transcribe_speech(tmp_path / "a.mp3", "m", []) is None

    def test_api_error(self, client, tmp_path):
        client.interactions.create.side_effect = errors.APIError(
            500, {"error": {"message": "boom"}}
        )
        backend = GeminiSpeechBackend(client=client)
        assert backend.transcribe_speech(tmp_path / "a.mp3", "m", []) is None
        client.files.delete.assert_called_once_with(name="files/1")

    def test_upload_fails(self, client, tmp_path):
        client.files.upload.side_effect = OSError("gone")
        backend = GeminiSpeechBackend(client=client)
        assert backend.transcribe_speech(tmp_path / "a.mp3", "m", []) is None

    def test_default_client(self):
        backend = GeminiSpeechBackend(api_key="key", language="de-DE")
        assert backend.describe()["provider"] == "google-gemini-developer-api"
        assert backend.language == "de-DE"


def test_word_segments_skips_other_annotations():
    annotations = [{"type": "citation"}, {"type": "word_info", "speaker": "spk:0"}]
    data = {"steps": [{"content": [{"annotations": annotations}, {}]}]}
    assert _word_segments(data) == []


def test_offset_invalid():
    assert _offset(None) is None
    assert _offset("abc") is None
