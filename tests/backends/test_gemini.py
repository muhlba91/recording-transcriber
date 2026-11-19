"""Tests for the Gemini backend."""

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import httpx
import pytest
from google.genai import errors

from recording_transcriber.backends import gemini as gemini_module
from recording_transcriber.backends.gemini import GeminiBackend, interaction_usage
from recording_transcriber.models.usage import Usage


def _delta(text, kind="text"):
    return SimpleNamespace(
        event_type="step.delta", delta=SimpleNamespace(type=kind, text=text)
    )


def _completed(status="completed", usage=None):
    return SimpleNamespace(
        event_type="interaction.completed",
        interaction=SimpleNamespace(status=status, usage=usage),
    )


def _usage():
    return SimpleNamespace(
        input_tokens_by_modality=[
            SimpleNamespace(modality="text", tokens=500),
            SimpleNamespace(modality="AUDIO", tokens=3600),
        ],
        total_output_tokens=900,
        total_thought_tokens=12,
    )


def _api_error():
    return errors.APIError(500, {"error": {"message": "boom", "status": "INTERNAL"}})


class TestGeminiBackend:
    @pytest.fixture
    def client(self):
        client = MagicMock()
        client.files.upload.return_value = SimpleNamespace(
            name="files/1", uri="https://f/1", state="FileState.ACTIVE"
        )
        yield client

    @pytest.fixture
    def backend(self, client):
        yield GeminiBackend(client=client)

    def test_describe(self, backend):
        assert set(backend.describe()) == {"provider", "endpoint", "logging"}

    def test_check(self, backend, client):
        assert backend.check("gemini-4-carbon")
        client.models.get.assert_called_once_with(model="gemini-4-carbon")

    def test_check_error(self, backend, client):
        client.models.get.side_effect = _api_error()
        assert not backend.check("gemini-4-carbon")

    def test_transcribe(self, backend, client, tmp_path):
        client.interactions.create.return_value = iter(
            [
                SimpleNamespace(event_type="interaction.created"),
                SimpleNamespace(
                    event_type="interaction.status_update", status="in_progress"
                ),
                _delta("sig", kind="thought_signature"),
                _delta('{"turns": '),
                _delta("[]}"),
                _completed(usage=_usage()),
            ]
        )
        response = backend.transcribe(
            tmp_path / "a.mp3", "prompt", {"type": "object"}, "gemini-4-carbon"
        )
        assert response.text == '{"turns": []}'
        assert response.status == "completed"
        assert response.usage == Usage(
            input_text_tokens=500,
            input_audio_tokens=3600,
            output_tokens=900,
            thought_tokens=12,
        )
        kwargs = client.interactions.create.call_args.kwargs
        assert kwargs["model"] == "gemini-4-carbon"
        assert kwargs["input"][0] == {"type": "text", "text": "prompt"}
        assert kwargs["input"][1]["uri"] == "https://f/1"
        assert kwargs["stream"] is True
        assert kwargs["generation_config"]["thinking_level"] == "low"
        assert kwargs["response_format"]["schema"] == {"type": "object"}
        client.files.delete.assert_called_once_with(name="files/1")

    def test_transcribe_waits_for_upload(self, backend, client, tmp_path, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)
        client.files.upload.return_value = SimpleNamespace(
            name="files/1", uri="u", state="PROCESSING"
        )
        client.files.get.return_value = SimpleNamespace(
            name="files/1", uri="u", state="ACTIVE"
        )
        client.interactions.create.return_value = iter([_completed()])
        response = backend.transcribe(tmp_path / "a.mp3", "p", {}, "m")
        assert response.status == "completed"
        assert response.usage == Usage()
        assert sleeps
        client.files.get.assert_called_with(name="files/1")

    def test_transcribe_upload_timeout(self, backend, client, tmp_path, monkeypatch):
        # every reading of the clock is far later than the last
        clock = itertools.count(step=10**6)
        monkeypatch.setattr(
            gemini_module,
            "time",
            SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda seconds: None),
        )
        client.files.upload.return_value = SimpleNamespace(
            name="files/1", uri="u", state="PROCESSING"
        )
        assert backend.transcribe(tmp_path / "a.mp3", "p", {}, "m") is None
        client.files.delete.assert_called_once_with(name="files/1")

    def test_transcribe_upload_failed(self, backend, client, tmp_path):
        client.files.upload.return_value = SimpleNamespace(
            name="files/1", uri="u", state="FAILED"
        )
        assert backend.transcribe(tmp_path / "a.mp3", "p", {}, "m") is None
        client.files.delete.assert_called_once_with(name="files/1")
        client.interactions.create.assert_not_called()

    def test_transcribe_upload_error(self, backend, client, tmp_path):
        client.files.upload.side_effect = httpx.ConnectError("offline")
        assert backend.transcribe(tmp_path / "a.mp3", "p", {}, "m") is None

    def test_transcribe_incomplete(self, backend, client, tmp_path):
        client.interactions.create.return_value = iter(
            [_delta('{"tu'), _completed(status="incomplete")]
        )
        assert (
            backend.transcribe(tmp_path / "a.mp3", "p", {}, "m").status == "incomplete"
        )

    def test_transcribe_stream_error_event(self, backend, client, tmp_path):
        event = SimpleNamespace(event_type="error")
        client.interactions.create.return_value = iter([event])
        assert backend.transcribe(tmp_path / "a.mp3", "p", {}, "m") is None
        client.files.delete.assert_called_once()

    def test_transcribe_api_error(self, backend, client, tmp_path):
        client.interactions.create.side_effect = _api_error()
        assert backend.transcribe(tmp_path / "a.mp3", "p", {}, "m") is None
        client.files.delete.assert_called_once()

    def test_complete(self, backend, client):
        client.interactions.create.return_value = iter([_delta("{}"), _completed()])
        assert backend.complete("prompt", {"a": 1}, "m").text == "{}"
        kwargs = client.interactions.create.call_args.kwargs
        assert kwargs["model"] == "m"
        assert kwargs["input"] == "prompt"
        assert kwargs["response_format"]["schema"] == {"a": 1}

    def test_complete_without_events(self, backend, client):
        client.interactions.create.return_value = iter([])
        response = backend.complete("prompt", {}, "m")
        assert (response.text, response.status) == ("", "unknown")
        assert response.usage == Usage()

    def test_complete_keeps_the_streamed_status(self, backend, client):
        # a completed event without status keeps the last status update
        client.interactions.create.return_value = iter(
            [
                SimpleNamespace(
                    event_type="interaction.status_update", status="in_progress"
                ),
                _completed(status=None),
            ]
        )
        assert backend.complete("prompt", {}, "m").status == "in_progress"

    def test_delete_error_is_swallowed(self, backend, client):
        client.files.delete.side_effect = _api_error()
        assert backend._delete("files/9") is None
        client.files.delete.assert_called_once_with(name="files/9")

    def test_close_deletes_leftovers(self, backend, client, tmp_path):
        client.interactions.create.return_value = iter([_completed()])
        client.files.delete.side_effect = [_api_error(), None, None]
        backend.transcribe(tmp_path / "a.mp3", "p", {}, "m")
        backend.close()
        backend.close()
        # the failed delete is retried once, then nothing is left
        assert client.files.delete.call_args_list == [call(name="files/1")] * 2

    def test_default_client(self):
        backend = GeminiBackend(api_key="test-key")
        assert backend.thinking_level == "low"


def test_interaction_usage_missing_counts():
    usage = SimpleNamespace(
        input_tokens_by_modality=[
            SimpleNamespace(modality="text", tokens=None),
            SimpleNamespace(modality="audio", tokens=None),
        ],
        total_output_tokens=None,
        total_thought_tokens=None,
    )
    assert interaction_usage(usage) == Usage()


def test_interaction_usage_missing_modality_counts_zero():
    def usage(modality):
        return SimpleNamespace(
            input_tokens_by_modality=[SimpleNamespace(modality=modality, tokens=7)],
            total_output_tokens=1,
            total_thought_tokens=2,
        )

    assert interaction_usage(usage("audio")) == Usage(
        input_audio_tokens=7, output_tokens=1, thought_tokens=2
    )
    assert interaction_usage(usage("text")) == Usage(
        input_text_tokens=7, output_tokens=1, thought_tokens=2
    )
