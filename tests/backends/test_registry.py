"""Tests for the backend registry."""

import pytest

from recording_transcriber.backends.chat_api import ChatApiBackend
from recording_transcriber.backends.gemini import GeminiBackend
from recording_transcriber.backends.gemini_speech import GeminiSpeechBackend
from recording_transcriber.backends.mistral_speech import MistralSpeechBackend
from recording_transcriber.backends.openai_speech import OpenAiSpeechBackend
from recording_transcriber.backends.registry import create_backends
from recording_transcriber.configuration.config import Config

_TEXT = ("polish", "reconcile", "protocol", "agenda")


def _transcribe(provider, model="m") -> dict:
    return {"transcribe": {"provider": provider, "model": model}}


@pytest.fixture
def keys(monkeypatch):
    """Set a different key per provider."""
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", "MISTRAL_API_KEY", "LITELLM_KEY"):
        monkeypatch.setenv(name, f"secret-{name}")


def _authorization(backend) -> str:
    return backend._headers()["Authorization"]


def test_create_backends(keys):
    backends = create_backends(Config())
    # polish only runs after gemini-transcribe
    assert set(backends) == {"transcribe", "reconcile", "protocol", "agenda"}
    first = backends["transcribe"]
    assert isinstance(first, GeminiBackend)
    assert all(backend is first for backend in backends.values())


@pytest.mark.parametrize(
    ("provider", "expected", "smart"),
    [
        ("gemini-transcribe", GeminiSpeechBackend, False),
        ("gemini-transcribe-smart", GeminiSpeechBackend, True),
        ("openai", OpenAiSpeechBackend, None),
        ("mistral", MistralSpeechBackend, None),
    ],
)
def test_create_speech_backends(keys, provider, expected, smart):
    config = Config(tasks=_transcribe(provider), transcript={"language": "de-DE"})
    backends = create_backends(config)
    assert isinstance(backends["transcribe"], expected)
    assert type(backends["reconcile"]) is GeminiBackend
    if smart is not None:
        assert backends["transcribe"].smart is smart
    assert set(backends) == set(config.used_tasks())


def test_create_backends_polish_for_diarized_gemini(keys):
    config = Config(
        tasks={**_transcribe("gemini-transcribe"), "polish": {"model": "p"}},
        transcript={"language": "de-DE"},
    )
    assert "polish" in create_backends(config)


def test_create_chat_backends(keys):
    config = Config(
        providers={
            "litellm": {
                "base_url": "https://llm.example/v1",
                "api_key_env": "LITELLM_KEY",
            }
        },
        tasks={
            "transcribe": {"provider": "openai", "model": "gpt-transcribe"},
            "reconcile": {"provider": "openai", "model": "g"},
            "protocol": {"provider": "mistral", "model": "m"},
            "agenda": {"provider": "litellm", "model": "x"},
        },
    )
    backends = create_backends(config)
    assert isinstance(backends["transcribe"], OpenAiSpeechBackend)
    assert isinstance(backends["reconcile"], ChatApiBackend)
    assert backends["reconcile"].describe()["endpoint"] == (
        "https://api.openai.com/v1/chat/completions"
    )
    assert backends["protocol"].describe()["endpoint"] == (
        "https://api.eu.mistral.ai/v1/chat/completions"
    )
    assert backends["agenda"].describe() == {
        "provider": "litellm",
        "endpoint": "https://llm.example/v1/chat/completions",
        "logging": "set by the endpoint's operator (custom provider)",
    }


def test_create_backends_shares_one_backend_per_provider(keys):
    config = Config(
        tasks={
            "reconcile": {"provider": "openai", "model": "a"},
            "protocol": {"provider": "openai", "model": "b"},
        }
    )
    backends = create_backends(config)
    assert backends["reconcile"] is backends["protocol"]


def test_create_backends_reports_every_missing_key(monkeypatch):
    for name in ("GEMINI_API_KEY", "MISTRAL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = Config(tasks=_transcribe("mistral"))
    assert create_backends(config) is None


def test_create_backends_unused_provider_needs_no_key(keys, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    # polish is unused for a Gemini transcribe model
    config = Config(tasks={"polish": {"provider": "openai", "model": "g"}})
    assert create_backends(config) is not None


def test_create_backends_language(keys):
    transcript = {"language": "de-DE"}
    gemini = create_backends(
        Config(tasks=_transcribe("gemini-transcribe"), transcript=transcript)
    )["transcribe"]
    mistral = create_backends(
        Config(tasks=_transcribe("mistral"), transcript=transcript)
    )["transcribe"]
    assert (gemini.language, mistral.language) == ("de-DE", "de")


def test_create_backends_thinking_level(keys):
    backend = create_backends(Config(thinking_level="high"))["transcribe"]
    assert backend.thinking_level == "high"


def test_create_backends_speech_keys(keys):
    openai = create_backends(Config(tasks=_transcribe("openai")))["transcribe"]
    mistral = create_backends(Config(tasks=_transcribe("mistral")))["transcribe"]
    assert _authorization(openai) == "Bearer secret-OPENAI_API_KEY"
    assert _authorization(mistral) == "Bearer secret-MISTRAL_API_KEY"


def test_create_backends_chat_settings(keys):
    config = Config(
        providers={
            "litellm": {
                "base_url": "https://llm.example/v1",
                "api_key_env": "LITELLM_KEY",
            }
        },
        tasks={
            "reconcile": {"provider": "openai", "model": "g"},
            "protocol": {"provider": "litellm", "model": "x"},
        },
    )
    backends = create_backends(config)
    openai, custom = backends["reconcile"], backends["protocol"]
    assert (openai.provider, custom.provider) == ("openai", "litellm")
    assert _authorization(openai) == "Bearer secret-OPENAI_API_KEY"
    assert _authorization(custom) == "Bearer secret-LITELLM_KEY"
