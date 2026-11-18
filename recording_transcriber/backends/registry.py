"""Backend registry: build the backends a configuration asks for."""

import logging
import os

from ..configuration.config import Config
from ..utils.const import (
    CUSTOM_LOGGING,
    MISTRAL_BASE_URL,
    MISTRAL_LOGGING,
    OPENAI_BASE_URL,
    OPENAI_LOGGING,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_TRANSCRIBE,
    PROVIDER_GEMINI_TRANSCRIBE_SMART,
    PROVIDER_MISTRAL,
    PROVIDER_OPENAI,
    TASK_TRANSCRIBE,
)
from .base import SpeechBackend, TextBackend, TranscriptionBackend
from .chat_api import ChatApiBackend
from .gemini import GeminiBackend
from .gemini_speech import GeminiSpeechBackend
from .mistral_speech import MistralSpeechBackend
from .openai_speech import OpenAiSpeechBackend

_LOGGER = logging.getLogger(__name__)

# the third-party speech-to-text classes
_SPEECH_APIS = {
    PROVIDER_OPENAI: OpenAiSpeechBackend,
    PROVIDER_MISTRAL: MistralSpeechBackend,
}
# the chat API of the third-party providers: base URL and data policy
_CHAT_APIS = {
    PROVIDER_OPENAI: (OPENAI_BASE_URL, OPENAI_LOGGING),
    PROVIDER_MISTRAL: (MISTRAL_BASE_URL, MISTRAL_LOGGING),
}


def create_backends(
    config: Config,
) -> dict[str, TranscriptionBackend | SpeechBackend | TextBackend] | None:
    """Return the backend of every task the run executes, keyed by task name, or
    None if a key is missing.

    A provider is built once; tasks on the same provider share its backend. OpenAI
    and Mistral are evaluation candidates only (design D6).

    config: the configuration"""
    missing = [name for name in config.key_variables() if not os.environ.get(name)]
    if missing:
        _LOGGER.error("Environment variable(s) not set: %s.", ", ".join(missing))
        return None
    built = {}
    backends = {}
    for task in config.used_tasks():
        provider = getattr(config.tasks, task).provider
        transcribe = task == TASK_TRANSCRIBE
        # a provider transcribes and completes text with different backends, except
        # Gemini, whose backend does both
        key = (provider, transcribe and provider != PROVIDER_GEMINI)
        if key not in built:
            built[key] = _create(config, provider, transcribe)
        backends[task] = built[key]
    return backends


def _create(
    config: Config, provider: str, transcribe: bool
) -> TranscriptionBackend | SpeechBackend | TextBackend:
    """Return the backend of a provider for the transcribe task or the text tasks.

    config: the configuration
    provider: the provider name
    transcribe: whether the backend transcribes audio (else it completes text)"""
    key = os.environ[config.key_variable(provider)]
    if provider == PROVIDER_GEMINI:
        return GeminiBackend(
            api_key=key,
            thinking_level=config.thinking_level,
        )
    if provider in (PROVIDER_GEMINI_TRANSCRIBE, PROVIDER_GEMINI_TRANSCRIBE_SMART):
        return GeminiSpeechBackend(
            api_key=key,
            smart=provider == PROVIDER_GEMINI_TRANSCRIBE_SMART,
            language=config.transcript.language,
        )
    if transcribe:
        return _SPEECH_APIS[provider](key, config.transcript.language)
    if provider in config.providers:
        return ChatApiBackend(
            provider, config.providers[provider].base_url, key, CUSTOM_LOGGING
        )
    base_url, data_policy = _CHAT_APIS[provider]
    # the data policy only lands in run.json
    return ChatApiBackend(provider, base_url, key, data_policy)  # pragma: no mutate
