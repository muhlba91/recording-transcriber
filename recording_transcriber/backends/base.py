"""Backend interfaces."""

from pathlib import Path
from typing import Protocol

from ..models.model_response import ModelResponse
from ..models.speech_result import SpeechResult


class TextBackend(Protocol):
    """A provider for text-only steps (speaker reconciliation, protocol)."""

    def complete(self, prompt: str, schema: dict, model: str) -> ModelResponse | None:
        """Return the JSON answer to a prompt, or None if the call failed.

        prompt: the instructions
        schema: the JSON schema of the answer
        model: the model id"""

    def describe(self) -> dict:
        """Return provider, endpoint and logging setting for run.json."""

    def check(self, model: str) -> bool:
        """Return whether the key works and the model exists (cheap, before real work).

        model: the model id"""

    def close(self) -> None:
        """Delete anything left at the provider (uploads) and release the client."""


class TranscriptionBackend(TextBackend, Protocol):
    """A provider that transcribes audio."""

    def transcribe(
        self, audio: Path, prompt: str, schema: dict, model: str
    ) -> ModelResponse | None:
        """Return the JSON transcription of an audio file, or None if the call failed.

        audio: the audio file
        prompt: the instructions
        schema: the JSON schema of the answer
        model: the model id"""


class SpeechBackend(Protocol):
    """A speech-to-text provider: audio in, segments out, no prompt."""

    def check(self, model: str) -> bool:
        """Return whether the key works and the model exists (cheap, before real work).

        model: the model id"""

    def transcribe_speech(
        self, audio: Path, model: str, vocabulary: list[str]
    ) -> SpeechResult | None:
        """Return the segments of an audio file, or None if the call failed.

        audio: the audio file
        model: the model id
        vocabulary: terms to prefer, if the model takes them"""

    def describe(self) -> dict:
        """Return provider, endpoint and logging setting for run.json."""

    def close(self) -> None:
        """Delete anything left at the provider (uploads)."""
