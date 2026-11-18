"""Base for speech-to-text APIs with an OpenAI-shaped /audio/transcriptions route."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import httpx

from ..models.speech_result import SpeechResult
from ..utils.const import REQUEST_TIMEOUT_MS

_LOGGER = logging.getLogger(__name__)


class SpeechApiBackend(ABC):
    """Upload a clip as multipart form data and parse the JSON answer.

    The audio is sent with the request and not stored as a file at the provider, so
    close() has nothing to delete. Subclasses set provider, url and data_policy and
    implement _form and _parse."""

    provider: str
    url: str
    data_policy: str

    def __init__(
        self,
        api_key: str,
        language: str = "en",  # pragma: no mutate (default)
        client: httpx.Client | None = None,
    ):
        """Initialize the backend.

        api_key: the provider key
        language: the BCP-47 language; these APIs take the primary subtag (default: en)
        client: a preconfigured client, used by tests (default: None)"""
        self.__api_key = api_key
        self.language = language.split("-")[0]
        self.__client = client or httpx.Client(timeout=REQUEST_TIMEOUT_MS / 1000)

    def describe(self) -> dict:
        """Return provider, endpoint and logging setting for run.json."""
        return {
            "provider": self.provider,
            "endpoint": self.url,
            "logging": self.data_policy,
        }

    def check(self, model: str) -> bool:
        """Return whether the key is set; a bad key or model fails the first call.

        These APIs have no cheap model lookup that works for every key type.

        model: the model id"""
        if not self.__api_key:
            _LOGGER.error("No API key for %s model %s.", self.provider, model)
            return False
        return True

    def transcribe_speech(
        self, audio: Path, model: str, vocabulary: list[str]
    ) -> SpeechResult | None:
        """Send the clip and return its segments, or None if the call failed.

        audio: the mp3 clip
        model: the model id
        vocabulary: terms to prefer"""
        try:
            with audio.open("rb") as file:
                # pragma: no mutate start (wire format)
                response = self.__client.post(
                    self.url,
                    headers=self._headers(),
                    files={"file": (audio.name, file, "audio/mpeg")},
                    data=self._form(model, vocabulary),
                )
                # pragma: no mutate end
        except (httpx.HTTPError, OSError) as error:
            _LOGGER.error(
                "Could not call %s model %s: %s.", self.provider, model, error
            )
            return None
        if response.status_code != httpx.codes.OK:
            _LOGGER.error(
                "%s model %s returned %s: %s.",
                self.provider,
                model,
                response.status_code,
                response.text[:300],
            )
            return None
        try:
            return self._parse(response.json())
        except (ValueError, KeyError, TypeError) as error:
            _LOGGER.error("Invalid answer from %s: %s.", self.provider, error)
            return None

    def _headers(self) -> dict:
        """Return the request headers with the key."""
        # header names are case-insensitive
        return {"Authorization": f"Bearer {self.__api_key}"}

    def close(self) -> None:
        """Close the HTTP client; nothing is stored at the provider."""
        self.__client.close()

    @abstractmethod
    def _form(self, model: str, vocabulary: list[str]) -> dict:
        """Return the form fields of the request.

        model: the model id
        vocabulary: terms to prefer"""

    @abstractmethod
    def _parse(self, data: dict) -> SpeechResult:
        """Return the segments of a JSON answer; raise on an unexpected shape.

        data: the JSON answer"""
