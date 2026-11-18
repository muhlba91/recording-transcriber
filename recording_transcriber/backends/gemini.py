"""Google Gemini backend (Developer API, Interactions API)."""

import logging
import time
from pathlib import Path

import httpx
from google import genai
from google.genai import errors

from ..models.model_response import ModelResponse
from ..models.usage import Usage
from ..utils.const import (
    GEMINI_ENDPOINT,
    GEMINI_LOGGING,
    GEMINI_PROVIDER,
    MAX_OUTPUT_TOKENS,
    REQUEST_TIMEOUT_MS,
    STATUS_UNKNOWN,
    UPLOAD_POLL_S,
    UPLOAD_TIMEOUT_S,
)

_LOGGER = logging.getLogger(__name__)


class GeminiBackend:
    """Transcribe and complete text with Gemini models.

    Audio goes through the Files API and is deleted right after each call."""

    def __init__(
        self,
        api_key: str | None = None,
        thinking_level: str = "low",
        client: genai.Client | None = None,
    ):
        """Initialize the backend.

        api_key: the paid-tier AI Studio key (default: None, only with a client)
        thinking_level: the Gemini thinking level (default: low)
        client: a preconfigured client, used by tests (default: None)"""
        self.thinking_level = thinking_level
        # the client also reads the key from the environment; key routing is tested
        # in the registry
        # pragma: no mutate start (client setup)
        self._client = client or genai.Client(
            api_key=api_key, http_options={"timeout": REQUEST_TIMEOUT_MS}
        )
        # pragma: no mutate end
        self._uploads = set()

    def describe(self) -> dict:
        """Return provider, endpoint and logging setting for run.json."""
        return {
            "provider": GEMINI_PROVIDER,
            "endpoint": GEMINI_ENDPOINT,
            "logging": GEMINI_LOGGING,
        }

    def check(self, model: str) -> bool:
        """Return whether the key works and the model exists.

        model: the model id"""
        try:
            self._client.models.get(model=model)
        except (errors.APIError, httpx.HTTPError) as error:
            _LOGGER.error("Could not access Gemini model %s: %s.", model, error)
            return False
        return True

    def transcribe(
        self, audio: Path, prompt: str, schema: dict, model: str
    ) -> ModelResponse | None:
        """Upload the audio, transcribe it and delete the upload.

        audio: the mp3 file
        prompt: the instructions
        schema: the JSON schema of the answer
        model: the model id"""
        uploaded = self._upload(audio)
        if uploaded is None:
            return None
        try:
            content = [
                {"type": "text", "text": prompt},
                {
                    "type": "audio",
                    "uri": uploaded.uri,
                    "mime_type": "audio/mpeg",
                },
            ]
            return self._stream(model, content, schema)
        finally:
            self._delete(uploaded.name)

    def complete(self, prompt: str, schema: dict, model: str) -> ModelResponse | None:
        """Return the JSON answer to a text prompt.

        prompt: the instructions
        schema: the JSON schema of the answer
        model: the model id"""
        return self._stream(model, prompt, schema)

    def close(self) -> None:
        """Delete uploads left by an interrupted call."""
        for name in list(self._uploads):
            self._delete(name)

    def _upload(self, audio: Path):
        """Upload a file and wait until it is usable; return None on failure.

        audio: the mp3 file"""
        try:
            uploaded = self._client.files.upload(
                file=audio,  # pragma: no mutate (request wiring)
                config={"mime_type": "audio/mpeg"},
            )
            self._uploads.add(uploaded.name)
            deadline = time.monotonic() + UPLOAD_TIMEOUT_S
            # state is an SDK enum (FileState.ACTIVE); its string form is stable
            while not str(uploaded.state).endswith("ACTIVE"):
                if (
                    str(uploaded.state).endswith("FAILED")
                    or time.monotonic() > deadline  # pragma: no mutate (threshold)
                ):
                    _LOGGER.error("Upload of %s did not become active.", audio.name)
                    self._delete(uploaded.name)
                    return None
                time.sleep(UPLOAD_POLL_S)
                uploaded = self._client.files.get(name=uploaded.name)
        except (errors.APIError, httpx.HTTPError, OSError) as error:
            _LOGGER.error("Could not upload %s to Gemini: %s.", audio.name, error)
            return None
        return uploaded

    def _delete(self, name: str) -> None:
        """Delete an uploaded file; failures are logged (Google deletes it after 48 h).

        name: the file resource name"""
        try:
            self._client.files.delete(name=name)
            self._uploads.discard(name)
        except (errors.APIError, httpx.HTTPError) as error:
            _LOGGER.error("Could not delete Gemini upload %s: %s.", name, error)

    def _stream(self, model: str, content, schema: dict) -> ModelResponse | None:
        """Run a streaming interaction and collect the text and final status.

        model: the model id
        content: the input (text or a list of content parts)
        schema: the JSON schema of the answer"""
        parts = []
        status = STATUS_UNKNOWN
        usage = Usage()
        try:
            # pragma: no mutate start (wire format)
            response_format = {"mime_type": "application/json", "schema": schema}
            # pragma: no mutate end
            stream = self._client.interactions.create(
                model=model,
                input=content,
                stream=True,
                generation_config={
                    "thinking_level": self.thinking_level,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                response_format=response_format,
            )
            for event in stream:
                # events always carry a type; the default is defensive
                event_type = getattr(event, "event_type", None)  # pragma: no mutate
                if event_type == "step.delta" and event.delta.type == "text":
                    parts.append(event.delta.text)
                elif event_type == "interaction.status_update":
                    status = event.status
                elif event_type == "interaction.completed":
                    status = event.interaction.status or status
                    usage = interaction_usage(event.interaction.usage)
                elif event_type == "error":
                    _LOGGER.error("Gemini stream error for %s: %s.", model, event)
                    return None
        except (errors.APIError, httpx.HTTPError) as error:
            _LOGGER.error("Could not call Gemini model %s: %s.", model, error)
            return None
        return ModelResponse(text="".join(parts), status=status, usage=usage)


def interaction_usage(usage) -> Usage:
    """Convert the Interactions API usage into Usage.

    usage: the usage object of the completed interaction (may be None)"""
    if usage is None:
        return Usage()
    by_modality = {
        str(item.modality).lower(): item.tokens or 0
        for item in usage.input_tokens_by_modality or []
    }
    return Usage(
        input_text_tokens=by_modality.get("text", 0),
        input_audio_tokens=by_modality.get("audio", 0),
        output_tokens=usage.total_output_tokens or 0,
        thought_tokens=usage.total_thought_tokens or 0,
    )
