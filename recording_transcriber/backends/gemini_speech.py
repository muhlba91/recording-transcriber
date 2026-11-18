"""Gemini speech-to-text backend (gemini-3.5-transcribe, candidates C5 and C6)."""

import logging
from pathlib import Path

import httpx
from google import genai
from google.genai import errors

from ..models.segment import Segment
from ..models.speech_result import SpeechResult
from ..utils.const import (
    REQUEST_TIMEOUT_MS,
    SENTENCE_END,
    SPEECH_MAX_TERMS,
    STATUS_COMPLETED,
)
from .gemini import GeminiBackend, interaction_usage

_LOGGER = logging.getLogger(__name__)


class GeminiSpeechBackend(GeminiBackend):
    """Transcribe with gemini-3.5-transcribe through the Interactions API.

    Diarized mode returns verbatim words with speakers and timestamps. Smart mode
    returns clean text with the vocabulary, but no speakers or timestamps (Google
    does not allow combining them). It reuses GeminiBackend's upload, delete, check
    and close; the inherited text methods are not used."""

    def __init__(
        self,
        api_key: str | None = None,
        smart: bool = False,
        language: str = "en-US",  # pragma: no mutate (default)
        client: genai.Client | None = None,
    ):
        """Initialize the backend.

        api_key: the paid-tier AI Studio key (default: None, only with a client)
        smart: use smart mode instead of diarization (default: False)
        language: the BCP-47 code; Google lists region codes like de-DE (default: en-US)
        client: a preconfigured client, used by tests (default: None)"""
        # transcription_config is only accepted on v1beta
        # the client also reads the key from the environment; key routing is tested
        # in the registry
        # pragma: no mutate start (client setup)
        super().__init__(
            client=client
            or genai.Client(
                api_key=api_key,
                http_options={"api_version": "v1beta", "timeout": REQUEST_TIMEOUT_MS},
            )
        )
        # pragma: no mutate end
        self.smart = smart
        self.language = language

    def transcribe_speech(
        self, audio: Path, model: str, vocabulary: list[str]
    ) -> SpeechResult | None:
        """Upload the clip, transcribe it and delete the upload.

        audio: the mp3 clip
        model: the model id
        vocabulary: terms to prefer (smart mode only)"""
        uploaded = self._upload(audio)
        if uploaded is None:
            return None
        try:
            response = self._client.interactions.create(
                model=model,
                input=[
                    {
                        "type": "audio",
                        "uri": uploaded.uri,
                        "mime_type": "audio/mpeg",
                    }
                ],
                generation_config={"transcription_config": self._config(vocabulary)},
            )
        except (errors.APIError, httpx.HTTPError) as error:
            _LOGGER.error("Could not call Gemini model %s: %s.", model, error)
            return None
        finally:
            self._delete(uploaded.name)
        if response.status != STATUS_COMPLETED:
            _LOGGER.error("Gemini model %s ended with %s.", model, response.status)
            return None
        data = response.model_dump()
        segments = (
            _text_segment(data.get("output_text"))
            if self.smart
            else _word_segments(data)
        )
        return SpeechResult(segments=segments, usage=interaction_usage(response.usage))

    # the transcription config is the provider's API
    def _config(self, vocabulary: list[str]) -> dict:  # pragma: no mutate block
        """Return the transcription_config.

        vocabulary: terms to prefer (smart mode only)"""
        if self.smart:
            config = {
                "language_codes": [self.language],
                "mode": {"type": "smart"},
            }
            if vocabulary:
                config["custom_vocabulary"] = vocabulary[:SPEECH_MAX_TERMS]
            return config
        return {
            "language_codes": [self.language],
            "mode": {
                "type": "verbatim",
                "diarization_mode": "speaker",
                "timestamp_granularities": ["word"],
            },
        }


def _text_segment(text: str | None) -> list[Segment]:
    """Return the text as one untimed segment without a speaker.

    text: the output text"""
    text = (text or "").strip()
    return [Segment(text=text)] if text else []


def _word_segments(data: dict) -> list[Segment]:
    """Group word annotations into segments: one per sentence and speaker.

    The words carry their punctuation, so the text is rebuilt from them.

    data: the interaction as a dict"""
    words = [
        annotation
        for step in data.get("steps") or []
        for part in step.get("content") or []
        for annotation in part.get("annotations") or []
        if annotation.get("type") == "word_info" and annotation.get("text")
    ]
    segments = []
    current = None
    for word in words:
        if current is None or current.speaker != word.get("speaker"):
            current = Segment(
                speaker=word.get("speaker"),
                start_s=_offset(word.get("start_offset")),
                text=word["text"],
            )
            segments.append(current)
        else:
            current.text = f"{current.text} {word['text']}"
        if word["text"].endswith(SENTENCE_END):
            current = None
    return segments


def _offset(value: str | None) -> float | None:
    """Return seconds from an offset like "1.500s", or None.

    value: the offset string"""
    try:
        return float(value.removesuffix("s"))
    except (AttributeError, ValueError):
        return None
