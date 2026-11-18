"""Mistral speech-to-text backend (evaluation candidate C10)."""

from ..models.segment import Segment
from ..models.speech_result import SpeechResult
from ..utils.const import (
    MISTRAL_BASE_URL,
    MISTRAL_LOGGING,
    MISTRAL_PROVIDER,
    SPEECH_MAX_TERMS,
    SPEECH_ROUTE,
)
from .speech_api import SpeechApiBackend


class MistralSpeechBackend(SpeechApiBackend):
    """Transcribe with Voxtral on the EU endpoint: diarized segments with start times."""

    provider = MISTRAL_PROVIDER
    url = MISTRAL_BASE_URL + SPEECH_ROUTE
    data_policy = MISTRAL_LOGGING

    # the form fields are the provider's API
    def _form(
        self, model: str, vocabulary: list[str]
    ) -> dict:  # pragma: no mutate block
        """Return the form fields; a list value is sent as repeated fields.

        model: the model id
        vocabulary: terms to prefer"""
        form = {
            "model": model,
            "language": self.language,
            "diarize": "true",
            "timestamp_granularities": "segment",
        }
        if vocabulary:
            form["context_bias"] = vocabulary[:SPEECH_MAX_TERMS]
        return form

    def _parse(self, data: dict) -> SpeechResult:
        """Return the diarized segments.

        data: the JSON answer"""
        segments = [
            Segment(
                speaker=segment.get("speaker_id"),
                start_s=float(segment["start"]),
                text=segment["text"].strip(),
            )
            for segment in data["segments"]
            if segment["text"].strip()
        ]
        return SpeechResult(segments=segments)
