"""OpenAI speech-to-text backend (evaluation candidate C9)."""

from ..models.segment import Segment
from ..models.speech_result import SpeechResult
from ..utils.const import (
    OPENAI_BASE_URL,
    OPENAI_LOGGING,
    OPENAI_PROVIDER,
    SPEECH_MAX_TERMS,
    SPEECH_ROUTE,
)
from .speech_api import SpeechApiBackend


class OpenAiSpeechBackend(SpeechApiBackend):
    """Transcribe with OpenAI (gpt-transcribe): text only, no speakers or timestamps."""

    provider = OPENAI_PROVIDER
    url = OPENAI_BASE_URL + SPEECH_ROUTE
    data_policy = OPENAI_LOGGING

    # the form fields are the provider's API
    def _form(
        self, model: str, vocabulary: list[str]
    ) -> dict:  # pragma: no mutate block
        """Return the form fields; the vocabulary goes into the prompt.

        model: the model id
        vocabulary: terms to prefer"""
        form = {"model": model, "language": self.language, "response_format": "json"}
        if vocabulary:
            form["prompt"] = ", ".join(vocabulary[:SPEECH_MAX_TERMS])
        return form

    def _parse(self, data: dict) -> SpeechResult:
        """Return the text as one untimed segment without a speaker.

        data: the JSON answer"""
        text = data["text"].strip()
        return SpeechResult(segments=[Segment(text=text)] if text else [])
