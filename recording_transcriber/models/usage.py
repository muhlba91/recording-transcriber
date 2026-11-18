"""Token usage."""

from pydantic import BaseModel


class Usage(BaseModel):
    """Token usage of one or more model calls."""

    input_text_tokens: int = 0
    input_audio_tokens: int = 0
    output_tokens: int = 0
    thought_tokens: int = 0
    # speech-to-text models bill per audio minute
    audio_seconds: float = 0

    def add(self, other: "Usage") -> "Usage":
        """Return the sum of this and another usage.

        other: the usage to add"""
        return Usage(
            input_text_tokens=self.input_text_tokens + other.input_text_tokens,
            input_audio_tokens=self.input_audio_tokens + other.input_audio_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            thought_tokens=self.thought_tokens + other.thought_tokens,
            audio_seconds=self.audio_seconds + other.audio_seconds,
        )
