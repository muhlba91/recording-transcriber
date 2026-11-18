"""A speech-to-text result."""

from pydantic import BaseModel

from .segment import Segment
from .usage import Usage


class SpeechResult(BaseModel):
    """The result of one speech-to-text call."""

    segments: list[Segment]
    usage: Usage = Usage()
