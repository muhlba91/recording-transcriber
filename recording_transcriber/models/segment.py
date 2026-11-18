"""A speech segment."""

from pydantic import BaseModel


class Segment(BaseModel):
    """A stretch of speech from a speech-to-text model, timed from the clip start.

    speaker is the model's label (only valid within one call) or None without
    diarization; start_s is None if the model returns no timestamps."""

    speaker: str | None = None
    start_s: float | None = None
    text: str
