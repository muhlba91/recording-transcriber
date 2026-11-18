"""A speaker turn."""

from pydantic import BaseModel


class Turn(BaseModel):
    """A speaker turn, timed in seconds from the start of the recording."""

    speaker_id: str
    start_s: float
    text: str
