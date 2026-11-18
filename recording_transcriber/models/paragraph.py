"""A transcript paragraph."""

from pydantic import BaseModel


class Paragraph(BaseModel):
    """A rendered paragraph: consecutive turns of one speaker."""

    start_s: float
    speaker: str
    speaker_ids: list[str]
    text: str
