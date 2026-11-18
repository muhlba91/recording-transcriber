"""A chunk of the prepared recording."""

from pydantic import BaseModel


class ChunkSpan(BaseModel):
    """A chunk of the prepared recording, in seconds from its start."""

    index: int
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Return the chunk length in seconds."""
        return self.end_s - self.start_s
