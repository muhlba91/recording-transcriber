"""A roster entry."""

from pydantic import BaseModel


class RosterEntry(BaseModel):
    """A voice the model identified, carried from chunk to chunk."""

    speaker_id: str
    name: str | None = None
    name_evidence: str | None = None
    voice: str = ""
