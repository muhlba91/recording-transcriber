"""A speaker assignment."""

from pydantic import BaseModel

from ..utils.const import CONFIDENCE_LOW


class SpeakerAssignment(BaseModel):
    """One person: the speaker ids that belong to them and their name.

    aliases are all forms the person was called in the recording ("Anna", "Dr. X")."""

    ids: list[str]
    name: str | None = None
    aliases: list[str] = []
    confidence: str = CONFIDENCE_LOW
    evidence: str = ""
