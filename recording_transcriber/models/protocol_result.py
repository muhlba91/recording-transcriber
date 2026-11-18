"""The protocol step's result."""

from pydantic import BaseModel

from .document import Document
from .usage import Usage


class ProtocolResult(BaseModel):
    """The protocol and agenda (None if their calls failed), usage and warnings."""

    protocol: Document | None = None
    agenda: Document | None = None
    protocol_usage: Usage = Usage()
    agenda_usage: Usage = Usage()
    warnings: list[str] = []
