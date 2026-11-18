"""The job context."""

from pydantic import BaseModel


class JobContext(BaseModel):
    """Optional context provided per job in the context/ folder."""

    glossary: list[str] = []
    attendees: str | None = None
    agenda: str | None = None
