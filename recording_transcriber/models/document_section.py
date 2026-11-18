"""A document section."""

from pydantic import BaseModel


class DocumentSection(BaseModel):
    """A section of a written document: a heading, prose and list items."""

    heading: str
    paragraphs: list[str] = []
    bullets: list[str] = []
