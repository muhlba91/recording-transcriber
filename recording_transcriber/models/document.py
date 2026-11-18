"""A written document (protocol or agenda)."""

from pydantic import BaseModel

from .document_section import DocumentSection


class Document(BaseModel):
    """A document written by a model from the transcript."""

    title: str
    sections: list[DocumentSection] = []
