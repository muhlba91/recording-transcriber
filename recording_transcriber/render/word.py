"""Word (.docx) writers: transcript, protocol and agenda."""

import re
from io import BytesIO

from docx import Document

from ..models.document import Document as WrittenDocument
from ..models.paragraph import Paragraph
from ..utils.const import QUALITY_HEADING
from ..utils.timestamps import format_timestamp

# control characters, lone surrogates and non-characters are invalid in the XML
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def transcript_docx(
    title: str, paragraphs: list[Paragraph], note: list[str] | None = None
) -> bytes:
    """Return the transcript as a .docx file.

    title: the document title
    paragraphs: the transcript paragraphs
    note: the quality note's sentences, shown under the title (default: None)"""
    document = Document()
    document.add_heading(title)
    if note:
        block = document.add_paragraph()
        block.add_run(QUALITY_HEADING).bold = True
        for sentence in note:
            block.add_run(f"\n{_CONTROL.sub('', sentence)}").italic = True
    for paragraph in paragraphs:
        block = document.add_paragraph()
        heading = f"[{format_timestamp(paragraph.start_s)}] {paragraph.speaker}:"
        block.add_run(_CONTROL.sub("", heading)).bold = True
        block.add_run(f" {_CONTROL.sub('', paragraph.text)}")
    return _bytes(document)


def document_docx(written: WrittenDocument) -> bytes:
    """Return a protocol or agenda as a .docx file.

    written: the document"""
    document = Document()
    document.add_heading(_CONTROL.sub("", written.title))
    for section in written.sections:
        document.add_heading(_CONTROL.sub("", section.heading), level=2)
        for text in section.paragraphs:
            document.add_paragraph(_CONTROL.sub("", text))
        for text in section.bullets:
            document.add_paragraph(_CONTROL.sub("", text), style="List Bullet")
    return _bytes(document)


def _bytes(document) -> bytes:
    """Return a python-docx document as file content.

    document: the python-docx document"""
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
