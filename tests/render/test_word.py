"""Tests for the Word writer."""

from io import BytesIO

from docx import Document

from recording_transcriber.models.document import Document as WrittenDocument
from recording_transcriber.models.document_section import DocumentSection
from recording_transcriber.models.paragraph import Paragraph
from recording_transcriber.render.word import document_docx, transcript_docx


def test_transcript_docx():
    data = transcript_docx(
        "Board",
        [Paragraph(start_s=65, speaker="Dr. A", speaker_ids=["S1"], text="Hallo.")],
    )
    document = Document(BytesIO(data))
    texts = [paragraph.text for paragraph in document.paragraphs]
    assert texts == ["Board", "[00:01:05] Dr. A: Hallo."]
    assert document.paragraphs[0].style.name == "Heading 1"
    assert document.paragraphs[1].runs[0].bold


def test_document_docx():
    data = document_docx(
        WrittenDocument(
            title="Board",
            sections=[
                DocumentSection(heading="Agenda", paragraphs=["Text."], bullets=["a"])
            ],
        )
    )
    document = Document(BytesIO(data))
    texts = [(p.style.name, p.text) for p in document.paragraphs]
    assert texts == [
        ("Heading 1", "Board"),
        ("Heading 2", "Agenda"),
        ("Normal", "Text."),
        ("List Bullet", "a"),
    ]


def test_transcript_docx_note():
    data = transcript_docx("Board", [], ["Checks passed.", "Two speakers."])
    paragraph = Document(BytesIO(data)).paragraphs[1]
    assert paragraph.text == "Quality notes\nChecks passed.\nTwo speakers."
    assert paragraph.runs[0].bold and paragraph.runs[1].italic


def test_docx_drops_control_characters():
    # invalid in the XML, and python-docx would not escape them
    written = WrittenDocument(
        title="Bo\x0bard",
        sections=[
            DocumentSection(
                heading="Ag\x01enda", paragraphs=["Te\x00xt."], bullets=["\x1fa"]
            )
        ],
    )
    texts = [p.text for p in Document(BytesIO(document_docx(written))).paragraphs]
    assert texts == ["Board", "Agenda", "Text.", "a"]
    paragraph = Paragraph(
        start_s=0, speaker="Dr.\x0c A", speaker_ids=["S1"], text="H\x08i"
    )
    data = transcript_docx("Board", [paragraph], ["Chec\x0bks."])
    texts = [p.text for p in Document(BytesIO(data)).paragraphs]
    assert texts == ["Board", "Quality notes\nChecks.", "[00:00:00] Dr. A: Hi"]
