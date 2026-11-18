"""Review items shared by the checklist (flags.md) and the transcript's quality note."""

import re

from ..models.chunk_result import ChunkResult
from ..models.paragraph import Paragraph
from ..models.speaker_mapping import SpeakerMapping
from ..utils.const import (
    EXCERPT_CHARS,
    RECONCILIATION_TEXT,
    UNCLEAR_DEDUP_S,
    WARNING_TEXT,
)

_UNCERTAIN = re.compile(r"\S+ \(\?\)")


def review_items(
    paragraphs: list[Paragraph], results: list[ChunkResult], unclear_marker: str
) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """Return the unclear passages (sorted) and uncertain words as (start_s, text).

    Each model note stands for one nearby unclear marker; further markers are listed
    with an excerpt.

    paragraphs: the transcript paragraphs
    results: all transcribed chunks
    unclear_marker: the transcript's marker for unintelligible speech"""
    marker = re.compile(re.escape(unclear_marker))
    notes = [
        (passage.start_s, passage.note)
        for result in results
        for passage in result.unclear
    ]
    unclear = list(notes)
    unmatched_notes = list(notes)
    uncertain = []
    for paragraph in paragraphs:
        for match in marker.finditer(paragraph.text):
            note = next(
                (
                    item
                    for item in unmatched_notes
                    if abs(item[0] - paragraph.start_s)  # pragma: no mutate
                    <= UNCLEAR_DEDUP_S
                ),
                None,
            )
            if note is not None:
                unmatched_notes.remove(note)
                continue
            unclear.append((paragraph.start_s, _excerpt(paragraph.text, match)))
        for match in _UNCERTAIN.finditer(paragraph.text):
            uncertain.append((paragraph.start_s, _excerpt(paragraph.text, match)))
    return sorted(unclear), uncertain


def technical_hints(results: list[ChunkResult], mapping: SpeakerMapping) -> list[str]:
    """Return the user-facing texts of the sanity-check and reconciliation warnings.

    results: all transcribed chunks
    mapping: the speaker mapping"""
    hints = [
        WARNING_TEXT[code].format(part=result.span.index + 1)
        for result in results
        for code in result.warnings
        if code in WARNING_TEXT
    ]
    if mapping.warnings:
        hints.append(RECONCILIATION_TEXT)
    return hints


def _excerpt(text: str, match: re.Match) -> str:
    """Return the text around a match, for locating it in the transcript.

    text: the paragraph text
    match: the match to show"""
    start = max(0, match.start() - EXCERPT_CHARS)
    end = min(len(text), match.end() + EXCERPT_CHARS)
    prefix = "…" if start > 0 else ""  # pragma: no mutate (threshold)
    suffix = "…" if end < len(text) else ""  # pragma: no mutate (threshold)
    return f"{prefix}{text[start:end]}{suffix}"
