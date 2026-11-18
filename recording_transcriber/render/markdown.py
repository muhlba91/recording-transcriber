"""Markdown writers: transcript, review checklist (flags), speaker list and written
documents (protocol, agenda)."""

import re

from ..models.chunk_result import ChunkResult
from ..models.document import Document
from ..models.paragraph import Paragraph
from ..models.speaker_mapping import SpeakerMapping
from ..utils.const import QUALITY_HEADING
from ..utils.timestamps import format_timestamp
from .review import review_items, technical_hints


def transcript_markdown(
    title: str, paragraphs: list[Paragraph], note: list[str] | None = None
) -> str:
    """Return the transcript as Markdown, one `[hh:mm:ss] Name:` paragraph per turn.

    title: the document title
    paragraphs: the transcript paragraphs
    note: the quality note's sentences, shown under the title (default: None)"""
    lines = [f"# {title}", ""]
    if note:
        lines += [f"> **{QUALITY_HEADING}**", ">"]
        lines += [f"> - {_line(sentence)}" for sentence in note] + [""]
    for paragraph in paragraphs:
        timestamp = format_timestamp(paragraph.start_s)
        lines += [f"**[{timestamp}] {paragraph.speaker}:** {_line(paragraph.text)}", ""]
    return "\n".join(lines)


def document_markdown(document: Document) -> str:
    """Return a protocol or agenda as Markdown.

    document: the document"""
    lines = [f"# {_line(document.title)}", ""]
    for section in document.sections:
        lines += [f"## {_line(section.heading)}", ""]
        for text in section.paragraphs:
            lines += [_line(text), ""]
        lines += [f"- {_line(text)}" for text in section.bullets]
        if section.bullets:
            lines.append("")
    return "\n".join(lines)


def flags_markdown(
    title: str,
    paragraphs: list[Paragraph],
    results: list[ChunkResult],
    mapping: SpeakerMapping,
    labels: dict[str, str],
    unclear_marker: str,
) -> str:
    """Return the review checklist: unclear passages, uncertain words and speakers.

    title: the document title
    paragraphs: the transcript paragraphs
    results: all transcribed chunks
    mapping: the speaker mapping
    labels: the display label per speaker id
    unclear_marker: the transcript's marker for unintelligible speech"""
    unclear, uncertain = review_items(paragraphs, results, unclear_marker)
    lines = [f"# Review checklist: {title}", ""]
    lines += _section("Unclear passages", unclear)
    lines += _section("Uncertain words", uncertain)
    lines += ["## Speakers", ""]
    lines += _speaker_flags(mapping, labels) or ["- All speakers named.", ""]
    hints = technical_hints(results, mapping)
    if hints:
        lines += ["## Technical notes", ""] + [f"- {hint}" for hint in hints] + [""]
    return "\n".join(lines)


def speakers_markdown(
    title: str,
    mapping: SpeakerMapping,
    labels: dict[str, str],
    results: list[ChunkResult],
    attendees: str | None,
) -> str:
    """Return the speaker list with names heard, evidence and voice descriptions.

    title: the document title
    mapping: the speaker mapping
    labels: the display label per speaker id
    results: all transcribed chunks (for the voice descriptions)
    attendees: the attendee list, to show who was not matched (may be None)"""
    voices = {
        entry.speaker_id: entry.voice for result in results for entry in result.roster
    }
    groups = mapping.group_of()
    lines = [
        f"# Speakers: {title}",
        "",
        "| Label | Called | IDs | Confidence | Evidence | Voice |",
        "|---|---|---|---|---|---|",
    ]
    for label in dict.fromkeys(labels.values()):
        ids = [speaker_id for speaker_id, value in labels.items() if value == label]
        group = groups.get(ids[0])
        cells = [
            label,
            ", ".join(group.aliases) if group else "",
            ", ".join(ids),
            group.confidence if group else "",
            group.evidence if group else "",
            voices.get(ids[0], ""),
        ]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    lines.append("")
    unmatched = _unmatched_attendees(attendees, labels, mapping)
    if unmatched:
        lines += ["## Attendees without a matched voice", ""]
        lines += [f"- {attendee}" for attendee in unmatched] + [""]
    return "\n".join(lines)


def _unmatched_attendees(
    attendees: str | None, labels: dict[str, str], mapping: SpeakerMapping
) -> list[str]:
    """Return attendees whose last name is not a word in any label or alias.

    attendees: the attendee list (one person per line, name first)
    labels: the display label per speaker id
    mapping: the speaker mapping"""
    if not attendees:
        return []
    names = [
        *labels.values(),
        *(alias for group in mapping.speakers for alias in group.aliases),
    ]
    heard = set(re.findall(r"\w+", " ".join(names).lower()))
    unmatched = []
    for line in attendees.splitlines():
        name = line.split(",")[0].strip()
        last_name = name.split()[-1].lower() if name else ""
        if last_name and last_name not in heard:
            unmatched.append(name)
    return unmatched


def _speaker_flags(mapping: SpeakerMapping, labels: dict[str, str]) -> list[str]:
    """Return checklist lines for speakers shown without a name.

    mapping: the speaker mapping
    labels: the display label per speaker id"""
    groups = mapping.group_of()
    lines = []
    seen = set()
    for speaker_id, label in labels.items():
        if mapping.is_named(speaker_id) or label in seen:
            continue
        seen.add(label)
        group = groups.get(speaker_id)
        if group is not None and group.name:
            lines.append(
                f"- {label}: possibly {group.name} (low confidence). {group.evidence}"
            )
        else:
            lines.append(f"- {label}: not identified.")
    return lines + [""] if lines else []


def _section(heading: str, items: list[tuple[float, str]]) -> list[str]:
    """Return a checklist section, or a "none" line if empty.

    heading: the section heading
    items: (seconds, text) entries"""
    lines = [f"## {heading}", ""]
    if not items:
        return lines + ["- None.", ""]  # pragma: no mutate (blank line)
    lines += [
        f"- [ ] [{format_timestamp(start)}] {_line(text)}" for start, text in items
    ]
    return lines + [""]


def _line(text: str) -> str:
    """Keep model text on one line, so it cannot start a heading or list.

    text: the text"""
    return " ".join(text.split())


def _cell(text: str) -> str:
    """Make text safe for a Markdown table cell.

    text: the cell text"""
    return _line(text).replace("|", "/")
