"""The quality note at the top of the transcript."""

from ..models.chunk_result import ChunkResult
from ..models.paragraph import Paragraph
from ..models.speaker_mapping import SpeakerMapping
from ..utils.const import (
    QUALITY_ALL_NAMED,
    QUALITY_CHECK,
    QUALITY_ESTIMATE,
    QUALITY_FLAGS,
    QUALITY_PASSED,
    QUALITY_SMALL_SHARE,
    QUALITY_UNNAMED,
)
from .review import review_items, technical_hints


def quality_note(
    paragraphs: list[Paragraph],
    results: list[ChunkResult],
    mapping: SpeakerMapping,
    unclear_marker: str,
    estimate: str | None,
) -> list[str]:
    """Return the quality note's sentences: only counted facts and a labeled estimate.

    The model's own uncertainty markers miss most errors, so no score is derived
    from them (design §4). Warning codes without a user-facing text (a split chunk)
    are not problems and do not fail the checks.

    paragraphs: the transcript paragraphs
    results: all transcribed chunks
    mapping: the speaker mapping
    unclear_marker: the transcript's marker for unintelligible speech
    estimate: the evaluated word accuracy of the model (None: not evaluated)"""
    hints = technical_hints(results, mapping)
    note = [QUALITY_CHECK.format(hints=" ".join(hints)) if hints else QUALITY_PASSED]
    words = {}
    named = set()
    for paragraph in paragraphs:
        count = len(paragraph.text.split())
        speaker = paragraph.speaker
        # a negated total gives the same share
        words[speaker] = words.get(speaker, 0) + count  # pragma: no mutate
        if any(mapping.is_named(identifier) for identifier in paragraph.speaker_ids):
            named.add(paragraph.speaker)
    unnamed = [label for label in words if label not in named]
    if unnamed:
        total = sum(words.values()) or 1  # pragma: no mutate (0 / any total is 0)
        percent = 100 * sum(words[label] for label in unnamed) / total
        # pragma: no mutate start (threshold)
        share = QUALITY_SMALL_SHARE if 0 < percent < 1 else round(percent)
        # pragma: no mutate end
        note.append(
            QUALITY_UNNAMED.format(named=len(named), unnamed=len(unnamed), share=share)
        )
    elif named:
        note.append(QUALITY_ALL_NAMED.format(named=len(named)))
    if estimate:
        note.append(
            QUALITY_ESTIMATE.format(estimate=estimate.strip().removesuffix("."))
        )
    unclear, uncertain = review_items(paragraphs, results, unclear_marker)
    note.append(QUALITY_FLAGS.format(unclear=len(unclear), uncertain=len(uncertain)))
    return note
