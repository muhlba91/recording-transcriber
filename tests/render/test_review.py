"""Tests for the review items."""

import pytest

from recording_transcriber.models.chunk_result import ChunkResult
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.paragraph import Paragraph
from recording_transcriber.models.unclear_passage import UnclearPassage
from recording_transcriber.render.review import review_items


def _result(*notes: UnclearPassage) -> ChunkResult:
    return ChunkResult(
        span=ChunkSpan(index=0, start_s=0, end_s=3600),
        model="m",
        turns=[],
        roster=[],
        unclear=list(notes),
    )


@pytest.mark.parametrize(
    ("start_s", "listed"),
    [(1000, 1), (879, 2)],
)
def test_review_items_note_covers_marker_within_two_minutes(start_s, listed):
    paragraph = Paragraph(start_s=start_s, speaker="A", speaker_ids=["S1"], text="[?]")
    unclear, _ = review_items(
        [paragraph], [_result(UnclearPassage(start_s=1000, note="Husten"))], "[?]"
    )
    assert len(unclear) == listed


@pytest.mark.parametrize(
    ("text", "excerpt"),
    [
        # the excerpt shows 60 characters on each side, with an ellipsis if cut off
        ("a" * 61 + "[?]" + "b" * 61, "…" + "a" * 60 + "[?]" + "b" * 60 + "…"),
        ("a" + "[?]" + "b", "a[?]b"),
    ],
)
def test_review_items_excerpt(text, excerpt):
    paragraph = Paragraph(start_s=0, speaker="A", speaker_ids=["S1"], text=text)
    unclear, _ = review_items([paragraph], [], "[?]")
    assert unclear == [(0, excerpt)]
