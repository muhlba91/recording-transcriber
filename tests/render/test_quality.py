"""Tests for the quality note."""

import pytest

from recording_transcriber.models.chunk_result import ChunkResult
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.paragraph import Paragraph
from recording_transcriber.models.speaker_assignment import SpeakerAssignment
from recording_transcriber.models.speaker_mapping import SpeakerMapping
from recording_transcriber.render.quality import quality_note


def _result(warnings=None) -> ChunkResult:
    return ChunkResult(
        span=ChunkSpan(index=0, start_s=0, end_s=60),
        model="m",
        turns=[],
        roster=[],
        warnings=warnings or [],
    )


def _paragraphs() -> list[Paragraph]:
    return [
        Paragraph(start_s=0, speaker="Dr. A", speaker_ids=["S1"], text="a b c d e f"),
        Paragraph(start_s=9, speaker="Sprecher 1", speaker_ids=["S2"], text="x y"),
        Paragraph(
            start_s=20, speaker="Dr. A", speaker_ids=["S1"], text="g (?) [unklar]"
        ),
    ]


_MAPPING = SpeakerMapping(
    speakers=[SpeakerAssignment(ids=["S1"], name="Dr. A", confidence="high")]
)


def test_quality_note():
    note = quality_note(_paragraphs(), [_result()], _MAPPING, "[unklar]", "about 95 %")
    assert note == [
        "Automatic checks: passed (no gaps, repetitions or timestamp problems found).",
        "Speakers: 1 named; 1 unnamed (18 % of the words).",
        (
            "Typical word accuracy for this setup, estimated from the evaluation and "
            "not measured on this recording: about 95 %."
        ),
        (
            "The model marked 1 unclear passage(s) and 1 uncertain word(s) (see "
            "flags.md); unmarked errors remain likely, most often in names and "
            "technical terms."
        ),
    ]


def test_quality_note_issues():
    mapping = SpeakerMapping(warnings=["RECONCILIATION_FAILED"])
    paragraphs = _paragraphs()[1:2]
    note = quality_note(paragraphs, [_result(["TAIL_GAP"])], mapping, "[unklar]", None)
    assert note[0] == (
        "Check recommended: The end of part 1 may be missing. Speaker reconciliation "
        "failed; names come straight from the transcription."
    )
    assert note[1] == "Speakers: 0 named; 1 unnamed (100 % of the words)."
    assert len(note) == 3


def test_quality_note_all_named():
    note = quality_note(_paragraphs()[:1], [_result()], _MAPPING, "[unklar]", None)
    assert note[1] == "Speakers: all 1 named."


@pytest.mark.parametrize(
    ("named_words", "unnamed_words", "share"),
    [(199, 1, "<1"), (99, 1, "1")],
)
def test_quality_note_share_of_unnamed_words(named_words, unnamed_words, share):
    paragraphs = [
        Paragraph(
            start_s=0, speaker="Dr. A", speaker_ids=["S1"], text="a " * named_words
        ),
        Paragraph(
            start_s=9,
            speaker="Sprecher 1",
            speaker_ids=["S2"],
            text="x " * unnamed_words,
        ),
    ]
    note = quality_note(paragraphs, [_result()], _MAPPING, "[unklar]", "95 %.")
    assert note[1] == f"Speakers: 1 named; 1 unnamed ({share} % of the words)."
    assert note[2].endswith(": 95 %.")


def test_quality_note_no_text():
    note = quality_note([], [_result()], SpeakerMapping(), "[unklar]", None)
    assert len(note) == 2
