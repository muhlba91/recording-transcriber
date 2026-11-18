"""Tests for the paragraph builder."""

from recording_transcriber.models.turn import Turn
from recording_transcriber.render.paragraphs import build_paragraphs


def test_build_paragraphs_merges_same_speaker():
    turns = [
        Turn(speaker_id="S1", start_s=0, text="Guten Morgen."),
        Turn(speaker_id="S3", start_s=5, text="Weiter."),
        Turn(speaker_id="S2", start_s=9, text="Danke."),
    ]
    paragraphs = build_paragraphs(
        turns, {"S1": "Dr. A", "S3": "Dr. A", "S2": "Sprecher 1"}
    )
    assert [(p.speaker, p.speaker_ids, p.text) for p in paragraphs] == [
        ("Dr. A", ["S1", "S3"], "Guten Morgen. Weiter."),
        ("Sprecher 1", ["S2"], "Danke."),
    ]


def test_build_paragraphs_long_talk_keeps_timestamps():
    turns = [
        Turn(speaker_id="S1", start_s=start, text="x") for start in (0, 80, 160, 200)
    ]
    paragraphs = build_paragraphs(turns, {"S1": "Dr. A"})
    assert [p.start_s for p in paragraphs] == [0, 160]


def test_build_paragraphs_unknown_id_keeps_id():
    paragraphs = build_paragraphs([Turn(speaker_id="S9", start_s=0, text="x")], {})
    assert paragraphs[0].speaker == "S9"
