"""Tests for the legacy import."""

from recording_transcriber.evaluation.legacy import legacy_record


def test_legacy_record():
    text = (
        "---\nfile: a.mp3.0.mp3\n\n"
        "Dr. A: Guten Morgen.\n[Publikum applaudiert]\n\n"
        "Dr. B: Danke [...] sehr.\n"
    )
    record = legacy_record(text)
    assert record["job"] == "legacy"
    assert record["speakers"] == []
    assert record["paragraphs"][0]["speaker_ids"] == ["Dr. A"]
    assert [(p["speaker"], p["text"], p["start_s"]) for p in record["paragraphs"]] == [
        ("Dr. A", "Guten Morgen. [Publikum applaudiert]", None),
        ("Dr. B", "Danke [...] sehr.", None),
    ]


def test_legacy_record_text_before_first_speaker():
    assert legacy_record("ohne Sprecher\n")["paragraphs"] == []
