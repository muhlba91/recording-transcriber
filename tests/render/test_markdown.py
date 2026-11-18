"""Tests for the Markdown writers."""

import pytest

from recording_transcriber.models.chunk_result import ChunkResult
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.document import Document
from recording_transcriber.models.document_section import DocumentSection
from recording_transcriber.models.paragraph import Paragraph
from recording_transcriber.models.roster_entry import RosterEntry
from recording_transcriber.models.speaker_assignment import SpeakerAssignment
from recording_transcriber.models.speaker_mapping import SpeakerMapping
from recording_transcriber.models.unclear_passage import UnclearPassage
from recording_transcriber.render.markdown import (
    document_markdown,
    flags_markdown,
    speakers_markdown,
    transcript_markdown,
)


class TestMarkdown:
    @pytest.fixture
    def data(self):
        paragraphs = [
            Paragraph(
                start_s=0,
                speaker="Dr. A",
                speaker_ids=["S1"],
                text="Die UACR (?) ist wichtig.",
            ),
            Paragraph(
                start_s=70,
                speaker="Sprecher 1",
                speaker_ids=["S2"],
                text="Das ist [unklar] so.",
            ),
            Paragraph(
                start_s=400,
                speaker="Dr. A",
                speaker_ids=["S1"],
                text="Später [unklar].\n# kein Titel",
            ),
        ]
        results = [
            ChunkResult(
                span=ChunkSpan(index=0, start_s=0, end_s=500),
                model="m",
                turns=[],
                roster=[
                    RosterEntry(speaker_id="S1", name="Dr. A", voice="tief | ruhig"),
                    RosterEntry(speaker_id="S2"),
                ],
                unclear=[UnclearPassage(start_s=30, note="Husten")],
                warnings=["TAIL_GAP", "SPLIT"],
            )
        ]
        mapping = SpeakerMapping(
            speakers=[
                SpeakerAssignment(
                    ids=["S1"],
                    name="Dr. A",
                    aliases=["Anna", "Frau Dr. A"],
                    confidence="high",
                    evidence="stellt sich vor",
                ),
                SpeakerAssignment(
                    ids=["S2"], name="Dr. B", confidence="low", evidence="vermutet"
                ),
            ],
            warnings=["RECONCILIATION_FAILED"],
        )
        labels = {"S1": "Dr. A", "S2": "Sprecher 1"}
        yield paragraphs, results, mapping, labels

    def test_transcript_markdown(self, data):
        text = transcript_markdown("Board", data[0], ["Eins.", "Zwei\nzeilig."])
        assert text == (
            "# Board\n\n> **Quality notes**\n>\n> - Eins.\n> - Zwei zeilig.\n\n"
            "**[00:00:00] Dr. A:** Die UACR (?) ist wichtig.\n\n"
            "**[00:01:10] Sprecher 1:** Das ist [unklar] so.\n\n"
            "**[00:06:40] Dr. A:** Später [unklar]. # kein Titel\n"
        )

    def test_flags_markdown(self, data):
        paragraphs, results, mapping, labels = data
        text = flags_markdown("Board", paragraphs, results, mapping, labels, "[unklar]")
        # the marker at 01:10 is next to the model note, so it is listed once
        assert text == (
            "# Review checklist: Board\n\n"
            "## Unclear passages\n\n"
            "- [ ] [00:00:30] Husten\n"
            "- [ ] [00:06:40] Später [unklar]. # kein Titel\n\n"
            "## Uncertain words\n\n"
            "- [ ] [00:00:00] Die UACR (?) ist wichtig.\n\n"
            "## Speakers\n\n"
            "- Sprecher 1: possibly Dr. B (low confidence). vermutet\n\n"
            "## Technical notes\n\n"
            "- The end of part 1 may be missing.\n"
            "- Speaker reconciliation failed; names come straight from the "
            "transcription.\n"
        )

    def test_flags_markdown_clean(self):
        mapping = SpeakerMapping(
            speakers=[SpeakerAssignment(ids=["S1"], name="Dr. A", confidence="high")]
        )
        text = flags_markdown("Board", [], [], mapping, {"S1": "Dr. A"}, "[unclear]")
        assert text.count("- None.\n") == 2
        assert text.endswith("## Speakers\n\n- All speakers named.\n")
        assert "Technical notes" not in text

    def test_flags_markdown_unidentified(self):
        text = flags_markdown(
            "Board",
            [],
            [],
            SpeakerMapping(),
            {"S1": "Sprecher 1", "S2": "Sprecher 2", "S3": "Sprecher 1"},
            "[unklar]",
        )
        # one line per label, not per id
        assert "## Speakers\n\n- Sprecher 1: not identified.\n" in text
        assert text.endswith("- Sprecher 2: not identified.\n")
        assert text.count("not identified") == 2

    def test_flags_markdown_long_excerpt(self):
        paragraph = Paragraph(
            start_s=0,
            speaker="A",
            speaker_ids=["S1"],
            text="a" * 100 + " [?] " + "b" * 100,
        )
        text = flags_markdown(
            "Board", [paragraph], [], SpeakerMapping(), {"S1": "A"}, "[?]"
        )
        # the marker is matched literally, not as a regex
        assert "…" + "a" * 59 + " [?] " + "b" * 59 + "…" in text

    def test_speakers_markdown(self, data):
        _, results, mapping, labels = data
        # matched by last name: "Dr. Anna A" and "Anna"; blank lines are skipped
        attendees = "Dr. Anna A, Moderation\nDr. Carl C\n\nBerta Anna\nAnna\n"
        text = speakers_markdown("Board", mapping, labels, results, attendees)
        assert text == (
            "# Speakers: Board\n\n"
            "| Label | Called | IDs | Confidence | Evidence | Voice |\n"
            "|---|---|---|---|---|---|\n"
            "| Dr. A | Anna, Frau Dr. A | S1 | high | stellt sich vor | tief / ruhig |\n"
            "| Sprecher 1 |  | S2 | low | vermutet |  |\n\n"
            "## Attendees without a matched voice\n\n- Dr. Carl C\n"
        )

    def test_speakers_markdown_without_roster_entry(self):
        text = speakers_markdown("Board", SpeakerMapping(), {"S1": "A"}, [], None)
        assert text.endswith("| A |  | S1 |  |  |  |\n")

    def test_speakers_markdown_shared_label(self, data):
        _, results, mapping, _ = data
        text = speakers_markdown(
            "Board", mapping, {"S1": "A", "S2": "A"}, results, None
        )
        assert text.endswith(
            "| A | Anna, Frau Dr. A | S1, S2 | high | stellt sich vor | tief / ruhig |\n"
        )

    def test_speakers_markdown_unmapped(self, data):
        _, results, _, _ = data
        text = speakers_markdown(
            "Board", SpeakerMapping(), {"S1": "Sprecher 1"}, results, None
        )
        assert "| Sprecher 1 |  | S1 |  |  | tief / ruhig |" in text
        assert "Attendees" not in text


def test_flags_markdown_note_covers_one_marker():
    paragraph = Paragraph(
        start_s=0, speaker="A", speaker_ids=["S1"], text="eins [?] zwei [?] drei"
    )
    result = ChunkResult(
        span=ChunkSpan(index=0, start_s=0, end_s=60),
        model="m",
        turns=[],
        roster=[],
        unclear=[UnclearPassage(start_s=5, note="Husten")],
    )
    text = flags_markdown("Board", [paragraph], [result], SpeakerMapping(), {}, "[?]")
    assert text.count("- [ ] ") == 2


def test_document_markdown():
    document = Document(
        title="Board",
        sections=[
            DocumentSection(heading="Agenda", bullets=["a", "# b"]),
            DocumentSection(heading="Topic", paragraphs=["One\ntwo."]),
        ],
    )
    assert document_markdown(document) == (
        "# Board\n\n## Agenda\n\n- a\n- # b\n\n## Topic\n\nOne two.\n"
    )


def test_transcript_markdown_note():
    text = transcript_markdown("Board", [], ["Checks passed."])
    assert "> **Quality notes**" in text
    assert text.endswith("> - Checks passed.\n")
