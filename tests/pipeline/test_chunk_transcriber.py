"""Tests for the chunk transcriber."""

import json

import pytest

from recording_transcriber.audio.ffmpeg import Ffmpeg
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.job_context import JobContext
from recording_transcriber.models.model_response import ModelResponse
from recording_transcriber.models.roster_entry import RosterEntry
from recording_transcriber.models.turn import Turn
from recording_transcriber.models.usage import Usage
from recording_transcriber.pipeline.chunk_transcriber import ChunkTranscriber
from recording_transcriber.utils import retry
from recording_transcriber.utils.const import MAX_ATTEMPTS
from tests.fakes import chunk_answer, completed
from tests.pipeline.recording import RecordingBackend

_GOOD = chunk_answer(
    [("S1", "00:00", "Guten Morgen, mein Name ist Huber."), ("S2", "00:06", "Danke.")],
    roster=[
        ("S1", "Dr. Huber", "00:00 Selbstvorstellung", "tief"),
        ("S2", "", "", "hell"),
    ],
    unclear=[("00:04", "Husten")],
)
_SPAN = ChunkSpan(index=0, start_s=0, end_s=8)
_TRUNCATED = ModelResponse(text='{"tu', status="incomplete")


class TestChunkTranscriber:
    @pytest.fixture
    def make(self, tmp_path):
        def make(*responses, context=None):
            backend = RecordingBackend(transcriptions=responses)
            context = context or JobContext(
                glossary=["KDIGO", "UACR"], attendees="Dr. Huber", agenda="TOP 1"
            )
            transcriber = ChunkTranscriber(
                backend=backend,
                ffmpeg=Ffmpeg(),
                model="m",
                context=context,
                silences=[(3.0, 5.0)],
                part_count=1,
                scratch_dir=tmp_path,
            )
            return transcriber, backend

        yield make

    @pytest.fixture
    def sleeps(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr(retry.time, "sleep", sleeps.append)
        yield sleeps

    def test_transcribe(self, make, tone_file):
        transcriber, backend = make(completed(_GOOD))
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert [turn.speaker_id for turn in result.turns] == ["S1", "S2"]
        assert result.roster[0].name == "Dr. Huber"
        assert result.roster[1].name is None
        assert result.roster[0].name_evidence == "00:00 Selbstvorstellung"
        assert result.roster[1].name_evidence is None
        assert [entry.voice for entry in result.roster] == ["tief", "hell"]
        assert result.unclear[0].start_s == 4
        assert result.warnings == []
        assert backend.calls == ["m"]
        prompt = backend.prompts[0]
        for data in ("Dr. Huber", "TOP 1", "KDIGO", "UACR"):
            assert data in prompt

    def test_transcribe_offsets_and_context(self, make, tone_file):
        transcriber, backend = make(completed(_GOOD))
        span = ChunkSpan(index=1, start_s=2, end_s=8)
        roster = [
            RosterEntry(speaker_id="S1", name="Dr. Müller", voice="tief"),
            RosterEntry(speaker_id="S2"),
        ]
        previous = [
            Turn(speaker_id="S1", start_s=1, text="Vorher."),
            Turn(speaker_id="S2", start_s=1.5, text="Danach."),
        ]
        result = transcriber.transcribe(tone_file, span, roster, previous)
        assert [turn.start_s for turn in result.turns] == [2, 8]
        prompt = backend.prompts[0]
        assert "00:00:02" in prompt and "00:06" in prompt
        assert "[00:00:01] S1: Vorher.\n[00:00:01] S2: Danach." in prompt
        # not escaped, one entry per line
        first, second = (json.dumps(e.model_dump(), ensure_ascii=False) for e in roster)
        assert f"{first}\n{second}\n" in prompt and "Müller" in prompt

    def test_transcribe_keeps_roster_entries_left_out(self, make, tone_file):
        transcriber, _ = make(completed(_GOOD))
        roster = [RosterEntry(speaker_id="S7", voice="leise")]
        result = transcriber.transcribe(tone_file, _SPAN, roster, [])
        assert [entry.speaker_id for entry in result.roster] == ["S1", "S2", "S7"]

    def test_transcribe_retries_with_backoff(self, make, tone_file, sleeps):
        transcriber, _ = make(
            None, ModelResponse(text="", status="failed"), completed(_GOOD)
        )
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert result.attempts == 3
        assert sleeps == [10, 20]

    def test_transcribe_gives_up_without_final_sleep(self, make, tone_file, sleeps):
        transcriber, _ = make(*[None] * MAX_ATTEMPTS)
        assert transcriber.transcribe(tone_file, _SPAN, [], []) is None
        assert len(sleeps) == MAX_ATTEMPTS - 1

    def test_transcribe_schema_error_reasked_at_once(self, make, tone_file, sleeps):
        transcriber, _ = make(completed("not json"), completed(_GOOD))
        assert transcriber.transcribe(tone_file, _SPAN, [], []).attempts == 2
        assert sleeps == []

    def test_transcribe_schema_error_twice(self, make, tone_file):
        transcriber, _ = make(
            completed("{}"), completed('{"turns": 1}'), completed(_GOOD)
        )
        assert transcriber.transcribe(tone_file, _SPAN, [], []) is None

    def test_transcribe_schema_error_twice_keeps_flagged(self, make, tone_file):
        transcriber, _ = make(
            completed(chunk_answer([])), completed("{}"), completed("{}")
        )
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert result.warnings == ["NO_TURNS"]

    def test_transcribe_invalid_timestamps_tolerated(self, make, tone_file):
        answer = chunk_answer(
            [
                ("S1", "later", "Hallo."),
                ("S1", "00:02", "Hallo."),
                ("S2", "soon", "Danke."),
                ("S1", "00:07", "Bitte."),
            ],
            unclear=[("n/a", "Husten")],
        )
        transcriber, _ = make(completed(answer))
        span = ChunkSpan(index=0, start_s=1, end_s=8)
        result = transcriber.transcribe(tone_file, span, [], [])
        # an unreadable first time is the chunk start
        assert [turn.start_s for turn in result.turns] == [1, 3, 3, 8]
        assert result.unclear == []

    def test_transcribe_sanity_retry_then_flag(self, make, tone_file):
        empty = chunk_answer([])
        transcriber, backend = make(
            completed(empty, tokens=1), completed(empty, tokens=2)
        )
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert result.warnings == ["NO_TURNS"]
        assert result.attempts == 2
        assert result.usage.output_tokens == 3
        assert backend.calls == ["m", "m"]

    def test_transcribe_sanity_keeps_better_answer(self, make, tone_file):
        gap = chunk_answer([("S1", "00:00", "Nur am Anfang.")])
        transcriber, _ = make(completed(chunk_answer([])), completed(gap))
        span = ChunkSpan(index=0, start_s=0, end_s=200)
        result = transcriber.transcribe(tone_file, span, [], [])
        assert result.warnings == ["TAIL_GAP"]

    def test_transcribe_sanity_retry_succeeds(self, make, tone_file):
        transcriber, _ = make(completed(chunk_answer([])), completed(_GOOD))
        assert transcriber.transcribe(tone_file, _SPAN, [], []).warnings == []

    def test_transcribe_flagged_kept_after_failures(self, make, tone_file):
        transcriber, _ = make(
            completed(chunk_answer([]), tokens=1), *[None] * (MAX_ATTEMPTS - 1)
        )
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert result.warnings == ["NO_TURNS"]
        assert result.usage.output_tokens == 1

    def test_transcribe_split_on_truncation(self, make, tone_file):
        first = chunk_answer(
            [("S1", "00:00", "Erster Teil.")], roster=[("S1", "", "", "tief")]
        )
        second = chunk_answer(
            [("S1", "00:01", "Zweiter Teil.")], roster=[("S1", "", "", "tief")]
        )
        transcriber, backend = make(_TRUNCATED, completed(first), completed(second))
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert [turn.start_s for turn in result.turns] == [0, 5]
        assert result.span == _SPAN
        assert result.warnings[0] == "SPLIT"
        assert result.attempts == 2
        assert "Erster Teil." in backend.prompts[2]

    def test_transcribe_split_on_truncation_merges_halves(self, make, tone_file):
        truncated = ModelResponse(
            text='{"tu', status="incomplete", usage=Usage(output_tokens=5)
        )
        first = chunk_answer([("S1", "00:00", "Eins.")], unclear=[("00:01", "Husten")])
        second = chunk_answer([("S1", "00:01", "Zwei.")], unclear=[("00:02", "Hupe")])
        transcriber, _ = make(truncated, completed(first), completed(second))
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert [passage.start_s for passage in result.unclear] == [1, 6]
        # the truncated call and both halves are billed
        assert result.usage.output_tokens == 25

    def test_transcribe_split_fails(self, make, tone_file):
        transcriber, backend = make(_TRUNCATED, _TRUNCATED, _TRUNCATED)
        assert transcriber.transcribe(tone_file, _SPAN, [], []) is None
        # the chunk and its first half are split; the quarter is truncated again
        assert backend.transcriptions == []

    def test_transcribe_splits_at_most_twice(self, make, tone_file):
        # every second half is truncated again: the chunk is split twice, then given up
        good = completed(_GOOD)
        transcriber, backend = make(_TRUNCATED, good, _TRUNCATED, good, _TRUNCATED)
        assert transcriber.transcribe(tone_file, _SPAN, [], []) is None
        assert backend.transcriptions == []

    def test_transcribe_split_second_half_fails(self, make, tone_file):
        transcriber, _ = make(_TRUNCATED, completed(_GOOD), *[None] * MAX_ATTEMPTS)
        assert transcriber.transcribe(tone_file, _SPAN, [], []) is None

    def test_transcribe_extract_fails(self, make, tmp_path):
        transcriber, _ = make()
        assert transcriber.transcribe(tmp_path / "missing.mp3", _SPAN, [], []) is None
