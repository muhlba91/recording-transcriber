"""Tests for the speech-to-text chunk transcriber."""

import pytest

from recording_transcriber.audio.ffmpeg import Ffmpeg
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.roster_entry import RosterEntry
from recording_transcriber.models.segment import Segment
from recording_transcriber.models.speech_result import SpeechResult
from recording_transcriber.models.usage import Usage
from recording_transcriber.pipeline.speech_transcriber import (
    SpeechTranscriber,
    build_turns,
)
from recording_transcriber.pipeline.turn_polisher import TurnPolisher
from recording_transcriber.utils.const import MAX_ATTEMPTS
from tests.fakes import FakeBackend, FakeSpeechBackend, completed, polish_answer

_SPAN = ChunkSpan(index=1, start_s=100, end_s=108)
_RESULT = SpeechResult(
    segments=[
        Segment(speaker="spk:0", start_s=0.0, text="äh Guten Morgen."),
        Segment(speaker="spk:1", start_s=4.0, text="Danke."),
    ],
    usage=Usage(input_audio_tokens=200),
)


class TestSpeechTranscriber:
    @pytest.fixture
    def make(self, tmp_path, tone_file):
        def make(results, polisher=None):
            backend = FakeSpeechBackend(results)
            transcriber = SpeechTranscriber(
                backend=backend,
                ffmpeg=Ffmpeg(),
                model="m",
                vocabulary=["KDIGO"],
                scratch_dir=tmp_path,
                polisher=polisher,
            )
            return transcriber, backend

        yield make

    def test_transcribe(self, make, tone_file):
        transcriber, backend = make([_RESULT])
        earlier = [RosterEntry(speaker_id="P1-spk0")]
        span = ChunkSpan(index=1, start_s=0, end_s=8)
        result = transcriber.transcribe(tone_file, span, earlier, [])
        assert [(t.speaker_id, t.start_s) for t in result.turns] == [
            ("P2-spk0", 0.0),
            ("P2-spk1", 4.0),
        ]
        # earlier parts' ids are not carried: labels only hold within one call
        assert [entry.speaker_id for entry in result.roster] == ["P2-spk0", "P2-spk1"]
        assert result.usage.audio_seconds == 8
        assert result.usage.input_audio_tokens == 200
        assert backend.calls == [("m", ["KDIGO"])]
        assert result.warnings == []

    def test_transcribe_retries_failed_call(self, make, tone_file):
        transcriber, _ = make([None, _RESULT])
        result = transcriber.transcribe(
            tone_file, ChunkSpan(index=0, start_s=0, end_s=8), [], []
        )
        assert result.attempts == 2

    def test_transcribe_succeeds_on_last_attempt(self, make, tone_file):
        transcriber, _ = make([None] * (MAX_ATTEMPTS - 1) + [_RESULT])
        result = transcriber.transcribe(tone_file, _SPAN, [], [])
        assert result.attempts == MAX_ATTEMPTS

    def test_transcribe_fails(self, make, tone_file):
        transcriber, _ = make([None] * MAX_ATTEMPTS)
        assert transcriber.transcribe(tone_file, _SPAN, [], []) is None

    def test_transcribe_extract_fails(self, make, tmp_path):
        transcriber, _ = make([_RESULT])
        assert transcriber.transcribe(tmp_path / "missing.mp3", _SPAN, [], []) is None

    def test_transcribe_polishes_verbatim(self, make, tone_file):
        text = FakeBackend(
            completions=[completed(polish_answer("Guten Morgen.", "Danke."))]
        )
        polisher = TurnPolisher(text, "r", [])
        transcriber, _ = make([_RESULT], polisher=polisher)
        result = transcriber.transcribe(
            tone_file, ChunkSpan(index=0, start_s=0, end_s=8), [], []
        )
        assert result.turns[0].text == "Guten Morgen."
        assert result.text_usage.output_tokens == 10

    def test_transcribe_flags_failed_polish(self, make, tone_file):
        text = FakeBackend(completions=[completed("{}")] * MAX_ATTEMPTS)
        polisher = TurnPolisher(text, "r", [])
        transcriber, _ = make([_RESULT], polisher=polisher)
        result = transcriber.transcribe(
            tone_file, ChunkSpan(index=0, start_s=0, end_s=8), [], []
        )
        assert result.turns[0].text == "äh Guten Morgen."
        assert result.warnings == ["POLISH_FAILED"]

    def test_transcribe_timed_checks_timing(self, make, tone_file):
        timed = SpeechResult(segments=[Segment(start_s=0, text="Nur am Anfang.")])
        transcriber, _ = make([timed])
        result = transcriber.transcribe(
            tone_file, ChunkSpan(index=0, start_s=0, end_s=200), [], []
        )
        assert result.warnings == ["TAIL_GAP"]

    def test_transcribe_untimed(self, make, tone_file):
        untimed = SpeechResult(segments=[Segment(text="Alles in einem.")])
        transcriber, _ = make([untimed])
        result = transcriber.transcribe(
            tone_file, ChunkSpan(index=0, start_s=0, end_s=200), [], []
        )
        assert [(t.speaker_id, t.start_s) for t in result.turns] == [("P1-X", 0.0)]
        # no timing checks without timestamps (the tail gap would always fire)
        assert result.warnings == []

    def test_transcribe_no_speech(self, make, tone_file):
        transcriber, _ = make([SpeechResult(segments=[])])
        result = transcriber.transcribe(
            tone_file, ChunkSpan(index=0, start_s=0, end_s=8), [], []
        )
        assert result.warnings == ["NO_TURNS"]


def test_build_turns_merges_until_sentence_end():
    segments = [
        Segment(speaker="a", start_s=0, text="Eins."),
        Segment(speaker="a", start_s=10, text="Zwei"),
        Segment(speaker="a", start_s=25, text="drei."),
        Segment(speaker="a", start_s=30, text="Vier."),
        Segment(speaker="b", start_s=32, text="Fünf."),
        Segment(speaker="b", start_s=None, text="Sechs."),
    ]
    turns = build_turns(segments, _SPAN)
    assert [(t.speaker_id, t.start_s, t.text) for t in turns] == [
        ("P2-a", 100, "Eins. Zwei drei."),
        ("P2-a", 130, "Vier."),
        ("P2-b", 132, "Fünf. Sechs."),
    ]
