"""Tests for the sanity checks."""

from recording_transcriber.models.chunk_result import ChunkResult
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.turn import Turn
from recording_transcriber.utils.checks import (
    chunk_problems,
    has_loop,
    words_per_minute,
    wpm_outliers,
)


def _result(index: int, words: int) -> ChunkResult:
    span = ChunkSpan(index=index, start_s=index * 60, end_s=index * 60 + 60)
    turn = Turn(speaker_id="S1", start_s=span.start_s, text="wort " * words)
    return ChunkResult(span=span, model="m", turns=[turn], roster=[])


def test_chunk_problems_ok():
    turns = [
        Turn(speaker_id="S1", start_s=100, text="Guten Morgen."),
        Turn(speaker_id="S2", start_s=250, text="Danke."),
        Turn(speaker_id="S1", start_s=400, text="Gern."),
        Turn(speaker_id="S2", start_s=450, text="Weiter."),
    ]
    assert chunk_problems(turns, start_s=100, duration_s=400) == []


def test_chunk_problems_no_turns():
    assert chunk_problems([], start_s=0, duration_s=60) == ["NO_TURNS"]


def test_chunk_problems_tail_gap():
    turns = [Turn(speaker_id="S1", start_s=0, text="Hallo.")]
    assert chunk_problems(turns, start_s=0, duration_s=600) == ["TAIL_GAP"]


def test_chunk_problems_loop():
    turns = [
        Turn(
            speaker_id="S1",
            start_s=0,
            text="eins zwei drei vier fünf sechs sieben acht " * 4,
        )
    ]
    assert chunk_problems(turns, start_s=0, duration_s=30) == ["REPETITION_LOOP"]


def test_has_loop_short_text():
    assert not has_loop("nur wenige Wörter")


def test_has_loop_varied_text():
    assert not has_loop(" ".join(f"wort{index}" for index in range(40)))


def test_has_loop_across_turns():
    # no turn repeats itself, the turns together do
    eight = "eins zwei drei vier fünf sechs sieben acht"
    turns = [Turn(speaker_id="S1", start_s=index, text=eight) for index in range(4)]
    assert chunk_problems(turns, start_s=0, duration_s=10) == ["REPETITION_LOOP"]
    assert chunk_problems(turns, 0, 600, timed=False) == ["REPETITION_LOOP"]


def test_words_per_minute():
    assert words_per_minute(_result(0, 120)) == 120
    result = _result(0, 120)
    result.span = ChunkSpan(index=0, start_s=0, end_s=120)
    assert words_per_minute(result) == 60
    # a chunk shorter than a second counts as one second
    result.span = ChunkSpan(index=0, start_s=0, end_s=0)
    assert words_per_minute(result) == 120 * 60


def test_wpm_outliers():
    results = [_result(0, 100), _result(1, 110), _result(2, 10)]
    assert wpm_outliers(results) == [2]


def test_wpm_outliers_too_few_chunks():
    assert wpm_outliers([_result(0, 100), _result(1, 1)]) == []


def test_chunk_problems_timestamps_backwards():
    turns = [
        Turn(speaker_id="S1", start_s=50, text="a"),
        Turn(speaker_id="S2", start_s=10, text="b"),
        Turn(speaker_id="S1", start_s=55, text="c"),
    ]
    assert chunk_problems(turns, start_s=0, duration_s=60) == ["TIMESTAMPS"]


def test_chunk_problems_timestamps_past_end():
    turns = [
        Turn(speaker_id="S1", start_s=0, text="a"),
        Turn(speaker_id="S1", start_s=90, text="b"),
    ]
    assert chunk_problems(turns, start_s=0, duration_s=60) == ["TIMESTAMPS"]


def test_chunk_problems_turn_gap():
    turns = [
        Turn(speaker_id="S1", start_s=0, text="a"),
        Turn(speaker_id="S1", start_s=400, text="b"),
    ]
    assert chunk_problems(turns, start_s=0, duration_s=420) == ["TURN_GAP"]


def test_chunk_problems_late_first_turn():
    turns = [Turn(speaker_id="S1", start_s=400, text="a")]
    assert chunk_problems(turns, start_s=0, duration_s=420) == ["TURN_GAP"]


def test_chunk_problems_untimed():
    turns = [Turn(speaker_id="P1-X", start_s=0, text="Ein langer Text.")]
    assert chunk_problems(turns, 0, 600, timed=False) == []


def test_chunk_problems_untimed_loop():
    text = " ".join(["eins zwei drei vier fünf sechs sieben acht"] * 5)
    turns = [Turn(speaker_id="P1-X", start_s=0, text=text)]
    assert chunk_problems(turns, 0, 600, timed=False) == ["REPETITION_LOOP"]
