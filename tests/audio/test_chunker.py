"""Tests for the chunk planning."""

from recording_transcriber.audio.chunker import best_cut, plan_chunks


def test_best_cut_longest_silence():
    silences = [(90, 91), (100, 104), (110, 111)]
    assert best_cut(silences, target_s=100, low_s=80, high_s=120) == 102


def test_best_cut_tie_closest_to_target():
    silences = [(80, 81), (99, 100)]
    assert best_cut(silences, target_s=100, low_s=70, high_s=130) == 99.5


def test_best_cut_tie_closest_below_target():
    # the closest silence is the lower one, so the tie-break is not just "the later"
    silences = [(95, 96), (119, 120)]
    assert best_cut(silences, target_s=100, low_s=70, high_s=130) == 95.5


def test_best_cut_no_silence_in_window():
    assert best_cut([(10, 12)], target_s=100, low_s=80, high_s=120) == 100


def test_plan_chunks_short_recording_not_split():
    spans = plan_chunks(28 * 60, [], target_s=25 * 60)
    assert len(spans) == 1
    assert spans[0].end_s == 28 * 60


def test_plan_chunks_split_at_silence():
    silences = [(1490, 1494), (2990, 2992)]
    spans = plan_chunks(3100, silences, target_s=1500)
    assert [(span.start_s, span.end_s) for span in spans] == [(0, 1492), (1492, 3100)]
    assert [span.index for span in spans] == [0, 1]


def test_plan_chunks_no_silence():
    spans = plan_chunks(4000, [], target_s=1500, window_s=180)
    assert [(span.start_s, span.end_s) for span in spans] == [
        (0, 1500),
        (1500, 2750),
        (2750, 4000),
    ]


def test_plan_chunks_no_tiny_last_chunk():
    # 28.1 min left and the longest pause near 28 min: cut near the middle instead
    silences = [(1679, 1682), (840, 841)]
    spans = plan_chunks(1686, silences, target_s=1500)
    assert [(span.start_s, span.end_s) for span in spans] == [(0, 840.5), (840.5, 1686)]
