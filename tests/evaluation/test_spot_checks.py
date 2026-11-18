"""Tests for the speaker spot-checks."""

from recording_transcriber.evaluation import spot_checks as spot_module
from recording_transcriber.evaluation.alignment import align
from recording_transcriber.evaluation.spot_checks import (
    _round_robin,
    find_spot_checks,
)
from recording_transcriber.evaluation.words import Words


def _words(turns):
    words = Words(language="de-DE")
    for speaker, start, end in turns:
        words.add(" ".join(["wort"] * 20), speaker, start, end)
    return words


def test_find_spot_checks(monkeypatch):
    monkeypatch.setattr(spot_module, "SPOT_SPACING_S", 10)
    turns = [("A", 0, 40), ("B", 40, 60), ("A", 60, 100), ("C", 100, 105)]
    reference = _words(turns)
    other = _words([("X", 0, 40), ("Y", 40, 60), ("Y", 60, 100), ("Y", 100, 105)])
    words = {"C1": reference, "C2": other}
    alignments = {"C2": align(reference.tokens, other.tokens)}
    times = reference.times
    checks = find_spot_checks(reference, words, alignments, times, boundary_s=50)
    # C is too short; A twice and B once, one late clip first
    assert [round(c.time_s) for c in checks] == [12, 42, 72]
    assert [c.labels["C1"] for c in checks] == ["A", "B", "A"]
    assert [c.labels["C2"] for c in checks] == ["X", "Y", "Y"]


def _timed(candidate, turns):
    words = Words(language="de-DE")
    for speaker, tokens, start, end in turns:
        words.add(" ".join(tokens), speaker, start, end)
    return words


def test_find_spot_checks_reads_the_label_at_the_clip_middle():
    tokens = [f"a{n}" for n in range(20)]
    reference = _timed("C1", [("A", tokens, 0, 40)])
    # an insertion shifts C2; the voice changes at the clip's middle token
    shifted = _timed(
        "C2", [("P", ["x", *tokens[:10]], 0, 20), ("Q", tokens[10:], 20, 40)]
    )
    # C3 stops before the clip: it gets its last speaker
    short = _timed("C3", [("R", tokens[:9], 0, 18), ("S", tokens[9:10], 18, 20)])
    words = {"C1": reference, "C2": shifted, "C3": short}
    alignments = {c: align(reference.tokens, words[c].tokens) for c in ("C2", "C3")}
    (check,) = find_spot_checks(reference, words, alignments, reference.times, 0)
    assert (check.time_s, check.labels) == (11.5, {"C1": "A", "C2": "Q", "C3": "S"})


def test_find_spot_checks_prefers_the_longest_stretch(monkeypatch):
    monkeypatch.setattr(spot_module, "SPOT_COUNT", 1)
    reference = _words([("A", 0, 20), ("B", 20, 60)])
    (check,) = find_spot_checks(
        reference, {"C1": reference}, {}, reference.times, boundary_s=1000
    )
    assert check.labels["C1"] == "B"


def test_round_robin_alternates_voices():
    stretches = [("A", 0, 50), ("A", 100, 40), ("B", 200, 30)]
    assert _round_robin(stretches, [], 3) == [("A", 0), ("B", 200), ("A", 100)]


def test_round_robin_spacing():
    assert _round_robin([("A", 0, 50), ("B", 10, 40)], [], 2) == [("A", 0)]


def test_round_robin_stops_at_count():
    assert _round_robin([("A", 0, 50), ("B", 100, 40)], [], 1) == [("A", 0)]


def test_round_robin_skips_a_close_clip_and_goes_on():
    stretches = [("A", 0, 50), ("B", 10, 40), ("A", 100, 30), ("C", 200, 20)]
    assert _round_robin(stretches, [], 3) == [("A", 0), ("C", 200), ("A", 100)]
