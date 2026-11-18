"""Tests for the word alignment."""

from recording_transcriber.evaluation.alignment import (
    align,
    pick_medoid,
    reference_times,
    word_error_rate,
)
from recording_transcriber.evaluation.words import Words


def _words(text, start=None):
    words = Words(language="de-DE")
    words.add(text, "A", start, None if start is None else start + len(text.split()))
    return words


def test_align():
    reference = ["a", "b", "c", "d", "e"]
    alignment = align(reference, ["a", "x", "c", "e"])
    # b -> x (replace), d deleted
    assert alignment.differs == {1, 3}
    assert alignment.deleted == {3}
    assert alignment.range(1, 2) == (1, 2)
    assert alignment.range(3, 4) == (3, 3)
    assert alignment.range(0, 5) == (0, 4)
    assert alignment.starts == [0, 1, 2, 3, 3, 4]


def test_align_deleted_words_share_one_start():
    alignment = align(["a", "b", "c", "d"], ["a", "d"])
    assert alignment.deleted == {1, 2}
    assert alignment.starts == [0, 1, 1, 1, 2]


def test_align_insert():
    alignment = align(["a", "b", "c"], ["a", "b", "x", "y", "c"])
    # the insertion belongs to the reference word before it
    assert alignment.differs == {1}
    assert alignment.range(1, 2) == (1, 4)
    assert alignment.starts == [0, 1, 4, 5]


def test_align_insert_at_start():
    alignment = align(["a"], ["x", "a"])
    assert alignment.differs == {0}
    assert alignment.range(0, 1) == (1, 2)


def test_word_error_rate():
    assert word_error_rate(["a", "x"], ["a", "b"]) == 0.5
    assert word_error_rate([], []) == 0
    assert word_error_rate(["a"], []) == 1


def test_pick_medoid():
    words = {
        "C0": _words("a b c d"),
        "C1": _words("a b y z", start=0),
        "C2": _words("a b c x", start=0),
    }
    assert pick_medoid(words, ["C1", "C2"]) == "C2"


def test_reference_times():
    reference = _words("a b c", start=0)
    anchor = _words("a b c", start=10)
    assert reference_times(reference, reference, None) == [0, 1, 2]
    alignment = align(reference.tokens, anchor.tokens)
    assert reference_times(reference, anchor, alignment) == [10, 11, 12]
    # a reference word after the anchor's end reads the anchor's last time
    short = _words("a b", start=0)
    alignment = align(reference.tokens, short.tokens)
    assert reference_times(reference, short, alignment) == [0, 1, 1]
    single = _words("a", start=0)
    alignment = align(reference.tokens, single.tokens)
    assert reference_times(reference, single, alignment) == [0, 0, 0]
