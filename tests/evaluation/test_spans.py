"""Tests for the disagreement spans."""

import pytest

from recording_transcriber.evaluation import spans as spans_module
from recording_transcriber.evaluation.alignment import align
from recording_transcriber.evaluation.spans import _is_medical, find_spans
from recording_transcriber.evaluation.words import Words


def _words(text):
    words = Words(language="de-DE")
    for position, word in enumerate(text.split()):
        words.add(word, "A", position * 10.0, position * 10.0 + 10)
    return words


def test_find_spans():
    reference = _words("wir geben Finerenon und dann noch das Medikament heute")
    words = {
        "C1": reference,
        "C2": _words("wir geben Finerenone und dann noch das Medikament heute"),
        "C3": _words("wir geben Fineron und dann noch des Medikament heute"),
    }
    alignments = {c: align(reference.tokens, words[c].tokens) for c in ("C2", "C3")}
    times = [time for time in reference.times]
    found = find_spans(reference, words, alignments, times, ["Finerenon"])
    assert [(s.start, s.end, s.medical) for s in found] == [(2, 3, True), (6, 7, False)]
    first = found[0]
    assert {v["text"]: v["candidates"] for v in first.variants.values()} == {
        "Finerenon": ["C1"],
        "Finerenone": ["C2"],
        "Fineron": ["C3"],
    }
    assert first.before == "wir geben" and first.after.startswith("und dann")
    assert first.time_s == 20


def test_find_spans_skips_long_regions(monkeypatch):
    monkeypatch.setattr(spans_module, "SPAN_MAX_TOKENS", 1)
    reference = _words("a b c d")
    words = {"C1": reference, "C2": _words("a x y d")}
    alignments = {"C2": align(reference.tokens, words["C2"].tokens)}
    assert find_spans(reference, words, alignments, reference.times, []) == []


def test_is_medical():
    assert _is_medical("UACR", [], "de-DE")
    assert _is_medical("30 mg", [], "de-DE")
    assert _is_medical("Hyperkaliämie", [], "de-DE")
    assert _is_medical("Finerenone", ["finerenon"], "de-DE")
    assert not _is_medical("und dann", ["finerenon"], "de-DE")
    # the suffix needs the minimum length: nine letters count, four do not
    assert _is_medical("Gastritis", [], "de-DE")
    assert not _is_medical("Rose", [], "de-DE")


@pytest.mark.parametrize(
    ("spelling", "start"),
    [
        # medical spans rank before spans with more variants
        (("UACR", "uakr"), 8),
        # otherwise more variants rank first
        (("uacr", "uakr"), 0),
    ],
)
def test_find_spans_ranking(monkeypatch, spelling, start):
    monkeypatch.setattr(spans_module, "SPAN_COUNT", 1)
    monkeypatch.setattr(spans_module, "LATE_SPAN_COUNT", 0)
    reference = _words(f"alpha x x x bravo x x x {spelling[0]} x")
    words = {
        "C1": reference,
        "C2": _words(f"beta x x x brave x x x {spelling[1]} x"),
        "C3": _words(f"gamma x x x bravo x x x {spelling[0]} x"),
    }
    alignments = {c: align(reference.tokens, words[c].tokens) for c in ("C2", "C3")}
    found = find_spans(reference, words, alignments, reference.times, [])
    assert [s.start for s in found] == [start]


def test_find_spans_context_and_variants():
    tokens = [f"w{n}" for n in range(40)]
    reference = _words(" ".join(tokens))
    other = tokens[:20] + ["neu", "wort"] + tokens[22:]
    # an extra word early on shifts everything: the span must follow the alignment
    words = {"C1": reference, "C2": _words(" ".join(["extra", *other]))}
    alignments = {"C2": align(reference.tokens, words["C2"].tokens)}
    (span,) = find_spans(reference, words, alignments, reference.times, [])
    assert (span.start, span.end, span.time_s, span.end_s) == (20, 22, 200, 210)
    assert span.before == " ".join(tokens[8:20])
    assert span.after == " ".join(tokens[22:34])
    assert {key: v["text"] for key, v in span.variants.items()} == {
        "w20 w21": "w20 w21",
        "neu wort": "neu wort",
    }


def test_find_spans_ignores_candidates():
    reference = _words("wir geben Finerenon heute")
    words = {"C1": reference, "C6": _words("wir geben Fineron heute")}
    alignments = {"C6": align(reference.tokens, words["C6"].tokens)}
    assert find_spans(reference, words, alignments, reference.times, [], {"C6"}) == []


def test_find_spans_counts_only_shown_variants(monkeypatch):
    monkeypatch.setattr(spans_module, "SPAN_MAX_VARIANTS", 2)
    reference = _words("wir geben Finerenon heute")
    words = {
        "C1": reference,
        "C2": _words("wir geben Fineron heute"),
        "C6": _words("wir geben Finerona heute"),
    }
    alignments = {c: align(reference.tokens, words[c].tokens) for c in ("C2", "C6")}
    found = find_spans(reference, words, alignments, reference.times, [], {"C6"})
    assert [(span.shown, len(span.variants)) for span in found] == [(2, 3)]
