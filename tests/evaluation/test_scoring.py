"""Tests for the scoring and decision rules."""

import pytest

from recording_transcriber.evaluation.leaderboard import leaderboard_markdown
from recording_transcriber.evaluation.scoring import (
    decide,
    name_score,
    score,
    speaker_score,
    word_score,
)
from tests.fakes import candidate_stats

_SPANS = [
    {
        "id": "w01",
        "medical": True,
        "variants": [
            {"key": "finerenon", "letter": "A", "candidates": ["C1"]},
            {"key": "fineron", "letter": "B", "candidates": ["C2", "C7"]},
        ],
    },
    {
        "id": "w02",
        "medical": False,
        "variants": [
            {"key": "und dann", "letter": "A", "candidates": ["C1", "C7"]},
            {"key": "und denn", "letter": "B", "candidates": ["C2"]},
        ],
    },
    {
        "id": "w03",
        "medical": False,
        "variants": [{"key": "x", "letter": "A", "candidates": ["C1", "C2", "C7"]}],
    },
]
_SPOTS = [
    {"id": "s01", "labels": {"C1": "Anna", "C2": "Sprecher 1", "C7": "A"}},
    {"id": "s02", "labels": {"C1": "Dr. Huber", "C2": "Sprecher 1", "C7": "A"}},
    {"id": "s03", "labels": {"C1": "Anna", "C2": "Sprecher 2", "C7": "A"}},
]


def test_word_score():
    answers = {
        "w01": {"choice": "A"},
        "w02": {"choice": "none", "text": "Und dann!"},
        "w03": {"choice": "unclear"},
    }
    assert word_score("C1", _SPANS, answers, "de-DE") == 100
    assert word_score("C2", _SPANS, answers, "de-DE") == 0
    assert word_score("C7", _SPANS, answers, "de-DE") == pytest.approx(100 / 3)
    assert word_score("C1", _SPANS, {}, "de-DE") is None
    # "none of these" without text counts as unanswered
    blank = {"w01": {"choice": "none", "text": " "}}
    assert word_score("C1", _SPANS, blank, "de-DE") is None
    assert word_score("C1", _SPANS, {"w01": {"choice": "none"}}, "de-DE") is None
    # an unanswered span does not stop the next ones from counting
    skipped = {"w01": {"choice": "unclear"}, "w02": {"choice": "A"}}
    assert word_score("C1", _SPANS, skipped, "de-DE") == 100


def test_speaker_score():
    answers = {"s01": "Anna Berger", "s02": "Markus Huber", "s03": "Anna Berger"}
    assert speaker_score("C1", candidate_stats(1), _SPOTS, answers, "de-DE") == 100
    assert speaker_score(
        "C2", candidate_stats(1), _SPOTS, answers, "de-DE"
    ) == pytest.approx(100 / 3)
    silent = candidate_stats(1, speakers=False)
    assert speaker_score("C1", silent, _SPOTS, answers, "de-DE") == 0
    assert (
        speaker_score("C1", candidate_stats(1), _SPOTS, {"s01": "x"}, "de-DE") is None
    )


def test_name_score():
    answers = {"s01": "Anna", "s02": "Markus Huber", "s03": "Unknown person 1"}
    assert name_score("C1", _SPOTS, answers, "de-DE") == 100
    assert name_score("C2", _SPOTS, answers, "de-DE") == 0
    assert name_score("C1", _SPOTS, {"s03": "Unknown person 2"}, "de-DE") is None
    # a three-letter name counts
    short = [{"id": "s01", "labels": {"C1": "Eva"}}]
    assert name_score("C1", short, {"s01": "Eva"}, "de-DE") == 100


def test_decide():
    candidates = {
        "C1": candidate_stats(0.6),
        "C4": candidate_stats(0.3),
        "C6": candidate_stats(0.3, eligible=False),
        "C7": candidate_stats(0.6, eligible=False),
        "C8": candidate_stats(0.6, eligible=False),
        "C10": candidate_stats(0.35, signoff=True),
        "C2": candidate_stats(0.2, disqualified=["REPETITION_LOOP"]),
    }

    def row(pre, final=None, word=50):
        return {"pre": pre, "final": final, "word": word}

    rows = {
        "C1": row(50, final=80),
        "C4": row(90),
        "C6": row(10, word=70),
        "C7": row(76),
        "C8": row(52, final=95),
        "C10": row(40, final=78),
        "C2": row(99, final=99),
    }
    decisions = decide(candidates, rows)
    # the final score decides when any candidate has one: C10 is within TIE_POINTS of
    # C1 and cheaper; C2 is disqualified and C8 not eligible
    assert decisions == {
        "winner": "C10",
        "production": "C1",
        "chunk_minutes": 60,
        "denoise": False,
        "test_h1": True,
    }


def test_decide_without_readability_or_rows():
    candidates = {
        "C1": candidate_stats(0.6),
        "C4": candidate_stats(None),
        "C6": candidate_stats(0.3, eligible=False),
    }
    rows = {
        "C1": {"pre": 50, "final": None, "word": None},
        "C4": {"pre": 60, "final": None, "word": None},
        "C6": {"pre": 10, "final": None, "word": None},
    }
    decisions = decide(candidates, rows)
    assert decisions["winner"] == "C4"
    assert decisions["chunk_minutes"] == 25 and not decisions["test_h1"]
    # a candidate without a word score cannot lead
    rows["C6"]["word"] = 50
    assert not decide(candidates, rows)["test_h1"]
    assert decide({}, {})["winner"] is None


def test_score_and_leaderboard():
    data = {
        "language": "de-DE",
        "reference": "C1",
        "anchor": "C1",
        "tokens": {"C1": "t2", "C2": "t1", "C7": "t3"},
        "candidates": {
            "C1": candidate_stats(0.6),
            "C2": candidate_stats(0.3, signoff=True),
            "C7": candidate_stats(
                0.6, eligible=False, disqualified=["dropped 40 s at 00:10:00"]
            ),
        },
        "spans": _SPANS,
        "spots": _SPOTS,
    }
    answers = {
        "words": {"w01": {"choice": "A"}, "w02": {"choice": "A"}},
        "speakers": {"s01": "Anna", "s02": "Markus Huber", "s03": "Anna"},
        "readability": {"t2": 5, "t1": 1, "unknown": 3},
    }
    result = score(data, answers)
    c1 = result["rows"]["C1"]
    assert (c1["word"], c1["speaker"], c1["readability"]) == (100, 100, 100)
    assert c1["pre"] == pytest.approx(0.53 * 100 + 0.35 * 100 + 0.12 * 50, abs=0.1)
    assert (c1["names"], result["rows"]["C2"]["names"]) == (100, 0)
    assert result["rows"]["C2"]["readability"] == 0
    assert result["rows"]["C7"]["final"] is None
    assert result["decisions"]["winner"] == "C1"
    markdown = leaderboard_markdown(data, result)
    assert (
        markdown.index("| C1 |") < markdown.index("| C2 |") < markdown.index("| C7 |")
    )
    assert "needs deployment sign-off" in markdown
    assert "not eligible, disqualified: dropped 40 s" in markdown
    assert "**Winner:** C1" in markdown


def test_decide_denoise_when_it_beats_the_baseline():
    candidates = {
        "C1": candidate_stats(0.6),
        "C8": candidate_stats(0.6, eligible=False),
    }
    rows = {
        "C1": {"pre": 50, "final": None, "word": 50},
        "C8": {"pre": 70, "final": None, "word": 70},
    }
    assert decide(candidates, rows)["denoise"]


def test_score_counts_a_missing_score_as_zero():
    data = {
        "language": "de-DE",
        "tokens": {},
        "candidates": {"C1": {**candidate_stats(0.6), "cost_score": 37.9}},
        "spans": [],
        "spots": [],
    }
    # only the cost is known: 0.12 x 37.9, rounded to one decimal
    assert score(data, {})["rows"]["C1"]["pre"] == 4.5


def test_decide_ignores_disqualified_whole_file():
    candidates = {
        "C1": candidate_stats(0.6),
        "C7": candidate_stats(0.6, eligible=False, disqualified=["dropped 40 s"]),
    }
    rows = {
        "C1": {"pre": 50, "final": None, "word": 50},
        "C7": {"pre": 90, "final": None, "word": 90},
    }
    assert decide(candidates, rows)["chunk_minutes"] == 25


def test_speaker_score_normalizes_typed_names():
    answers = {"s01": "Anna Berger", "s02": "markus  huber", "s03": "anna berger."}
    assert speaker_score("C1", candidate_stats(1), _SPOTS, answers, "de-DE") == 100


def test_name_score_matches_any_name_word():
    answers = {"s02": "Huber (OA)"}
    assert name_score("C1", _SPOTS, answers, "de-DE") == 100


def test_word_score_hidden_variant_never_matches():
    spans = [
        {
            "id": "w01",
            "medical": False,
            "variants": [
                {"key": "a", "letter": "A", "candidates": ["C1"]},
                {"key": "", "letter": "", "candidates": ["C6"]},
            ],
        }
    ]
    assert word_score("C6", spans, {"w01": {"choice": "A"}}, "de-DE") == 0
