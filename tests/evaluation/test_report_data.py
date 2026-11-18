"""Tests for the evaluation data."""

import json
from string import ascii_lowercase

import pytest

from recording_transcriber.evaluation import report_data
from recording_transcriber.evaluation.alignment import align
from recording_transcriber.evaluation.manifest import Manifest
from recording_transcriber.evaluation.report_data import (
    _cost_per_hour,
    _dropped,
    _finely_timed,
    _sample,
    build_data,
)
from recording_transcriber.evaluation.words import Words
from tests.fakes import write_candidate


def _run(total, warnings=()):
    return {
        "started": "2026-11-15T10:00:00+00:00",
        "finished": "2026-11-15T10:02:00+00:00",
        "chunks": [{"end": "00:00:30"}, {"end": "01:00:00"}],
        "cost_usd": {"total": total},
        "warnings": list(warnings),
    }


_TEXT = "wir geben heute Finerenon und dann noch das Medikament am Abend"


@pytest.fixture
def evaluation(tmp_path):
    write_candidate(tmp_path / "C1", [("S1", 0, _TEXT)], _run(0.5), {"S1": "Anna"})
    write_candidate(
        tmp_path / "C2",
        [("S1", 0, _TEXT.replace("Finerenon", "Finerenone"))],
        _run(0.25, ["part 2: REPETITION_LOOP"]),
    )
    write_candidate(tmp_path / "C9", [("P1-X", 0, _TEXT)], _run(0.3))
    write_candidate(tmp_path / "C0", [("Dr. A", None, _TEXT.replace("dann", "denn"))])
    (tmp_path / "C0" / "work" / "chunk_00.json").unlink()
    (tmp_path / "C0" / "out" / "run.json").unlink(missing_ok=True)
    manifest = Manifest.model_validate(
        {
            "candidates": {
                "C1": {"label": "Argon", "transcribe": {"model": "a"}},
                "C2": {"label": "Carbon", "transcribe": {"model": "c"}},
                "C9": {
                    "label": "OpenAI",
                    "transcribe": {"model": "o"},
                    "speakers": False,
                    "signoff": True,
                },
                "C5": {"label": "missing", "transcribe": {"model": "t"}},
            }
        }
    )
    yield manifest, tmp_path


def test_build_data(evaluation, monkeypatch):
    monkeypatch.setattr(report_data, "SAMPLE_S", 3)
    manifest, folder = evaluation
    data = build_data(manifest, folder, ["Finerenon"], "de-DE")
    assert set(data["candidates"]) == {"C0", "C1", "C2", "C9"}
    assert data["reference"] == "C1"
    c1, c2, c9, c0 = (data["candidates"][c] for c in ("C1", "C2", "C9", "C0"))
    assert c1["cost_per_hour"] == 0.5 and c1["wall_s"] == 120
    assert c9["cost_score"] == 100 and c1["cost_score"] == 60
    assert c2["disqualified"] == ["REPETITION_LOOP"]
    assert not c9["speakers"] and c9["signoff"]
    assert not c0["eligible"] and c0["cost_per_hour"] is None and c0["cost_score"] == 0
    assert c0["label"] == "legacy output (gemini-2.5-pro-preview, 2025)"
    assert c1["label"] == "Argon"
    assert c0["speakers"] and c0["coverage"] == 100
    # the sample is the candidate's own text, cut after the aligned words
    assert c1["sample"] == "Anna: wir geben heute Finerenon und dann noch das"
    # C2 is disqualified, so only C0's "denn" opens a span
    assert [s["timestamp"] for s in data["spans"]] == ["00:00:02"]
    assert data["spans"][0]["variants"][0]["key"]
    assert json.dumps(data)


def test_build_data_without_timed_candidates(evaluation):
    manifest, folder = evaluation
    for candidate in ("C1", "C2", "C9"):
        (folder / candidate / "work" / "chunk_00.json").unlink()
    assert build_data(manifest, folder, [], "de-DE") is None


def test_build_data_reference_needs_speakers(evaluation):
    manifest, folder = evaluation
    manifest.candidates["C1"].speakers = False
    manifest.candidates["C2"].eligible = False
    # C9 has no speakers either: no candidate can be the reference
    assert build_data(manifest, folder, [], "de-DE") is None


def test_finely_timed():
    words = Words(language="de-DE")
    words.add(" ".join(["wort"] * 500), "X", 0, 1500)
    assert not _finely_timed(words)
    words.add("noch mehr", "X", 1500, 1510)
    assert not _finely_timed(words)
    timed = Words(language="de-DE")
    for minute in range(3):
        timed.add("ein satz", "A", minute * 60, minute * 60 + 30)
    assert _finely_timed(timed)
    assert not _finely_timed(Words(language="de-DE"))


@pytest.mark.parametrize(
    ("starts", "expected"),
    [([1000, 1075, 1150], True), ([0, 90, 181], False)],
)
def test_finely_timed_turns_per_minute(starts, expected):
    words = Words(language="de-DE")
    for start in starts:
        words.add("wort", "A", start, start + 1)
    assert _finely_timed(words) is expected


def _speech(prefix):
    return [f"{prefix}{letter}" for letter in ascii_lowercase[:20]]


@pytest.fixture
def lineup(tmp_path, monkeypatch):
    """Three candidates over 2.5 minutes: C2 is the reference, C1 has the finest
    turns (the time anchor) and one different word, C3 drops a passage."""
    monkeypatch.setattr(report_data, "SAMPLE_S", 60)
    first, second, third = _speech("u"), _speech("v"), _speech("y")
    changed = [*first[7:9], "ujx", *first[10:]]
    turns = {
        "C1": [
            ("S1", 0, first[:7]),
            ("S1", 35, changed),
            ("S2", 70, second[:10]),
            ("S2", 105, second[10:]),
            ("S1", 140, third),
        ],
        "C2": [("S1", 0, first), ("S2", 70, second), ("S1", 140, third)],
        "C3": [("S1", 0, first[:10]), ("S2", 70, second[6:]), ("S1", 140, third)],
    }
    costs = {"C1": 0.5, "C2": 0.25, "C3": 0.3}
    for identifier, spoken in turns.items():
        write_candidate(
            tmp_path / identifier,
            [(speaker, start, " ".join(words)) for speaker, start, words in spoken],
            _run(costs[identifier]),
            {"S1": "Anna", "S2": "Ben"} if identifier == "C1" else None,
        )
    manifest = Manifest.model_validate(
        {
            "candidates": {
                identifier: {"label": identifier.lower(), "transcribe": {"model": "m"}}
                for identifier in turns
            }
        }
    )
    yield manifest, tmp_path


def test_build_data_details(lineup):
    manifest, folder = lineup
    data = build_data(manifest, folder, [], "de-DE")
    assert (data["language"], data["reference"], data["anchor"]) == (
        "de-DE",
        "C2",
        "C1",
    )
    candidates = data["candidates"]
    assert candidates["C3"]["disqualified"] == ["dropped 44 s at 00:00:43"]
    assert candidates["C3"]["coverage"] == 73.3
    assert candidates["C2"]["coverage"] == 100
    assert candidates["C1"]["sample"] == (
        "Anna: ua ub uc ud ue uf ug\n\nAnna: uh ui ujx uk ul um un uo up uq"
    )
    # a candidate that dropped words gets a sample of as many of its own words
    assert candidates["C3"]["sample"] == "S1: ua ub uc ud ue uf ug uh ui uj"
    # the span's times come from the anchor's turns, through the alignment
    (span,) = data["spans"]
    assert span["timestamp"] == "00:00:40"
    assert [(v["text"], v["candidates"]) for v in span["variants"]] == [
        ("ujx", ["C1"]),
        ("uj", ["C2", "C3"]),
    ]
    assert [spot["labels"] for spot in data["spots"]] == [
        {"C1": "Anna", "C2": "S1", "C3": "S1"},
        {"C1": "Ben", "C2": "S2", "C3": "S2"},
    ]


def test_dropped():
    alignment = align(list("abcdef"), list("af"))
    assert _dropped(alignment, [0, 10, 20, 30, 40, 50]) == [(10, 30)]
    assert _dropped(None, [0]) == []
    assert _dropped(align(list("abc"), list("ac")), [0, 40, 80]) == []
    # a run that reaches the end
    assert _dropped(align(list("abcdef"), list("a")), [0, 10, 20, 30, 40, 50]) == [
        (10, 40)
    ]


def test_cost_per_hour():
    ends = ["00:30:00", "01:00:00", "01:30:00"]
    run = {"cost_usd": {"total": 1}, "chunks": [{"end": end} for end in ends]}
    assert _cost_per_hour(run) == 0.667
    assert _cost_per_hour({"cost_usd": {"total": 1}, "chunks": []}) is None


def test_sample_stops_at_budget(tmp_path):
    write_candidate(tmp_path, [("A", 0, "eins zwei"), ("B", 5, "drei vier")])
    words = Words(language="de-DE")
    words.add("eins zwei drei", "A", 0, 3)
    assert _sample(words, 3, tmp_path) == "A: eins zwei\n\nB: drei"
    assert _sample(words, 2, tmp_path) == "A: eins zwei"


def test_build_data_anchor_without_drops(evaluation, monkeypatch):
    manifest, folder = evaluation
    monkeypatch.setattr(report_data, "MAX_DROPPED_S", 1)
    # C9 gets speakers and finer turns than C1 but drops most words: not the anchor
    manifest.candidates["C9"].speakers = True
    write_candidate(folder / "C9", [("A", 0, "wir"), ("A", 0.4, "geben")], _run(0.3))
    data = build_data(manifest, folder, [], "de-DE")
    assert data["anchor"] != "C9"
