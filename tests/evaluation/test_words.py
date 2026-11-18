"""Tests for the candidate words."""

import json

import pytest

from recording_transcriber.evaluation.words import Words, _table, load_words, normalize
from tests.fakes import write_candidate


@pytest.fixture(autouse=True)
def fresh_tables():
    _table.cache_clear()


def test_normalize_german():
    assert normalize("SGLT2-Hemmer,", "de-DE") == ["sglt2", "hemmer"]
    assert normalize("Drei", "de-DE") == ["3"]
    assert normalize("äh,", "de-DE") == []
    assert normalize("30%", "de-DE") == normalize("30 %", "de-DE") == ["30", "prozent"]
    assert normalize("2,5", "de-DE") == normalize("2.5", "de-DE") == ["2.5"]
    assert normalize("25", "de-DE") == ["25"]
    assert normalize("Ende.", "de-DE") == ["ende"]
    # "eh" is a word in Austrian German ("eh klar"), not a filler
    assert normalize("eh", "de-DE") == ["eh"]


def test_normalize_german_bare_subtag():
    assert normalize("Drei", "de") == ["3"]


def test_normalize_english():
    assert normalize("Seven,", "en-US") == ["7"]
    assert normalize("Twenty-five", "en-US") == ["20", "5"]
    assert normalize("Um,", "en-GB") == []
    assert normalize("erm", "en") == []
    assert normalize("30%", "en-US") == normalize("30 %", "en-US") == ["30", "percent"]
    assert normalize("2.5", "en-US") == ["2.5"]
    assert normalize("1,000", "en-US") == normalize("1000", "en-US") == ["1000"]
    assert normalize("thousand", "en-US") == ["1000"]
    # German words stay words in English transcripts
    assert normalize("drei", "en-US") == ["drei"]
    assert normalize("äh", "en-US") == ["äh"]


def test_normalize_other_language():
    assert normalize("Trois,", "fr-FR") == ["trois"]
    assert normalize("uh", "fr-FR") == ["uh"]
    assert normalize("2,5", "fr-FR") == ["2.5"]
    assert normalize("30%", "fr-FR") == ["30", "%"]
    assert normalize("peut-être", "fr-CA") == ["peut", "être"]


def test_words_add_english():
    words = Words(language="en-US")
    words.add("Um, three SGLT2-inhibitors.", "A", None, None)
    assert words.tokens == ["3", "sglt2", "inhibitors"]


def test_words_add():
    words = Words(language="de-DE")
    words.add("Äh, Guten [unklar] Morgen (?) SGLT2-Hemmer.", "A", 10, 18)
    assert words.tokens == ["guten", "morgen", "sglt2", "hemmer"]
    assert words.raw_words == ["Guten", "Morgen", "SGLT2-Hemmer."]
    assert words.times == [10, 12, 14, 16]
    assert words.text(1, 4) == "Morgen SGLT2-Hemmer."
    assert words.timed and words.parts == 1


def test_words_add_untimed():
    words = Words(language="de-DE")
    words.add("Hallo Welt", "A", None, None)
    # a half-known stretch is untimed too
    words.add("eins", "A", None, 5)
    words.add("zwei", "A", 5, None)
    assert words.times == [None] * 4
    assert not words.timed
    assert not Words(language="de-DE").timed


def test_load_words_timed(tmp_path):
    write_candidate(
        tmp_path,
        [("S1", 0, "Eins zwei."), ("S2", 4, "Drei vier fünf.")],
        labels={"S1": "Anna"},
    )
    words = load_words(tmp_path, "de-DE")
    assert words.speakers == ["Anna", "Anna", "S2", "S2", "S2"]
    assert words.times[:2] == [0, 2]
    # the last turn is timed at WORDS_PER_SECOND
    assert words.times[2:] == [4.0, 4.4, 4.8]


def test_load_words_turn_without_a_paragraph_keeps_its_speaker_id(tmp_path):
    write_candidate(tmp_path, [("S1", 0, "Eins.")])
    chunk = tmp_path / "work" / "chunk_00.json"
    record = json.loads(chunk.read_text())
    record["turns"].append({"speaker_id": "Z", "start_s": 5, "text": "Zwei."})
    chunk.write_text(json.dumps(record))
    assert load_words(tmp_path, "de-DE").speakers == ["S1", "Z"]


def test_load_words_turn_times(tmp_path):
    write_candidate(tmp_path, [("S1", 0, "a b"), ("S2", 10, "c")])
    # a turn lasts until the next one starts; the words spread over it
    assert load_words(tmp_path, "de-DE").times == [0, 5, 10]


def test_load_words_untimed(tmp_path):
    write_candidate(tmp_path, [("Dr. A", None, "Hallo.")])
    (tmp_path / "work" / "chunk_00.json").unlink()
    words = load_words(tmp_path, "de-DE")
    assert words.tokens == ["hallo"] and not words.timed
    assert words.speakers == ["Dr. A"]


def test_load_words_missing(tmp_path):
    assert load_words(tmp_path, "de-DE") is None


def test_load_words_without_plan_is_untimed(tmp_path):
    write_candidate(tmp_path, [("S1", 0, "Hallo.")])
    (tmp_path / "work" / "chunks.json").unlink()
    assert not load_words(tmp_path, "de-DE").timed


def test_load_words_ignores_chunks_outside_plan(tmp_path):
    write_candidate(tmp_path, [("S1", 0, "Hallo.")])
    stale = tmp_path / "work" / "chunk_01.json"
    stale.write_text(
        json.dumps({"turns": [{"speaker_id": "S1", "start_s": 9, "text": "alt"}]})
    )
    assert load_words(tmp_path, "de-DE").tokens == ["hallo"]
