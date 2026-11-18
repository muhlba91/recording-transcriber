"""A candidate transcript as normalized, timed words."""

import json
import logging
import re
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from ..utils.const import (
    CHUNK_FILE,
    OUT_DIR,
    PLAN_FILE,
    TRANSCRIPT_JSON,
    WORDS_PER_SECOND,
    WORK_DIR,
)

_LOGGER = logging.getLogger(__name__)

# markers and notes that are not speech: [unklar], [Applaus], "Wort (?)"
_NOT_SPEECH = re.compile(r"\[[^\]]*\]|\(\?\)")
_SEPARATORS = re.compile(r"[-/–]")
_PUNCTUATION = re.compile(r"[^\w.]|(?<!\d)\.|\.(?!\d)")
# a decimal comma and point are the same number ("2,5" and "2.5") and differ from "25"
_DECIMAL_COMMA = re.compile(r"(?<=\d)[.,](?=\d)")
# an English comma between digits groups thousands ("1,000"); only the point is decimal
_THOUSANDS_COMMA = re.compile(r"(?<=\d),(?=\d)")


@dataclass(frozen=True)
class _Table:
    """The language-specific part of the normalization.

    fillers: the filler words to drop
    numbers: the number words (and "%") by their replacement
    decimal: the separator between digits to normalize
    replacement: what the separator becomes ("." or nothing for thousands)"""

    fillers: frozenset[str]
    numbers: dict[str, str]
    decimal: re.Pattern
    replacement: str


_GERMAN = _Table(
    fillers=frozenset({"äh", "ähm", "öh", "öhm", "ehm", "hm", "hmm", "mhm"}),
    numbers={
        "null": "0",
        "eins": "1",
        "zwei": "2",
        "drei": "3",
        "vier": "4",
        "fünf": "5",
        "sechs": "6",
        "sieben": "7",
        "acht": "8",
        "neun": "9",
        "zehn": "10",
        "elf": "11",
        "zwölf": "12",
        "zwanzig": "20",
        "dreißig": "30",
        "hundert": "100",
        "%": "prozent",
    },
    decimal=_DECIMAL_COMMA,
    replacement=".",
)
_ENGLISH = _Table(
    fillers=frozenset({"uh", "um", "er", "erm", "hm", "hmm", "mhm"}),
    numbers={
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
        "thousand": "1000",
        "%": "percent",
    },
    decimal=_THOUSANDS_COMMA,
    replacement="",
)
# other languages: no fillers or number words, both separators are decimal
_GENERIC = _Table(
    fillers=frozenset(), numbers={}, decimal=_DECIMAL_COMMA, replacement="."
)
_TABLES = {"de": _GERMAN, "en": _ENGLISH}


@dataclass
class Words:
    """One candidate's transcript as normalized tokens for alignment.

    Every token keeps the raw word it came from (for display), the speaker label and
    an estimated time (None for untimed transcripts). The language (BCP-47, like
    "de-DE") picks the normalization."""

    language: str
    raw_words: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)
    raw_index: list[int] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    times: list[float | None] = field(default_factory=list)
    # the number of timed stretches (turns); more means finer timestamps
    parts: int = 0

    @property
    def timed(self) -> bool:
        """Return whether the tokens carry times."""
        return bool(self.times) and self.times[0] is not None

    def add(
        self, text: str, speaker: str, start_s: float | None, end_s: float | None
    ) -> None:
        """Append a stretch of speech, spreading its time evenly over its tokens.

        text: the text
        speaker: the speaker label
        start_s: the start time (None if untimed)
        end_s: the end time (None if untimed)"""
        first = len(self.tokens)
        self.parts += start_s is not None
        for raw in _NOT_SPEECH.sub(" ", text).split():
            pieces = normalize(raw, self.language)
            if not pieces:
                continue
            self.raw_words.append(raw)
            for piece in pieces:
                self.tokens.append(piece)
                self.raw_index.append(len(self.raw_words) - 1)
                self.speakers.append(speaker)
        count = len(self.tokens) - first
        for position in range(count):
            if start_s is None or end_s is None:
                self.times.append(None)
            else:
                self.times.append(start_s + (end_s - start_s) * position / count)

    def text(self, start: int, end: int) -> str:
        """Return the raw words behind the tokens [start, end).

        start: the first token
        end: the token after the last"""
        indices = dict.fromkeys(self.raw_index[start:end])
        return " ".join(self.raw_words[index] for index in indices)


def normalize(word: str, language: str) -> list[str]:
    """Return the comparison tokens of a raw word: lower case, no punctuation or
    fillers, digits for number words, a decimal point for decimals, "%" as its own
    word; hyphenated words split.

    Fillers, number words and the decimal notation depend on the primary language
    subtag (de, en); other languages get the generic normalization.

    word: the raw word
    language: the transcript language, BCP-47 like de-DE"""
    table = _table(language.split("-")[0].lower())
    tokens = []
    text = table.decimal.sub(table.replacement, word.lower()).replace("%", " % ")
    for piece in _SEPARATORS.sub(" ", text).split():
        piece = "%" if piece == "%" else _PUNCTUATION.sub("", piece)
        piece = table.numbers.get(piece, piece)
        if piece and piece not in table.fillers:
            tokens.append(piece)
    return tokens


@cache
def _table(primary: str) -> _Table:
    """Return the normalization table of a primary language subtag; warn once if
    there is none.

    primary: the lower-case primary subtag"""
    table = _TABLES.get(primary)
    if table is None:
        _LOGGER.warning(
            "No normalization tables for language %s; using generic ones.", primary
        )
        return _GENERIC
    return table


def load_words(folder: Path, language: str) -> Words | None:
    """Load a candidate's words; None if it has no transcript.

    Turns come from the work/ cache of the current plan (finer timestamps than the
    paragraphs), labels from out/transcript.json. A transcript without a cache (the
    legacy one) is untimed.

    folder: the candidate's job folder
    language: the transcript language"""
    transcript = folder / OUT_DIR / TRANSCRIPT_JSON
    if not transcript.is_file():
        return None
    paragraphs = json.loads(transcript.read_text(encoding="utf-8"))["paragraphs"]
    labels = {
        speaker_id: paragraph["speaker"]
        for paragraph in paragraphs
        for speaker_id in paragraph["speaker_ids"]
    }
    turns = [
        turn
        for path in _chunk_files(folder / WORK_DIR)
        for turn in json.loads(path.read_text(encoding="utf-8"))["turns"]
    ]
    words = Words(language=language)
    if not turns:
        for paragraph in paragraphs:
            words.add(paragraph["text"], paragraph["speaker"], None, None)
        return words
    for position, turn in enumerate(turns):
        start = turn["start_s"]
        # pragma: no mutate start (threshold)
        if position + 1 < len(turns) and turns[position + 1]["start_s"] > start:
            end = turns[position + 1]["start_s"]
        else:
            end = start + len(turn["text"].split()) / WORDS_PER_SECOND
        # pragma: no mutate end
        label = labels.get(turn["speaker_id"], turn["speaker_id"])
        words.add(turn["text"], label, start, end)
    return words


def _chunk_files(work_dir: Path) -> list[Path]:
    """Return the cached chunk files of the current plan, in order.

    work_dir: the candidate's work/ folder"""
    plan = work_dir / PLAN_FILE
    if not plan.is_file():
        return []
    count = len(json.loads(plan.read_text(encoding="utf-8"))["spans"])
    paths = [work_dir / CHUNK_FILE.format(index=index) for index in range(count)]
    return [path for path in paths if path.is_file()]
