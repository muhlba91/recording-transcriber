"""Transcriber sanity checks for model answers."""

from collections import Counter
from itertools import pairwise
from statistics import median

from ..models.chunk_result import ChunkResult
from ..models.turn import Turn
from .const import (
    LOOP_NGRAM,
    LOOP_REPEATS,
    MAX_TAIL_GAP_S,
    MAX_TURN_GAP_S,
    TIMESTAMP_SLACK_S,
    WARNING_LOOP,
    WARNING_NO_TURNS,
    WARNING_TAIL_GAP,
    WARNING_TIMESTAMPS,
    WARNING_TURN_GAP,
    WPM_MAX_RATIO,
    WPM_MIN_CHUNKS,
    WPM_MIN_RATIO,
)


def chunk_problems(
    turns: list[Turn], start_s: float, duration_s: float, timed: bool = True
) -> list[str]:
    """Return the problem codes for one transcribed chunk; empty if it looks sane.

    turns: the chunk's turns, timed from the start of the recording
    start_s: the chunk start in the recording
    duration_s: the chunk length
    timed: whether the turns carry real timestamps (default: True)"""
    if not turns:
        return [WARNING_NO_TURNS]
    if not timed:
        return [WARNING_LOOP] if has_loop(" ".join(t.text for t in turns)) else []
    problems = []
    offsets = [turn.start_s - start_s for turn in turns]
    backwards = any(
        later < earlier - TIMESTAMP_SLACK_S  # pragma: no mutate (threshold)
        for earlier, later in pairwise(offsets)
    )
    if backwards or max(offsets) > duration_s + TIMESTAMP_SLACK_S:  # pragma: no mutate
        problems.append(WARNING_TIMESTAMPS)
    if duration_s - max(offsets) > MAX_TAIL_GAP_S:  # pragma: no mutate
        problems.append(WARNING_TAIL_GAP)
    gaps = [later - earlier for earlier, later in pairwise(offsets)]
    if max([offsets[0], *gaps]) > MAX_TURN_GAP_S:  # pragma: no mutate (threshold)
        problems.append(WARNING_TURN_GAP)
    if has_loop(" ".join(turn.text for turn in turns)):
        problems.append(WARNING_LOOP)
    return problems


def has_loop(text: str) -> bool:
    """Return whether any word n-gram repeats suspiciously often (a model loop).

    text: the text to check"""
    words = text.casefold().split()
    ngrams = Counter(
        tuple(words[index : index + LOOP_NGRAM])
        # an extra, shorter n-gram at the end never repeats
        for index in range(len(words) - LOOP_NGRAM + 1)  # pragma: no mutate
    )
    return any(count >= LOOP_REPEATS for count in ngrams.values())


def words_per_minute(result: ChunkResult) -> float:
    """Return the spoken words per minute of a chunk.

    result: the transcribed chunk"""
    words = sum(len(turn.text.split()) for turn in result.turns)
    return words / max(result.span.duration_s / 60, 1 / 60)


def wpm_outliers(results: list[ChunkResult]) -> list[int]:
    """Return the positions of chunks far off the recording's median words per minute.

    Only meaningful with a few chunks, so fewer than WPM_MIN_CHUNKS return nothing.

    results: all transcribed chunks"""
    if len(results) < WPM_MIN_CHUNKS:
        return []
    rates = [words_per_minute(result) for result in results]
    middle = median(rates)
    # pragma: no mutate start (the ratios are thresholds)
    return [
        position
        for position, rate in enumerate(rates)
        if not WPM_MIN_RATIO * middle <= rate <= WPM_MAX_RATIO * middle
    ]
    # pragma: no mutate end
