"""Silence-aligned chunk planning."""

from ..models.chunk_span import ChunkSpan
from ..utils.const import CHUNK_WINDOW_S


def best_cut(
    silences: list[tuple[float, float]], target_s: float, low_s: float, high_s: float
) -> float:
    """Return the middle of the longest silence in [low_s, high_s], or target_s if none.

    Ties go to the silence closest to the target.

    silences: the silent stretches as (start, end) seconds
    target_s: the preferred cut
    low_s: the earliest allowed cut
    high_s: the latest allowed cut"""
    candidates = [
        (end - start, -abs((start + end) / 2 - target_s), (start + end) / 2)
        for start, end in silences
        # the window limits are tuning
        if low_s <= (start + end) / 2 <= high_s  # pragma: no mutate
    ]
    if not candidates:
        return target_s
    return max(candidates)[2]


def plan_chunks(
    duration_s: float,
    silences: list[tuple[float, float]],
    target_s: float,
    window_s: float = CHUNK_WINDOW_S,
) -> list[ChunkSpan]:
    """Split a recording into chunks of about target_s, cut at silences.

    A remainder up to target_s + window_s stays one chunk (25 min target: recordings up
    to 28 min are not split). A remainder shorter than two chunks is cut near its
    middle, so there is never a tiny last chunk.

    duration_s: the recording length
    silences: the silent stretches as (start, end) seconds
    target_s: the preferred chunk length
    window_s: how far a cut may move from the target (default: CHUNK_WINDOW_S)"""
    spans = []
    start = 0.0
    while duration_s - start > target_s + window_s:
        remaining = duration_s - start
        target = start + min(target_s, remaining / 2)
        cut = best_cut(silences, target, target - window_s, target + window_s)
        spans.append(ChunkSpan(index=len(spans), start_s=start, end_s=cut))
        start = cut
    spans.append(ChunkSpan(index=len(spans), start_s=start, end_s=duration_s))
    return spans
