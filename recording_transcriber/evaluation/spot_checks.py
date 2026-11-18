"""Pick the speaker spot-checks the reviewer judges (report part 2)."""

from bisect import bisect_left
from dataclasses import dataclass, field
from itertools import groupby

from ..utils.const import LATE_SPOT_COUNT, SPOT_COUNT, SPOT_S, SPOT_SPACING_S
from .alignment import Alignment
from .words import Words


@dataclass
class SpotCheck:
    """A SPOT_S clip of one voice and the speaker label every candidate gives it."""

    time_s: float
    labels: dict[str, str] = field(default_factory=dict)


def find_spot_checks(
    reference: Words,
    words: dict[str, Words],
    alignments: dict[str, Alignment],
    times: list[float],
    boundary_s: float,
) -> list[SpotCheck]:
    """Return up to SPOT_COUNT spot-checks in recording order.

    Clips come from long stretches of one reference voice. Voices take turns, longest
    stretch first, so every voice appears and some twice (which tests "same person").
    LATE_SPOT_COUNT clips come after the first chunk boundary, where ids may drift.

    reference: the reference words
    words: every candidate's words (including the reference)
    alignments: every other candidate aligned to the reference
    times: the time per reference token
    boundary_s: the end of the reference's first chunk"""
    stretches = _stretches(reference, times)
    late = _round_robin(
        [s for s in stretches if s[1] >= boundary_s],  # pragma: no mutate (threshold)
        [],
        LATE_SPOT_COUNT,
    )
    picked = _round_robin(stretches, late, SPOT_COUNT)
    checks = []
    for start in sorted(start for _, start in picked):
        position = bisect_left(times, start + SPOT_S / 2)
        check = SpotCheck(time_s=start)
        for candidate, value in words.items():
            alignment = alignments.get(candidate)
            index = position if alignment is None else alignment.starts[position]
            check.labels[candidate] = value.speakers[
                min(index, len(value.speakers) - 1)
            ]
        checks.append(check)
    return checks


def _stretches(reference: Words, times: list[float]) -> list[tuple[str, float, float]]:
    """Return (label, clip start, stretch length) for every stretch of one voice that
    holds a SPOT_S clip, longest first.

    reference: the reference words
    times: the time per reference token"""
    stretches = []
    for label, group in groupby(
        range(len(reference.tokens)), lambda position: reference.speakers[position]
    ):
        positions = list(group)
        first = times[positions[0]]
        length = times[positions[-1]] - first
        if length >= SPOT_S:  # pragma: no mutate (threshold)
            stretches.append((label, first + length / 2 - SPOT_S / 2, length))
    return sorted(stretches, key=lambda stretch: -stretch[2])


def _round_robin(
    stretches: list[tuple[str, float, float]],
    chosen: list[tuple[str, float]],
    count: int,
) -> list[tuple[str, float]]:
    """Pick clips voice by voice until count, keeping SPOT_SPACING_S apart.

    stretches: (label, clip start, length), longest first
    chosen: (label, clip start) already picked
    count: the total to reach"""
    chosen = list(chosen)
    remaining = list(stretches)
    while len(chosen) < count and remaining:
        picked_labels = set()
        # one clip per voice and round; a voice already picked waits for the next
        for stretch in list(remaining):
            label, start, _ = stretch
            if len(chosen) >= count:
                break
            if label in picked_labels:
                continue
            remaining.remove(stretch)
            if any(
                abs(start - other) < SPOT_SPACING_S  # pragma: no mutate (threshold)
                for _, other in chosen
            ):
                continue
            picked_labels.add(label)
            chosen.append((label, start))
    return chosen
