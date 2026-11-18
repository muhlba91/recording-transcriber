"""Word alignment of candidates against a reference candidate."""

from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from .words import Words


@dataclass
class Alignment:
    """A candidate aligned to the reference tokens.

    starts[i] is the candidate token that lines up with reference token i, and
    starts[len(reference)] is the candidate's length, so reference tokens [i, j) map
    to candidate tokens [starts[i], starts[j]). An insertion belongs to the reference
    token before it."""

    starts: list[int]
    differs: set[int]
    deleted: set[int]

    def range(self, start: int, end: int) -> tuple[int, int]:
        """Return the candidate tokens aligned to reference tokens [start, end).

        start: the first reference token
        end: the reference token after the last"""
        return self.starts[start], self.starts[end]


def word_error_rate(hypothesis: list[str], reference: list[str]) -> float:
    """Return the word error rate of a token list against a reference.

    hypothesis: the tokens to rate
    reference: the reference tokens"""
    return Levenshtein.distance(hypothesis, reference) / max(len(reference), 1)


def pick_medoid(words: dict[str, Words], pool: list[str]) -> str:
    """Return the candidate from the pool with the lowest total word error rate against
    all others: the most typical transcript, used as the reference.

    words: the candidates' words
    pool: the candidates that may be the reference"""

    def total_error(candidate: str) -> float:
        """Return the candidate's total word error rate against the others.

        candidate: the candidate id"""
        return sum(
            word_error_rate(words[other].tokens, words[candidate].tokens)
            for other in words
            if other != candidate
        )

    return min(pool, key=total_error)


def align(reference: list[str], candidate: list[str]) -> Alignment:
    """Align a candidate's tokens to the reference tokens.

    reference: the reference tokens
    candidate: the candidate tokens"""
    starts = [0] * (len(reference) + 1)  # pragma: no mutate (all overwritten below)
    differs = set()
    deleted = set()
    for operation in Levenshtein.opcodes(reference, candidate):
        tag = operation.tag
        source = range(operation.src_start, operation.src_end)
        # equal and replace blocks are as long as their target; a deleted word has none
        for position in source:
            offset = 0 if tag == "delete" else position - operation.src_start
            starts[position] = operation.dest_start + offset
        if tag == "insert":
            differs.add(max(operation.src_start - 1, 0))
        elif tag != "equal":
            differs.update(source)
        if tag == "delete":
            deleted.update(source)
    starts[len(reference)] = len(candidate)
    return Alignment(starts=starts, differs=differs, deleted=deleted)


def reference_times(
    reference: Words, anchor: Words, alignment: Alignment | None
) -> list[float]:
    """Return a time per reference token, read from the anchor candidate.

    The anchor is the timed candidate with the finest turns; the reference itself
    when alignment is None.

    reference: the reference words
    anchor: the anchor words
    alignment: the anchor aligned to the reference (None if anchor is the reference)"""
    if alignment is None:
        return [time or 0.0 for time in reference.times]
    last = len(anchor.times) - 1
    return [
        anchor.times[min(alignment.starts[position], last)] or 0.0
        for position in range(len(reference.tokens))
    ]
