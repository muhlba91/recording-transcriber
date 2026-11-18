"""Pick the disagreement spans the reviewer judges (report part 1)."""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from ..utils.const import (
    LATE_SPAN_COUNT,
    LATE_SPAN_S,
    MEDICAL_MIN_CHARS,
    MEDICAL_RATIO,
    MEDICAL_SUFFIXES,
    SPAN_CONTEXT_TOKENS,
    SPAN_COUNT,
    SPAN_MAX_TOKENS,
    SPAN_MAX_VARIANTS,
    SPAN_MERGE_GAP,
    SPAN_SPACING_S,
)
from .alignment import Alignment
from .words import Words, normalize

# acronyms and terms with digits: UACR, eGFR, SGLT2
_ACRONYM = re.compile(r"[A-ZÄÖÜ].*[A-ZÄÖÜ]|\d")


@dataclass
class Span:
    """A stretch of the reference where candidates wrote different words.

    variants maps each distinct normalized text to its display text and candidates."""

    start: int
    end: int
    time_s: float
    end_s: float
    before: str
    after: str
    medical: bool = False
    variants: dict[str, dict] = field(default_factory=dict)
    # the number of variants shown to the reviewer
    shown: int = 0


def find_spans(
    reference: Words,
    words: dict[str, Words],
    alignments: dict[str, Alignment],
    times: list[float],
    glossary: list[str],
    ignore: set[str] | None = None,
) -> list[Span]:
    """Return up to SPAN_COUNT spans, most informative first, in recording order.

    Medical spans rank first, then spans with more variants. Spans keep SPAN_SPACING_S
    apart, and LATE_SPAN_COUNT of them start after LATE_SPAN_S.

    reference: the reference words
    words: every candidate's words (including the reference)
    alignments: every other candidate aligned to the reference
    times: the time per reference token
    glossary: the merged glossary
    ignore: candidates whose differences do not open a span; their variants are
        still shown (default: None)"""
    terms = [
        token
        for term in glossary
        for token in normalize(term, reference.language)
        if len(token) > 3  # pragma: no mutate (threshold)
    ]
    spans = []
    ignore = ignore or set()
    deciding = {c: a for c, a in alignments.items() if c not in ignore}
    for start, end in _regions(deciding, len(reference.tokens)):
        span = _span(reference, words, alignments, times, start, end)
        # variants only ignored candidates wrote are hidden on the page
        span.shown = sum(
            bool(set(v["candidates"]) - ignore) for v in span.variants.values()
        )
        if 2 <= span.shown <= SPAN_MAX_VARIANTS:
            span.medical = any(
                _is_medical(variant["text"], terms, reference.language)
                for variant in span.variants.values()
            )
            spans.append(span)
    ranked = sorted(spans, key=lambda span: (not span.medical, -span.shown))
    late = _spread(
        [span for span in ranked if span.time_s >= LATE_SPAN_S],  # pragma: no mutate
        [],
        LATE_SPAN_COUNT,  # pragma: no mutate (tuning)
    )
    chosen = _spread(ranked, late, SPAN_COUNT)
    return sorted(chosen, key=lambda span: span.start)


def _regions(alignments: dict[str, Alignment], length: int) -> list[tuple[int, int]]:
    """Return the reference ranges where any candidate differs, merged across small
    gaps and limited to SPAN_MAX_TOKENS.

    alignments: every other candidate aligned to the reference
    length: the number of reference tokens"""
    differs = sorted(set().union(*(a.differs for a in alignments.values())))
    regions = []
    for position in differs:
        if regions and position - regions[-1][1] <= SPAN_MERGE_GAP:  # pragma: no mutate
            regions[-1][1] = position + 1
        else:
            regions.append([position, position + 1])
    return [
        (start, end)
        for start, end in regions
        if end - start <= SPAN_MAX_TOKENS and end <= length  # pragma: no mutate
    ]


def _span(
    reference: Words,
    words: dict[str, Words],
    alignments: dict[str, Alignment],
    times: list[float],
    start: int,
    end: int,
) -> Span:
    """Return the span for a reference range with every candidate's variant.

    reference: the reference words
    words: every candidate's words
    alignments: every other candidate aligned to the reference
    times: the time per reference token
    start: the first reference token
    end: the reference token after the last"""
    span = Span(
        start=start,
        end=end,
        time_s=times[start],
        end_s=times[end - 1],
        before=reference.text(max(start - SPAN_CONTEXT_TOKENS, 0), start),
        after=reference.text(end, end + SPAN_CONTEXT_TOKENS),
    )
    for candidate, value in words.items():
        alignment = alignments.get(candidate)
        first, last = (start, end) if alignment is None else alignment.range(start, end)
        key = " ".join(value.tokens[first:last])
        variant = span.variants.setdefault(
            key, {"text": value.text(first, last), "candidates": []}
        )
        variant["candidates"].append(candidate)
    return span


def _is_medical(text: str, terms: list[str], language: str) -> bool:
    """Return whether a variant holds a glossary term, an acronym or a medical-looking
    long word.

    text: the variant's raw text
    terms: the normalized glossary tokens
    language: the transcript language"""
    for raw in text.split():
        tokens = normalize(raw, language)
        if _ACRONYM.search(raw) and tokens:
            return True
        for token in tokens:
            if any(
                fuzz.ratio(token, term) >= MEDICAL_RATIO  # pragma: no mutate
                for term in terms
            ):
                return True
            if len(token) >= MEDICAL_MIN_CHARS and token.endswith(MEDICAL_SUFFIXES):
                return True
    return False


def _spread(ranked: list[Span], chosen: list[Span], count: int) -> list[Span]:
    """Add spans in rank order, skipping those too close to a chosen one.

    ranked: the candidates in rank order
    chosen: spans already chosen
    count: the total to reach"""
    chosen = list(chosen)
    for span in ranked:
        if len(chosen) >= count:
            break
        if all(
            abs(span.time_s - other.time_s) >= SPAN_SPACING_S  # pragma: no mutate
            for other in chosen
        ):
            chosen.append(span)
    return chosen
