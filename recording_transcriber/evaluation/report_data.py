"""Collect everything the report and the scoring need from the candidate runs."""

import json
import logging
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

from ..utils.const import (
    DISQUALIFYING_WARNINGS,
    LEGACY_ID,
    MAX_DROPPED_S,
    MIN_TURNS_PER_MINUTE,
    OUT_DIR,
    RUN_JSON,
    SAMPLE_S,
    SPOT_COUNT,
    TRANSCRIPT_JSON,
)
from ..utils.timestamps import format_timestamp, parse_timestamp
from .alignment import Alignment, align, pick_medoid, reference_times
from .manifest import Manifest
from .spans import find_spans
from .spot_checks import find_spot_checks
from .words import Words, load_words

_LOGGER = logging.getLogger(__name__)


def build_data(
    manifest: Manifest, out_dir: Path, glossary: list[str], language: str
) -> dict | None:
    """Return the evaluation data: candidates, disagreement spans and spot-checks.

    The data names the candidates; the report page only gets anonymous tokens.

    manifest: the candidates
    out_dir: the evaluation folder with one folder per candidate
    glossary: the merged glossary
    language: the transcript language"""
    words = {}
    for identifier in [LEGACY_ID, *manifest.candidates]:
        value = load_words(out_dir / identifier, language)
        if value is None:
            _LOGGER.warning("Candidate %s has no transcript; skipping it.", identifier)
        elif value.tokens:
            words[identifier] = value
    # the reference and time anchor need speakers, real timestamps and the default
    # chunking (C7's single chunk would leave no chunk boundary for the spot-checks)
    pool = [
        identifier
        for identifier, value in words.items()
        if _finely_timed(value)
        and (spec := manifest.candidates.get(identifier)) is not None
        and spec.speakers
        and spec.eligible
    ]
    if not pool:
        _LOGGER.error("No timed candidate with speakers in %s.", out_dir)
        return None
    reference_id = pick_medoid(words, pool)
    reference = words[reference_id]
    alignments = {
        identifier: align(reference.tokens, value.tokens)
        for identifier, value in words.items()
        if identifier != reference_id
    }
    # an anchor that dropped passages would give all words there one time
    own_times = reference_times(reference, reference, None)  # pragma: no mutate
    complete = [
        identifier
        for identifier in pool
        if not _dropped(alignments.get(identifier), own_times)
    ]
    anchor_id = max(complete or pool, key=lambda identifier: words[identifier].parts)
    times = reference_times(reference, words[anchor_id], alignments.get(anchor_id))
    runs = {identifier: _run(out_dir / identifier) for identifier in words}
    boundary = _first_boundary(runs[reference_id])
    sample_end = bisect_left(times, SAMPLE_S)
    candidates = {
        identifier: _candidate(
            identifier,
            manifest,
            words[identifier],
            alignments.get(identifier),
            times,
            runs[identifier],
            sample_end,
            out_dir / identifier,
        )
        for identifier in words
    }
    _add_cost_scores(candidates)
    # a candidate that dropped minutes differs everywhere there; it would merge
    # whole passages into regions too long to judge
    disqualified = {
        i for i, candidate in candidates.items() if candidate["disqualified"]
    }
    spans = find_spans(reference, words, alignments, times, glossary, disqualified)
    spots = find_spot_checks(reference, words, alignments, times, boundary)
    if len(spots) < SPOT_COUNT:  # pragma: no mutate (only the warning)
        _LOGGER.warning("Only %s speaker spot-checks found.", len(spots))
    _LOGGER.info(
        "Reference %s, time anchor %s: %s spans, %s spot-checks.",
        reference_id,
        anchor_id,
        len(spans),
        len(spots),
    )
    # the keys are read by the report writer, the values are display rounding
    # pragma: no mutate block
    return {
        "language": language,
        "reference": reference_id,
        "anchor": anchor_id,
        "candidates": candidates,
        "spans": [
            {
                "id": f"w{number:02d}",
                "start_s": round(span.time_s, 1),
                "end_s": round(span.end_s, 1),
                "timestamp": format_timestamp(span.time_s),
                "before": span.before,
                "after": span.after,
                "medical": span.medical,
                "variants": [
                    {"key": key, **variant} for key, variant in span.variants.items()
                ],
            }
            for number, span in enumerate(spans, start=1)
        ],
        "spots": [
            {
                "id": f"s{number:02d}",
                "start_s": round(spot.time_s, 1),
                "timestamp": format_timestamp(spot.time_s),
                "labels": spot.labels,
            }
            for number, spot in enumerate(spots, start=1)
        ],
    }


def _candidate(
    identifier: str,
    manifest: Manifest,
    words: Words,
    alignment: Alignment | None,
    times: list[float],
    run: dict,
    sample_end: int,
    folder: Path,
) -> dict:
    """Return one candidate's facts: settings, cost, checks and readability sample.

    identifier: the candidate id
    manifest: the candidates
    words: the candidate's words
    alignment: the candidate aligned to the reference (None for the reference)
    times: the time per reference token
    run: the candidate's run.json (empty for the legacy output)
    sample_end: the first reference token after the readability sample
    folder: the candidate folder"""
    spec = manifest.candidates.get(identifier)
    dropped = _dropped(alignment, times)
    warnings = sorted(
        {
            code
            for code in DISQUALIFYING_WARNINGS
            for w in run.get("warnings", [])
            if w.endswith(code)
        }
    )
    reasons = warnings + [
        f"dropped {length:.0f} s at {format_timestamp(start)}"
        for start, length in dropped
    ]
    sample_stop = sample_end if alignment is None else alignment.starts[sample_end]
    return {
        "label": spec.label if spec else "legacy output (gemini-2.5-pro-preview, 2025)",
        "eligible": bool(spec and spec.eligible),
        "speakers": bool(spec is None or spec.speakers),
        "signoff": bool(spec and spec.signoff),
        "disqualified": reasons,
        "coverage": round(100 * (1 - len(alignment.deleted) / len(times)), 1)
        if alignment
        else 100.0,
        "cost_per_hour": _cost_per_hour(run),
        "wall_s": _wall_seconds(run),
        "sample": _sample(words, sample_stop, folder),
    }


def _dropped(
    alignment: Alignment | None, times: list[float]
) -> list[tuple[float, float]]:
    """Return (start, length) of every run of deleted reference words of at least
    MAX_DROPPED_S.

    alignment: the candidate aligned to the reference (None for the reference)
    times: the time per reference token"""
    if alignment is None:
        return []
    dropped = []
    run_start = None
    # one step past the end closes a run that reaches it
    for position in range(len(times) + 1):  # pragma: no mutate
        if position in alignment.deleted:
            run_start = position if run_start is None else run_start
            continue
        if run_start is not None:
            length = times[position - 1] - times[run_start]
            if length >= MAX_DROPPED_S:
                dropped.append((times[run_start], length))
            run_start = None
    return dropped


def _sample(words: Words, stop: int, folder: Path) -> str:
    """Return the readability sample: the candidate's own paragraphs, as written
    (markers, fillers), cut after as many words as tokens [0, stop) hold.

    Cutting by aligned words keeps samples of timed and untimed candidates equal.

    words: the candidate's words
    stop: the token after the sample
    folder: the candidate folder"""
    budget = len(set(words.raw_index[:stop]))
    record = json.loads(
        (folder / OUT_DIR / TRANSCRIPT_JSON).read_text(encoding="utf-8")
    )
    lines = []
    for paragraph in record["paragraphs"]:
        if budget <= 0:
            break
        text = paragraph["text"].split()[:budget]
        budget -= len(text)
        lines.append(f"{paragraph['speaker']}: {' '.join(text)}")
    return "\n\n".join(lines)


def _add_cost_scores(candidates: dict) -> None:
    """Add the cost score: 100 x the cheapest $/h among the candidates that can win,
    divided by the candidate's.

    candidates: the candidate facts"""
    costs = [
        value["cost_per_hour"]
        for value in candidates.values()
        if value["eligible"] and not value["disqualified"] and value["cost_per_hour"]
    ]
    cheapest = min(costs, default=None)
    for value in candidates.values():
        cost = value["cost_per_hour"]
        # pragma: no mutate start (display rounding)
        value["cost_score"] = (
            round(100 * cheapest / cost, 1) if cheapest and cost else 0.0
        )
        # pragma: no mutate end


def _run(folder: Path) -> dict:
    """Return a candidate's run.json, or an empty dict.

    folder: the candidate folder"""
    path = folder / OUT_DIR / RUN_JSON
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# it only places the late spot-checks
def _first_boundary(run: dict) -> float:  # pragma: no mutate block
    """Return the end of the first chunk in seconds (0 without chunks).

    run: the run.json"""
    chunks = run.get("chunks")
    if not chunks:
        return 0.0
    return parse_timestamp(chunks[0]["end"]) or 0.0


def _cost_per_hour(run: dict) -> float | None:
    """Return the estimated cost per audio hour, or None without a price.

    run: the run.json"""
    total = (run.get("cost_usd") or {}).get("total")
    chunks = run.get("chunks") or []
    duration = parse_timestamp(chunks[-1]["end"]) if chunks else None
    if total is None or not duration:
        return None
    return round(total / (duration / 3600), 3)  # pragma: no mutate (rounding)


def _wall_seconds(run: dict) -> float | None:
    """Return the run's wall time in seconds (a resumed run only counts the rest).

    run: the run.json"""
    if "started" not in run:
        return None
    started = datetime.fromisoformat(run["started"])
    finished = datetime.fromisoformat(run["finished"])
    return (finished - started).total_seconds()


def _finely_timed(words: Words) -> bool:
    """Return whether a transcript has real timestamps: at least MIN_TURNS_PER_MINUTE
    timed turns per minute (an untimed answer has one turn per chunk).

    words: the candidate's words"""
    if not words.timed:
        return False
    minutes = max((words.times[-1] - words.times[0]) / 60, 1)
    return words.parts / minutes >= MIN_TURNS_PER_MINUTE
