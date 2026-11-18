"""Score the candidates from the reviewer's answers (design §5)."""

import math
from itertools import combinations

from ..utils.const import (
    ANSWER_NONE,
    ANSWER_UNCLEAR,
    FINAL_WEIGHTS,
    H1_WORD_POINTS,
    MEDICAL_WEIGHT,
    PRE_WEIGHTS,
    ROLE_ARGON,
    ROLE_DENOISED,
    ROLE_WHOLE_FILE,
    ROLE_WORDS_ONLY_PROBE,
    TIE_POINTS,
    UNKNOWN_PERSON,
    WHOLE_FILE_POINTS,
)
from .words import normalize


def score(data: dict, answers: dict) -> dict:
    """Return the scores per candidate and the decisions.

    data: the evaluation data (report_data.json)
    answers: the reviewer's answers (answers.json)"""
    tokens = {value: key for key, value in data["tokens"].items()}
    readability = {
        tokens[token]: rating
        for token, rating in (answers.get("readability") or {}).items()
        if token in tokens
    }
    language = data["language"]
    rows = {}
    for identifier, candidate in data["candidates"].items():
        row = {
            "word": word_score(
                identifier, data["spans"], answers.get("words") or {}, language
            ),
            "speaker": speaker_score(
                identifier,
                candidate,
                data["spots"],
                answers.get("speakers") or {},
                language,
            ),
            "names": name_score(
                identifier, data["spots"], answers.get("speakers") or {}, language
            ),
            "cost": candidate["cost_score"],
            "readability": None,
            "final": None,
        }
        row["pre"] = _weighted(row, PRE_WEIGHTS)
        if identifier in readability:
            row["readability"] = (readability[identifier] - 1) / 4 * 100
            row["final"] = _weighted(row, FINAL_WEIGHTS)
        rows[identifier] = row
    return {"rows": rows, "decisions": decide(data["candidates"], rows)}


def word_score(
    identifier: str, spans: list[dict], answers: dict, language: str
) -> float | None:
    """Return the weighted share of answered spans the candidate got right.

    identifier: the candidate id
    spans: the disagreement spans
    answers: the answer per span id
    language: the transcript language"""
    total = hits = 0
    for span in spans:
        answer = answers.get(span["id"]) or {}
        choice = answer.get("choice")
        typed = " ".join(normalize(answer.get("text", ""), language))
        # "none of these" without text counts as unanswered, as on the page
        if (
            not choice
            or choice == ANSWER_UNCLEAR
            or (choice == ANSWER_NONE and not typed)
        ):
            continue
        weight = MEDICAL_WEIGHT if span["medical"] else 1
        total += weight
        variant = next(v for v in span["variants"] if identifier in v["candidates"])
        if choice == ANSWER_NONE:
            correct = typed == variant["key"]
        else:
            correct = variant["letter"] == choice
        hits += weight if correct else 0
    return 100 * hits / total if total else None


def speaker_score(
    identifier: str, candidate: dict, spots: list[dict], answers: dict, language: str
) -> float | None:
    """Return the pairwise same/different agreement with the reviewer's speakers.

    identifier: the candidate id
    candidate: the candidate facts
    spots: the spot-checks
    answers: the answer per spot id
    language: the transcript language"""
    answered = [spot for spot in spots if answers.get(spot["id"])]
    pairs = list(combinations(answered, 2))
    if not pairs:
        return None
    if not candidate["speakers"]:
        return 0.0
    # typed names may differ in case, spacing or punctuation
    said = {
        spot["id"]: tuple(normalize(answers[spot["id"]], language)) for spot in answered
    }
    agree = sum(
        (said[a["id"]] == said[b["id"]])
        == (a["labels"][identifier] == b["labels"][identifier])
        for a, b in pairs
    )
    return 100 * agree / len(pairs)


def name_score(
    identifier: str, spots: list[dict], answers: dict, language: str
) -> float | None:
    """Return the share of named spot-checks where the label shares a name word
    (three letters or more) with the answer.

    Answers like "Unknown person 2" are skipped.

    identifier: the candidate id
    spots: the spot-checks
    answers: the answer per spot id
    language: the transcript language"""
    named = [
        (
            spot,
            {
                word
                for word in normalize(answers[spot["id"]], language)
                if len(word) >= 3
            },
        )
        for spot in spots
        if answers.get(spot["id"])
        and not answers[spot["id"]].startswith(UNKNOWN_PERSON)
    ]
    named = [(spot, words) for spot, words in named if words]
    if not named:
        return None
    hits = sum(
        bool(words & set(normalize(spot["labels"][identifier], language)))
        for spot, words in named
    )
    return 100 * hits / len(named)


def decide(candidates: dict, rows: dict) -> dict:
    """Apply the decision rules: winner, Google default, chunk length, denoise, H1.

    Among candidates with a final score (readability rated) the final score decides,
    otherwise the pre-score; scores of different kinds are never compared.

    candidates: the candidate facts
    rows: the scores per candidate"""
    contenders = [
        identifier
        for identifier, candidate in candidates.items()
        if candidate["eligible"] and not candidate["disqualified"]
    ]

    def pick(pool: list[str]) -> str | None:
        """Return the best of the pool; within TIE_POINTS the cheaper.

        pool: the candidate ids"""
        rated = [i for i in pool if rows[i]["final"] is not None]
        key = "final" if rated else "pre"
        pool = rated or pool
        if not pool:
            return None
        best = max(rows[i][key] for i in pool)
        close = [
            i
            for i in pool
            if best - rows[i][key] <= TIE_POINTS  # pragma: no mutate
        ]
        # within TIE_POINTS the cheaper wins
        return min(close, key=lambda i: candidates[i]["cost_per_hour"] or math.inf)

    winner = pick(contenders)
    google = pick([i for i in contenders if not candidates[i]["signoff"]])
    return {
        "winner": winner,
        "production": google,
        "chunk_minutes": 60
        if _leads(candidates, rows, ROLE_WHOLE_FILE, ROLE_ARGON, WHOLE_FILE_POINTS)
        else 25,
        "denoise": _leads(candidates, rows, ROLE_DENOISED, ROLE_ARGON, TIE_POINTS),
        "test_h1": winner is not None
        and _leads(
            candidates, rows, ROLE_WORDS_ONLY_PROBE, winner, H1_WORD_POINTS, "word"
        ),
    }


def _leads(
    candidates: dict,
    rows: dict,
    candidate: str,
    baseline: str,
    points: float,
    key: str = "pre",
) -> bool:
    """Return whether a candidate leads a baseline by at least the points.

    candidates: the candidate facts
    rows: the scores per candidate
    candidate: the candidate id
    baseline: the baseline id
    points: the lead needed
    key: the score to compare (default: pre)"""
    lead = _lead(candidates, rows, candidate, baseline, key)
    return lead >= points  # pragma: no mutate (threshold)


def _lead(
    candidates: dict, rows: dict, candidate: str, baseline: str, key: str
) -> float:
    """Return how many points a candidate leads a baseline; -inf if either is missing
    or the candidate is disqualified (a truncating whole-file run must not win).

    candidates: the candidate facts
    rows: the scores per candidate
    candidate: the candidate id
    baseline: the baseline id
    key: the score to compare"""
    if candidate not in rows or baseline not in rows:
        return -math.inf
    if candidates[candidate]["disqualified"]:
        return -math.inf
    first, second = rows[candidate][key], rows[baseline][key]
    if first is None or second is None:
        return -math.inf
    return first - second


def _weighted(row: dict, weights: dict) -> float:
    """Return the weighted sum of scores; a missing score counts as 0.

    row: the scores
    weights: the weight per score"""
    return round(sum((row[key] or 0) * weight for key, weight in weights.items()), 1)
