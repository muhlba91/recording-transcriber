"""Render leaderboard.md from the scores."""

from ..utils.const import FINAL_WEIGHTS, PRE_WEIGHTS


def leaderboard_markdown(data: dict, result: dict) -> str:
    """Return the leaderboard: one row per candidate, best first, then the decisions.

    data: the evaluation data
    result: the scores and decisions from score()"""
    rows = result["rows"]
    candidates = data["candidates"]

    def rank(identifier: str) -> tuple:
        row = rows[identifier]
        # rated candidates first, by final score; scores of different kinds never mix
        return (row["final"] is None, -(row["final"] or 0), -row["pre"])

    lines = [
        "# Leaderboard",
        "",
        (
            f"Reference transcript: {data['reference']}, time anchor: {data['anchor']}. "
            f"{len(data['spans'])} word spans, {len(data['spots'])} speaker spot-checks."
        ),
        (
            f"Pre-score = {_formula(PRE_WEIGHTS)}; final = {_formula(FINAL_WEIGHTS)} "
            "(top two only). All scores 0-100."
        ),
        "",
        (
            "| ID | Candidate | Word | Speaker | Cost $/h | Cost | Readability | Pre | "
            "Final | Names | Coverage | Wall | Notes |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for identifier in sorted(rows, key=rank):
        row = rows[identifier]
        candidate = candidates[identifier]
        notes = []
        if not candidate["eligible"]:
            notes.append("not eligible")
        if candidate["signoff"]:
            notes.append("needs deployment sign-off")
        if candidate["disqualified"]:
            notes.append("disqualified: " + "; ".join(candidate["disqualified"]))
        lines.append(
            f"| {identifier} | {candidate['label']} | {_number(row['word'])} | "
            f"{_number(row['speaker'])} | {_number(candidate['cost_per_hour'], 2)} | "
            f"{_number(row['cost'])} | {_number(row['readability'])} | "
            f"{_number(row['pre'])} | {_number(row['final'])} | "
            f"{_number(row['names'])} | {_number(candidate['coverage'])} | "
            f"{_minutes(candidate['wall_s'])} | {', '.join(notes)} |"
        )
    decisions = result["decisions"]
    lines += [
        "",
        "## Decisions",
        "",
        f"- **Winner:** {decisions['winner'] or 'none'}",
        (
            f"- **Production default (Google, no sign-off needed):** "
            f"{decisions['production'] or 'none'}"
        ),
        f"- **chunk_minutes:** {decisions['chunk_minutes']} (C7 whole file vs C1)",
        f"- **denoise:** {str(decisions['denoise']).lower()} (C8 vs C1)",
        (
            f"- **Test hybrid H1:** {'yes' if decisions['test_h1'] else 'no'} "
            "(C6 word score vs the winner)"
        ),
        "",
    ]
    return "\n".join(lines)


def _formula(weights: dict) -> str:
    """Return a weight dict as "0.53 word + ...".

    weights: the weight per score"""
    return " + ".join(f"{weight} {key}" for key, weight in weights.items())


def _number(value: float | None, digits: int = 0) -> str:
    """Return a rounded number, or "-" for None.

    value: the number
    digits: the decimals (default: 0)"""
    return "-" if value is None else f"{value:.{digits}f}"


def _minutes(seconds: float | None) -> str:
    """Return seconds as "12 min", or "-".

    seconds: the duration"""
    return "-" if seconds is None else f"{seconds / 60:.0f} min"
