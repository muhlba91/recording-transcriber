"""Tests for the leaderboard."""

from recording_transcriber.evaluation.leaderboard import leaderboard_markdown
from tests.fakes import candidate_stats


def test_leaderboard_empty_values():
    data = {
        "reference": "C1",
        "anchor": "C1",
        "candidates": {"C1": dict(candidate_stats(None), wall_s=None)},
        "spans": [],
        "spots": [],
    }
    result = {
        "rows": {
            "C1": {
                k: None
                for k in ("word", "speaker", "names", "cost", "readability", "final")
            }
            | {"pre": 0}
        },
        "decisions": {
            "winner": None,
            "production": None,
            "chunk_minutes": 25,
            "denoise": False,
            "test_h1": False,
        },
    }
    markdown = leaderboard_markdown(data, result)
    assert "| C1 | x | - | - | - |" in markdown
    assert "**Winner:** none" in markdown
    assert "- **Production default (Google, no sign-off needed):** none\n" in markdown
    assert "- **denoise:** false (C8 vs C1)\n" in markdown
    assert "- **Test hybrid H1:** no (C6" in markdown


def test_leaderboard_markdown():
    data = {
        "reference": "C1",
        "anchor": "C2",
        "candidates": {
            "C1": dict(candidate_stats(0.6), coverage=100.0),
            "C2": dict(candidate_stats(0.3, signoff=True), wall_s=90),
            "C3": dict(candidate_stats(None, eligible=False), wall_s=None),
            "C4": dict(candidate_stats(0.5, disqualified=["x", "y"])),
        },
        "spans": [{}] * 3,
        "spots": [{}] * 2,
    }

    def row(word, speaker, readability, pre, final, names):
        return {
            "word": word,
            "speaker": speaker,
            "cost": 50.0,
            "readability": readability,
            "pre": pre,
            "final": final,
            "names": names,
        }

    result = {
        "rows": {
            "C1": row(80.4, 70.0, None, 70.2, None, 33.3),
            "C2": row(90.0, 60.0, 0.0, 60.0, 0.0, None),
            "C3": row(None, None, None, 80.0, None, None),
            "C4": row(1.0, 2.0, 100.0, 1.0, 0.6, 100.0),
        },
        "decisions": {
            "winner": "C4",
            "production": "C1",
            "chunk_minutes": 60,
            "denoise": True,
            "test_h1": True,
        },
    }
    markdown = leaderboard_markdown(data, result)
    assert "3 word spans, 2 speaker spot-checks." in markdown
    rows = [line for line in markdown.splitlines() if line.startswith("| C")]
    assert rows == [
        (
            "| C4 | x | 1 | 2 | 0.50 | 50 | 100 | 1 | 1 | 100 | 99 | 10 min | "
            "disqualified: x; y |"
        ),
        (
            "| C2 | x | 90 | 60 | 0.30 | 50 | 0 | 60 | 0 | - | 99 | 2 min | "
            "needs deployment sign-off |"
        ),
        "| C3 | x | - | - | - | 50 | - | 80 | - | - | 99 | - | not eligible |",
        "| C1 | x | 80 | 70 | 0.60 | 50 | - | 70 | - | 33 | 100 | 10 min |  |",
    ]
    assert "- **Winner:** C4\n" in markdown
    assert "- **Production default (Google, no sign-off needed):** C1\n" in markdown
    assert "- **chunk_minutes:** 60 (C7 whole file vs C1)\n" in markdown
    assert "- **denoise:** true (C8 vs C1)\n" in markdown
    assert "- **Test hybrid H1:** yes (C6 word score vs the winner)\n" in markdown
