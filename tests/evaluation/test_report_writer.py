"""Tests for the report writer."""

import json
import re

import pytest

from recording_transcriber.audio.ffmpeg import Ffmpeg
from recording_transcriber.evaluation.report_writer import write_report


def _data():
    return {
        "candidates": {
            "C1": {
                "eligible": True,
                "disqualified": [],
                "speakers": True,
                "cost_score": 50,
                "sample": "Anna: a",
            },
            "C2": {
                "eligible": True,
                "disqualified": ["x"],
                "speakers": True,
                "cost_score": 100,
                "sample": "b",
            },
            "C0": {
                "eligible": False,
                "disqualified": [],
                "speakers": True,
                "cost_score": 0,
                "sample": "c",
            },
        },
        "spans": [
            {
                "id": "w01",
                "start_s": 1.0,
                "end_s": 2.0,
                "before": "vor",
                "after": "nach </script>",
                "medical": True,
                "variants": [
                    {"key": "a", "text": "a", "candidates": ["C1", "C0"]},
                    {"key": "b", "text": "b", "candidates": ["C2"]},
                ],
            }
        ],
        "spots": [
            {
                "id": "s01",
                "start_s": 0.0,
                "labels": {"C1": "Anna", "C2": "Sprecher 1", "C0": "Dr. A"},
            }
        ],
    }


def test_write_report(tmp_path, tone_file):
    report = tmp_path / "report"
    (report / "old").mkdir(parents=True)
    (report / "index.html").write_text("an earlier report")
    data = _data()
    assert write_report(data, [tone_file], ["Dr. Müller", "Dr. Huber"], report)
    assert sorted(path.name for path in (report / "clips").iterdir()) == [
        "s01.mp3",
        "w01.mp3",
    ]
    assert not (report / "old").exists()
    assert sorted(data["tokens"].values()) == ["t1", "t2", "t3"]
    # C2 is disqualified: its variant gets no letter and is not shown
    letters = {v["text"]: v["letter"] for v in data["spans"][0]["variants"]}
    assert letters == {"a": "A", "b": ""}
    page = (report / "index.html").read_text()
    payload = json.loads(re.search(r"const DATA = (.*);\n", page).group(1))
    assert payload["id"] == data["report"]
    assert [v["text"] for v in payload["spans"][0]["variants"]] == ["a"]
    # only attendees: candidate labels would hint at who wrote what
    assert payload["names"] == ["Dr. Huber", "Dr. Müller"] and "Dr. Müller" in page
    assert payload["unknown"] == [f"Unknown person {number}" for number in range(1, 9)]
    # only candidates that can win; the page never names a candidate
    assert [c["token"] for c in payload["candidates"]] == [data["tokens"]["C1"]]
    assert "C1" not in json.dumps(payload) and "<\\/script>" in page
    tokens = data["tokens"]
    (span,) = payload["spans"]
    assert span["clip"] == "clips/w01.mp3" and span["medical"]
    assert span["variants"] == [
        {"letter": "A", "text": "a", "who": [tokens["C1"], tokens["C0"]]}
    ]
    assert payload["spots"][0]["who"][tokens["C0"]] == "Dr. A"
    assert Ffmpeg().duration(report / "clips" / "w01.mp3") == pytest.approx(
        8, abs=0.5
    )  # 0-8 s: padded, clamped to the start


def test_write_report_prepare_fails(tmp_path):
    assert not write_report(
        _data(), [tmp_path / "missing.mp3"], [], tmp_path / "report"
    )


def test_write_report_extract_fails(tmp_path, tone_file, monkeypatch):
    ffmpeg = Ffmpeg()
    monkeypatch.setattr(ffmpeg, "extract", lambda *args: False)
    assert not write_report(
        _data(), [tone_file], [], tmp_path / "report", ffmpeg=ffmpeg
    )
    data = _data()
    data["spans"] = []
    assert not write_report(data, [tone_file], [], tmp_path / "report", ffmpeg=ffmpeg)


def test_write_report_keeps_unrelated_folder(tmp_path, tone_file):
    (tmp_path / "notes.txt").write_text("keep")
    assert not write_report(_data(), [tone_file], [], tmp_path)
    assert (tmp_path / "notes.txt").is_file()


def test_write_report_spot_extract_fails(tmp_path, tone_file, monkeypatch):
    ffmpeg = Ffmpeg()
    extract = ffmpeg.extract
    monkeypatch.setattr(
        ffmpeg, "extract", lambda a, s, e, t: t.stem != "s01" and extract(a, s, e, t)
    )
    assert not write_report(_data(), [tone_file], [], tmp_path / "r", ffmpeg=ffmpeg)
