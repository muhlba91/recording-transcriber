"""Tests for the eval commands."""

import json
from types import SimpleNamespace

import pytest

from recording_transcriber.evaluation import commands
from tests.fakes import write_candidate


class TestCommands:
    def test_run_command(self, pipelines, job, tmp_path):
        path = tmp_path / "candidates.yaml"
        path.write_text(
            "candidates:\n  C1: {label: a, transcribe: {model: gemini-4-argon}}\n"
        )
        assert commands.run_command(path, job, None, tmp_path / "eval")
        assert pipelines == [("gemini-4-argon", "C1")]
        assert not commands.run_command(tmp_path / "missing.yaml", job, None, tmp_path)

    def test_run_command_only_and_config(self, manifest, pipelines, job, tmp_path):
        path = tmp_path / "candidates.yaml"
        path.write_text(
            "candidates:\n"
            "  C1: {label: a, transcribe: {model: gemini-4-argon}}\n"
            "  C2: {label: b, transcribe: {model: gemini-4-carbon}}\n"
        )
        assert commands.run_command(path, job, None, tmp_path / "eval", ["C2"])
        assert pipelines == [("gemini-4-carbon", "C2")]
        # an unreadable config stops the run
        broken = tmp_path / "config.yaml"
        broken.write_text("chunk_minutes: [")
        assert not commands.run_command(path, job, broken, tmp_path / "eval")

    @pytest.fixture
    def evaluation(self, job, tmp_path, tone_file):
        """Two finished candidates, a manifest and a config with a glossary."""
        (job / "audio" / "rec.mp3").write_bytes(tone_file.read_bytes())
        (job / "context" / "attendees.txt").write_text("Dr. Huber, Moderation\n\n")
        out = tmp_path / "eval"
        text = "wir geben heute Finerenon und dann noch das Medikament"
        write_candidate(out / "C1", [("S1", 0, text)])
        write_candidate(out / "C2", [("S1", 0, text.replace("Finerenon", "Fineron"))])
        manifest = tmp_path / "candidates.yaml"
        manifest.write_text(
            "candidates:\n  C1: {label: a, transcribe: {model: m}}\n  C2: {label: b, transcribe: {model: n}}\n"
        )
        config = tmp_path / "config.yaml"
        config.write_text("glossary_global: glossary.txt\n")
        (tmp_path / "glossary.txt").write_text("Finerenon\n")
        yield SimpleNamespace(
            manifest=manifest, out=out, report=tmp_path / "report", config=config
        )

    @pytest.fixture
    def reported(self, evaluation, job):
        assert commands.report_command(
            evaluation.manifest,
            job,
            evaluation.config,
            evaluation.out,
            evaluation.report,
        )
        yield evaluation

    def test_report_command(self, reported):
        data = json.loads((reported.out / "report_data.json").read_text())
        # the glossary of the config makes the span medical
        assert data["spans"][0]["medical"]
        # the attendees fill the speaker dropdown
        assert "Dr. Huber" in (reported.report / "index.html").read_text()

    def test_report_command_invalid_config(self, evaluation, job, tmp_path):
        evaluation.config.write_text("chunk_minutes: [")
        assert not commands.report_command(
            evaluation.manifest, job, evaluation.config, evaluation.out, tmp_path / "x"
        )

    def test_score_command(self, reported, tmp_path):
        data = json.loads((reported.out / "report_data.json").read_text())
        answers = tmp_path / "answers.json"
        letter = next(
            v["letter"] for v in data["spans"][0]["variants"] if "C1" in v["candidates"]
        )
        answers.write_text(
            json.dumps({"report": data["report"], "words": {"w01": {"choice": letter}}})
        )
        assert commands.score_command(reported.out, answers)
        assert "**Winner:** C1" in (reported.out / "leaderboard.md").read_text()
        assert (
            json.loads((reported.out / "leaderboard.json").read_text())["rows"]["C1"][
                "word"
            ]
            == 100
        )

    def test_report_command_failures(self, job, tmp_path):
        path = tmp_path / "candidates.yaml"
        path.write_text("candidates:\n  C1: {label: a, transcribe: {model: m}}\n")
        assert not commands.report_command(
            tmp_path / "x.yaml", job, None, tmp_path, tmp_path / "r"
        )
        assert not commands.report_command(path, job, None, tmp_path, tmp_path / "r")

    def test_report_command_write_fails(self, job, tmp_path, monkeypatch):
        out = tmp_path / "eval"
        write_candidate(out / "C1", [("S1", 0, "a b c")])
        path = tmp_path / "candidates.yaml"
        path.write_text("candidates:\n  C1: {label: a, transcribe: {model: m}}\n")
        monkeypatch.setattr(commands, "write_report", lambda *args: False)
        assert not commands.report_command(path, job, None, out, tmp_path / "r")

    def test_score_command_failures(self, tmp_path):
        assert not commands.score_command(tmp_path, tmp_path / "answers.json")
        (tmp_path / "report_data.json").write_text('{"report": "abc"}')
        (tmp_path / "answers.json").write_text('{"report": "other"}')
        assert not commands.score_command(tmp_path, tmp_path / "answers.json")

    def test_names(self):
        assert commands._names("Dr. Huber, Moderation, Klinik\n\nAnna Berger\n") == [
            "Dr. Huber",
            "Anna Berger",
        ]
        assert commands._names(None) == []
