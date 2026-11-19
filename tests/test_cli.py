"""Tests for the command line interface."""

from pathlib import Path

import pytest

from recording_transcriber import cli


class TestCli:
    @pytest.fixture
    def calls(self, monkeypatch):
        calls = {}

        class FakePipeline:
            def __init__(self, config, job, backends):
                calls["config"] = config
                calls["job"] = job
                calls["backends"] = backends

            def run(self):
                return True

        monkeypatch.setattr(cli, "Pipeline", FakePipeline)
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        yield calls

    def test_main(self, calls, tmp_path):
        assert (
            cli.main(
                ["run", "--job", str(tmp_path / "job"), "--model", "gemini-4-carbon"]
            )
            == 0
        )
        assert calls["config"].tasks.transcribe.model == "gemini-4-carbon"
        assert calls["job"].name == "job"
        assert "transcribe" in calls["backends"]

    def test_main_config(self, calls, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("chunk_minutes: 40\n")
        argv = ["run", "--job", str(tmp_path / "job"), "--config", str(config)]
        assert cli.main(argv) == 0
        assert calls["config"].chunk_minutes == 40

    def test_run_job_config(self, calls, tmp_path):
        config = tmp_path / "app" / "config.yaml"
        config.parent.mkdir()
        config.write_text("chunk_minutes: 40\n")
        assert cli.run_job(tmp_path / "jobs" / "board", config)
        assert calls["config"].chunk_minutes == 40
        assert calls["config"].folder == config.parent

    def test_run_job_provider_keeps_folder(self, calls, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        config = tmp_path / "config.yaml"
        config.write_text("chunk_minutes: 40\n")
        assert cli.run_job(tmp_path / "job", config, provider="mistral")
        assert calls["config"].folder == tmp_path

    def test_run_job_invalid_config(self, calls, tmp_path):
        assert not cli.run_job(tmp_path, Path("/nonexistent.yaml"))

    def test_run_job_missing_key(self, calls, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY")
        assert not cli.run_job(tmp_path)

    def test_main_provider(self, calls, tmp_path, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
        argv = ["run", "--job", str(tmp_path / "job"), "--provider", "mistral"]
        assert cli.main(argv) == 0
        assert calls["config"].tasks.transcribe.provider == "mistral"

    def test_main_provider_invalid_for_config(self, calls, tmp_path):
        argv = [
            "run",
            "--job",
            str(tmp_path / "job"),
            "--provider",
            "gemini-transcribe",
        ]
        assert cli.main(argv) == 1

    def test_main_model_empty(self, calls, tmp_path):
        assert cli.main(["run", "--job", str(tmp_path / "job"), "--model", ""]) == 1
        assert "config" not in calls

    def test_main_failure(self, calls, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY")
        assert cli.main(["run", "--job", str(tmp_path / "job")]) == 1


class TestMain:
    @pytest.fixture
    def patched(self, monkeypatch):
        seen = {"run_job": [], "eval": [], "logging": []}
        outcome = {"run_job": True, "eval": True}

        def run_job(*args):
            seen["run_job"].append(args)
            return outcome["run_job"]

        def run_eval(args):
            seen["eval"].append(args)
            return outcome["eval"]

        monkeypatch.setattr(cli, "run_job", run_job)
        monkeypatch.setattr(cli, "_run_eval", run_eval)
        monkeypatch.setattr(cli, "setup_logging", seen["logging"].append)
        yield seen, outcome

    @pytest.mark.parametrize(("success", "code"), [(True, 0), (False, 1)])
    def test_run_exit_code(self, patched, success, code):
        seen, outcome = patched
        outcome["run_job"] = success
        argv = ["run", "--job", "j", "--config", "c.yaml", "--model", "m", "--verbose"]
        assert cli.main(argv) == code
        assert len(seen["run_job"]) == 1
        assert seen["logging"] == [True]

    @pytest.mark.parametrize(("success", "code"), [(True, 0), (False, 1)])
    def test_eval_exit_code(self, patched, success, code):
        seen, outcome = patched
        outcome["eval"] = success
        assert cli.main(["eval", "score", "--answers", "a.json"]) == code
        assert len(seen["eval"]) == 1
        assert seen["logging"] == [False]
        assert seen["run_job"] == []


class TestEvalCli:
    @pytest.fixture
    def calls(self, monkeypatch):
        from recording_transcriber.evaluation import commands

        calls = []
        for name in ("run_command", "report_command", "score_command"):
            monkeypatch.setattr(
                commands,
                name,
                lambda *args, name=name: calls.append((name, args)) or True,
            )
        yield calls

    def test_eval_run(self, calls):
        argv = ["eval", "run", "--job", "j", "--config", "c.yaml", "--only", "C1, C5"]
        assert cli.main(argv) == 0
        name, args = calls[0]
        assert name == "run_command"
        assert Path("j") in args and Path("c.yaml") in args
        assert args[4] == ["C1", "C5"]

    def test_eval_run_all(self, calls):
        assert cli.main(["eval", "run", "--job", "j"]) == 0
        assert calls[0][1][4] is None

    def test_eval_report(self, calls):
        argv = ["eval", "report", "--job", "j", "--report", "r", "--config", "c.yaml"]
        assert cli.main(argv) == 0
        name, args = calls[0]
        assert name == "report_command"
        assert {Path("j"), Path("c.yaml"), Path("r")} <= set(args)

    def test_eval_score(self, calls):
        assert cli.main(["eval", "score", "--answers", "a.json", "--out", "o"]) == 0
        assert calls[0] == ("score_command", (Path("o"), Path("a.json")))
