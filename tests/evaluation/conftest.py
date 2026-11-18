"""Shared fixtures for the evaluation tests."""

import pytest

from recording_transcriber.evaluation import runner
from recording_transcriber.evaluation.manifest import Manifest


@pytest.fixture
def manifest(tmp_path):
    legacy = tmp_path / "legacy.txt"
    legacy.write_text("Dr. A: Hallo.\n")
    yield Manifest.model_validate(
        {
            "legacy": legacy,
            "candidates": {
                "C1": {"label": "a", "transcribe": {"model": "gemini-4-argon"}},
                "C9": {
                    "label": "b",
                    "transcribe": {"provider": "openai", "model": "gpt-transcribe"},
                },
            },
        }
    )


@pytest.fixture
def pipelines(monkeypatch):
    runs = []

    class FakePipeline:
        def __init__(self, config, job, backends):
            assert backends
            self.config, self.job = config, job

        def run(self):
            runs.append((self.config.tasks.transcribe.model, self.job.root.name))
            return self.config.tasks.transcribe.provider != "openai"

    monkeypatch.setattr(runner, "Pipeline", FakePipeline)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    yield runs


@pytest.fixture
def job(tmp_path):
    job = tmp_path / "job"
    (job / "audio").mkdir(parents=True)
    (job / "context").mkdir()
    yield job
