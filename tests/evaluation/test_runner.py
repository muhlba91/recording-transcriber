"""Tests for the candidate runner."""

import json

import pytest

from recording_transcriber.configuration.config import Config
from recording_transcriber.evaluation import runner


def test_run_candidates(manifest, pipelines, job, tmp_path):
    out = tmp_path / "eval"
    assert not runner.run_candidates(manifest, job, Config(), out)
    assert pipelines == [("gemini-4-argon", "C1"), ("gpt-transcribe", "C9")]
    assert (out / "C1" / "audio").resolve() == job / "audio"
    legacy = json.loads((out / "C0" / "out" / "transcript.json").read_text())
    assert legacy["paragraphs"][0]["speaker"] == "Dr. A"
    # a rerun keeps the existing links
    assert runner.run_candidates(manifest, job, Config(), out, only=["C1"])


def test_run_candidates_skips_finished(manifest, pipelines, job, tmp_path):
    out = tmp_path / "eval"
    (out / "C1" / "out").mkdir(parents=True)
    (out / "C1" / "out" / "run.json").write_text("{}")
    runner.run_candidates(manifest, job, Config(), out, only=["C1"])
    assert pipelines == []


def test_run_candidates_unknown_id(manifest, pipelines, job, tmp_path):
    assert not runner.run_candidates(manifest, job, Config(), tmp_path, only=["C99"])


def test_link_sources_replaces_other_job(job, tmp_path):
    other = tmp_path / "other"
    (other / "audio").mkdir(parents=True)
    folder = tmp_path / "eval" / "C1"
    folder.mkdir(parents=True)
    (folder / "audio").symlink_to(other / "audio", target_is_directory=True)
    runner._link_sources(job, folder)
    assert (folder / "audio").resolve() == (job / "audio").resolve()


def test_run_candidates_failures(manifest, pipelines, job, tmp_path, monkeypatch):
    manifest.legacy = tmp_path / "missing.txt"
    monkeypatch.delenv("OPENAI_API_KEY")
    manifest.candidates["C1"].transcribe.provider = "whisper"
    assert runner.run_candidates(manifest, job, Config(), tmp_path / "eval") is False
    assert pipelines == []


def test_run_candidates_only_legacy(manifest, pipelines, job, tmp_path):
    out = tmp_path / "eval"
    assert runner.run_candidates(manifest, job, Config(), out, only=["C0"]) is True
    assert (out / "C0" / "out" / "transcript.json").is_file()
    # the import also works into an existing folder
    assert runner.run_candidates(manifest, job, Config(), out, only=["C0"]) is True
    assert pipelines == []


def test_run_candidates_only_legacy_missing(manifest, pipelines, job, tmp_path):
    manifest.legacy = tmp_path / "missing.txt"
    out = tmp_path / "eval"
    assert runner.run_candidates(manifest, job, Config(), out, only=["C0"]) is False


def test_run_candidates_only_other_candidate_skips_legacy(
    manifest, pipelines, job, tmp_path
):
    out = tmp_path / "eval"
    assert runner.run_candidates(manifest, job, Config(), out, only=["C1"]) is True
    assert not (out / "C0").exists()
    assert pipelines == [("gemini-4-argon", "C1")]


@pytest.mark.parametrize(
    ("finished", "broken", "only"),
    [(False, False, ["C9"]), (True, False, None), (False, True, None)],
    ids=["selected", "earlier finished", "earlier backend fails"],
)
def test_run_candidates_later_candidate_runs(
    manifest, pipelines, job, tmp_path, finished, broken, only
):
    out = tmp_path / "eval"
    if finished:
        (out / "C1" / "out").mkdir(parents=True)
        (out / "C1" / "out" / "run.json").write_text("{}")
    if broken:
        manifest.candidates["C1"].transcribe.provider = "whisper"
    runner.run_candidates(manifest, job, Config(), out, only=only)
    assert pipelines == [("gpt-transcribe", "C9")]


def test_run_candidates_pipeline_failure(manifest, pipelines, job, tmp_path):
    out = tmp_path / "eval"
    assert runner.run_candidates(manifest, job, Config(), out, only=["C9"]) is False


@pytest.mark.parametrize(
    ("present", "missing"), [("audio", "context"), ("context", "audio")]
)
def test_link_sources_skips_missing_folder(tmp_path, present, missing):
    job = tmp_path / "job"
    (job / present).mkdir(parents=True)
    folder = tmp_path / "eval" / "C1"
    runner._link_sources(job, folder)
    assert (folder / present).is_symlink()
    assert not (folder / missing).exists()
