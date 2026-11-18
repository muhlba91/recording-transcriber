"""Tests for the notebook helpers and the Colab notebook."""

import ast
import json
import logging
import os
import sys
import time
import types
from pathlib import Path

import pytest

from recording_transcriber import notebook

_NOTEBOOK = Path(__file__).parent.parent / "colab" / "recording-transcriber.ipynb"


def _job(workspace: Path, name: str, finished: bool = False, age: int = 0) -> Path:
    folder = workspace / name
    (folder / "out").mkdir(parents=True)
    (folder / "rec.mp3").write_text("x")
    if finished:
        (folder / "out" / "transcript.md").write_text("x")
    stamp = time.time() - age
    os.utime(folder, (stamp, stamp))
    return folder


def test_prepare_workspace_creates_readme(tmp_path):
    workspace = tmp_path / "My Drive" / "Transcriptions"
    assert notebook.prepare_workspace(workspace) == []
    assert "Run all" in (workspace / "README.txt").read_text()


def test_prepare_workspace_keeps_readme(tmp_path):
    workspace = tmp_path / "Transcriptions"
    notebook.prepare_workspace(workspace)
    (workspace / "README.txt").write_text("mine")
    notebook.prepare_workspace(workspace)
    assert (workspace / "README.txt").read_text() == "mine"


def test_prepare_workspace_lists_jobs_newest_first(tmp_path):
    workspace = tmp_path / "Transcriptions"
    (workspace / "_old").mkdir(parents=True)
    (workspace / "notes only").mkdir()
    older = _job(workspace, "b", age=100)
    newer = _job(workspace, "a", age=0)
    assert notebook.prepare_workspace(workspace) == [newer, older]


def test_workspace_path(tmp_path):
    assert notebook.workspace_path(tmp_path, " Jobs ") == tmp_path / "Jobs"
    for name in ("", ".", "..", "/content/drive/Shareddrives/x", "a/b", "a\\b"):
        assert notebook.workspace_path(tmp_path, name) is None


def test_pick_job_newest_unfinished(tmp_path):
    _job(tmp_path, "done", finished=True, age=0)
    todo = _job(tmp_path, "todo", age=50)
    _job(tmp_path, "older", age=100)
    assert notebook.pick_job(tmp_path) == todo


def test_pick_job_by_name(tmp_path):
    done = _job(tmp_path, "done", finished=True)
    assert notebook.pick_job(tmp_path, " done ") == done
    assert notebook.pick_job(tmp_path, "missing") is None


def test_pick_job_none_left(tmp_path):
    _job(tmp_path, "done", finished=True)
    assert notebook.pick_job(tmp_path) is None


def test_required_keys(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("tasks:\n  agenda: {provider: openai, model: m}\n")
    assert notebook.required_keys(config) == ["GEMINI_API_KEY", "OPENAI_API_KEY"]
    assert notebook.required_keys(tmp_path / "missing.yaml") is None


class TestRun:
    @pytest.fixture
    def calls(self, monkeypatch):
        calls = []

        def fake_run_job(job_dir, config_path):
            notebook._LOGGER.warning("running %s", job_dir.name)
            calls.append((job_dir, config_path))
            return True

        # the real one reconfigures the root logger
        monkeypatch.setattr(notebook, "setup_logging", calls.append)
        monkeypatch.setattr(notebook, "run_job", fake_run_job)
        yield calls

    def test_run(self, calls, tmp_path):
        job = tmp_path / "jobs" / "board"
        config = tmp_path / "app" / "config.yaml"
        handlers = list(logging.getLogger().handlers)
        assert notebook.run(job, config)
        # again after a disconnect: work/ exists
        assert notebook.run(job, config, verbose=True)
        assert calls == [False, (job, config), True, (job, config)]
        assert (job / "work" / "run.log").is_file()
        # the log file handler is gone again
        assert logging.getLogger().handlers == handlers

    def test_run_inside_app_folder(self, calls, tmp_path):
        job = tmp_path / "app" / "board"
        assert not notebook.run(job, tmp_path / "app" / "config.yaml")
        assert calls == [False]
        assert not job.exists()


def _code_cells() -> list[str]:
    cells = json.loads(_NOTEBOOK.read_text())["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def test_notebook_cells_compile():
    cells = _code_cells()
    assert len(cells) == 5
    for source in cells:
        compile(source, "cell", "exec")
    # outputs are never committed: the notebook must stay free of job data
    raw = json.loads(_NOTEBOOK.read_text())
    assert all(not cell.get("outputs") for cell in raw["cells"])
    assert raw["metadata"]["colab"]["private_outputs"]


@pytest.fixture
def install_cell() -> dict:
    # the install cell's imports, constants and helpers, without the install itself
    tree = ast.parse(_code_cells()[1])
    nodes = [node for node in tree.body if not isinstance(node, ast.If)]
    namespace = {}
    exec(compile(ast.Module(nodes, []), "cell", "exec"), namespace)  # noqa: S102
    return namespace


def _wheel(folder: Path, version: str) -> str:
    name = f"recording_transcriber-{version}-py3-none-any.whl"
    (folder / name).write_text("")
    return name


def test_notebook_pick_wheel(install_cell, tmp_path):
    pick_wheel = install_cell["pick_wheel"]
    assert pick_wheel(tmp_path) is None
    for version in ("1.2.0", "1.10.0", "1.9.3"):
        _wheel(tmp_path, version)
    _wheel(tmp_path, "latest")
    assert pick_wheel(tmp_path).name == _wheel(tmp_path, "1.10.0")
    assert pick_wheel(tmp_path, "1.2.0").name == _wheel(tmp_path, "1.2.0")
    assert pick_wheel(tmp_path, "2.0.0") is None
    # a release candidate is the highest version, and can be pinned
    candidate = _wheel(tmp_path, "1.11.0rc1")
    assert pick_wheel(tmp_path).name == candidate
    assert pick_wheel(tmp_path, "1.11.0rc1").name == candidate


def test_notebook_pinned_version(install_cell):
    pinned_version = install_cell["pinned_version"]
    assert pinned_version({}) is None
    assert pinned_version({"app_version": None}) is None
    assert pinned_version({"app_version": "1.2.0"}) == "1.2.0"
    with pytest.raises(TypeError, match="Quote app_version"):
        pinned_version({"app_version": 1.0})
    with pytest.raises(RuntimeError, match="not a version"):
        pinned_version({"app_version": "latest"})


class _SecretNotFoundError(Exception):
    pass


class _NotebookAccessError(Exception):
    pass


@pytest.fixture
def key_cell(monkeypatch):
    """Run the key cell against fake Colab secrets; return a runner."""
    secrets = {}

    class Userdata:
        SecretNotFoundError = _SecretNotFoundError
        NotebookAccessError = _NotebookAccessError

        @staticmethod
        def get(name):
            value = secrets.get(name)
            if value is None:
                raise _SecretNotFoundError(name)
            if isinstance(value, Exception):
                raise value
            return value

    colab = types.ModuleType("google.colab")
    colab.userdata = Userdata
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(notebook, "required_keys", lambda path: secrets.pop("needs"))
    for name in ("GEMINI_API_KEY", "LITELLM_KEY"):
        # the cell writes os.environ: setenv first, so that the teardown removes it
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)

    def run(needs, **values):
        secrets.update(values, needs=needs)
        exec(compile(_code_cells()[2], "cell", "exec"), {"APP": Path("/app")})  # noqa: S102

    return run


def test_notebook_keys_cell(key_cell):
    key_cell(["GEMINI_API_KEY", "LITELLM_KEY"], GEMINI_API_KEY="g", LITELLM_KEY="h")
    assert (os.environ["GEMINI_API_KEY"], os.environ["LITELLM_KEY"]) == ("g", "h")


def test_notebook_keys_cell_google_fallback(key_cell):
    key_cell(["GEMINI_API_KEY"], GOOGLE_API_KEY="old")
    assert os.environ["GEMINI_API_KEY"] == "old"


def test_notebook_keys_cell_reports_every_missing_secret(key_cell):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY, LITELLM_KEY"):
        key_cell(["GEMINI_API_KEY", "LITELLM_KEY"])


def test_notebook_keys_cell_needs_notebook_access(key_cell):
    with pytest.raises(RuntimeError, match="Notebook access"):
        key_cell(["LITELLM_KEY"], LITELLM_KEY=_NotebookAccessError())


def test_notebook_keys_cell_invalid_config(key_cell):
    with pytest.raises(RuntimeError, match="config.yaml"):
        key_cell(None)
