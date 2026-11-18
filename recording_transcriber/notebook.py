"""Helpers for the Colab notebook: the workspace, picking a job and running it."""

import logging
from pathlib import Path

from .cli import run_job, setup_logging
from .configuration.config import Config
from .configuration.job import Job
from .utils.const import (
    IGNORED_JOB_PREFIXES,
    LOG_FORMAT,
    RUN_LOG,
    WORKSPACE_README,
    WORKSPACE_README_TEXT,
)

_LOGGER = logging.getLogger(__name__)


def workspace_path(my_drive: Path, name: str) -> Path | None:
    """Return the workspace folder in My Drive, or None if the name is not a plain
    folder name (an absolute path or ".." could point into the app folder).

    my_drive: the mounted My Drive
    name: the workspace folder's name"""
    name = name.strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        _LOGGER.error(
            "WORKSPACE must be a plain folder name in My Drive, like Transcriptions."
        )
        return None
    return my_drive / name


def prepare_workspace(workspace: Path) -> list[Path]:
    """Create the workspace with a README on first use; return its job folders.

    Job folders are the subfolders with recordings, most recently changed first;
    names starting with "." or "_" are skipped.

    workspace: the folder in My Drive that holds one folder per job"""
    workspace.mkdir(parents=True, exist_ok=True)
    readme = workspace / WORKSPACE_README
    if not readme.exists():
        readme.write_text(WORKSPACE_README_TEXT, encoding="utf-8")
    folders = [
        path
        for path in workspace.iterdir()
        if path.is_dir()
        and not path.name.startswith(IGNORED_JOB_PREFIXES)
        and Job(path).audio_files()
    ]
    return sorted(folders, key=lambda path: path.stat().st_mtime, reverse=True)


def pick_job(workspace: Path, name: str = "") -> Path | None:
    """Return the named job folder, or else the newest one without a transcript.

    Logs the jobs found, or what to do if there is none.

    workspace: the folder in My Drive that holds one folder per job
    name: the job folder's name (default: empty, the newest unfinished job)"""
    folders = prepare_workspace(workspace)
    for folder in folders:
        _LOGGER.info(
            "Found job %s (%s).",
            folder.name,
            "done" if Job(folder).finished else "to do",
        )
    if name.strip():
        folder = workspace / name.strip()
        if folder in folders:
            return folder
        _LOGGER.error(
            "No job folder named %s with recordings in %s.", name.strip(), workspace
        )
        return None
    for folder in folders:
        if not Job(folder).finished:
            return folder
    _LOGGER.error(
        "No job to do: put the recordings into a new folder in %s, then run all "
        "cells again. To redo a finished job, enter its name in the job field.",
        workspace,
    )
    return None


def required_keys(config_path: Path) -> list[str] | None:
    """Return the names of the API key variables the configuration needs, or None
    and log an error if it is invalid.

    config_path: the config.yaml in the app folder"""
    config = Config.load(config_path)
    return None if config is None else config.key_variables()


def run(job_dir: Path, config_path: Path, verbose: bool = False) -> bool:
    """Run a job with console logging and a copy of the log in work/run.log.

    Running it again after a disconnect resumes from the cache.

    job_dir: the job folder
    config_path: the config.yaml in the app folder
    verbose: log debug messages (default: False)"""
    setup_logging(verbose)
    # checked before the log is written: the app folder only holds the
    # maintainer's files (the pipeline repeats the check)
    if job_dir.resolve().is_relative_to(config_path.parent.resolve()):
        _LOGGER.error(
            "The job folder %s is inside the shared app folder; move it to "
            "My Drive/Transcriptions.",
            job_dir,
        )
        return False
    log_file = Job(job_dir).work_dir / RUN_LOG
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter(LOG_FORMAT))  # pragma: no mutate (cosmetic)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        return run_job(job_dir, config_path)
    finally:
        root.removeHandler(handler)
        handler.close()
