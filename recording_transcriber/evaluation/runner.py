"""Run every candidate of a manifest as its own job."""

import logging
from pathlib import Path

from ..backends.registry import create_backends
from ..configuration.config import Config
from ..configuration.job import Job
from ..pipeline.pipeline import Pipeline
from ..utils.const import (
    CANDIDATE_SOURCE_DIRS,
    LEGACY_ID,
    OUT_DIR,
    RUN_JSON,
    TRANSCRIPT_JSON,
)
from ..utils.serialize import json_text
from .legacy import legacy_record
from .manifest import Manifest

_LOGGER = logging.getLogger(__name__)


def run_candidates(
    manifest: Manifest,
    job_dir: Path,
    config: Config,
    out_dir: Path,
    only: list[str] | None = None,
) -> bool:
    """Run the candidates into out_dir/<id>/; return False if any failed.

    Each candidate folder links to the job's audio and context, so nothing is
    copied. A candidate with an out/run.json is finished and skipped (its wall time
    stays); delete its folder to run it again. A failed one resumes from work/.

    manifest: the candidates
    job_dir: the job folder with audio/ and context/
    config: the base configuration
    out_dir: the evaluation folder
    only: the candidate ids to run (default: None, all)"""
    unknown = set(only or []) - set(manifest.candidates) - {LEGACY_ID}
    if unknown:
        _LOGGER.error("Unknown candidates: %s.", ", ".join(sorted(unknown)))
        return False
    succeeded = True
    if manifest.legacy is not None and (only is None or LEGACY_ID in only):
        succeeded = _import_legacy(manifest.legacy, out_dir / LEGACY_ID)
    for identifier in manifest.candidates:
        folder = out_dir / identifier
        if only is not None and identifier not in only:
            continue
        if (folder / OUT_DIR / RUN_JSON).is_file():
            _LOGGER.info("Candidate %s is finished; skipping it.", identifier)
            continue
        _LOGGER.info(
            "Candidate %s: %s.", identifier, manifest.candidates[identifier].label
        )
        candidate_config = manifest.config_for(identifier, config)
        backends = (
            None if candidate_config is None else create_backends(candidate_config)
        )
        if backends is None:
            succeeded = False
            continue
        _link_sources(job_dir, folder)
        if not Pipeline(candidate_config, Job(folder), backends).run():
            _LOGGER.error("Candidate %s failed.", identifier)
            succeeded = False
    return succeeded


def _link_sources(job_dir: Path, folder: Path) -> None:
    """Link the job's audio and context folders into a candidate folder; a link to
    another job is replaced.

    job_dir: the job folder
    folder: the candidate folder"""
    folder.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATE_SOURCE_DIRS:
        link = folder / name
        source = (job_dir / name).resolve()
        if not source.is_dir():
            continue
        if link.is_symlink() and link.resolve() != source:
            link.unlink()
        if not link.is_symlink():
            # the flag only matters on Windows
            link.symlink_to(source, target_is_directory=True)


def _import_legacy(path: Path, folder: Path) -> bool:
    """Write the legacy transcript as the C0 candidate's transcript.json.

    path: the legacy transcription.txt
    folder: the C0 folder"""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        _LOGGER.error("Could not read the legacy transcript %s: %s.", path, error)
        return False
    target = folder / OUT_DIR / TRANSCRIPT_JSON
    target.parent.mkdir(parents=True, exist_ok=True)
    record = legacy_record(text)
    target.write_text(json_text(record), encoding="utf-8")
    _LOGGER.info(
        "Candidate %s: legacy output, %s paragraphs.",
        LEGACY_ID,
        len(record["paragraphs"]),
    )
    return True
