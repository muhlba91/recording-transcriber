"""The eval subcommands: run, report, score."""

import json
import logging
from pathlib import Path

from ..configuration.config import Config
from ..configuration.job import Job
from ..utils.const import LEADERBOARD_JSON, LEADERBOARD_MD, REPORT_DATA_FILE
from ..utils.serialize import json_text
from .leaderboard import leaderboard_markdown
from .manifest import Manifest
from .report_data import build_data
from .report_writer import write_report
from .runner import run_candidates
from .scoring import score

_LOGGER = logging.getLogger(__name__)


def run_command(
    manifest_path: Path,
    job_dir: Path,
    config_path: Path | None,
    out_dir: Path,
    only: list[str] | None = None,
) -> bool:
    """Run the candidates; return False if any failed.

    manifest_path: the candidates file
    job_dir: the job folder with audio/ and context/
    config_path: the base config.yaml (default: None, defaults)
    out_dir: the evaluation folder
    only: the candidate ids to run (default: None, all)"""
    manifest = Manifest.load(manifest_path)
    config = Config.load(config_path)
    if manifest is None or config is None:
        return False
    return run_candidates(manifest, job_dir, config, out_dir, only)


def report_command(
    manifest_path: Path,
    job_dir: Path,
    config_path: Path | None,
    out_dir: Path,
    report_dir: Path,
) -> bool:
    """Align the candidates and write the report and out_dir/report_data.json.

    manifest_path: the candidates file
    job_dir: the job folder with audio/ and context/
    config_path: the base config.yaml (default: None, defaults)
    out_dir: the evaluation folder
    report_dir: the report folder to write (replaced)"""
    manifest = Manifest.load(manifest_path)
    config = Config.load(config_path)
    if manifest is None or config is None:
        return False
    job = Job(job_dir)
    context = job.context(config.glossary_global)
    data = build_data(manifest, out_dir, context.glossary, config.transcript.language)
    if data is None:
        return False
    names = _names(context.attendees)
    if not write_report(data, job.audio_files(), names, report_dir):
        return False
    path = out_dir / REPORT_DATA_FILE
    path.write_text(json_text(data), encoding="utf-8")
    _LOGGER.info("Report written to %s; key and data in %s.", report_dir, path)
    return True


def score_command(out_dir: Path, answers_path: Path) -> bool:
    """Score the answers and write out_dir/leaderboard.md and leaderboard.json.

    out_dir: the evaluation folder with report_data.json
    answers_path: the reviewer's answers.json"""
    try:
        data = json.loads((out_dir / REPORT_DATA_FILE).read_text(encoding="utf-8"))
        answers = json.loads(answers_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _LOGGER.error("Could not read the report data or answers: %s.", error)
        return False
    if answers.get("report") != data["report"]:
        _LOGGER.error(
            "The answers belong to another report: %s.", answers.get("report")
        )
        return False
    result = score(data, answers)
    (out_dir / LEADERBOARD_MD).write_text(
        leaderboard_markdown(data, result), encoding="utf-8"
    )
    (out_dir / LEADERBOARD_JSON).write_text(json_text(result), encoding="utf-8")
    _LOGGER.info("Winner: %s.", result["decisions"]["winner"])
    return True


def _names(attendees: str | None) -> list[str]:
    """Return the names from the attendee list ("name, role, organization" lines).

    attendees: the attendee list"""
    return [
        line.split(",")[0].strip()
        for line in (attendees or "").splitlines()
        if line.strip()
    ]
