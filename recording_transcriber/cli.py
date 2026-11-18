"""Command line interface: recording-transcriber run | eval."""

import argparse
import logging
from pathlib import Path

from pydantic import ValidationError

from .backends.registry import create_backends
from .configuration.config import Config
from .configuration.job import Job
from .pipeline.pipeline import Pipeline
from .utils.const import LOG_FORMAT, PROVIDERS

_LOGGER = logging.getLogger(__name__)


def run_job(
    job_dir: Path,
    config_path: Path | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> bool:
    """Run one job; return False and log an error if it fails.

    job_dir: the job folder
    config_path: the config.yaml (default: None, the defaults)
    model: overrides tasks.transcribe.model (default: None)
    provider: overrides tasks.transcribe.provider (default: None)"""
    config = Config.load(config_path)
    if config is None:
        return False
    if model is not None or provider is not None:
        # validate again: e.g. Gemini's transcribe model needs a region code
        data = config.model_dump()
        if model is not None:
            data["tasks"]["transcribe"]["model"] = model
        if provider is not None:
            data["tasks"]["transcribe"]["provider"] = provider
        try:
            override = Config.model_validate(data)
        except ValidationError as error:
            _LOGGER.error("Invalid model or provider override: %s.", error)
            return False
        # keep the app-folder guard: validation drops the loaded file's folder
        config = override.with_folder(config.folder)
    backends = create_backends(config)
    if backends is None:
        return False
    _LOGGER.info(
        "Job %s with %s (config: %s).",
        job_dir.name,
        config.tasks.transcribe.model,
        config_path or "defaults",
    )
    return Pipeline(config, Job(job_dir), backends).run()


def main(argv: list[str] | None = None) -> int:
    """Parse the arguments and run the command; return the exit code.

    argv: the arguments (default: None, uses sys.argv)"""
    parser = argparse.ArgumentParser(prog="recording-transcriber", description=__doc__)
    subparser = parser.add_subparsers(dest="command", required=True)
    run_parser = subparser.add_parser("run", help="transcribe a job folder")
    run_parser.add_argument("--job", type=Path, required=True, help="the job folder")
    run_parser.add_argument("--config", type=Path, help="the config.yaml")
    run_parser.add_argument("--model", help="override tasks.transcribe.model")
    run_parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        help="override tasks.transcribe.provider",
    )
    run_parser.add_argument("--verbose", action="store_true", help="debug logging")
    _add_eval_parser(subparser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.command == "eval":
        return 0 if _run_eval(args) else 1
    return 0 if run_job(args.job, args.config, args.model, args.provider) else 1


def setup_logging(verbose: bool = False) -> None:  # pragma: no mutate block
    """Log to the console with timestamps; quiet the per-request HTTP logs.

    verbose: log debug messages (default: False)"""
    # logging configuration is cosmetic, like the log statements (no mutate block)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        force=True,
    )
    # per-request logs are noise; google-genai logs through its own "httpx2" logger
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _add_eval_parser(subparser: argparse._SubParsersAction) -> None:
    """Add the eval subcommands run, report and score (design §5).

    subparser: the subparsers of the main parser"""
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--out", type=Path, default=Path("runs/eval"), help="the evaluation folder"
    )
    common_parser.add_argument("--verbose", action="store_true", help="debug logging")
    candidates_parser = argparse.ArgumentParser(add_help=False)
    candidates_parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("eval/candidates.yaml"),
        help="the candidates",
    )
    candidates_parser.add_argument(
        "--job", type=Path, required=True, help="the job with audio/ and context/"
    )
    candidates_parser.add_argument("--config", type=Path, help="the base config.yaml")
    eval_parser = subparser.add_parser("eval", help="compare candidate models")
    steps_parser = eval_parser.add_subparsers(dest="step", required=True)
    run_parser = steps_parser.add_parser(
        "run", parents=[common_parser, candidates_parser], help="run the candidates"
    )
    run_parser.add_argument("--only", help="comma-separated candidate ids")
    report_parser = steps_parser.add_parser(
        "report",
        parents=[common_parser, candidates_parser],
        help="build the reviewer's report",
    )
    report_parser.add_argument(
        "--report", type=Path, default=Path("report"), help="the report folder"
    )
    score_parser = steps_parser.add_parser(
        "score", parents=[common_parser], help="score the answers"
    )
    score_parser.add_argument(
        "--answers", type=Path, required=True, help="the answers.json"
    )


def _run_eval(args: argparse.Namespace) -> bool:
    """Run an eval subcommand; the evaluation extra is only imported here.

    args: the parsed arguments"""
    from .evaluation import commands

    if args.step == "run":
        only = [i.strip() for i in args.only.split(",")] if args.only else None
        return commands.run_command(
            args.manifest, args.job, args.config, args.out, only
        )
    if args.step == "report":
        return commands.report_command(
            args.manifest, args.job, args.config, args.out, args.report
        )
    return commands.score_command(args.out, args.answers)
