"""JSON records: transcript.json and run.json."""

import platform
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

from .. import prompts
from ..configuration.config import Config
from ..models.chunk_result import ChunkResult
from ..models.paragraph import Paragraph
from ..models.protocol_result import ProtocolResult
from ..models.speaker_mapping import SpeakerMapping
from ..models.usage import Usage
from ..utils.const import WARNING_NO_PRICE
from ..utils.cost import estimate_cost
from ..utils.timestamps import format_timestamp


def transcript_record(
    job: str, paragraphs: list[Paragraph], mapping: SpeakerMapping
) -> dict:
    """Return the machine-readable transcript.

    job: the job name
    paragraphs: the transcript paragraphs
    mapping: the speaker mapping"""
    return {
        "job": job,
        "speakers": [group.model_dump() for group in mapping.speakers],
        "paragraphs": [
            {"timestamp": format_timestamp(paragraph.start_s), **paragraph.model_dump()}
            for paragraph in paragraphs
        ],
    }


def run_record(
    job: str,
    config: Config,
    backends: dict,
    results: list[ChunkResult],
    mapping: SpeakerMapping,
    started: datetime,
    finished: datetime,
    protocol: ProtocolResult | None = None,
) -> dict:
    """Return the run record: providers, models, settings, tokens, cost and warnings.

    job: the job name
    config: the configuration
    backends: describe() of the backend of each executed task, keyed by task
    results: all transcribed chunks
    mapping: the speaker mapping
    started: the run start
    finished: the run end
    protocol: the protocol step's result (default: None, the step was off)"""
    transcription = Usage()
    polish = Usage()
    for result in results:
        transcription = transcription.add(result.usage)
        polish = polish.add(result.text_usage)
    tasks = config.tasks
    costs = {
        "transcription": estimate_cost(
            transcription, config.prices.get(tasks.transcribe.model)
        ),
        "reconciliation": estimate_cost(
            mapping.usage, config.prices.get(tasks.reconcile.model)
        ),
    }
    # speech-to-text candidates only: the polish pass that removes fillers
    if polish != Usage():
        costs["polish"] = estimate_cost(polish, config.prices.get(tasks.polish.model))
    usage = {
        "transcription": transcription.model_dump(),
        "polish": polish.model_dump(),
        "reconciliation": mapping.usage.model_dump(),
    }
    warnings = list(mapping.warnings)
    if protocol is not None:
        steps = {
            "protocol": (tasks.protocol.model, protocol.protocol_usage),
            "agenda": (tasks.agenda.model, protocol.agenda_usage),
        }
        for step, (model, step_usage) in steps.items():
            costs[step] = estimate_cost(step_usage, config.prices.get(model))
            usage[step] = step_usage.model_dump()
        warnings += protocol.warnings
    models = {
        "transcription": tasks.transcribe.model,
        "reconciliation": tasks.reconcile.model,
        "polish": tasks.polish.model,
        "protocol": tasks.protocol.model,
        "agenda": tasks.agenda.model,
    }
    warnings += [
        f"{WARNING_NO_PRICE}: {model}"
        for model in sorted(
            {models[step] for step, cost in costs.items() if cost is None}
        )
    ]
    known = [cost for cost in costs.values() if cost is not None]
    # the costs are rounded already; this only removes float noise
    total = round(sum(known), 4) if known else None  # pragma: no mutate
    return {
        "job": job,
        "app_version": _app_version(),
        "config_app_version": config.app_version,
        "python": platform.python_version(),
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        # provider and model per task; "service" is what the backend reports
        "tasks": {
            task: {
                "provider": getattr(tasks, task).provider,
                "model": getattr(tasks, task).model,
                "service": backend["provider"],
                "endpoint": backend["endpoint"],
                "logging": backend["logging"],
            }
            for task, backend in backends.items()
        },
        "settings": {
            "chunk_minutes": config.target_chunk_minutes,
            "denoise": config.denoise,
            "write_protocol": config.write_protocol,
            "thinking_level": config.thinking_level,
            "transcript": config.transcript.model_dump(),
            "meeting": config.meeting.model_dump(),
            "prompts_replaced": _replaced_prompts(config),
        },
        "chunks": [
            {
                "index": result.span.index,
                "start": format_timestamp(result.span.start_s),
                "end": format_timestamp(result.span.end_s),
                "model": result.model,
                "attempts": result.attempts,
                "turns": len(result.turns),
                "warnings": result.warnings,
                "usage": result.usage.model_dump(),
            }
            for result in results
        ],
        "speakers": len(mapping.speakers),
        "usage": usage,
        # audio counted at the text input rate; Google may bill it up to 2x
        "cost_usd": {**costs, "total": total},
        "warnings": warnings
        + [f"part {r.span.index + 1}: {w}" for r in results for w in r.warnings],
    }


def _app_version() -> str:
    """Return the installed transcriber version."""
    try:
        # the distribution name is a fixed identifier, not behaviour (and it is
        # normalized case-insensitively)
        return version("recording-transcriber")  # pragma: no mutate
    except PackageNotFoundError:
        return "unknown"


def _replaced_prompts(config: Config) -> list[str]:
    """Return the names of the prompts replaced from prompts_dir.

    config: the configuration"""
    if config.prompts_dir is None:
        return []
    return [
        name
        for name in prompts.names()
        if (config.prompts_dir / f"{name}.md").is_file()
    ]
