"""Tests for the JSON records."""

from datetime import UTC, datetime

from recording_transcriber.configuration.config import Config, Price
from recording_transcriber.models.chunk_result import ChunkResult
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.paragraph import Paragraph
from recording_transcriber.models.protocol_result import ProtocolResult
from recording_transcriber.models.speaker_assignment import SpeakerAssignment
from recording_transcriber.models.speaker_mapping import SpeakerMapping
from recording_transcriber.models.turn import Turn
from recording_transcriber.models.usage import Usage
from recording_transcriber.render import records
from recording_transcriber.render.records import run_record, transcript_record

_SERVICE = {"provider": "p", "endpoint": "e", "logging": "l"}
_BACKENDS = {"transcribe": _SERVICE, "reconcile": _SERVICE}
_NESTED = {"tasks", "settings", "chunks", "usage"}
# depend on the environment
_ENVIRONMENT = {"app_version", "config_app_version", "python"}


def _result() -> ChunkResult:
    return ChunkResult(
        span=ChunkSpan(index=0, start_s=0, end_s=60),
        model="gemini-4-carbon",
        turns=[],
        roster=[],
        usage=Usage(input_audio_tokens=1_000_000),
        warnings=["TAIL_GAP"],
    )


def test_transcript_record():
    mapping = SpeakerMapping(
        speakers=[SpeakerAssignment(ids=["S1"], name="Dr. A", confidence="high")]
    )
    record = transcript_record(
        "job",
        [Paragraph(start_s=61, speaker="Dr. A", speaker_ids=["S1"], text="x")],
        mapping,
    )
    assert record["job"] == "job"
    assert record["paragraphs"][0]["timestamp"] == "00:01:01"
    assert record["speakers"][0]["name"] == "Dr. A"


def test_run_record():
    config = Config(
        tasks={"transcribe": {"model": "gemini-4-carbon"}},
        prices={"gemini-4-carbon": Price(input=2, output=10)},
    )
    mapping = SpeakerMapping(
        model="gemini-4-carbon",
        usage=Usage(output_tokens=100_000),
        warnings=["RECONCILIATION_FAILED"],
    )
    now = datetime(2026, 11, 14, tzinfo=UTC)
    record = run_record("job", config, _BACKENDS, [_result()], mapping, now, now)
    assert record["tasks"]["transcribe"] == {
        "provider": "gemini",
        "model": "gemini-4-carbon",
        "service": "p",
        "endpoint": "e",
        "logging": "l",
    }
    assert set(record["tasks"]) == {"transcribe", "reconcile"}
    assert record["cost_usd"] == {
        "transcription": 2.0,
        "reconciliation": 1.0,
        "total": 3.0,
    }
    assert record["warnings"] == ["RECONCILIATION_FAILED", "part 1: TAIL_GAP"]
    assert record["chunks"][0]["end"] == "00:01:00"


def test_run_record_fields():
    config = Config(
        app_version="1.0.0",
        tasks={"transcribe": {"model": "gemini-4-carbon"}},
        prices={"gemini-4-carbon": Price(input=2.5, output=10)},
        chunk_minutes=30,
        denoise=True,
        thinking_level="high",
    )
    result = _result()
    result.span = ChunkSpan(index=1, start_s=60, end_s=125)
    result.attempts = 2
    result.turns = [Turn(speaker_id="S1", start_s=61, text="x")]
    mapping = SpeakerMapping(
        speakers=[SpeakerAssignment(ids=["S1"], name="A", confidence="high")],
        usage=Usage(output_tokens=100_000),
    )
    started = datetime(2026, 11, 14, 8, 0, 1, tzinfo=UTC)
    finished = datetime(2026, 11, 14, 9, 30, tzinfo=UTC)
    record = run_record("job", config, _BACKENDS, [result], mapping, started, finished)
    assert {
        key: record[key] for key in record if key not in _NESTED | _ENVIRONMENT
    } == {
        "job": "job",
        "started": "2026-11-14T08:00:01+00:00",
        "finished": "2026-11-14T09:30:00+00:00",
        "speakers": 1,
        "cost_usd": {"transcription": 2.5, "reconciliation": 1.0, "total": 3.5},
        "warnings": ["part 2: TAIL_GAP"],
    }
    assert record["settings"] == {
        "chunk_minutes": 30,
        "denoise": True,
        "write_protocol": True,
        "thinking_level": "high",
        "transcript": config.transcript.model_dump(),
        "meeting": config.meeting.model_dump(),
        "prompts_replaced": [],
    }
    assert record["chunks"] == [
        {
            "index": 1,
            "start": "00:01:00",
            "end": "00:02:05",
            "model": "gemini-4-carbon",
            "attempts": 2,
            "turns": 1,
            "warnings": ["TAIL_GAP"],
            "usage": Usage(input_audio_tokens=1_000_000).model_dump(),
        }
    ]
    assert record["usage"] == {
        "transcription": Usage(input_audio_tokens=1_000_000).model_dump(),
        "polish": Usage().model_dump(),
        "reconciliation": Usage(output_tokens=100_000).model_dump(),
    }


def test_run_record_without_prices():
    now = datetime(2026, 11, 14, tzinfo=UTC)
    record = run_record(
        "job", Config(), _BACKENDS, [_result()], SpeakerMapping(), now, now
    )
    assert record["cost_usd"]["total"] is None


def test_app_version_unknown(monkeypatch):
    def missing(name):
        raise records.PackageNotFoundError(name)

    monkeypatch.setattr(records, "version", missing)
    assert records._app_version() == "unknown"


def test_run_record_polish():
    config = Config(
        tasks={
            "transcribe": {
                "provider": "gemini-transcribe",
                "model": "gemini-3.5-transcribe",
            }
        },
        transcript={"language": "de-DE"},
        prices={
            "gemini-3.5-transcribe": Price(per_minute=0.005),
            "gemini-4-carbon": Price(input=2, output=10),
        },
    )
    result = _result()
    result.usage = Usage(audio_seconds=600)
    result.text_usage = Usage(output_tokens=100_000)
    now = datetime(2026, 11, 14, tzinfo=UTC)
    record = run_record("job", config, _BACKENDS, [result], SpeakerMapping(), now, now)
    assert record["cost_usd"]["transcription"] == 0.05
    assert record["cost_usd"]["polish"] == 1.0
    assert record["cost_usd"]["total"] == 1.05


def test_run_record_protocol():
    config = Config(
        prices={
            "gemini-4-neon": Price(input=3, output=15),
            "gemini-4-carbon": Price(input=2, output=10),
        }
    )
    mapping = SpeakerMapping(warnings=["RECONCILIATION_FAILED"])
    protocol = ProtocolResult(
        protocol_usage=Usage(output_tokens=100_000),
        agenda_usage=Usage(output_tokens=200_000),
        warnings=["PROTOCOL_FAILED"],
    )
    now = datetime(2026, 11, 14, tzinfo=UTC)
    record = run_record("job", config, _BACKENDS, [], mapping, now, now, protocol)
    assert record["cost_usd"]["protocol"] == 1.0
    assert record["cost_usd"]["agenda"] == 2.0
    assert record["usage"]["agenda"]["output_tokens"] == 200_000
    assert record["warnings"] == ["RECONCILIATION_FAILED", "PROTOCOL_FAILED"]


def test_run_record_settings(tmp_path):
    folder = tmp_path / "prompts"
    folder.mkdir()
    (folder / "polish.md").write_text("$turns $glossary $language $language_notes")
    config = Config(prompts_dir=folder, meeting={"type": "board"})
    now = datetime(2026, 11, 14, tzinfo=UTC)
    settings = run_record("job", config, _BACKENDS, [], SpeakerMapping(), now, now)[
        "settings"
    ]
    assert settings["prompts_replaced"] == ["polish"]
    assert settings["meeting"]["type"] == "board"
    record = run_record("job", Config(), _BACKENDS, [], SpeakerMapping(), now, now)
    assert record["settings"]["prompts_replaced"] == []


def test_run_record_no_price():
    config = Config(tasks={"transcribe": {"model": "gemini-4-neon"}})
    now = datetime(2026, 11, 14, tzinfo=UTC)
    record = run_record(
        "job", config, _BACKENDS, [_result()], SpeakerMapping(), now, now
    )
    assert record["cost_usd"]["total"] is None
    assert record["warnings"] == [
        "NO_PRICE: gemini-4-carbon",
        "NO_PRICE: gemini-4-neon",
        "part 1: TAIL_GAP",
    ]
