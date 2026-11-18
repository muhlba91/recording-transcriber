"""Tests for the speaker reconciler."""

import json

from recording_transcriber.models.chunk_result import ChunkResult
from recording_transcriber.models.chunk_span import ChunkSpan
from recording_transcriber.models.roster_entry import RosterEntry
from recording_transcriber.models.turn import Turn
from recording_transcriber.pipeline.speaker_reconciler import SpeakerReconciler
from recording_transcriber.utils.const import MAX_ATTEMPTS
from tests.fakes import FakeBackend, completed
from tests.pipeline.recording import RecordingBackend

_RESULTS = [
    ChunkResult(
        span=ChunkSpan(index=0, start_s=0, end_s=60),
        model="m",
        turns=[Turn(speaker_id="S1", start_s=5, text="Ich bin Dr. Huber.")],
        roster=[
            RosterEntry(speaker_id="S1", name="Dr. Huber", name_evidence="00:05"),
            RosterEntry(speaker_id="S2", name="Dr. Raten"),
        ],
    )
]
_MAPPING = json.dumps(
    {
        "speakers": [
            {
                "ids": ["S1", "S2"],
                "name": "Dr. Maria Huber",
                "aliases": ["Maria", " ", "Frau Dr. Huber"],
                "confidence": "high",
                "evidence": "00:05",
            }
        ]
    }
)


def test_reconcile():
    backend = RecordingBackend(completions=[completed(_MAPPING)])
    mapping = SpeakerReconciler(backend, "m", "Dr. Huber, Linz").reconcile(_RESULTS)
    assert mapping.model == "m"
    assert mapping.speakers[0].ids == ["S1", "S2"]
    assert mapping.speakers[0].aliases == ["Maria", "Frau Dr. Huber"]
    assert mapping.speakers[0].evidence == "00:05"
    assert mapping.usage.output_tokens == 10
    assert backend.calls == ["m"]
    assert "[00:00:05] S1: Ich bin Dr. Huber." in backend.prompts[0]
    assert "Dr. Huber, Linz" in backend.prompts[0]


def test_reconcile_prompt_lists_every_part():
    second = ChunkResult(
        span=ChunkSpan(index=1, start_s=60, end_s=120),
        model="m",
        turns=[
            Turn(speaker_id="S1", start_s=65, text="Weiter."),
            Turn(speaker_id="S2", start_s=70, text="Ja."),
        ],
        roster=[RosterEntry(speaker_id="S1", name="Dr. Müller", voice="tief")],
    )
    backend = FakeBackend(completions=[completed(_MAPPING)])
    SpeakerReconciler(backend, "m", None).reconcile([*_RESULTS, second])
    prompt = backend.prompts[0]
    # the second part's roster is not ASCII-escaped
    assert "Dr. Müller" in prompt
    assert "[00:01:05] S1: Weiter." in prompt and "[00:01:10] S2: Ja." in prompt


def test_reconcile_retries():
    backend = FakeBackend(completions=[None, completed("{}"), completed(_MAPPING)])
    mapping = SpeakerReconciler(backend, "m", None).reconcile(_RESULTS)
    assert mapping.warnings == []
    assert mapping.usage.output_tokens == 20


def test_reconcile_fallback():
    backend = FakeBackend(completions=[*[None] * (MAX_ATTEMPTS - 1), completed("[]")])
    mapping = SpeakerReconciler(backend, "m", None).reconcile(_RESULTS)
    assert mapping.warnings == ["RECONCILIATION_FAILED"]
    assert mapping.model == "m"
    assert mapping.usage.output_tokens == 10
    assert [group.evidence for group in mapping.speakers] == ["00:05", ""]
    assert [(group.name, group.confidence) for group in mapping.speakers] == [
        ("Dr. Huber", "medium"),
        ("Dr. Raten", "low"),
    ]
