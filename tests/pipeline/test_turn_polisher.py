"""Tests for the clean-up pass."""

import json

from recording_transcriber.models.model_response import ModelResponse
from recording_transcriber.models.turn import Turn
from recording_transcriber.pipeline.turn_polisher import TurnPolisher
from recording_transcriber.utils.const import MAX_ATTEMPTS
from tests.fakes import FakeBackend, completed, polish_answer
from tests.pipeline.recording import RecordingBackend

_TURNS = [
    Turn(speaker_id="P1-a", start_s=0, text="äh Guten Morgen"),
    Turn(speaker_id="P1-b", start_s=5, text="ähm"),
    Turn(speaker_id="P1-a", start_s=9, text="Finerenone"),
]


def test_polish():
    backend = RecordingBackend(
        completions=[completed(polish_answer("Guten Morgen.", "", "Finerenon."))]
    )
    turns, usage = TurnPolisher(backend, "m", ["Finerenon", "KDIGO"]).polish(_TURNS)
    assert [(t.speaker_id, t.start_s, t.text) for t in turns] == [
        ("P1-a", 0, "Guten Morgen."),
        ("P1-a", 9, "Finerenon."),
    ]
    assert usage.output_tokens == 10
    assert backend.calls == ["m"]
    prompt = backend.prompts[0]
    assert "Finerenon" in prompt and "KDIGO" in prompt
    # the numbers are what the answer refers to
    assert all(
        line in prompt for line in ("0: äh Guten Morgen", "1: ähm", "2: Finerenone")
    )


def test_polish_keeps_skipped_turns():
    answer = json.dumps({"turns": [{"index": 0, "text": "Guten Morgen."}]})
    backend = FakeBackend(completions=[completed(answer)])
    turns, _ = TurnPolisher(backend, "m", []).polish(_TURNS)
    assert [turn.text for turn in turns] == ["Guten Morgen.", "ähm", "Finerenone"]
    assert "(none)" in backend.prompts[0]


def test_polish_retries_invalid_answer():
    backend = FakeBackend(
        completions=[
            completed("not json"),
            ModelResponse(text="", status="incomplete"),
            None,
            completed(polish_answer("a", "b", "c")),
        ]
    )
    turns, _ = TurnPolisher(backend, "m", []).polish(_TURNS)
    assert [turn.text for turn in turns] == ["a", "b", "c"]


def test_polish_fails():
    backend = FakeBackend(completions=[completed("{}")] * MAX_ATTEMPTS)
    turns, usage = TurnPolisher(backend, "m", []).polish(_TURNS)
    assert turns is None
    assert usage.output_tokens == 10 * MAX_ATTEMPTS
