"""Fake backends that replay queued, hand-built responses."""

import json

from recording_transcriber.configuration.config import Config
from recording_transcriber.models.model_response import ModelResponse
from recording_transcriber.models.usage import Usage
from recording_transcriber.utils.const import TASK_TRANSCRIBE


def backends(transcription, text=None, config=None) -> dict:
    """Return a backend per task the configuration executes: the transcription one
    for transcribe, and one for the text tasks.

    transcription: the backend of the transcribe task
    text: the backend of the text tasks (default: None, the transcription backend)
    config: the configuration (default: None, the defaults)"""
    tasks = (config or Config()).used_tasks()
    return {
        TASK_TRANSCRIBE: transcription,
        **{task: text or transcription for task in tasks if task != TASK_TRANSCRIBE},
    }


def chunk_answer(turns, roster=None, unclear=None) -> str:
    """Return a chunk answer in the model's JSON format.

    turns: (speaker_id, MM:SS, text) tuples
    roster: (speaker_id, name, evidence, voice) tuples
    unclear: (MM:SS, note) tuples"""
    return json.dumps(
        {
            "turns": [
                {"speaker_id": s, "start": start, "text": text}
                for s, start, text in turns
            ],
            "roster": [
                {"speaker_id": s, "name": n, "name_evidence": e, "voice": v}
                for s, n, e, v in roster or []
            ],
            "unclear": [
                {"start": start, "note": note} for start, note in unclear or []
            ],
        }
    )


def completed(text: str, tokens: int = 10) -> ModelResponse:
    """Return a completed response.

    text: the answer
    tokens: the output tokens"""
    return ModelResponse(
        text=text,
        status="completed",
        usage=Usage(input_audio_tokens=100, input_text_tokens=50, output_tokens=tokens),
    )


def document_answer(title: str, heading: str = "Thema", text: str = "Text.") -> str:
    """Return a protocol or agenda answer in the model's JSON format.

    title: the document title
    heading: the heading of its one section
    text: the section's one paragraph"""
    return json.dumps(
        {
            "title": title,
            "sections": [{"heading": heading, "paragraphs": [text], "bullets": []}],
        }
    )


class FakeBackend:
    """Replay queued responses; records every call."""

    def __init__(self, transcriptions=None, completions=None, check=True):
        """Initialize the fake.

        transcriptions: responses for transcribe(), in order (callables get the prompt)
        completions: responses for complete(), in order
        check: the result of check()"""
        self.transcriptions = list(transcriptions or [])
        self.completions = list(completions or [])
        self.check_result = check
        self.prompts = []
        self.closed = False

    def describe(self) -> dict:
        """Return fake provider metadata."""
        return {"provider": "fake", "endpoint": "none", "logging": "none"}

    def check(self, model):
        """Return the configured check result."""
        return self.check_result

    def transcribe(self, audio, prompt, schema, model):
        """Return the next queued transcription."""
        assert audio.is_file()
        self.prompts.append(prompt)
        response = self.transcriptions.pop(0)
        return response(prompt) if callable(response) else response

    def complete(self, prompt, schema, model):
        """Return the next queued completion."""
        self.prompts.append(prompt)
        return self.completions.pop(0)

    def close(self):
        """Remember that close() was called."""
        self.closed = True


class FakeSpeechBackend:
    """Replay queued speech-to-text results; records every call."""

    def __init__(self, results=None):
        """Initialize the fake.

        results: SpeechResult or None per call, in order"""
        self.results = list(results or [])
        self.calls = []
        self.closed = False

    def describe(self) -> dict:
        """Return fake provider metadata."""
        return {"provider": "fake-speech", "endpoint": "none", "logging": "none"}

    def check(self, model):
        """Accept every model."""
        return True

    def transcribe_speech(self, audio, model, vocabulary):
        """Return the next queued result."""
        assert audio.is_file()
        self.calls.append((model, vocabulary))
        return self.results.pop(0)

    def close(self):
        """Remember that close() was called."""
        self.closed = True


def polish_answer(*texts) -> str:
    """Return a clean-up answer in the model's JSON format.

    texts: the cleaned text per turn index"""
    return json.dumps(
        {"turns": [{"index": i, "text": text} for i, text in enumerate(texts)]}
    )


def write_candidate(folder, turns, run=None, labels=None):
    """Write a candidate run: out/transcript.json, work/chunks.json with one chunk,
    work/chunk_00.json and out/run.json.

    folder: the candidate folder
    turns: (speaker_id, start_s, text) tuples
    run: the run.json content (default: None, not written)
    labels: the display label per speaker id (default: None, the id itself)"""
    labels = labels or {}
    (folder / "out").mkdir(parents=True, exist_ok=True)
    (folder / "work").mkdir(parents=True, exist_ok=True)
    paragraphs = [
        {
            "start_s": start,
            "speaker": labels.get(speaker, speaker),
            "speaker_ids": [speaker],
            "text": text,
        }
        for speaker, start, text in turns
    ]
    (folder / "out" / "transcript.json").write_text(
        json.dumps({"paragraphs": paragraphs})
    )
    chunk = {
        "turns": [
            {"speaker_id": speaker, "start_s": start, "text": text}
            for speaker, start, text in turns
        ]
    }
    (folder / "work" / "chunk_00.json").write_text(json.dumps(chunk))
    (folder / "work" / "chunks.json").write_text(json.dumps({"spans": [{}]}))
    if run is not None:
        (folder / "out" / "run.json").write_text(json.dumps(run))


def candidate_stats(cost, eligible=True, speakers=True, signoff=False, disqualified=()):
    """Return the data of one candidate as the report data holds it.

    cost: the cost per hour (None: unknown)"""
    return {
        "label": "x",
        "eligible": eligible,
        "speakers": speakers,
        "signoff": signoff,
        "disqualified": list(disqualified),
        "coverage": 99.0,
        "cost_per_hour": cost,
        "cost_score": 100 * 0.3 / cost if cost else 0.0,
        "wall_s": 600,
    }
