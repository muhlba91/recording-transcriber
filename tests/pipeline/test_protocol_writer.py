"""Tests for the protocol writer."""

import json

import pytest

from recording_transcriber.configuration.config import Config
from recording_transcriber.configuration.job import Job
from recording_transcriber.models.job_context import JobContext
from recording_transcriber.models.paragraph import Paragraph
from recording_transcriber.pipeline.protocol_writer import ProtocolWriter
from recording_transcriber.utils.const import MAX_ATTEMPTS
from tests.fakes import FakeBackend, completed, document_answer
from tests.pipeline.changes import changes
from tests.pipeline.recording import RecordingBackend

_FAILURES = [None] * MAX_ATTEMPTS


def _parts() -> list[list[Paragraph]]:
    return [
        [
            Paragraph(start_s=0, speaker="Dr. A", speaker_ids=["S1"], text="Eins."),
            Paragraph(start_s=3, speaker="Dr. B", speaker_ids=["S3"], text="Drei."),
        ],
        [],
        [Paragraph(start_s=65, speaker="Sprecher 1", speaker_ids=["S2"], text="Zwei.")],
    ]


class TestProtocolWriter:
    @pytest.fixture
    def job(self, tmp_path):
        yield Job(tmp_path / "board")

    def _writer(
        self, job, backend, examples=None, config=None, context=None
    ) -> ProtocolWriter:
        config = config or Config(transcript={"language": "de-DE", "notes": "Austrian"})
        context = context or JobContext(
            glossary=["UACR", "KDIGO"], attendees="Dr. A, Linz", agenda="TOP 1"
        )
        return ProtocolWriter(backend, backend, job, config, context, examples or [])

    def _backend(self) -> RecordingBackend:
        return RecordingBackend(
            completions=[
                completed(document_answer("Teil 1")),
                completed(document_answer("Teil 3")),
                completed(document_answer("Protokoll", "Begrüßung", "Final.")),
                completed(document_answer("Übersicht", "Agenda")),
            ]
        )

    def test_write(self, job):
        backend = self._backend()
        examples = ["## Kopf\nBeispiel", "Zweites"]
        result = self._writer(job, backend, examples).write(_parts())
        assert result.protocol.sections[0].paragraphs == ["Final."]
        assert result.agenda.title == "Übersicht"
        assert result.warnings == []
        assert result.protocol_usage.output_tokens == 30
        assert result.agenda_usage.output_tokens == 10
        first, third, final, agenda = backend.prompts
        assert "[00:00:00] Dr. A: Eins." in first and "Dr. B: Drei." in first
        assert "Sprecher 1: Zwei." in third
        for data in ("de-DE", "Austrian", "UACR", "TOP 1"):
            assert data in first
        for data in ("Teil 1", "Teil 3", "Beispiel", "Zweites", "Dr. A, Linz"):
            assert data in final
        assert "Eins." in agenda and "Zwei." in agenda and "TOP 1" in agenda
        assert backend.calls == ["gemini-4-carbon"] * 4

    def test_write_uses_cache(self, job):
        self._writer(job, self._backend()).write(_parts())
        cached = FakeBackend()
        result = self._writer(job, cached).write(_parts())
        assert cached.prompts == []
        assert result.protocol.title == "Protokoll"
        assert result.protocol_usage.output_tokens == 30

    @pytest.mark.parametrize(("task", "redone"), [("protocol", 3), ("agenda", 1)])
    @pytest.mark.parametrize(
        "change", changes("model", "provider", "base_url", "same_endpoint")
    )
    def test_write_task_settings_change_is_redone(self, job, task, redone, change):
        config = Config(
            providers={
                "lab": {"base_url": "https://a/v1", "api_key_env": "K"},
                "lab2": {"base_url": "https://a/v1", "api_key_env": "K"},
            },
            tasks={task: {"provider": "lab", "model": "m"}},
        )
        self._writer(job, self._backend(), config=config).write(_parts())
        change(config, task)
        backend = self._backend()
        self._writer(job, backend, config=config).write(_parts())
        # only the calls of the changed task are repeated
        assert len(backend.prompts) == redone

    @pytest.mark.parametrize(
        "change", changes("thinking_level", "meeting_type", "writer_role")
    )
    def test_write_settings_change_is_redone(self, job, change):
        config = Config()
        self._writer(job, self._backend(), config=config).write(_parts())
        change(config, "protocol")
        backend = self._backend()
        self._writer(job, backend, config=config).write(_parts())
        assert len(backend.prompts) == 4

    def test_write_changed_part_is_redone(self, job):
        self._writer(job, self._backend()).write(_parts())
        parts = _parts()
        parts[2][0].text = "Drei."
        backend = FakeBackend(
            completions=[
                completed(document_answer("Teil 3", text="Neu.")),
                completed(document_answer("Protokoll")),
                completed(document_answer("Übersicht")),
            ]
        )
        self._writer(job, backend).write(parts)
        # part 1 comes from the cache; part 3, the join and the agenda are redone
        assert len(backend.prompts) == 3
        assert "Neu." in backend.prompts[1]

    def test_write_failed_part(self, job):
        backend = FakeBackend(
            completions=[*_FAILURES, completed(document_answer("Übersicht"))]
        )
        result = self._writer(job, backend).write(_parts())
        assert result.protocol is None
        assert result.agenda is not None
        assert result.warnings == ["PROTOCOL_FAILED"]
        # the first part failed, so neither the third part nor the join was asked
        assert len(backend.prompts) == MAX_ATTEMPTS + 1
        assert not (job.work_dir / "protocol_part_00.json").exists()

    def test_write_failed_later_part(self, job):
        backend = FakeBackend(
            completions=[
                completed(document_answer("Teil 1")),
                *_FAILURES,
                completed(document_answer("Übersicht")),
            ]
        )
        result = self._writer(job, backend).write(_parts())
        # a protocol with part 3 missing would look complete: no join is asked
        assert result.protocol is None
        assert result.warnings == ["PROTOCOL_FAILED"]
        assert len(backend.prompts) == 1 + MAX_ATTEMPTS + 1

    def test_write_invalid_answers(self, job):
        empty = json.dumps({"title": "x", "sections": []})
        backend = FakeBackend(
            completions=[completed("{}")] * MAX_ATTEMPTS
            + [completed(empty)] * MAX_ATTEMPTS
        )
        result = self._writer(job, backend).write([_parts()[0]])
        assert result.protocol is None and result.agenda is None
        assert result.warnings == ["PROTOCOL_FAILED", "AGENDA_FAILED"]

    def test_write_no_text(self, job):
        backend = FakeBackend()
        result = self._writer(job, backend).write([[], []])
        assert backend.prompts == []
        assert result.warnings == ["PROTOCOL_FAILED", "AGENDA_FAILED"]

    def test_write_outdated_cache(self, job):
        self._writer(job, self._backend()).write(_parts())
        path = job.work_dir / "agenda.json"
        data = json.loads(path.read_text())
        path.write_text(json.dumps({"fingerprint": data["fingerprint"]}))
        backend = FakeBackend(completions=[completed(document_answer("Neu"))])
        result = self._writer(job, backend).write(_parts())
        assert result.agenda.title == "Neu"
        assert len(backend.prompts) == 1

    def test_write_drops_blank_items(self, job):
        answer = json.dumps(
            {
                "title": "Übersicht",
                "sections": [
                    {"heading": " Agenda ", "paragraphs": [" "], "bullets": ["a"]},
                    {"heading": "Leer", "paragraphs": [""], "bullets": []},
                    {"heading": " ", "paragraphs": ["x"], "bullets": []},
                ],
            }
        )
        backend = FakeBackend(completions=[*_FAILURES, completed(answer)])
        result = self._writer(job, backend).write([_parts()[0]])
        assert [s.model_dump() for s in result.agenda.sections] == [
            {"heading": "Agenda", "paragraphs": [], "bullets": ["a"]}
        ]
