"""Tests for the job folder."""

import zipfile

import pytest
from docx import Document

from recording_transcriber.configuration import job as job_module
from recording_transcriber.configuration.job import Job
from recording_transcriber.utils.const import MAX_GLOSSARY_TERMS


class TestJob:
    @pytest.fixture
    def job(self, tmp_path):
        root = tmp_path / "jobs" / "board-2026"
        (root / "audio").mkdir(parents=True)
        (root / "context").mkdir()
        yield Job(root)

    def test_name(self, job):
        assert job.name == "board-2026"

    def test_audio_files(self, job):
        for name in ("b.MP3", "a.m4a", "notes.txt"):
            (job.audio_dir / name).write_text("x")
        assert [path.name for path in job.audio_files()] == ["a.m4a", "b.MP3"]

    def test_audio_files_no_folder(self, tmp_path):
        assert Job(tmp_path / "missing").audio_files() == []

    def test_context(self, job, tmp_path):
        shared = tmp_path / "glossary_global.txt"
        shared.write_text("KDIGO\nUACR\n")
        (job.context_dir / "glossary.txt").write_text(
            "# comment\n  # indented\nUACR\n\nFinerenon\n"
        )
        (job.context_dir / "attendees.txt").write_text(
            "# seed\nDr. Huber, Moderation\n"
        )
        (job.context_dir / "agenda.txt").write_text("# seed\n1. Opening\n")
        context = job.context(shared)
        assert context.glossary == ["KDIGO", "UACR", "Finerenon"]
        assert context.attendees == "Dr. Huber, Moderation"
        assert context.agenda == "1. Opening"

    @pytest.mark.parametrize("extra", [0, 5])
    def test_context_glossary_limit(self, job, extra):
        terms = "\n".join(f"term{index}" for index in range(MAX_GLOSSARY_TERMS + extra))
        (job.context_dir / "glossary.txt").write_text(terms)
        assert len(job.context().glossary) == MAX_GLOSSARY_TERMS

    def test_context_empty_file(self, job):
        (job.context_dir / "agenda.txt").write_text("  \n")
        assert job.context().agenda is None

    def test_cache_roundtrip(self, job):
        job.cache("chunk_00.json", {"text": "Spital", "ä": [1]})
        assert job.cached("chunk_00.json") == {"text": "Spital", "ä": [1]}
        # UTF-8 on disk, not \u escapes
        assert "ä" in (job.work_dir / "chunk_00.json").read_text(encoding="utf-8")
        assert not list(job.work_dir.glob(".*.tmp"))

    def test_cached_missing(self, job):
        assert job.cached("missing.json") is None

    def test_cached_not_an_object(self, job):
        job.cache("list.json", [1])
        assert job.cached("list.json") is None

    def test_cached_broken(self, job):
        job.work_dir.mkdir()
        (job.work_dir / "broken.json").write_text("{")
        assert job.cached("broken.json") is None

    def test_write_output(self, job):
        job.write_output("a.md", "text")
        job.write_output("b.docx", b"bytes")
        assert (job.out_dir / "a.md").read_text() == "text"
        assert (job.out_dir / "b.docx").read_bytes() == b"bytes"

    def test_audio_files_natural_order(self, job):
        job.audio_dir.mkdir(parents=True, exist_ok=True)
        for name in ("Teil 10.mp3", "Teil 2.mp3", "teil 1.mp3"):
            (job.audio_dir / name).write_text("x")
        assert [path.name for path in job.audio_files()] == [
            "teil 1.mp3",
            "Teil 2.mp3",
            "Teil 10.mp3",
        ]


class TestJobEncoding:
    @pytest.fixture
    def job(self, tmp_path):
        (tmp_path / "context").mkdir()
        yield Job(tmp_path)

    def test_context_bom(self, job):
        (job.context_dir / "attendees.txt").write_bytes("﻿# seed\nDr. Huber\n".encode())
        assert job.context().attendees == "Dr. Huber"

    def test_context_cp1252(self, job):
        (job.context_dir / "glossary.txt").write_bytes(
            "Hyperkaliämie\n".encode("cp1252")
        )
        assert job.context().glossary == ["Hyperkaliämie"]

    def test_context_unmappable_byte(self, job):
        # 0x81 has no meaning in cp1252; the rest of the file stays readable
        (job.context_dir / "attendees.txt").write_bytes(b"Dr. M\xfcller\x81")
        assert job.context().attendees.startswith("Dr. Müller")

    def test_context_missing_global_glossary(self, job, tmp_path):
        assert job.context(tmp_path / "missing.txt").glossary == []

    def _example(self, job, name, heading, text, level=2):
        folder = job.context_dir / "examples"
        folder.mkdir(exist_ok=True)
        document = Document()
        document.add_heading(heading, level=level)
        document.add_paragraph("")
        document.add_paragraph(text)
        document.save(folder / name)

    def test_examples(self, job):
        folder = job.context_dir / "examples"
        self._example(job, "über.docx", "Diskussion", "Text B.")
        self._example(job, "a.docx", "Begrüßung", "Text A.")
        # Titles are headings too; the order is natural ("teil 2" before "teil 10")
        self._example(job, "teil 10.docx", "Titel", "Text 10.", level=0)
        self._example(job, "teil 2.docx", "Kopf", "Text 2.")
        # a valid copy: only the name keeps the lock file out
        (folder / "~$a.docx").write_bytes((folder / "a.docx").read_bytes())
        (folder / "broken.docx").write_text("not a docx")
        assert job.examples() == [
            "## Begrüßung\nText A.",
            "## Kopf\nText 2.",
            "## Titel\nText 10.",
            "## Diskussion\nText B.",
        ]

    def test_examples_broken_xml(self, job):
        self._example(job, "a.docx", "Kopf", "Text.")
        broken = job.context_dir / "examples" / "b.docx"
        source_path = job.context_dir / "examples" / "a.docx"
        with (
            zipfile.ZipFile(source_path) as source,
            zipfile.ZipFile(broken, "w") as target,
        ):
            for item in source.infolist():
                data = source.read(item)
                if item.filename == "word/document.xml":
                    data = b"<w:document"
                target.writestr(item, data)
        assert job.examples() == ["## Kopf\nText."]

    def test_examples_limit(self, job, monkeypatch):
        monkeypatch.setattr(job_module, "MAX_EXAMPLE_CHARS", 11)
        self._example(job, "a.docx", "Kopf", "Text")
        self._example(job, "b.docx", "Kopf", "Zweiter Text.")
        # the first example is cut, the budget is used up, the second is skipped
        assert job.examples() == ["## Kopf\nTex"]

    def test_flat_folder(self, tmp_path):
        job = Job(tmp_path / "board")
        job.root.mkdir()
        (job.root / "rec.mp3").write_text("x")
        (job.root / "attendees.txt").write_text("Dr. A\n")
        assert [path.name for path in job.audio_files()] == ["rec.mp3"]
        assert job.context().attendees == "Dr. A"
        assert not job.finished
        job.write_output("transcript.md", "x")
        assert job.finished

    def test_empty_subfolders_fall_back(self, tmp_path):
        job = Job(tmp_path / "board")
        (job.root / "audio").mkdir(parents=True)
        (job.root / "context").mkdir()
        (job.root / "rec.mp3").write_text("x")
        assert job.audio_dir == job.root
        assert job.context_dir == job.root
        assert [path.name for path in job.audio_files()] == ["rec.mp3"]

    def test_examples_no_folder(self, job):
        assert job.examples() == []
