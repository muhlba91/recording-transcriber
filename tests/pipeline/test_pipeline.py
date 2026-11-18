"""Tests for the pipeline."""

import json

import pytest
from docx import Document

from recording_transcriber.audio import chunker
from recording_transcriber.audio.ffmpeg import Ffmpeg
from recording_transcriber.configuration.config import Config
from recording_transcriber.configuration.job import Job
from recording_transcriber.models.model_response import ModelResponse
from recording_transcriber.models.segment import Segment
from recording_transcriber.models.speech_result import SpeechResult
from recording_transcriber.pipeline import pipeline as pipeline_module
from recording_transcriber.pipeline.pipeline import Pipeline
from recording_transcriber.utils.const import (
    MAX_ATTEMPTS,
    PREPARED_FILE,
)
from tests.conftest import make_audio
from tests.fakes import (
    FakeBackend,
    FakeSpeechBackend,
    backends,
    chunk_answer,
    completed,
    document_answer,
    polish_answer,
)
from tests.pipeline.changes import changes
from tests.pipeline.recording import RecordingBackend

_MAPPING = json.dumps(
    {
        "speakers": [
            {
                "ids": ["S1"],
                "name": "Dr. Huber",
                "aliases": [],
                "confidence": "high",
                "evidence": "00:00",
            }
        ]
    }
)
_FAILURES = [None] * MAX_ATTEMPTS


def _answer(text: str) -> str:
    return chunk_answer(
        [("S1", "00:00", text), ("S2", "00:02", "Danke.")],
        roster=[("S1", "Dr. Huber", "00:00", "tief"), ("S2", "", "", "hell")],
    )


def _backend(*texts, mapping=True) -> RecordingBackend:
    return RecordingBackend(
        transcriptions=[completed(_answer(text)) for text in texts],
        completions=[completed(_MAPPING)] if mapping else [],
    )


class TestPipeline:
    @pytest.fixture
    def job(self, tmp_path, monkeypatch):
        # short chunks, so the 8 s file has two chunks cut at the silence
        monkeypatch.setattr(
            pipeline_module,
            "plan_chunks",
            lambda duration, silences, target: chunker.plan_chunks(
                duration, silences, 3.5, window_s=1.5
            ),
        )
        root = tmp_path / "jobs" / "board"
        make_audio(
            root / "audio" / "rec.mp3", [("tone", 3), ("silence", 2), ("tone", 3)]
        )
        (root / "context").mkdir()
        yield Job(root)

    @pytest.fixture
    def config(self, tmp_path):
        yield Config(write_protocol=False, local_dir=tmp_path / "scratch")

    @pytest.fixture
    def ran(self, job, config):
        """Run the pipeline once; return the backend."""
        backend = _backend("Teil eins.", "Teil zwei.")
        assert Pipeline(config, job, backends(backend, backend)).run()
        yield backend

    def test_run_outputs(self, ran, job):
        names = sorted(path.name for path in job.out_dir.iterdir())
        assert names == [
            "flags.md",
            "run.json",
            "speakers.md",
            "transcript.docx",
            "transcript.json",
            "transcript.md",
        ]
        transcript = (job.out_dir / "transcript.md").read_text()
        assert transcript.startswith("# board\n")
        assert "Dr. Huber:** Teil eins." in transcript
        assert "Speakers: 1 named; 1 unnamed" in transcript
        assert "Speaker 1:** Danke." in transcript
        assert (
            json.loads((job.out_dir / "transcript.json").read_text())["job"] == "board"
        )
        flags = (job.out_dir / "flags.md").read_text()
        assert flags.startswith("# Review checklist: board")
        assert (job.out_dir / "speakers.md").read_text().startswith("# Speakers: board")
        document = Document(str(job.out_dir / "transcript.docx"))
        assert document.paragraphs[0].text == "board"
        assert "Quality" in document.paragraphs[1].text

    def test_run_record(self, ran, job):
        run = json.loads((job.out_dir / "run.json").read_text())
        assert len(run["chunks"]) == 2
        # the first chunk is cut at the silence, not at the target length
        assert run["chunks"][0]["end"] == "00:00:04"
        assert run["job"] == "board"
        # started and finished carry the UTC offset
        assert run["started"].endswith("+00:00") and run["finished"].endswith("+00:00")

    def test_run_cleans_up(self, ran, job, tmp_path):
        assert ran.closed
        assert "part 2 of 2" in ran.prompts[1]
        # audio never lands in the job folder, and local copies are gone after success
        assert [path.name for path in job.root.rglob("*.mp3")] == ["rec.mp3"]
        assert not (tmp_path / "scratch" / "board").exists()

    def test_run_splits_a_truncated_chunk_at_the_silence(self, job, config):
        truncated = ModelResponse(text='{"tu', status="incomplete")
        backend = RecordingBackend(
            transcriptions=[
                truncated,
                completed(_answer("Eins.")),
                completed(_answer("Zwei.")),
                completed(_answer("Drei.")),
            ],
            completions=[completed(_MAPPING)],
        )
        assert Pipeline(config, job, backends(backend, backend)).run()
        transcript = (job.out_dir / "transcript.md").read_text()
        assert "Eins." in transcript and "Zwei." in transcript
        assert "Drei." in transcript

    def test_run_resumes(self, job, config):
        failing = FakeBackend(
            transcriptions=[completed(_answer("Teil eins.")), *_FAILURES]
        )
        assert not Pipeline(config, job, backends(failing, failing)).run()
        assert (job.work_dir / "chunk_00.json").is_file()
        assert failing.closed
        backend = _backend("Teil zwei.")
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert len(backend.prompts) == 2
        assert "Teil eins." in backend.prompts[0]
        # a second rerun only re-renders: everything comes from the cache
        cached = FakeBackend()
        assert Pipeline(config, job, backends(cached, cached)).run()
        assert cached.prompts == []

    def test_run_model_change_ignores_cache(self, job, config):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        config.tasks.transcribe.model = "gemini-4-argon"
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert len(other.prompts) == 3

    def test_run_prompt_override_is_used_and_ignores_cache(self, job, config, tmp_path):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        folder = tmp_path / "prompts"
        folder.mkdir()
        (folder / "transcribe.md").write_text(
            "Custom for $meeting_type: part $part_number of $part_count. "
            "$part_start $part_duration $attendees $agenda $glossary $roster "
            "$previous $language $language_notes $unclear_marker $writer_role"
        )
        config.prompts_dir = folder
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert other.prompts[0].startswith("Custom for meeting: part 1 of 2.")

    @pytest.mark.parametrize(
        "change",
        changes(
            "model",
            "provider",
            "base_url",
            "same_endpoint",
            "meeting_type",
            "writer_role",
        ),
    )
    def test_run_reconcile_settings_change_ignores_cache(self, job, tmp_path, change):
        config = Config(
            write_protocol=False,
            local_dir=tmp_path / "scratch",
            providers={
                "lab": {"base_url": "https://a/v1", "api_key_env": "K"},
                "lab2": {"base_url": "https://a/v1", "api_key_env": "K"},
            },
            tasks={"reconcile": {"provider": "lab", "model": "m"}},
        )
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        change(config, "reconcile")
        other = _backend("a", "b")
        assert Pipeline(config, job, backends(other, other)).run()
        # the mapping is asked again (its answer is consumed)
        assert other.completions == []

    def test_run_context_change_ignores_cache(self, job, config):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        (job.context_dir / "glossary.txt").write_text("KDIGO\n")
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert "KDIGO" in other.prompts[0]

    def test_run_context_reaches_every_step(self, job, config, tmp_path):
        glossary = tmp_path / "global.txt"
        glossary.write_text("UACR\n")
        config.glossary_global = glossary
        config.tasks.reconcile.model = "other-model"
        (job.root / "attendees.txt").write_text("Dr. Huber\nDr. Müller\n")
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert "UACR" in backend.prompts[0]
        # the attendees help the speaker mapping, which runs on the configured model
        assert "Dr. Huber\nDr. Müller" in backend.prompts[-1]
        assert backend.calls[-1] == "other-model"
        speakers = (job.out_dir / "speakers.md").read_text()
        assert "## Attendees without a matched voice\n\n- Dr. Müller\n" in speakers

    def test_run_reconcile_prompt_override_ignores_cache(self, job, config, tmp_path):
        backend = _backend("a", "b")
        pipeline = Pipeline(config, job, backends(backend, backend))
        assert pipeline.run()
        folder = tmp_path / "prompts"
        folder.mkdir()
        template = pipeline.renderer.template("reconcile_speakers")
        (folder / "reconcile_speakers.md").write_text(f"{template}\nExtra rule.\n")
        config.prompts_dir = folder
        other = _backend(mapping=True)
        assert Pipeline(config, job, backends(other, other)).run()
        # only the mapping is asked again, with the replacement template
        assert len(other.prompts) == 1 and "Extra rule." in other.prompts[0]

    def test_run_redoes_later_chunks(self, job, config):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        (job.work_dir / "chunk_00.json").unlink()
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert len(other.prompts) == 3

    def test_run_outdated_cache_files(self, job, config):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        (job.work_dir / "chunk_00.json").write_text('{"span": 1}')
        (job.work_dir / "mapping.json").write_text('{"speakers": 1}')
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert len(other.prompts) == 3

    def test_run_failed_reconciliation_not_cached(self, job, config):
        backend = FakeBackend(
            transcriptions=[completed(_answer("a")), completed(_answer("b"))],
            completions=_FAILURES,
        )
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert "reconciliation failed" in (job.out_dir / "flags.md").read_text()
        assert not (job.work_dir / "mapping.json").exists()
        retry = FakeBackend(completions=[completed(_MAPPING)])
        assert Pipeline(config, job, backends(retry, retry)).run()
        assert (job.work_dir / "mapping.json").exists()

    def test_run_denoise_change_ignores_cache(self, job, config):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        (job.work_dir / "chunks.json").write_text("[]")
        config.denoise = True
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert len(other.prompts) == 3

    def test_run_language_change_ignores_cache(self, job, config):
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        config.transcript.language = "de-DE"
        config.transcript.speaker_label = "Sprecher {number}"
        other = _backend("c", "d")
        assert Pipeline(config, job, backends(other, other)).run()
        assert len(other.prompts) == 3
        assert "Sprecher 1:** Danke." in (job.out_dir / "transcript.md").read_text()

    def test_run_drops_stale_chunks(self, job, config):
        backend = _backend("a", "b")
        (job.work_dir).mkdir(parents=True)
        for index in (2, 5):
            (job.work_dir / f"chunk_{index:02d}.json").write_text("{}")
        assert Pipeline(config, job, backends(backend, backend)).run()
        # the plan has two chunks, so chunk 2 is already one too many
        assert not (job.work_dir / "chunk_02.json").exists()
        assert not (job.work_dir / "chunk_05.json").exists()
        assert (job.work_dir / "chunk_01.json").exists()

    def test_run_protocol(self, job, tmp_path):
        config = Config(local_dir=tmp_path / "scratch")
        backend = _backend("Teil eins.", "Teil zwei.")
        backend.completions += [
            completed(document_answer("Teil 1")),
            completed(document_answer("Teil 2")),
            completed(document_answer("Protokoll", "Begrüßung")),
            completed(document_answer("Übersicht", "Agenda")),
        ]
        assert Pipeline(config, job, backends(backend, backend)).run()
        names = {path.name for path in job.out_dir.iterdir()}
        assert {"protocol.md", "protocol.docx", "agenda.md", "agenda.docx"} <= names
        assert "## Begrüßung" in (job.out_dir / "protocol.md").read_text()
        # the protocol sees the reconciled names
        # two chunks and the speaker pass come first, then the first protocol part
        assert "Dr. Huber: Teil eins." in backend.prompts[3]
        run = json.loads((job.out_dir / "run.json").read_text())
        assert run["usage"]["protocol"]["output_tokens"] == 30
        assert run["warnings"] == []

    def test_run_protocol_fails(self, job, tmp_path):
        config = Config(local_dir=tmp_path / "scratch")
        backend = _backend("a", "b")
        backend.completions += _FAILURES * 2
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert (job.out_dir / "transcript.md").is_file()
        assert not (job.out_dir / "protocol.md").exists()
        run = json.loads((job.out_dir / "run.json").read_text())
        assert run["warnings"] == ["PROTOCOL_FAILED", "AGENDA_FAILED"]

    def test_run_protocol_error(self, job, tmp_path, monkeypatch):
        def broken(*args):
            raise ValueError("bad text")

        monkeypatch.setattr(pipeline_module, "document_docx", broken)
        job.write_output("agenda.md", "old")
        config = Config(local_dir=tmp_path / "scratch")
        backend = _backend("a", "b")
        backend.completions += [
            completed(document_answer("Teil 1")),
            completed(document_answer("Teil 2")),
            completed(document_answer("Protokoll")),
            completed(document_answer("Übersicht")),
        ]
        assert Pipeline(config, job, backends(backend, backend)).run()
        names = {path.name for path in job.out_dir.iterdir()}
        assert "protocol.md" not in names and "agenda.md" not in names
        run = json.loads((job.out_dir / "run.json").read_text())
        assert run["warnings"] == ["PROTOCOL_FAILED", "AGENDA_FAILED"]
        assert not (tmp_path / "scratch" / "board").exists()

    def test_run_without_protocol_removes_old_one(self, job, config):
        job.write_output("protocol.md", "old")
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert not (job.out_dir / "protocol.md").exists()

    def test_run_quality_note_off(self, job, config):
        config.transcript.quality_note = False
        config.accuracy_estimates = {"gemini-4-hydrogen": "about 95 %"}
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert "Quality notes" not in (job.out_dir / "transcript.md").read_text()

    def test_run_quality_note_estimate(self, job, config):
        config.accuracy_estimates = {"gemini-4-hydrogen": "about 95 %"}
        backend = _backend("a", "b")
        assert Pipeline(config, job, backends(backend, backend)).run()
        assert "about 95 %" in (job.out_dir / "transcript.md").read_text()

    def test_run_local_dir_inside_job(self, job):
        backend = FakeBackend()
        assert not Pipeline(
            Config(write_protocol=False, local_dir=job.root / "tmp"),
            job,
            backends(backend, backend),
        ).run()
        assert not job.out_dir.exists()
        assert not (job.root / "tmp").exists()

    def test_run_job_inside_app_folder(self, job, tmp_path):
        config_path = job.root.parent / "config.yaml"
        config_path.write_text("write_protocol: false\n")
        config = Config.load(config_path)
        config.local_dir = tmp_path / "scratch"
        backend = FakeBackend()
        assert not Pipeline(config, job, backends(backend, backend)).run()
        assert not job.out_dir.exists()
        assert not (tmp_path / "scratch").exists()

    def test_run_no_audio(self, tmp_path, config):
        backend = FakeBackend()
        assert not Pipeline(
            config, Job(tmp_path / "empty"), backends(backend, backend)
        ).run()
        # closed on an early return too
        assert backend.closed

    def test_run_check_fails(self, job, config):
        backend = FakeBackend(check=False)
        assert not Pipeline(config, job, backends(backend, backend)).run()

    def test_run_checks_every_backend_and_model_once(self, job, config):
        checked = []

        class Recording(FakeBackend):
            def check(self, model):
                checked.append((id(self), model))
                return True

        first, second = Recording(transcriptions=_FAILURES), Recording()
        config.write_protocol = True
        config.tasks.reconcile.model = "other"
        # two backends with the same model are both checked
        config.tasks.agenda.model = "gemini-4-hydrogen"
        assert not Pipeline(config, job, backends(first, second)).run()
        assert checked == [
            (id(first), "gemini-4-hydrogen"),
            (id(second), "other"),
            (id(second), "gemini-4-argon"),
            (id(second), "gemini-4-hydrogen"),
        ]
        assert first.closed and second.closed

    def test_run_text_backend_check_fails_early(self, job, config):
        transcription = _backend("a", "b")
        text = FakeBackend(check=False)
        assert not Pipeline(config, job, backends(transcription, text)).run()
        assert transcription.prompts == []
        assert text.closed

    def test_run_plans_once_per_fingerprint(self, job, config, monkeypatch):
        prepared = []
        original = Ffmpeg.prepare

        def spy(self, sources, target, denoise):
            prepared.append(denoise)
            return original(self, sources, target, denoise)

        monkeypatch.setattr(Ffmpeg, "prepare", spy)

        def run():
            backend = _backend("a", "b")
            assert Pipeline(config, job, backends(backend, backend)).run()

        run()
        run()
        # the same inputs reuse the plan; other audio settings plan again
        assert prepared == [False]
        config.denoise = True
        run()
        assert prepared == [False, True]

    def test_run_replans_an_incomplete_plan(self, job, config):
        backend = _backend("a", "b")
        pipeline = Pipeline(config, job, backends(backend, backend))
        assert pipeline.run()
        plan = json.loads((job.work_dir / "chunks.json").read_text())
        for key in ("spans", "silences"):
            partial = {k: v for k, v in plan.items() if k != key}
            (job.work_dir / "chunks.json").write_text(json.dumps(partial))
            other = _backend("a", "b")
            assert Pipeline(config, job, backends(other, other)).run()
            assert key in json.loads((job.work_dir / "chunks.json").read_text())

    def test_run_default_scratch_dir(self, job, tmp_path, monkeypatch):
        monkeypatch.setattr(
            pipeline_module.tempfile, "gettempdir", lambda: str(tmp_path / "tmp")
        )
        config = Config(write_protocol=False)
        backend = _backend("a", "b")
        pipeline = Pipeline(config, job, backends(backend, backend))
        assert pipeline.run()
        assert pipeline._scratch.is_relative_to(tmp_path / "tmp")
        assert not pipeline._scratch.is_relative_to(job.root)

    def test_run_prepare_fails(self, job, config, monkeypatch):
        backend = FakeBackend()
        pipeline = Pipeline(config, job, backends(backend, backend))
        monkeypatch.setattr(pipeline.ffmpeg, "prepare", lambda *args: False)
        assert not pipeline.run()

    def test_run_duration_fails(self, job, config, monkeypatch):
        backend = FakeBackend()
        pipeline = Pipeline(config, job, backends(backend, backend))
        monkeypatch.setattr(pipeline.ffmpeg, "duration", lambda path: None)
        assert not pipeline.run()

    def test_run_prepare_fails_on_resume(self, job, config, monkeypatch):
        failing = FakeBackend(transcriptions=[completed(_answer("a")), *_FAILURES])
        pipeline = Pipeline(config, job, backends(failing, failing))
        assert not pipeline.run()
        (pipeline._scratch / PREPARED_FILE).unlink()
        monkeypatch.setattr(pipeline.ffmpeg, "prepare", lambda *args: False)
        assert not pipeline.run()


def test_run_flags_wpm_outlier(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "wpm_outliers", lambda results: [0])
    job = Job(tmp_path / "board")
    make_audio(job.audio_dir / "rec.mp3", [("tone", 3)])
    backend = _backend("a")
    assert Pipeline(
        Config(write_protocol=False, local_dir=tmp_path / "scratch"),
        job,
        backends(backend, backend),
    ).run()
    assert "WPM_OUTLIER" in (job.out_dir / "run.json").read_text()


def test_fingerprint(tmp_path):
    first = tmp_path / "a.mp3"
    first.write_bytes(b"12")
    renamed = tmp_path / "b.mp3"
    renamed.write_bytes(b"12")
    longer = tmp_path / "c.mp3"
    longer.write_bytes(b"123")
    fingerprints = {
        pipeline_module._fingerprint([first], Config()),
        pipeline_module._fingerprint([renamed], Config()),
        pipeline_module._fingerprint([longer], Config()),
        pipeline_module._fingerprint([first, renamed], Config()),
        pipeline_module._fingerprint([first], Config(chunk_minutes=30)),
        pipeline_module._fingerprint([first], Config(denoise=True)),
    }
    assert len(fingerprints) == 6


class TestPipelineSpeech:
    @pytest.fixture
    def job(self, tmp_path):
        job = Job(tmp_path / "board")
        make_audio(job.audio_dir / "rec.mp3", [("tone", 3)])
        yield job

    def _result(self):
        return SpeechResult(
            segments=[
                Segment(speaker="spk:0", start_s=0, text="äh Guten Morgen."),
                Segment(speaker="spk:1", start_s=2, text="Danke."),
            ]
        )

    def test_run_verbatim(self, job, tmp_path):
        config = Config(
            write_protocol=False,
            tasks={
                "transcribe": {
                    "provider": "gemini-transcribe",
                    "model": "gemini-3.5-transcribe",
                }
            },
            transcript={"language": "de-DE"},
            local_dir=tmp_path / "scratch",
            chunk_minutes=60,
        )
        speech = FakeSpeechBackend([self._result()])
        mapping = json.dumps(
            {
                "speakers": [
                    {
                        "ids": ["P1-spk0"],
                        "name": "Dr. Huber",
                        "aliases": [],
                        "confidence": "high",
                        "evidence": "",
                    }
                ]
            }
        )
        text = FakeBackend(
            completions=[
                completed(polish_answer("Guten Morgen.", "Danke.")),
                completed(mapping),
            ]
        )
        pipeline = Pipeline(config, job, backends(speech, text, config))
        assert pipeline.run()
        transcript = (job.out_dir / "transcript.md").read_text()
        assert "Dr. Huber:** Guten Morgen." in transcript
        run = json.loads((job.out_dir / "run.json").read_text())
        assert run["tasks"]["transcribe"]["provider"] == "gemini-transcribe"
        assert run["tasks"]["polish"]["provider"] == "gemini"
        assert run["usage"]["transcription"]["audio_seconds"] == pytest.approx(
            3, abs=0.2
        )
        assert run["usage"]["polish"]["output_tokens"] == 10
        assert "P1-spk0" in text.prompts[1]
        assert speech.closed

    def test_run_verbatim_uses_glossary_model_and_prompt(self, job, tmp_path):
        glossary = tmp_path / "glossary.txt"
        glossary.write_text("Finerenon\n")
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "polish.md").write_text("Custom polish. $glossary $turns")
        config = Config(
            write_protocol=False,
            tasks={
                "transcribe": {
                    "provider": "gemini-transcribe",
                    "model": "gemini-3.5-transcribe",
                },
                "polish": {"model": "polish-model"},
            },
            transcript={"language": "de-DE"},
            glossary_global=glossary,
            prompts_dir=prompts,
            local_dir=tmp_path / "scratch",
        )
        speech = FakeSpeechBackend([self._result()])
        text = RecordingBackend(
            completions=[
                completed(polish_answer("Guten Morgen.", "Danke.")),
                completed(_MAPPING),
            ]
        )
        assert Pipeline(config, job, backends(speech, text, config)).run()
        # the speech model gets the glossary, the polish step its own model and prompt
        assert "Finerenon" in speech.calls[0][1]
        assert text.calls[0] == "polish-model"
        assert text.prompts[0].startswith("Custom polish. Finerenon")

    def test_run_without_clean_up(self, job, tmp_path):
        config = Config(
            write_protocol=False,
            tasks={"transcribe": {"provider": "mistral", "model": "voxtral"}},
            local_dir=tmp_path / "scratch",
        )
        speech = FakeSpeechBackend([self._result()])
        text = FakeBackend(completions=[completed(_MAPPING)])
        assert Pipeline(config, job, backends(speech, text, config)).run()
        assert len(text.prompts) == 1
        assert "äh Guten Morgen." in (job.out_dir / "transcript.md").read_text()

    def test_attendees_change_asks_for_the_mapping_again(self, job, tmp_path):
        config = Config(
            write_protocol=False,
            tasks={"transcribe": {"provider": "mistral", "model": "voxtral"}},
            local_dir=tmp_path / "scratch",
        )
        speech = FakeSpeechBackend([self._result()])
        text = FakeBackend(completions=[completed(_MAPPING)])
        assert Pipeline(config, job, backends(speech, text, config)).run()
        # the chunks of a speech-to-text model do not depend on the attendees
        (job.root / "attendees.txt").write_text("Dr. Huber\n")
        speech = FakeSpeechBackend()
        text = FakeBackend(completions=[completed(_MAPPING)])
        assert Pipeline(config, job, backends(speech, text, config)).run()
        assert speech.calls == [] and text.completions == []

    def test_backend_change_ignores_cache(self, job, tmp_path):
        config = Config(
            write_protocol=False,
            tasks={"transcribe": {"provider": "mistral", "model": "voxtral"}},
            local_dir=tmp_path / "scratch",
        )
        first = FakeSpeechBackend([self._result()])
        assert Pipeline(
            config,
            job,
            backends(first, FakeBackend(completions=[completed(_MAPPING)])),
        ).run()
        config.tasks.transcribe.provider = "openai"
        second = FakeSpeechBackend([self._result()])
        assert Pipeline(
            config,
            job,
            backends(second, FakeBackend(completions=[completed(_MAPPING)])),
        ).run()
        assert len(second.calls) == 1

    @pytest.mark.parametrize(
        "change",
        changes("model", "provider", "base_url", "transcribe_model", "thinking_level"),
    )
    def test_polish_settings_change_ignores_cache(self, job, tmp_path, change):
        config = Config(
            write_protocol=False,
            tasks={
                "transcribe": {
                    "provider": "gemini-transcribe",
                    "model": "gemini-3.5-transcribe",
                },
                "polish": {"provider": "lab", "model": "m"},
            },
            providers={"lab": {"base_url": "https://a/v1", "api_key_env": "K"}},
            transcript={"language": "de-DE"},
            local_dir=tmp_path / "scratch",
        )
        runs = []
        for _ in range(2):
            speech = FakeSpeechBackend([self._result()])
            text = FakeBackend(
                completions=[completed(polish_answer("a", "b")), completed(_MAPPING)]
            )
            assert Pipeline(config, job, backends(speech, text, config)).run()
            runs.append(len(speech.calls))
            change(config, "polish")
        # without the change the second run would reuse the cached chunk (0 calls)
        assert runs == [1, 1]

    def test_failed_polish_is_retried(self, job, tmp_path):
        config = Config(
            write_protocol=False,
            tasks={
                "transcribe": {
                    "provider": "gemini-transcribe",
                    "model": "gemini-3.5-transcribe",
                }
            },
            transcript={"language": "de-DE"},
            local_dir=tmp_path / "scratch",
        )
        speech = FakeSpeechBackend([self._result()])
        text = FakeBackend(
            completions=[completed("{}")] * MAX_ATTEMPTS + [completed(_MAPPING)]
        )
        assert Pipeline(config, job, backends(speech, text, config)).run()
        assert "not polished" in (job.out_dir / "flags.md").read_text()
        # cached for the evaluation, but a rerun redoes the chunk
        assert (job.work_dir / "chunk_00.json").exists()
        speech = FakeSpeechBackend([self._result()])
        text = FakeBackend(
            completions=[completed(polish_answer("a", "b")), completed(_MAPPING)]
        )
        assert Pipeline(config, job, backends(speech, text, config)).run()
        assert len(speech.calls) == 1
        assert "not polished" not in (job.out_dir / "flags.md").read_text()


def test_speech_chunks_are_not_redone_in_cascade(tmp_path, monkeypatch):
    monkeypatch.setattr(
        pipeline_module,
        "plan_chunks",
        lambda duration, silences, target: chunker.plan_chunks(
            duration, silences, 3.5, window_s=1.5
        ),
    )
    job = Job(tmp_path / "board")
    make_audio(job.audio_dir / "rec.mp3", [("tone", 3), ("silence", 2), ("tone", 3)])
    config = Config(
        write_protocol=False,
        tasks={"transcribe": {"provider": "mistral", "model": "voxtral"}},
        local_dir=tmp_path / "scratch",
    )
    result = SpeechResult(segments=[Segment(speaker="a", start_s=0, text="Hallo.")])

    def run(results):
        speech = FakeSpeechBackend(results)
        text = FakeBackend(completions=[completed(_MAPPING)])
        assert Pipeline(config, job, backends(speech, text, config)).run()
        return speech

    run([result, result])
    (job.work_dir / "chunk_00.json").unlink()
    assert len(run([result]).calls) == 1
