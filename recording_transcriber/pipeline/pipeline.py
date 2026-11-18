"""The transcription pipeline: prepare, chunk, transcribe, reconcile, render, protocol."""

import json
import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from ..audio.chunker import plan_chunks
from ..audio.ffmpeg import Ffmpeg
from ..backends.base import SpeechBackend, TextBackend, TranscriptionBackend
from ..configuration.config import Config
from ..configuration.job import Job
from ..models.chunk_result import ChunkResult
from ..models.chunk_span import ChunkSpan
from ..models.job_context import JobContext
from ..models.protocol_result import ProtocolResult
from ..models.speaker_mapping import SpeakerMapping
from ..prompts.prompt_renderer import PromptRenderer
from ..render.markdown import (
    document_markdown,
    flags_markdown,
    speakers_markdown,
    transcript_markdown,
)
from ..render.paragraphs import build_paragraphs
from ..render.quality import quality_note
from ..render.records import run_record, transcript_record
from ..render.word import document_docx, transcript_docx
from ..utils.checks import wpm_outliers
from ..utils.const import (
    AGENDA_DOCX,
    AGENDA_MD,
    CHUNK_FILE,
    CHUNK_GLOB,
    FLAGS_MD,
    MAPPING_FILE,
    PLAN_FILE,
    PREPARED_FILE,
    PREPARED_PARTIAL_FILE,
    PREVIOUS_TURNS,
    PROTOCOL_DOCX,
    PROTOCOL_MD,
    RUN_JSON,
    SPEAKERS_MD,
    TASK_AGENDA,
    TASK_POLISH,
    TASK_PROTOCOL,
    TASK_RECONCILE,
    TASK_TRANSCRIBE,
    TRANSCRIPT_DOCX,
    TRANSCRIPT_JSON,
    TRANSCRIPT_MD,
    WARNING_AGENDA,
    WARNING_POLISH,
    WARNING_PROTOCOL,
    WARNING_WPM,
)
from ..utils.serialize import digest, json_text
from .chunk_transcriber import ChunkTranscriber
from .protocol_writer import ProtocolWriter
from .speaker_reconciler import SpeakerReconciler
from .speech_transcriber import SpeechTranscriber
from .turn_polisher import TurnPolisher

_LOGGER = logging.getLogger(__name__)


class Pipeline:
    """Run a job: prepare, chunk, transcribe, reconcile speakers, render, protocol.

    Every finished step is cached in work/, so a rerun resumes where it stopped.
    Audio is only kept in a local scratch folder, never in the job folder."""

    def __init__(
        self,
        config: Config,
        job: Job,
        backends: dict[str, TranscriptionBackend | SpeechBackend | TextBackend],
        ffmpeg: Ffmpeg | None = None,
    ):
        """Initialize the pipeline.

        config: the configuration
        job: the job folder
        backends: the backend of each task the run executes, by task name (the
            transcribe one is a general or speech-to-text model)
        ffmpeg: the ffmpeg wrapper (default: None, found automatically)"""
        self.config = config
        self.job = job
        self.backends = backends
        self.transcription = backends[TASK_TRANSCRIBE]
        self.renderer = PromptRenderer(config)
        self.ffmpeg = ffmpeg or Ffmpeg()
        # set by every run before they are read
        self._sources = []
        self._fingerprint = ""
        self._scratch = Path()

    def run(self) -> bool:
        """Run the job; return False and log an error if it fails. Every backend is
        closed afterwards (Gemini deletes its uploads)."""
        try:
            return self._run()
        finally:
            for backend in {id(b): b for b in self.backends.values()}.values():
                backend.close()

    def _run(self) -> bool:
        """Run the job; see run()."""
        started = datetime.now(UTC)
        self._sources = self.job.audio_files()
        if not self._sources:
            _LOGGER.error("No audio files found in %s.", self.job.audio_dir)
            return False
        self._fingerprint = _fingerprint(self._sources, self.config)
        # pragma: no mutate start (folder name)
        base = (
            self.config.local_dir
            or Path(tempfile.gettempdir()) / "recording-transcriber"
        )
        # pragma: no mutate end
        self._scratch = base / self.job.name / self._fingerprint
        if self._scratch.resolve().is_relative_to(self.job.root.resolve()):
            _LOGGER.error("local_dir must not be inside the job folder: %s.", base)
            return False
        # the app folder (shared drive) only ever holds the maintainer's files
        app = self.config.folder
        if app is not None and any(
            path.resolve().is_relative_to(app.resolve())
            for path in (self.job.root, self._scratch)
        ):
            _LOGGER.error(
                "The job folder and local_dir must be outside the shared app folder "
                "%s; move the job to My Drive/Transcriptions.",
                app,
            )
            return False
        context = self.job.context(self.config.glossary_global)
        if not self._check_backends():
            return False
        results = self._transcribe(context)
        if results is None:
            return False
        mapping = self._reconcile(results, context)
        self._render(results, mapping, context, started)
        # audio is confidential: remove the local copies once the transcript exists;
        # after a failure they stay for the resume
        shutil.rmtree(self._scratch.parent, ignore_errors=True)
        _LOGGER.info("Done: %s.", self.job.out_dir)
        return True

    def _check_backends(self) -> bool:
        """Return whether every backend of the run accepts its model(s); log why not."""
        pairs = {
            (id(self.backends[task]), getattr(self.config.tasks, task).model): (
                self.backends[task],
                getattr(self.config.tasks, task).model,
            )
            for task in self.backends
        }
        return all(backend.check(model) for backend, model in pairs.values())

    def _prepared(self) -> Path | None:
        """Return the prepared recording, converting it on first use."""
        target = self._scratch / PREPARED_FILE
        if target.is_file():
            return target
        _LOGGER.info("Preparing audio: %s file(s).", len(self._sources))
        partial = self._scratch / PREPARED_PARTIAL_FILE
        if not self.ffmpeg.prepare(self._sources, partial, self.config.denoise):
            return None
        partial.replace(target)
        return target

    def _plan(self) -> dict | None:
        """Return the chunk plan (spans and silences), cached in work/."""
        cached = self.job.cached(PLAN_FILE)
        if (
            cached is not None
            and cached.get("fingerprint") == self._fingerprint
            and all(key in cached for key in ("spans", "silences"))
        ):
            return cached
        audio = self._prepared()
        if audio is None:
            return None
        duration = self.ffmpeg.duration(audio)
        if duration is None:
            return None
        silences = self.ffmpeg.silences(audio)
        # pragma: no mutate start (the tests plan their own chunk length)
        spans = plan_chunks(duration, silences, self.config.target_chunk_minutes * 60)
        # pragma: no mutate end
        plan = {
            "fingerprint": self._fingerprint,
            "duration_s": duration,  # pragma: no mutate (only for the record)
            "silences": silences,
            "spans": [span.model_dump() for span in spans],
        }
        self.job.cache(PLAN_FILE, plan)
        return plan

    def _transcribe(self, context: JobContext) -> list[ChunkResult] | None:
        """Transcribe all chunks in order, reusing cached ones.

        context: the job context"""
        plan = self._plan()
        if plan is None:
            return None
        spans = [ChunkSpan.model_validate(span) for span in plan["spans"]]
        transcriber = self._transcriber(context, plan, len(spans))
        inputs_hash = self._inputs_hash(context)
        results = []
        roster = []
        previous = []
        self._drop_stale_chunks(len(spans))
        # a redone chunk changes the roster for every later chunk, so redo those too;
        # speech-to-text chunks are independent
        redone = False
        for span in spans:
            name = CHUNK_FILE.format(index=span.index)
            result = None if redone else self._cached_chunk(name, span, inputs_hash)
            if result is None:
                audio = self._prepared()
                if audio is None:
                    return None
                _LOGGER.info("Transcribing part %s of %s.", span.index + 1, len(spans))
                result = transcriber.transcribe(audio, span, roster, previous)
                if result is None:
                    return None
                result.inputs_hash = inputs_hash
                # json mode only matters for types JSON lacks (none in a chunk today)
                record = result.model_dump(mode="json")
                self.job.cache(name, record)
                redone = not self.config.speech_to_text
            results.append(result)
            roster = result.roster
            previous = result.turns[-PREVIOUS_TURNS:]
        for position in wpm_outliers(results):
            results[position].warnings.append(WARNING_WPM)
        return results

    def _drop_stale_chunks(self, count: int) -> None:
        """Delete cached chunks beyond the current plan (left by an older plan).

        count: the number of planned chunks"""
        for path in self.job.work_dir.glob(CHUNK_GLOB):
            if int(path.stem.split("_")[1]) >= count:
                path.unlink()

    def _transcriber(
        self, context: JobContext, plan: dict, part_count: int
    ) -> ChunkTranscriber | SpeechTranscriber:
        """Return the chunk transcriber for the configured backend.

        Both have transcribe(audio, span, roster, previous) -> ChunkResult | None.

        context: the job context
        plan: the chunk plan
        part_count: the number of chunks"""
        if not self.config.speech_to_text:
            return ChunkTranscriber(
                backend=self.transcription,
                ffmpeg=self.ffmpeg,
                model=self.config.tasks.transcribe.model,
                context=context,
                silences=[tuple(silence) for silence in plan["silences"]],
                part_count=part_count,
                scratch_dir=self._scratch,
                renderer=self.renderer,
            )
        polisher = None
        if TASK_POLISH in self.backends:
            polisher = TurnPolisher(
                self.backends[TASK_POLISH],
                self.config.tasks.polish.model,
                context.glossary,
                self.renderer,
            )
        return SpeechTranscriber(
            backend=self.transcription,
            ffmpeg=self.ffmpeg,
            model=self.config.tasks.transcribe.model,
            vocabulary=context.glossary,
            scratch_dir=self._scratch,
            polisher=polisher,
        )

    def _inputs_hash(self, context: JobContext) -> str:
        """Return a hash of everything that shapes a chunk's answer.

        Covers audio and chunk settings (plan fingerprint), the transcript language,
        the transcribe provider and model, the shared prompt settings and only the
        context and prompts the configured backend uses, so other edits keep the cache.

        context: the job context"""
        transcribe = self.config.tasks.transcribe
        parts = [
            self._fingerprint,
            self.renderer.settings(),
            self.config.thinking_level,
            transcribe.provider,
            transcribe.model,
        ]
        if not self.config.speech_to_text:
            return digest(
                *parts, context.model_dump_json(), self.renderer.template("transcribe")
            )
        # speech-to-text models only take the glossary; the polish pass has its own
        # provider and model
        parts += context.glossary
        if TASK_POLISH in self.backends:
            polish = self.config.tasks.polish
            parts += [
                self.renderer.template("polish"),
                polish.provider,
                self.config.endpoint(polish.provider),
                polish.model,
            ]
        return digest(*parts)

    def _cached_chunk(
        self, name: str, span: ChunkSpan, inputs_hash: str
    ) -> ChunkResult | None:
        """Return a cached chunk if span, model and inputs still match.

        name: the cache file name
        span: the planned chunk
        inputs_hash: the hash of the current prompt inputs"""
        cached = self.job.cached(name)
        if cached is None:
            return None
        try:
            result = ChunkResult.model_validate(cached)
        except ValidationError as error:
            _LOGGER.warning("Ignoring outdated cache file %s: %s.", name, error)
            return None
        if result.span != span or result.inputs_hash != inputs_hash:
            _LOGGER.info("Ignoring cached %s: audio, model or context changed.", name)
            return None
        # a failed polish is cached (the evaluation reads it) but tried again
        if WARNING_POLISH in result.warnings:
            _LOGGER.info(
                "Redoing part %s: the polish failed last time.", span.index + 1
            )
            return None
        _LOGGER.info("Resuming part %s from cache.", span.index + 1)
        return result

    def _reconcile(
        self, results: list[ChunkResult], context: JobContext
    ) -> SpeakerMapping:
        """Return the speaker mapping, cached in work/ for the same chunks.

        A failed reconciliation is not cached, so a rerun tries again.

        results: all transcribed chunks
        context: the job context"""
        settings = self.config.tasks.reconcile
        payload = [result.model_dump(mode="json") for result in results]
        fingerprint = digest(
            json.dumps(payload),
            context.attendees or "",  # pragma: no mutate (any constant works)
            settings.provider,
            self.config.endpoint(settings.provider),
            settings.model,
            self.renderer.template("reconcile_speakers"),
            # also in the chunks' inputs hash, so a change shows in the payload already
            self.renderer.settings(),  # pragma: no mutate
        )
        cached = self.job.cached(MAPPING_FILE)
        if cached is not None:
            try:
                mapping = SpeakerMapping.model_validate(cached)
            except ValidationError as error:
                _LOGGER.warning("Ignoring outdated %s: %s.", MAPPING_FILE, error)
                mapping = None
            if mapping is not None and mapping.fingerprint == fingerprint:
                _LOGGER.info("Using cached speaker mapping.")
                return mapping
        _LOGGER.info("Reconciling speakers.")
        reconciler = SpeakerReconciler(
            self.backends[TASK_RECONCILE],
            settings.model,
            context.attendees,
            self.renderer,
        )
        mapping = reconciler.reconcile(results)
        mapping.fingerprint = fingerprint
        if not mapping.warnings:
            record = mapping.model_dump(mode="json")
            self.job.cache(MAPPING_FILE, record)
        return mapping

    def _render(
        self,
        results: list[ChunkResult],
        mapping: SpeakerMapping,
        context: JobContext,
        started: datetime,
    ) -> None:
        """Write all files to out/: the transcript files first, so a failing or
        interrupted protocol step never costs the transcript; run.json last.

        results: all transcribed chunks
        mapping: the speaker mapping
        context: the job context
        started: the run start"""
        # model order within a chunk; timestamps may be imperfect and are flagged
        turns = [turn for result in results for turn in result.turns]
        speaker_ids = list(dict.fromkeys(turn.speaker_id for turn in turns))
        labels = mapping.labels(speaker_ids, self.config.transcript.speaker_label)
        paragraphs = build_paragraphs(turns, labels)
        title = self.job.name
        note = None  # pragma: no mutate (the renderers treat "" like None)
        if self.config.transcript.quality_note:
            note = quality_note(
                paragraphs,
                results,
                mapping,
                self.config.transcript.unclear_marker,
                self.config.accuracy_estimates.get(self.config.tasks.transcribe.model),
            )
        outputs = {
            TRANSCRIPT_MD: transcript_markdown(title, paragraphs, note),
            TRANSCRIPT_DOCX: transcript_docx(title, paragraphs, note),
            TRANSCRIPT_JSON: json_text(transcript_record(title, paragraphs, mapping)),
            FLAGS_MD: flags_markdown(
                title,
                paragraphs,
                results,
                mapping,
                labels,
                self.config.transcript.unclear_marker,
            ),
            SPEAKERS_MD: speakers_markdown(
                title, mapping, labels, results, context.attendees
            ),
        }
        for name, content in outputs.items():
            self.job.write_output(name, content)
        protocol = None
        if self.config.write_protocol:
            protocol = self._protocol(results, labels, context)
        else:
            self._remove_documents()
        record = run_record(
            job=title,
            config=self.config,
            backends={
                task: backend.describe() for task, backend in self.backends.items()
            },
            results=results,
            mapping=mapping,
            started=started,
            finished=datetime.now(UTC),
            protocol=protocol,
        )
        self.job.write_output(RUN_JSON, json_text(record))

    def _protocol(
        self, results: list[ChunkResult], labels: dict[str, str], context: JobContext
    ) -> ProtocolResult:
        """Write the protocol and the agenda to out/ and return the step's result.

        A failed document leaves no file behind, so an older one never looks current.
        Any error is logged: the transcript is written, so the run still succeeds.

        results: all transcribed chunks
        labels: the display label per speaker id
        context: the job context"""
        _LOGGER.info("Writing the protocol and the agenda.")
        self._remove_documents()
        try:
            writer = ProtocolWriter(
                self.backends[TASK_PROTOCOL],
                self.backends[TASK_AGENDA],
                self.job,
                self.config,
                context,
                self.job.examples(),
            )
            parts = [build_paragraphs(result.turns, labels) for result in results]
            protocol = writer.write(parts)
            documents = {
                (PROTOCOL_MD, PROTOCOL_DOCX): protocol.protocol,
                (AGENDA_MD, AGENDA_DOCX): protocol.agenda,
            }
            for (markdown_name, docx_name), document in documents.items():
                if document is not None:
                    self.job.write_output(markdown_name, document_markdown(document))
                    self.job.write_output(docx_name, document_docx(document))
        # the protocol is a bonus: never fail the run, skip run.json or keep audio
        except Exception:
            _LOGGER.exception("Could not write the protocol and the agenda.")
            self._remove_documents()
            return ProtocolResult(warnings=[WARNING_PROTOCOL, WARNING_AGENDA])
        return protocol

    def _remove_documents(self) -> None:
        """Delete the protocol and agenda files of an earlier run from out/."""
        for name in (PROTOCOL_MD, PROTOCOL_DOCX, AGENDA_MD, AGENDA_DOCX):
            (self.job.out_dir / name).unlink(missing_ok=True)


def _fingerprint(sources: list[Path], config: Config) -> str:
    """Return a short hash of the inputs that decide the chunk plan.

    sources: the recordings
    config: the configuration"""
    parts = [f"{path.name}:{path.stat().st_size}" for path in sources]
    parts += [
        f"chunk_minutes={config.target_chunk_minutes}",
        f"denoise={config.denoise}",
    ]
    # the hash length is tuning
    return digest(*parts, length=12)  # pragma: no mutate
