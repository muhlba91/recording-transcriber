"""Transcribe one chunk with retries, sanity checks and splitting."""

import json
import logging
from pathlib import Path

from ..audio.chunker import best_cut
from ..audio.ffmpeg import Ffmpeg
from ..backends.base import TranscriptionBackend
from ..models.chunk_result import ChunkResult
from ..models.chunk_span import ChunkSpan
from ..models.job_context import JobContext
from ..models.roster_entry import RosterEntry
from ..models.turn import Turn
from ..models.unclear_passage import UnclearPassage
from ..models.usage import Usage
from ..prompts.prompt_renderer import PromptRenderer
from ..prompts.schemas import CHUNK_SCHEMA
from ..utils.checks import chunk_problems
from ..utils.const import (
    MAX_ATTEMPTS,
    MAX_SPLIT_DEPTH,
    PREVIOUS_TURNS,
    PROMPT_PLACEHOLDER,
    STATUS_COMPLETED,
    STATUS_INCOMPLETE,
    WARNING_SPLIT,
)
from ..utils.retry import wait_before
from ..utils.roster import merge_roster
from ..utils.timestamps import format_minutes, format_timestamp, parse_timestamp

_LOGGER = logging.getLogger(__name__)


class ChunkTranscriber:
    """Transcribe chunks of one prepared recording.

    Failed calls are retried with exponential backoff, a schema error and a failed
    sanity check are re-asked once, and a truncated answer splits the chunk in half."""

    def __init__(
        self,
        backend: TranscriptionBackend,
        ffmpeg: Ffmpeg,
        model: str,
        context: JobContext,
        silences: list[tuple[float, float]],
        part_count: int,
        scratch_dir: Path,
        renderer: PromptRenderer | None = None,
    ):
        """Initialize the transcriber.

        backend: the transcription backend
        ffmpeg: the ffmpeg wrapper used to cut the chunk audio
        model: the model id
        context: the job context (glossary, attendees, agenda)
        silences: the silent stretches of the recording, used to split chunks
        part_count: the number of chunks, for the prompt
        scratch_dir: the local folder for chunk audio (never the job folder)
        renderer: fills the prompt template (default: None, the defaults)"""
        self.backend = backend
        self.ffmpeg = ffmpeg
        self.model = model
        self.context = context
        self.silences = silences
        self.part_count = part_count
        self.scratch_dir = scratch_dir
        self.renderer = renderer or PromptRenderer()

    def transcribe(
        self,
        audio: Path,
        span: ChunkSpan,
        roster: list[RosterEntry],
        previous: list[Turn],
    ) -> ChunkResult | None:
        """Transcribe a chunk; return None if it fails after all retries.

        audio: the prepared recording
        span: the chunk to transcribe
        roster: the roster so far
        previous: the last turns of the previous chunk"""
        return self._transcribe(audio, span, roster, previous, depth=0)

    def _transcribe(
        self,
        audio: Path,
        span: ChunkSpan,
        roster: list[RosterEntry],
        previous: list[Turn],
        depth: int,
    ) -> ChunkResult | None:
        """Transcribe a chunk or a split part of it.

        audio: the prepared recording
        span: the time range to transcribe
        roster: the roster so far
        previous: the last turns before this range
        depth: how often the chunk was split already"""
        clip = self.scratch_dir / f"chunk_{span.index:02d}_{span.start_s:.0f}.mp3"
        if not self.ffmpeg.extract(audio, span.start_s, span.end_s, clip):
            return None
        prompt = self._prompt(span, roster, previous)
        usage = Usage()
        flagged = None
        schema_retried = False
        backoff = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            # a schema re-ask follows at once; a failed call waits first
            if backoff:
                wait_before(attempt)
            backoff = False
            # pragma: no mutate start (request wiring)
            response = self.backend.transcribe(clip, prompt, CHUNK_SCHEMA, self.model)
            # pragma: no mutate end
            if response is not None:
                usage = usage.add(response.usage)
            if response is not None and response.status == STATUS_INCOMPLETE:
                if depth >= MAX_SPLIT_DEPTH:
                    _LOGGER.error(
                        "Part %s is still truncated after splitting.", span.index + 1
                    )
                    return None
                _LOGGER.warning("Part %s was truncated; splitting it.", span.index + 1)
                return self._split(audio, span, roster, previous, depth, usage)
            if response is None or response.status != STATUS_COMPLETED:
                backoff = True
                continue
            result = self._parse(response.text, span)
            if result is None:
                if schema_retried:
                    break
                schema_retried = True
                continue
            result.roster = merge_roster(result.roster, roster)
            result.attempts = attempt
            result.usage = usage
            result.warnings = chunk_problems(
                result.turns, span.start_s, span.duration_s
            )
            if not result.warnings:
                return result
            # one retry for a failed sanity check, then keep the better answer, flagged
            if flagged is not None:
                better = min(flagged, result, key=_badness)
                better.usage = usage
                better.attempts = attempt
                return better
            _LOGGER.warning(
                "Part %s failed checks %s; retrying.", span.index + 1, result.warnings
            )
            flagged = result
        if flagged is not None:
            flagged.usage = usage
            return flagged
        _LOGGER.error("Could not transcribe part %s.", span.index + 1)
        return None

    def _split(
        self,
        audio: Path,
        span: ChunkSpan,
        roster: list[RosterEntry],
        previous: list[Turn],
        depth: int,
        usage: Usage,
    ) -> ChunkResult | None:
        """Transcribe both halves of a chunk and merge them.

        audio: the prepared recording
        span: the chunk to split
        roster: the roster so far
        previous: the last turns before the chunk
        depth: how often the chunk was split already
        usage: the usage of the truncated call"""
        quarter = span.duration_s / 4  # pragma: no mutate (tuning)
        cut = best_cut(
            self.silences,
            span.start_s + 2 * quarter,  # pragma: no mutate (tuning)
            span.start_s + quarter,  # pragma: no mutate (tuning)
            span.end_s - quarter,  # pragma: no mutate (tuning)
        )
        first = ChunkSpan(index=span.index, start_s=span.start_s, end_s=cut)
        second = ChunkSpan(index=span.index, start_s=cut, end_s=span.end_s)
        left = self._transcribe(audio, first, roster, previous, depth + 1)
        if left is None:
            return None
        tail = (previous + left.turns)[-PREVIOUS_TURNS:]
        right = self._transcribe(audio, second, left.roster, tail, depth + 1)
        if right is None:
            return None
        return ChunkResult(
            span=span,
            model=self.model,
            turns=left.turns + right.turns,
            roster=right.roster,
            unclear=left.unclear + right.unclear,
            usage=usage.add(left.usage).add(right.usage),
            attempts=left.attempts + right.attempts,
            warnings=[WARNING_SPLIT, *left.warnings, *right.warnings],
        )

    def _prompt(
        self, span: ChunkSpan, roster: list[RosterEntry], previous: list[Turn]
    ) -> str:
        """Render the transcription prompt for a chunk.

        span: the time range to transcribe
        roster: the roster so far
        previous: the last turns before this range"""
        roster_text = "\n".join(
            json.dumps(entry.model_dump(), ensure_ascii=False) for entry in roster
        )
        previous_text = "\n".join(
            f"[{format_timestamp(turn.start_s)}] {turn.speaker_id}: {turn.text}"
            for turn in previous
        )
        # pragma: no mutate start (prompt layout)
        return self.renderer.render(
            "transcribe",
            part_number=str(span.index + 1),
            part_count=str(self.part_count),
            part_start=format_timestamp(span.start_s),
            part_duration=format_minutes(span.duration_s),
            attendees=self.context.attendees or PROMPT_PLACEHOLDER,
            agenda=self.context.agenda or PROMPT_PLACEHOLDER,
            glossary="\n".join(self.context.glossary) or PROMPT_PLACEHOLDER,
            roster=roster_text or "(empty: this is the first part)",
            previous=previous_text or "(none: this is the start of the recording)",
        )
        # pragma: no mutate end

    def _parse(self, text: str, span: ChunkSpan) -> ChunkResult | None:
        """Parse the model's JSON into a ChunkResult with recording-relative times.

        A turn with an unreadable timestamp keeps the previous turn's time and an
        unclear note with one is dropped, so a single typo does not cost a full call.

        text: the model output
        span: the time range that was transcribed"""
        try:
            data = json.loads(text)
            turns = []
            offset = 0.0
            for turn in data["turns"]:
                offset = _seconds(turn["start"], offset)
                if turn["text"].strip():
                    turns.append(
                        Turn(
                            speaker_id=turn["speaker_id"].strip(),
                            start_s=span.start_s + offset,
                            text=turn["text"].strip(),
                        )
                    )
            roster = [
                RosterEntry(
                    speaker_id=entry["speaker_id"].strip(),
                    name=entry["name"].strip() or None,
                    name_evidence=entry["name_evidence"].strip() or None,
                    voice=entry["voice"].strip(),
                )
                for entry in data["roster"]
            ]
            unclear = [
                UnclearPassage(
                    start_s=span.start_s + start,
                    note=passage["note"].strip(),
                )
                for passage in data["unclear"]
                if (start := parse_timestamp(passage["start"])) is not None
            ]
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            _LOGGER.warning("Invalid answer for part %s: %s.", span.index + 1, error)
            return None
        return ChunkResult(
            span=span,
            model=self.model,
            turns=turns,
            roster=roster,
            unclear=unclear,
        )


def _badness(result: ChunkResult) -> tuple[int, int]:
    """Return a sort key: fewer warnings first, then more turns.

    result: a transcribed chunk"""
    return len(result.warnings), -len(result.turns)


def _seconds(value: str, fallback: float) -> float:
    """Return a chunk-relative timestamp in seconds, or the fallback if unreadable.

    Out-of-range values are kept; the sanity check flags them.

    value: the MM:SS timestamp from the model
    fallback: the value for an unreadable timestamp"""
    seconds = parse_timestamp(value)
    return fallback if seconds is None else seconds
