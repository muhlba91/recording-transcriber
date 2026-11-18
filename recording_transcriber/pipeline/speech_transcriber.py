"""Transcribe one chunk with a speech-to-text model."""

import logging
import re
from pathlib import Path

from ..audio.ffmpeg import Ffmpeg
from ..backends.base import SpeechBackend
from ..models.chunk_result import ChunkResult
from ..models.chunk_span import ChunkSpan
from ..models.roster_entry import RosterEntry
from ..models.segment import Segment
from ..models.speech_result import SpeechResult
from ..models.turn import Turn
from ..models.usage import Usage
from ..utils.checks import chunk_problems
from ..utils.const import (
    MAX_ATTEMPTS,
    SENTENCE_END,
    SPEECH_NO_SPEAKER,
    SPEECH_SPEAKER_ID,
    SPEECH_TURN_S,
    WARNING_POLISH,
)
from ..utils.retry import wait_before
from .turn_polisher import TurnPolisher

_LOGGER = logging.getLogger(__name__)


class SpeechTranscriber:
    """Transcribe chunks with a speech-to-text model (evaluation candidates C5, C6,
    C9, C10).

    Speaker labels only hold within one call, so every id carries the part number and
    the speaker reconciliation links them. A verbatim answer is polished by a text
    model afterwards."""

    def __init__(
        self,
        backend: SpeechBackend,
        ffmpeg: Ffmpeg,
        model: str,
        vocabulary: list[str],
        scratch_dir: Path,
        polisher: TurnPolisher | None = None,
    ):
        """Initialize the transcriber.

        backend: the speech-to-text backend
        ffmpeg: the ffmpeg wrapper used to cut the chunk audio
        model: the model id
        vocabulary: terms to prefer, if the model takes them
        scratch_dir: the local folder for chunk audio (never the job folder)
        polisher: polishes verbatim answers (default: None, no polish)"""
        self.backend = backend
        self.ffmpeg = ffmpeg
        self.model = model
        self.vocabulary = vocabulary
        self.scratch_dir = scratch_dir
        self.polisher = polisher

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
        roster: unused; speech-to-text labels only hold within one call
        previous: unused; speech-to-text models take no context"""
        clip = self.scratch_dir / f"chunk_{span.index:02d}_{span.start_s:.0f}.mp3"
        if not self.ffmpeg.extract(audio, span.start_s, span.end_s, clip):
            return None
        result, attempts = self._call(clip)
        if result is None:
            _LOGGER.error("Could not transcribe part %s.", span.index + 1)
            return None
        turns = build_turns(result.segments, span)
        warnings = []
        text_usage = Usage()
        if self.polisher is not None and turns:
            polished, text_usage = self.polisher.polish(turns)
            if polished is None:
                warnings.append(WARNING_POLISH)
            else:
                turns = polished
        speaker_ids = list(dict.fromkeys(turn.speaker_id for turn in turns))
        timed = any(segment.start_s is not None for segment in result.segments)
        return ChunkResult(
            span=span,
            model=self.model,
            turns=turns,
            # labels only hold within one call, so earlier parts' ids are not carried
            roster=[RosterEntry(speaker_id=speaker_id) for speaker_id in speaker_ids],
            # speech-to-text models bill per audio minute
            usage=result.usage.add(Usage(audio_seconds=span.duration_s)),
            text_usage=text_usage,
            attempts=attempts,
            warnings=warnings
            + chunk_problems(turns, span.start_s, span.duration_s, timed=timed),
        )

    def _call(self, clip: Path) -> tuple[SpeechResult | None, int]:
        """Call the backend with retries; return the result (None if every call
        failed) and the number of attempts.

        clip: the chunk audio"""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            wait_before(attempt)
            result = self.backend.transcribe_speech(clip, self.model, self.vocabulary)
            if result is not None:
                return result, attempt
        return None, MAX_ATTEMPTS


def build_turns(segments: list[Segment], span: ChunkSpan) -> list[Turn]:
    """Merge segments into turns: a new turn on a speaker change, or at a sentence end
    once the turn is SPEECH_TURN_S long.

    An untimed segment gets the time of the turn before it (the chunk start first).

    segments: the segments, timed from the clip start
    span: the chunk the clip was cut from"""
    turns = []
    offset = 0.0
    for segment in segments:
        offset = segment.start_s if segment.start_s is not None else offset
        speaker_id = SPEECH_SPEAKER_ID.format(
            part=span.index + 1,
            label=re.sub(r"\W", "", segment.speaker or "") or SPEECH_NO_SPEAKER,
        )
        previous = turns[-1] if turns else None
        if (
            previous is not None
            and previous.speaker_id == speaker_id
            and not (
                previous.text.endswith(SENTENCE_END)
                and span.start_s + offset - previous.start_s  # pragma: no mutate
                >= SPEECH_TURN_S
            )
        ):
            previous.text = f"{previous.text} {segment.text}"
        else:
            turns.append(
                Turn(
                    speaker_id=speaker_id,
                    start_s=span.start_s + offset,
                    text=segment.text,
                )
            )
    return turns
