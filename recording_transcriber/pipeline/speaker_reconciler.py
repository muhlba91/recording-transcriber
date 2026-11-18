"""Reconcile speaker ids across chunks with one text-only call."""

import json
import logging

from ..backends.base import TextBackend
from ..models.chunk_result import ChunkResult
from ..models.speaker_assignment import SpeakerAssignment
from ..models.speaker_mapping import SpeakerMapping
from ..prompts.prompt_renderer import PromptRenderer
from ..prompts.schemas import MAPPING_SCHEMA
from ..utils.const import (
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    PROMPT_PLACEHOLDER,
    WARNING_RECONCILIATION,
)
from ..utils.retry import complete_with_retry
from ..utils.timestamps import format_timestamp

_LOGGER = logging.getLogger(__name__)


class SpeakerReconciler:
    """Merge drifted speaker ids and name people from evidence in the recording.

    The attendee list is optional and only corrects spellings. The model only returns
    the mapping; the code applies it deterministically."""

    def __init__(
        self,
        backend: TextBackend,
        model: str,
        attendees: str | None,
        renderer: PromptRenderer | None = None,
    ):
        """Initialize the reconciler.

        backend: the text backend
        model: the model id
        attendees: the attendee list, for spelling (may be None)
        renderer: fills the prompt template (default: None, the defaults)"""
        self.backend = backend
        self.model = model
        self.attendees = attendees
        self.renderer = renderer or PromptRenderer()

    def reconcile(self, results: list[ChunkResult]) -> SpeakerMapping:
        """Return the speaker mapping; falls back to the last roster if the call fails.

        results: all transcribed chunks"""
        # pragma: no mutate start (request wiring)
        speakers, usage = complete_with_retry(
            self.backend, self._prompt(results), MAPPING_SCHEMA, self.model, _parse
        )
        # pragma: no mutate end
        if speakers is not None:
            return SpeakerMapping(model=self.model, speakers=speakers, usage=usage)
        _LOGGER.warning("Speaker reconciliation failed; using the roster names.")
        return SpeakerMapping(
            model=self.model,
            speakers=_roster_fallback(results),
            usage=usage,
            warnings=[WARNING_RECONCILIATION],
        )

    def _prompt(self, results: list[ChunkResult]) -> str:
        """Render the reconciliation prompt.

        results: all transcribed chunks"""
        # pragma: no mutate start (prompt layout)
        rosters = "\n\n".join(
            f"Part {result.span.index + 1} "
            f"(from {format_timestamp(result.span.start_s)}):\n"
            + "\n".join(
                json.dumps(entry.model_dump(), ensure_ascii=False)
                for entry in result.roster
            )
            for result in results
        )
        transcript = "\n".join(
            f"[{format_timestamp(turn.start_s)}] {turn.speaker_id}: {turn.text}"
            for result in results
            for turn in result.turns
        )
        # pragma: no mutate end
        return self.renderer.render(
            "reconcile_speakers",
            attendees=self.attendees or PROMPT_PLACEHOLDER,
            rosters=rosters,
            transcript=transcript,
        )


def _parse(text: str) -> list[SpeakerAssignment] | None:
    """Parse the model's JSON mapping; return None if invalid.

    text: the model output"""
    try:
        data = json.loads(text)
        return [
            SpeakerAssignment(
                ids=[speaker_id.strip() for speaker_id in group["ids"]],
                name=group["name"].strip() or None,
                aliases=[alias.strip() for alias in group["aliases"] if alias.strip()],
                confidence=group["confidence"],
                evidence=group["evidence"].strip(),
            )
            for group in data["speakers"]
        ]
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        _LOGGER.warning("Invalid speaker mapping: %s.", error)
        return None


def _roster_fallback(results: list[ChunkResult]) -> list[SpeakerAssignment]:
    """Return one group per roster entry of the last chunk, keeping cited names.

    results: all transcribed chunks"""
    return [
        SpeakerAssignment(
            ids=[entry.speaker_id],
            name=entry.name,
            confidence=CONFIDENCE_MEDIUM
            if entry.name and entry.name_evidence
            else CONFIDENCE_LOW,
            evidence=entry.name_evidence or "",
        )
        for entry in results[-1].roster
    ]
