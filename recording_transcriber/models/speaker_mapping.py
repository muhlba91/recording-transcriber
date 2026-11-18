"""The speaker mapping."""

from pydantic import BaseModel

from ..utils.const import NAMED_CONFIDENCE
from .speaker_assignment import SpeakerAssignment
from .usage import Usage


class SpeakerMapping(BaseModel):
    """The reconciled speakers, cached as work/mapping.json."""

    model: str | None = None
    fingerprint: str = ""
    speakers: list[SpeakerAssignment] = []
    usage: Usage = Usage()
    warnings: list[str] = []

    def group_of(self) -> dict[str, SpeakerAssignment]:
        """Return the group per speaker id; a duplicate id keeps its first group."""
        groups = {}
        for group in self.speakers:
            for speaker_id in group.ids:
                groups.setdefault(speaker_id, group)
        return groups

    def is_named(self, speaker_id: str) -> bool:
        """Return whether the speaker id is shown with a name.

        speaker_id: the speaker id"""
        group = self.group_of().get(speaker_id)
        return (
            group is not None
            and bool(group.name)
            and group.confidence in NAMED_CONFIDENCE
        )

    def labels(self, speaker_ids: list[str], unknown: str) -> dict[str, str]:
        """Return the display label per speaker id.

        Named people with high or medium confidence get their name. Everyone else
        gets the unknown label ("Speaker {number}"), numbered by first appearance.

        speaker_ids: all speaker ids in order of first appearance
        unknown: the label template for unnamed speakers"""
        groups = self.group_of()
        labels = {}
        group_labels = {}
        number = 0
        for speaker_id in speaker_ids:
            group = groups.get(speaker_id)
            # ids of one group share a label; an unmapped id is its own group
            key = tuple(group.ids) if group is not None else (speaker_id,)
            if key not in group_labels:
                if self.is_named(speaker_id):
                    group_labels[key] = group.name
                else:
                    number += 1
                    group_labels[key] = unknown.format(number=number)
            labels[speaker_id] = group_labels[key]
        return labels
