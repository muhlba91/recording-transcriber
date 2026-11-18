"""Transcriber roster utils."""

from ..models.roster_entry import RosterEntry


def merge_roster(
    roster: list[RosterEntry], previous_roster: list[RosterEntry]
) -> list[RosterEntry]:
    """Return the new roster plus earlier entries the model left out, so no id is lost.

    roster: the roster from the model
    previous_roster: the roster before this chunk"""
    returned = {entry.speaker_id for entry in roster}
    kept = [entry for entry in previous_roster if entry.speaker_id not in returned]
    return roster + kept
