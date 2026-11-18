"""Tests for the roster utils."""

from recording_transcriber.models.roster_entry import RosterEntry
from recording_transcriber.utils.roster import merge_roster


def test_merge_roster_keeps_left_out_entries():
    new = [RosterEntry(speaker_id="S1", name="A")]
    old = [RosterEntry(speaker_id="S1"), RosterEntry(speaker_id="S2")]
    merged = merge_roster(new, old)
    assert [(e.speaker_id, e.name) for e in merged] == [("S1", "A"), ("S2", None)]
