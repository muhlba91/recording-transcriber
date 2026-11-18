"""Tests for the speaker mapping."""

from recording_transcriber.models.speaker_assignment import SpeakerAssignment
from recording_transcriber.models.speaker_mapping import SpeakerMapping

_LABEL = "Sprecher {number}"


def test_labels():
    mapping = SpeakerMapping(
        speakers=[
            SpeakerAssignment(ids=["S1", "S4"], name="Dr. Eva Lang", confidence="high"),
            SpeakerAssignment(ids=["S2"], name="Dr. Huber", confidence="low"),
            SpeakerAssignment(ids=["S3"], name=None, confidence="medium"),
        ]
    )
    labels = mapping.labels(["S2", "S1", "S3", "S4", "S5"], _LABEL)
    assert labels == {
        "S2": "Sprecher 1",
        "S1": "Dr. Eva Lang",
        "S3": "Sprecher 2",
        "S4": "Dr. Eva Lang",
        "S5": "Sprecher 3",
    }


def test_labels_empty_mapping():
    assert SpeakerMapping().labels(["S1", "S2"], _LABEL) == {
        "S1": "Sprecher 1",
        "S2": "Sprecher 2",
    }


def test_labels_unnamed_group_shares_label():
    mapping = SpeakerMapping(speakers=[SpeakerAssignment(ids=["S1", "S2"])])
    assert mapping.labels(["S1", "S2"], _LABEL) == {
        "S1": "Sprecher 1",
        "S2": "Sprecher 1",
    }


def test_group_of_first_group_wins():
    first = SpeakerAssignment(ids=["S1"], name="A", confidence="high")
    mapping = SpeakerMapping(speakers=[first, SpeakerAssignment(ids=["S1"], name="B")])
    assert mapping.group_of()["S1"] is first
    assert mapping.labels(["S1"], _LABEL) == {"S1": "A"}


def test_is_named():
    mapping = SpeakerMapping(
        speakers=[
            SpeakerAssignment(ids=["S1"], name="A", confidence="medium"),
            SpeakerAssignment(ids=["S2"], name="B", confidence="low"),
        ]
    )
    assert mapping.is_named("S1")
    assert not mapping.is_named("S2")
    assert not mapping.is_named("S3")
