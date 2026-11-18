"""Tests for the serialization utils."""

from recording_transcriber.utils.serialize import digest, json_text


def test_json_text():
    assert json_text({"name": "Jänner"}) == '{\n  "name": "Jänner"\n}\n'


def test_digest():
    # the order and the boundaries of the parts count
    assert digest("a", "b") != digest("b", "a")
    assert digest("a", "b") != digest("ab")
    assert len(digest("a", length=12)) == 12
