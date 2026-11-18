"""Tests for the timestamp utils."""

import pytest

from recording_transcriber.utils.timestamps import (
    format_minutes,
    format_timestamp,
    parse_timestamp,
)


@pytest.mark.parametrize(
    ("seconds", "text"),
    [(3725.9, "01:02:05"), (3600, "01:00:00"), (7202, "02:00:02"), (59, "00:00:59")],
)
def test_format_timestamp(seconds, text):
    assert format_timestamp(seconds) == text


def test_format_timestamp_negative():
    assert format_timestamp(-3) == "00:00:00"


def test_format_minutes():
    assert format_minutes(3725) == "62:05"
    assert format_minutes(0) == "00:00"
    assert format_minutes(-3) == "00:00"


def test_parse_timestamp():
    assert parse_timestamp("01:30") == 90
    assert parse_timestamp("1:02:03") == 3723
    assert parse_timestamp(" 45.5 ") == 45.5


def test_parse_timestamp_invalid():
    assert parse_timestamp("abc") is None
    assert parse_timestamp("1:2:3:4") is None
    assert parse_timestamp("-1:00") is None
