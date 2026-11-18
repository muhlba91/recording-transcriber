"""Tests for the usage model."""

from recording_transcriber.models.usage import Usage


def test_add():
    total = Usage(
        input_text_tokens=1,
        input_audio_tokens=2,
        output_tokens=3,
        thought_tokens=4,
        audio_seconds=1.5,
    ).add(
        Usage(
            input_text_tokens=10,
            input_audio_tokens=20,
            output_tokens=30,
            thought_tokens=40,
            audio_seconds=2,
        )
    )
    assert total == Usage(
        input_text_tokens=11,
        input_audio_tokens=22,
        output_tokens=33,
        thought_tokens=44,
        audio_seconds=3.5,
    )
