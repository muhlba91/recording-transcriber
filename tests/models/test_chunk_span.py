"""Tests for the chunk span."""

from recording_transcriber.models.chunk_span import ChunkSpan


def test_duration():
    assert ChunkSpan(index=0, start_s=10, end_s=70).duration_s == 60
