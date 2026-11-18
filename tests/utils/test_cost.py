"""Tests for the cost estimate."""

from recording_transcriber.configuration.config import Price
from recording_transcriber.models.usage import Usage
from recording_transcriber.utils.cost import estimate_cost


def test_estimate_cost():
    usage = Usage(
        input_text_tokens=500_000,
        input_audio_tokens=500_000,
        output_tokens=50_000,
        thought_tokens=50_000,
    )
    assert estimate_cost(usage, Price(input=2, output=10)) == 3.0


def test_estimate_cost_per_million_tokens():
    usage = Usage(input_text_tokens=1_000_000)
    assert estimate_cost(usage, Price(input=1000)) == 1000


def test_estimate_cost_no_price():
    assert estimate_cost(Usage(), None) is None


def test_estimate_cost_per_minute():
    usage = Usage(audio_seconds=600, input_audio_tokens=15_000)
    assert estimate_cost(usage, Price(per_minute=0.005)) == 0.05
