"""Transcriber cost utils."""

from ..configuration.config import Price
from ..models.usage import Usage


def estimate_cost(usage: Usage, price: Price | None) -> float | None:
    """Return the estimated cost in USD, or None without a price.

    Audio is counted at the text input rate; Google may bill it up to 2x.

    usage: the token usage
    price: the model's price per 1M tokens and per audio minute"""
    if price is None:
        return None
    tokens_in = usage.input_text_tokens + usage.input_audio_tokens
    tokens_out = usage.output_tokens + usage.thought_tokens
    tokens = (tokens_in * price.input + tokens_out * price.output) / 1_000_000
    minutes = usage.audio_seconds / 60 * price.per_minute
    return round(tokens + minutes, 4)  # pragma: no mutate (rounding)
