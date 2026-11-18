"""Transcriber serialization utils."""

import hashlib
import json


def json_text(data: dict) -> str:
    """Return pretty JSON that keeps non-ASCII text readable, with a final newline.

    data: the JSON document"""
    return json.dumps(data, ensure_ascii=False, indent=2) + "\n"


# the cache key format is internal
def digest(*parts: str, length: int = 16) -> str:  # pragma: no mutate block
    """Return a short SHA-256 hex digest of the parts, for cache keys.

    parts: the values that make up the key
    length: the number of hex characters (default: 16)"""
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:length]
