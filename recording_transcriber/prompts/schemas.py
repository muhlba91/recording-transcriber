"""JSON schemas for structured model output."""

from ..utils.const import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM

_STRING = {"type": "string"}

# timestamps are MM:SS strings: models get those right more reliably than seconds
CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker_id": _STRING,
                    "start": _STRING,
                    "text": _STRING,
                },
                "required": ["speaker_id", "start", "text"],
            },
        },
        "roster": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speaker_id": _STRING,
                    "name": _STRING,
                    "name_evidence": _STRING,
                    "voice": _STRING,
                },
                "required": ["speaker_id", "name", "name_evidence", "voice"],
            },
        },
        "unclear": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"start": _STRING, "note": _STRING},
                "required": ["start", "note"],
            },
        },
    },
    "required": ["turns", "roster", "unclear"],
}

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "speakers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ids": {"type": "array", "items": _STRING},
                    "name": _STRING,
                    "aliases": {"type": "array", "items": _STRING},
                    "confidence": {
                        "type": "string",
                        "enum": [CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW],
                    },
                    "evidence": _STRING,
                },
                "required": ["ids", "name", "aliases", "confidence", "evidence"],
            },
        },
    },
    "required": ["speakers"],
}

POLISH_SCHEMA = {
    "type": "object",
    "properties": {
        "turns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"index": {"type": "integer"}, "text": _STRING},
                "required": ["index", "text"],
            },
        },
    },
    "required": ["turns"],
}

# protocol parts, the combined protocol and the agenda share one plain-text structure
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": _STRING,
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": _STRING,
                    "paragraphs": {"type": "array", "items": _STRING},
                    "bullets": {"type": "array", "items": _STRING},
                },
                "required": ["heading", "paragraphs", "bullets"],
            },
        },
    },
    "required": ["title", "sections"],
}
