"""Transcriber timestamp utils."""


def format_timestamp(seconds: float) -> str:
    """Format seconds as hh:mm:ss.

    seconds: the time in seconds"""
    total = max(0, int(seconds))
    return f"{total // 3600:02d}:{total % 3600 // 60:02d}:{total % 60:02d}"


def format_minutes(seconds: float) -> str:
    """Format seconds as MM:SS, the format the models use within a chunk.

    seconds: the time in seconds"""
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def parse_timestamp(value: str) -> float | None:
    """Parse SS, MM:SS or H:MM:SS into seconds; return None if invalid.

    value: the timestamp string"""
    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        return None
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        return None
    if any(number < 0 for number in numbers):
        return None
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds
