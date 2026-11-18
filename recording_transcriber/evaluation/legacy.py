"""Import the legacy notebook output (candidate C0)."""

import re

# "Dr. Eva Lang: text"; header lines are "---" and "file: <chunk>.mp3"
_SPEAKER_LINE = re.compile(r"^([^:\[\]]{2,60}):\s+(.*)$")
_HEADER = re.compile(r"^(---|file:\s)")


def legacy_record(text: str) -> dict:
    """Return the legacy transcript in the shape of transcript.json, without times.

    Lines without a speaker (e.g. "[Publikum applaudiert]") continue the paragraph.

    text: the legacy transcription.txt"""
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _HEADER.match(line):
            continue
        match = _SPEAKER_LINE.match(line)
        if match is not None:
            speaker = match.group(1).strip()
            paragraphs.append(
                {
                    "start_s": None,
                    "speaker": speaker,
                    "speaker_ids": [speaker],
                    "text": match.group(2).strip(),
                }
            )
        elif paragraphs:
            paragraphs[-1]["text"] += f" {line}"
    return {"job": "legacy", "speakers": [], "paragraphs": paragraphs}
