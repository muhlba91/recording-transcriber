"""Write the offline report: index.html and the audio clips (design §5)."""

import json
import logging
import random
import secrets
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path
from string import ascii_uppercase

from ..audio.ffmpeg import Ffmpeg
from ..utils.const import (
    CLIP_MAX_S,
    CLIP_PAD_AFTER_S,
    CLIP_PAD_BEFORE_S,
    MEDICAL_WEIGHT,
    PRE_WEIGHTS,
    PREPARED_FILE,
    READABILITY_COUNT,
    REPORT_CLIPS,
    REPORT_PAGE,
    SPOT_S,
    UNKNOWN_PERSON,
    UNKNOWN_PERSON_COUNT,
)

_LOGGER = logging.getLogger(__name__)


def write_report(
    data: dict,
    sources: list[Path],
    attendees: list[str],
    report_dir: Path,
    ffmpeg: Ffmpeg | None = None,
) -> bool:
    """Anonymize the candidates, write the page and cut the clips.

    Adds the report id, the candidate tokens and the variant letters to data, which
    the scoring needs later.

    data: the evaluation data
    sources: the recordings
    attendees: the attendee names, for the speaker dropdown
    report_dir: the folder to write (replaced)
    ffmpeg: the ffmpeg wrapper (default: None, found automatically)"""
    ffmpeg = ffmpeg or Ffmpeg()
    data["report"] = secrets.token_hex(4)
    # any random order works
    shuffle = random.Random(data["report"])
    identifiers = list(data["candidates"])
    shuffle.shuffle(identifiers)
    data["tokens"] = {
        identifier: f"t{n}" for n, identifier in enumerate(identifiers, 1)
    }
    disqualified = {i for i, c in data["candidates"].items() if c["disqualified"]}
    for span in data["spans"]:
        shuffle.shuffle(span["variants"])
        # a variant only disqualified candidates wrote is not worth the reviewer's
        # time; it gets no letter, so it never matches an answer
        shown = [v for v in span["variants"] if set(v["candidates"]) - disqualified]
        for variant in span["variants"]:
            variant["letter"] = ""
        for index, variant in enumerate(shown[: len(ascii_uppercase)]):
            variant["letter"] = ascii_uppercase[index]
    if report_dir.exists():
        # only ever replace an earlier report, never an unrelated folder
        if any(report_dir.iterdir()) and not (report_dir / REPORT_PAGE).is_file():
            _LOGGER.error(
                "%s exists and is not a report; not replacing it.", report_dir
            )
            return False
        _LOGGER.warning(
            "Replacing %s: answers to the old report no longer fit.", report_dir
        )
        shutil.rmtree(report_dir)
    (report_dir / REPORT_CLIPS).mkdir(parents=True)
    if not _cut_clips(data, sources, report_dir / REPORT_CLIPS, ffmpeg):
        return False
    page = files(__package__).joinpath("report_page.html").read_text(encoding="utf-8")
    payload = json.dumps(_page_data(data, attendees), ensure_ascii=False)
    # keep "</script>" inside a string from ending the script block
    page = page.replace("__DATA__", payload.replace("</", "<\\/"))
    (report_dir / REPORT_PAGE).write_text(page, encoding="utf-8")
    return True


# the keys are read by the page script
def _page_data(data: dict, attendees: list[str]) -> dict:  # pragma: no mutate block
    """Return what the page needs, with candidates as anonymous tokens only.

    data: the evaluation data
    attendees: the attendee names"""
    tokens = data["tokens"]
    return {
        "id": data["report"],
        "weights": PRE_WEIGHTS,
        "medical_weight": MEDICAL_WEIGHT,
        "readability_count": READABILITY_COUNT,
        # only the attendee list: names from the candidates would hint at who wrote
        # what; other names are typed in
        "names": sorted(set(attendees)),
        "unknown": [
            f"{UNKNOWN_PERSON} {number}"
            for number in range(1, UNKNOWN_PERSON_COUNT + 1)
        ],
        "spans": [
            {
                "id": span["id"],
                "clip": f"{REPORT_CLIPS}/{span['id']}.mp3",
                "before": span["before"],
                "after": span["after"],
                "medical": span["medical"],
                "variants": [
                    {
                        "letter": variant["letter"],
                        "text": variant["text"],
                        "who": [tokens[c] for c in variant["candidates"]],
                    }
                    for variant in span["variants"]
                    if variant["letter"]
                ],
            }
            for span in data["spans"]
        ],
        "spots": [
            {
                "id": spot["id"],
                "clip": f"{REPORT_CLIPS}/{spot['id']}.mp3",
                "who": {tokens[c]: label for c, label in spot["labels"].items()},
            }
            for spot in data["spots"]
        ],
        # only candidates that can win are ever shown in part 3
        "candidates": [
            {
                "token": tokens[identifier],
                "speakers": candidate["speakers"],
                "cost_score": candidate["cost_score"],
                "sample": candidate["sample"],
            }
            for identifier, candidate in data["candidates"].items()
            if candidate["eligible"] and not candidate["disqualified"]
        ],
    }


def _cut_clips(data: dict, sources: list[Path], target: Path, ffmpeg: Ffmpeg) -> bool:
    """Cut one clip per span and spot-check from the prepared recording.

    The recording is prepared again (same timeline as the candidates) in a temp
    folder that is deleted afterwards.

    data: the evaluation data
    sources: the recordings
    target: the clips folder
    ffmpeg: the ffmpeg wrapper"""
    # pragma: no mutate start (folder name)
    scratch = Path(tempfile.mkdtemp(prefix="recording-transcriber-report-"))
    # pragma: no mutate end
    try:
        audio = scratch / PREPARED_FILE
        # the clips are cut from the plain recording
        if not ffmpeg.prepare(sources, audio, denoise=False):  # pragma: no mutate
            return False
        for span in data["spans"]:
            start = max(span["start_s"] - CLIP_PAD_BEFORE_S, 0)
            end = min(
                max(span["end_s"], span["start_s"]) + CLIP_PAD_AFTER_S,
                start + CLIP_MAX_S,
            )
            if not ffmpeg.extract(audio, start, end, target / f"{span['id']}.mp3"):
                return False
        for spot in data["spots"]:
            start = spot["start_s"]
            if not ffmpeg.extract(
                audio, start, start + SPOT_S, target / f"{spot['id']}.mp3"
            ):
                return False
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    _LOGGER.info("Cut %s clips.", len(data["spans"]) + len(data["spots"]))
    return True
