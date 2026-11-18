"""ffmpeg wrapper."""

import logging
import re
import shutil
import subprocess
from pathlib import Path

from ..utils.const import (
    BITRATE,
    DENOISE_FILTER,
    SAMPLE_RATE,
    SILENCE_MIN_S,
    SILENCE_NOISE,
)

_LOGGER = logging.getLogger(__name__)

_DURATION = re.compile(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)")
_SILENCE_START = re.compile(r"silence_start: (-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end: (\d+(?:\.\d+)?)")


def find_binary() -> str:
    """Return the system ffmpeg if installed, else the imageio-ffmpeg binary."""
    system = shutil.which("ffmpeg")
    if system is not None:
        return system
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


class Ffmpeg:
    """Run ffmpeg for probing, converting, silence detection and cutting."""

    def __init__(self, binary: str | None = None):
        """Initialize the wrapper.

        binary: the ffmpeg executable (default: None, found automatically)"""
        self.binary = binary or find_binary()

    def duration(self, path: Path) -> float | None:
        """Return the duration in seconds, parsed from `ffmpeg -i` (no ffprobe needed).

        path: the audio file"""
        result = self._run(["-i", str(path)], check=False)
        match = _DURATION.search(result.stderr) if result is not None else None
        if match is None:
            _LOGGER.error("Could not read the duration of %s.", path)
            return None
        hours, minutes, seconds = match.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    def prepare(self, sources: list[Path], target: Path, denoise: bool) -> bool:
        """Join the sources into one 16 kHz mono mp3 with loudness normalization.

        sources: the recordings, in order
        target: the mp3 to write
        denoise: whether to apply the denoise filter"""
        inputs = []
        for source in sources:
            inputs += ["-i", str(source)]
        filters = [DENOISE_FILTER] if denoise else []
        chain = ",".join([*filters, "loudnorm"])
        joined = "".join(f"[{index}:a]" for index in range(len(sources)))
        graph = f"{joined}concat=n={len(sources)}:v=0:a=1,{chain}[out]"
        target.parent.mkdir(parents=True, exist_ok=True)
        args = [
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-b:a",
            BITRATE,
            "-codec:a",
            "libmp3lame",
            str(target),
        ]
        return self._run(args) is not None

    def silences(self, path: Path) -> list[tuple[float, float]]:
        """Return the silent stretches as (start, end) seconds.

        path: the audio file"""
        detector = f"silencedetect=noise={SILENCE_NOISE}:d={SILENCE_MIN_S}"
        # pragma: no mutate start (ffmpeg arguments)
        result = self._run(["-i", str(path), "-af", detector, "-f", "null", "-"])
        # pragma: no mutate end
        if result is None:
            return []
        starts = [
            max(0.0, float(value)) for value in _SILENCE_START.findall(result.stderr)
        ]
        ends = [float(value) for value in _SILENCE_END.findall(result.stderr)]
        # a start without an end (the file ends in silence) is dropped
        return list(zip(starts, ends, strict=False))

    def extract(self, path: Path, start_s: float, end_s: float, target: Path) -> bool:
        """Copy a time range into a new file without re-encoding.

        path: the source audio
        start_s: the range start in seconds
        end_s: the range end in seconds
        target: the file to write"""
        target.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(path),
            "-c",
            "copy",
            str(target),
        ]
        return self._run(args) is not None

    def _run(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess | None:
        """Run ffmpeg; return None and log an error if it fails.

        args: the arguments after the binary
        check: whether a non-zero exit code is an error (default: True)"""
        command = [self.binary, "-hide_banner", "-nostdin", "-y", *args]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            _LOGGER.error("Could not run ffmpeg: %s.", error)
            return None
        if check and result.returncode != 0:
            _LOGGER.error("ffmpeg failed: %s.", result.stderr.strip()[-500:])
            return None
        return result
