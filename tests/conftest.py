"""Shared fixtures."""

import os
import socket
import subprocess
from pathlib import Path

import pytest

from recording_transcriber.audio.ffmpeg import find_binary


def make_audio(path: Path, pattern: list[tuple[str, float]]) -> Path:
    """Write an mp3 of tones and silences.

    path: the file to write
    pattern: ("tone" | "silence", seconds) segments"""
    inputs = []
    for kind, seconds in pattern:
        source = "sine=frequency=440" if kind == "tone" else "anullsrc=r=16000:cl=mono"
        inputs += ["-f", "lavfi", "-t", str(seconds), "-i", source]
    streams = "".join(f"[{index}:a]" for index in range(len(pattern)))
    graph = f"{streams}concat=n={len(pattern)}:v=0:a=1[out]"
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            find_binary(),
            "-hide_banner",
            "-y",
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "[out]",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keep every test away from real providers: no API keys from the environment
    (mise loads the real ones) and no network connections."""
    for name in list(os.environ):
        if name.endswith("_KEY"):
            monkeypatch.delenv(name)
    # also skips the macOS proxy lookup, which crashes in mutmut's forked runs
    monkeypatch.setenv("NO_PROXY", "*")

    def refuse(*args, **kwargs):
        raise OSError("tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Skip retry backoff."""
    monkeypatch.setattr("time.sleep", lambda seconds: None)


@pytest.fixture(scope="session")
def tone_file(tmp_path_factory) -> Path:
    """An 8 s file: 3 s tone, 2 s silence, 3 s tone."""
    return make_audio(
        tmp_path_factory.mktemp("audio") / "tone.mp3",
        [("tone", 3), ("silence", 2), ("tone", 3)],
    )
