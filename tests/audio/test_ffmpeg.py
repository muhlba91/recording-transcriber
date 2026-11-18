"""Tests for the ffmpeg wrapper."""

import subprocess
from pathlib import Path

import pytest

from recording_transcriber.audio import ffmpeg as ffmpeg_module
from recording_transcriber.audio.ffmpeg import Ffmpeg, find_binary


class TestFfmpeg:
    def test_find_binary_system(self, monkeypatch):
        monkeypatch.setattr(
            ffmpeg_module.shutil,
            "which",
            lambda name: "/usr/bin/ffmpeg" if name == "ffmpeg" else None,
        )
        assert find_binary() == "/usr/bin/ffmpeg"

    def test_find_binary_fallback(self, monkeypatch):
        monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda name: None)
        assert Path(find_binary()).is_file()

    def test_duration(self, tone_file):
        assert abs(Ffmpeg().duration(tone_file) - 8) < 0.2

    def test_duration_missing_file(self, tmp_path):
        assert Ffmpeg().duration(tmp_path / "missing.mp3") is None

    def test_duration_missing_binary(self, tmp_path):
        wrapper = Ffmpeg(binary=str(tmp_path / "no-ffmpeg"))
        assert wrapper.duration(tmp_path / "x.mp3") is None

    def test_duration_hours_minutes_seconds(self, monkeypatch, tmp_path):
        output = subprocess.CompletedProcess([], 1, stderr="  Duration: 01:02:03.50, s")
        monkeypatch.setattr(Ffmpeg, "_run", lambda self, args, check=True: output)
        assert Ffmpeg("ffmpeg").duration(tmp_path / "x.mp3") == 3723.5

    def test_silences(self, tone_file):
        silences = Ffmpeg().silences(tone_file)
        assert len(silences) == 1
        start, end = silences[0]
        assert 2.8 < start < 3.3 and 4.7 < end < 5.2

    @pytest.mark.parametrize("denoise", [True, False])
    def test_prepare_joins_files(self, tone_file, tmp_path, denoise):
        target = tmp_path / "out" / "prepared.mp3"
        wrapper = Ffmpeg()
        assert wrapper.prepare([tone_file, tone_file], target, denoise=denoise)
        assert abs(wrapper.duration(target) - 16) < 0.5

    @pytest.mark.parametrize("denoise", [True, False])
    def test_prepare_applies_denoise_only_when_asked(
        self, monkeypatch, tmp_path, denoise
    ):
        calls = []
        monkeypatch.setattr(
            Ffmpeg, "_run", lambda self, args, check=True: calls.append(args)
        )
        Ffmpeg("ffmpeg").prepare([tmp_path / "a.mp3"], tmp_path / "out.mp3", denoise)
        assert ("afftdn" in " ".join(calls[0])) is denoise

    def test_extract(self, tone_file, tmp_path):
        target = tmp_path / "chunks" / "deep" / "clip.mp3"
        wrapper = Ffmpeg()
        assert wrapper.extract(tone_file, 1, 4, target)
        assert abs(wrapper.duration(target) - 3) < 0.3

    def test_silences_parsing(self, monkeypatch, tmp_path):
        stderr = (
            "silence_start: -0.0069\nsilence_end: 2.5 | silence_duration: 2.5\n"
            "silence_start: 7.25\nsilence_end: 8.5 | silence_duration: 1.25\n"
            "silence_start: 10\n"
        )
        output = subprocess.CompletedProcess([], 0, stderr=stderr)
        monkeypatch.setattr(Ffmpeg, "_run", lambda self, args, check=True: output)
        # a start without an end (the file ends in silence) is dropped
        assert Ffmpeg("ffmpeg").silences(tmp_path / "x.mp3") == [(0, 2.5), (7.25, 8.5)]

    def test_run_failure(self, tmp_path):
        assert not Ffmpeg().extract(tmp_path / "missing.mp3", 0, 1, tmp_path / "x.mp3")

    def test_run_missing_binary(self, tmp_path):
        wrapper = Ffmpeg(binary=str(tmp_path / "no-ffmpeg"))
        assert wrapper.silences(tmp_path / "x.mp3") == []
