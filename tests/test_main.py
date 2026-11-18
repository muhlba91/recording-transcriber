"""Tests for python -m recording_transcriber."""

import runpy
import sys

import pytest


def test_main_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["recording-transcriber", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("recording_transcriber", run_name="__main__")
    assert exit_info.value.code == 0
    assert "run" in capsys.readouterr().out
