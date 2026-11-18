"""Tests for the retry utils."""

from recording_transcriber.utils import retry
from recording_transcriber.utils.const import BACKOFF_BASE_S, MAX_ATTEMPTS
from tests.fakes import FakeBackend, completed


def test_wait_before(monkeypatch):
    sleeps = []
    monkeypatch.setattr(retry.time, "sleep", sleeps.append)
    for attempt in range(1, 5):
        retry.wait_before(attempt)
    # nothing before the first attempt, then the wait doubles
    assert len(sleeps) == 3
    assert sleeps[0] == BACKOFF_BASE_S
    assert sleeps[1:] == [sleeps[0] * 2, sleeps[0] * 4]


def test_complete_with_retry():
    backend = FakeBackend(completions=[None, completed("bad"), completed("good")])
    result, usage = retry.complete_with_retry(
        backend, "p", {}, "m", lambda text: text if text == "good" else None
    )
    assert result == "good"
    assert usage.output_tokens == 20
    assert len(backend.prompts) == 3


def test_complete_with_retry_fails():
    backend = FakeBackend(completions=[completed("bad")] * MAX_ATTEMPTS)
    result, usage = retry.complete_with_retry(backend, "p", {}, "m", lambda text: None)
    assert result is None
    assert usage.output_tokens == 10 * MAX_ATTEMPTS
