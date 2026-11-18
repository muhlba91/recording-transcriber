"""A fake backend that also records what every call was given."""

from tests.fakes import FakeBackend


class RecordingBackend(FakeBackend):
    """Replay queued responses; records the model of every call."""

    def __init__(self, transcriptions=None, completions=None, check=True):
        """Initialize the fake.

        transcriptions: responses for transcribe(), in order
        completions: responses for complete(), in order
        check: the result of check()"""
        super().__init__(transcriptions, completions, check)
        self.calls = []

    def transcribe(self, audio, prompt, schema, model):
        """Record the call, then return the next queued transcription."""
        self.calls.append(model)
        return super().transcribe(audio, prompt, schema, model)

    def complete(self, prompt, schema, model):
        """Record the call, then return the next queued completion."""
        self.calls.append(model)
        return super().complete(prompt, schema, model)
