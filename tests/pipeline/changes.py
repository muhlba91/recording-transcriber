"""Settings changes that invalidate the cache, shared by the pipeline tests."""

import pytest

# every change takes the configuration and the task it applies to (task-independent
# changes ignore it); the base configuration has the custom providers "lab" and "lab2"
_CHANGES = {
    "model": lambda config, task: setattr(getattr(config.tasks, task), "model", "x"),
    "provider": lambda config, task: setattr(
        getattr(config.tasks, task), "provider", "gemini"
    ),
    "base_url": lambda config, task: setattr(
        config.providers["lab"], "base_url", "https://b/v1"
    ),
    "same_endpoint": lambda config, task: setattr(
        getattr(config.tasks, task), "provider", "lab2"
    ),
    "transcribe_model": lambda config, task: setattr(
        config.tasks.transcribe, "model", "gemini-3.6"
    ),
    "thinking_level": lambda config, task: setattr(config, "thinking_level", "high"),
    "meeting_type": lambda config, task: setattr(config.meeting, "type", "board"),
    "writer_role": lambda config, task: setattr(
        config.meeting, "writer_role", "a clerk"
    ),
}


def changes(*names: str) -> list:
    """Return the named settings changes as pytest parameters.

    names: the keys of the changes to use"""
    return [pytest.param(_CHANGES[name], id=name) for name in names]
