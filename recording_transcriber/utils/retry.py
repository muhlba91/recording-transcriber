"""Transcriber retry utils."""

import time
from collections.abc import Callable

from ..backends.base import TextBackend
from ..models.usage import Usage
from .const import BACKOFF_BASE_S, MAX_ATTEMPTS, STATUS_COMPLETED


def wait_before(attempt: int) -> None:
    """Sleep with exponential backoff before a retry (10, 20, 40 s).

    Nothing happens before the first attempt.

    attempt: the attempt number, starting at 1"""
    if attempt > 1:
        time.sleep(BACKOFF_BASE_S * 2 ** (attempt - 2))


def complete_with_retry[T](
    backend: TextBackend,
    prompt: str,
    schema: dict,
    model: str,
    parse: Callable[[str], T | None],
) -> tuple[T | None, Usage]:
    """Return the parsed answer to a text prompt and the usage of all attempts.

    The answer is None if every attempt failed. A failed call, an incomplete answer
    and an answer the parser rejects are retried.

    backend: the text backend
    prompt: the instructions
    schema: the JSON schema of the answer
    model: the model id
    parse: turns the answer text into the result, or None if it is invalid"""
    usage = Usage()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        wait_before(attempt)
        # pragma: no mutate start (request wiring)
        response = backend.complete(prompt, schema, model)
        # pragma: no mutate end
        if response is not None:
            usage = usage.add(response.usage)
        if response is None or response.status != STATUS_COMPLETED:
            continue
        result = parse(response.text)
        if result is not None:
            return result, usage
    return None, usage
