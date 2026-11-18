"""A raw model response."""

from pydantic import BaseModel

from .usage import Usage


class ModelResponse(BaseModel):
    """The raw result of one model call.

    status is the provider's final state, e.g. completed or incomplete (truncated)."""

    text: str
    status: str
    usage: Usage = Usage()
