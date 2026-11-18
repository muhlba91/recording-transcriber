"""An unclear passage."""

from pydantic import BaseModel


class UnclearPassage(BaseModel):
    """A passage the model could not understand."""

    start_s: float
    note: str
