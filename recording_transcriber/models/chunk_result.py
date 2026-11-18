"""A transcribed chunk."""

from pydantic import BaseModel

from .chunk_span import ChunkSpan
from .roster_entry import RosterEntry
from .turn import Turn
from .unclear_passage import UnclearPassage
from .usage import Usage


class ChunkResult(BaseModel):
    """A transcribed chunk, cached as work/chunk_NN.json."""

    span: ChunkSpan
    model: str
    # hash of audio, settings, prompt and context; a change invalidates the cache
    inputs_hash: str = ""
    turns: list[Turn]
    roster: list[RosterEntry]
    unclear: list[UnclearPassage] = []
    usage: Usage = Usage()
    # the polish pass over a verbatim speech-to-text answer (priced with the polish model)
    text_usage: Usage = Usage()
    attempts: int = 1
    warnings: list[str] = []
