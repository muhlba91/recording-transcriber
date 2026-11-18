"""Polish verbatim speech-to-text turns into clean verbatim, one text call per chunk."""

import json
import logging

from ..backends.base import TextBackend
from ..models.turn import Turn
from ..models.usage import Usage
from ..prompts.prompt_renderer import PromptRenderer
from ..prompts.schemas import POLISH_SCHEMA
from ..utils.const import PROMPT_PLACEHOLDER
from ..utils.retry import complete_with_retry

_LOGGER = logging.getLogger(__name__)


class TurnPolisher:
    """Turn verbatim text into clean verbatim, keeping speakers and timestamps.

    The model only returns new texts by index; the code keeps everything else."""

    def __init__(
        self,
        backend: TextBackend,
        model: str,
        glossary: list[str],
        renderer: PromptRenderer | None = None,
    ):
        """Initialize the polisher.

        backend: the text backend
        model: the model id
        glossary: the merged glossary
        renderer: fills the prompt template (default: None, the defaults)"""
        self.backend = backend
        self.model = model
        self.glossary = glossary
        self.renderer = renderer or PromptRenderer()

    def polish(self, turns: list[Turn]) -> tuple[list[Turn] | None, Usage]:
        """Return the cleaned turns (None if every call failed) and the usage.

        A turn that becomes empty is dropped; a turn the model skipped stays as is.

        turns: the verbatim turns of one chunk"""
        # pragma: no mutate start (prompt layout)
        prompt = self.renderer.render(
            "polish",
            glossary="\n".join(self.glossary) or PROMPT_PLACEHOLDER,
            turns="\n".join(
                f"{index}: {turn.text}" for index, turn in enumerate(turns)
            ),
        )
        # pragma: no mutate end
        # pragma: no mutate start (request wiring)
        texts, usage = complete_with_retry(
            self.backend, prompt, POLISH_SCHEMA, self.model, _parse
        )
        # pragma: no mutate end
        if texts is not None:
            return _apply(turns, texts), usage
        _LOGGER.warning("Polish failed; keeping the verbatim text.")
        return None, usage


def _parse(text: str) -> dict[int, str] | None:
    """Return the cleaned text per turn index, or None if the answer is invalid.

    text: the model output"""
    try:
        return {
            int(turn["index"]): turn["text"].strip()
            for turn in json.loads(text)["turns"]
        }
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        _LOGGER.warning("Invalid polish answer: %s.", error)
        return None


def _apply(turns: list[Turn], texts: dict[int, str]) -> list[Turn]:
    """Return the turns with their cleaned texts.

    turns: the verbatim turns
    texts: the cleaned text per index"""
    cleaned = [
        turn.model_copy(update={"text": texts.get(index, turn.text)})
        for index, turn in enumerate(turns)
    ]
    return [turn for turn in cleaned if turn.text]
