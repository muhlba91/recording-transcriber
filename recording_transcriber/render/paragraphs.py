"""Build transcript paragraphs from turns."""

from ..models.paragraph import Paragraph
from ..models.turn import Turn
from ..utils.const import MERGE_GAP_S


def build_paragraphs(turns: list[Turn], labels: dict[str, str]) -> list[Paragraph]:
    """Apply the speaker labels and merge close consecutive turns of the same person.

    A paragraph spans at most MERGE_GAP_S, so long talks keep timestamps for navigation
    and review.

    turns: all turns in recording order
    labels: the display label per speaker id"""
    paragraphs = []
    for turn in turns:
        speaker = labels.get(turn.speaker_id, turn.speaker_id)
        previous = paragraphs[-1] if paragraphs else None
        if (
            previous is not None
            and previous.speaker == speaker
            and turn.start_s - previous.start_s <= MERGE_GAP_S  # pragma: no mutate
        ):
            previous.text = f"{previous.text} {turn.text}"
            if turn.speaker_id not in previous.speaker_ids:
                previous.speaker_ids.append(turn.speaker_id)
        else:
            paragraphs.append(
                Paragraph(
                    start_s=turn.start_s,
                    speaker=speaker,
                    speaker_ids=[turn.speaker_id],
                    text=turn.text,
                )
            )
    return paragraphs
