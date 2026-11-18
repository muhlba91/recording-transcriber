"""Write the protocol and the agenda from the finished transcript."""

import logging

from pydantic import ValidationError

from ..backends.base import TextBackend
from ..configuration.config import Config
from ..configuration.job import Job
from ..models.document import Document
from ..models.job_context import JobContext
from ..models.paragraph import Paragraph
from ..models.protocol_result import ProtocolResult
from ..models.usage import Usage
from ..prompts.prompt_renderer import PromptRenderer
from ..prompts.schemas import DOCUMENT_SCHEMA
from ..render.markdown import document_markdown
from ..utils.const import (
    AGENDA_CACHE_FILE,
    PROMPT_PLACEHOLDER,
    PROTOCOL_FILE,
    PROTOCOL_PART_FILE,
    PROTOCOL_PART_HEADING,
    TASK_AGENDA,
    TASK_PROTOCOL,
    WARNING_AGENDA,
    WARNING_PROTOCOL,
)
from ..utils.retry import complete_with_retry
from ..utils.serialize import digest
from ..utils.timestamps import format_timestamp

_LOGGER = logging.getLogger(__name__)


class ProtocolWriter:
    """Write a protocol per transcript part, join them, and write the agenda.

    Every answer is cached in work/ under a hash of its prompt, provider and model, so
    a rerun only repeats failed or changed calls."""

    def __init__(
        self,
        protocol_backend: TextBackend,
        agenda_backend: TextBackend,
        job: Job,
        config: Config,
        context: JobContext,
        examples: list[str],
    ):
        """Initialize the writer.

        protocol_backend: the backend of the protocol task
        agenda_backend: the backend of the agenda task
        job: the job folder (for the cache)
        config: the configuration
        context: the job context
        examples: the text of the example protocols (style references)"""
        self.backends = {TASK_PROTOCOL: protocol_backend, TASK_AGENDA: agenda_backend}
        self.renderer = PromptRenderer(config)
        self.job = job
        self.config = config
        self.context = context
        self.examples = examples

    def write(self, parts: list[list[Paragraph]]) -> ProtocolResult:
        """Return the protocol and the agenda; a failed one is None with a warning.

        parts: the transcript paragraphs per part, with display names"""
        result = ProtocolResult()
        sections = []
        failed = False
        for index, paragraphs in enumerate(parts):
            if not paragraphs:
                continue
            prompt = self._prompt(
                "protocol_part",
                part_number=str(index + 1),  # pragma: no mutate (prompt text)
                part_count=str(len(parts)),  # pragma: no mutate (prompt text)
                transcript=_transcript(paragraphs),
            )
            name = PROTOCOL_PART_FILE.format(index=index)
            section, usage = self._call(name, prompt, TASK_PROTOCOL)
            result.protocol_usage = result.protocol_usage.add(usage)
            if section is None:
                # a protocol with a part missing would look complete; leave it out
                failed = True
                break
            # pragma: no mutate start (prompt text)
            heading = PROTOCOL_PART_HEADING.format(number=index + 1)
            # pragma: no mutate end
            sections.append(f"{heading}\n\n{document_markdown(section)}")
        if sections and not failed:
            # pragma: no mutate start (prompt layout)
            prompt = self._prompt(
                "protocol",
                examples="\n\n---\n\n".join(self.examples) or PROMPT_PLACEHOLDER,
                parts="\n\n".join(sections),
            )
            # pragma: no mutate end
            result.protocol, usage = self._call(PROTOCOL_FILE, prompt, TASK_PROTOCOL)
            result.protocol_usage = result.protocol_usage.add(usage)
        if result.protocol is None:
            _LOGGER.warning("Could not write the protocol.")
            result.warnings.append(WARNING_PROTOCOL)
        paragraphs = [paragraph for part in parts for paragraph in part]
        if paragraphs:
            prompt = self._prompt("agenda", transcript=_transcript(paragraphs))
            result.agenda, result.agenda_usage = self._call(
                AGENDA_CACHE_FILE, prompt, TASK_AGENDA
            )
        if result.agenda is None:
            _LOGGER.warning("Could not write the agenda.")
            result.warnings.append(WARNING_AGENDA)
        return result

    def _prompt(self, name: str, **values: str) -> str:
        """Render a template with the language settings and the job context.

        name: the template name without .md
        values: the template's own placeholder values"""
        # pragma: no mutate start (prompt layout)
        return self.renderer.render(
            name,
            attendees=self.context.attendees or PROMPT_PLACEHOLDER,
            agenda=self.context.agenda or PROMPT_PLACEHOLDER,
            glossary="\n".join(self.context.glossary) or PROMPT_PLACEHOLDER,
            **values,
        )
        # pragma: no mutate end

    def _call(self, name: str, prompt: str, task: str) -> tuple[Document | None, Usage]:
        """Return the document for a prompt from the cache or the model, and its usage.

        Only successful answers are cached, so a rerun tries a failed call again.

        name: the cache file name
        prompt: the rendered prompt
        task: the task (protocol or agenda)"""
        settings = getattr(self.config.tasks, task)
        fingerprint = digest(
            prompt,
            settings.provider,
            self.config.endpoint(settings.provider),
            settings.model,
            self.config.thinking_level,
        )
        cached = self.job.cached(name)
        if cached is not None and cached.get("fingerprint") == fingerprint:
            try:
                return (
                    Document.model_validate(cached["document"]),
                    Usage.model_validate(cached["usage"]),
                )
            except (KeyError, ValidationError) as error:
                _LOGGER.warning("Ignoring outdated %s: %s.", name, error)
        # pragma: no mutate start (request wiring)
        document, usage = complete_with_retry(
            self.backends[task], prompt, DOCUMENT_SCHEMA, settings.model, _parse
        )
        # pragma: no mutate end
        if document is not None:
            self.job.cache(
                name,
                {
                    "fingerprint": fingerprint,
                    "document": document.model_dump(),
                    "usage": usage.model_dump(),
                },
            )
        return document, usage


def _transcript(paragraphs: list[Paragraph]) -> str:
    """Return paragraphs as `[hh:mm:ss] Name: text` lines.

    paragraphs: the transcript paragraphs"""
    return "\n".join(  # pragma: no mutate (prompt layout)
        f"[{format_timestamp(paragraph.start_s)}] {paragraph.speaker}: {paragraph.text}"
        for paragraph in paragraphs
    )


def _parse(text: str) -> Document | None:
    """Parse the model's JSON document without blank items; return None if invalid or
    empty.

    text: the model output"""
    try:
        document = Document.model_validate_json(text)
    except ValidationError as error:
        _LOGGER.warning("Invalid document answer: %s.", error)
        return None
    for section in document.sections:
        section.heading = section.heading.strip()
        section.paragraphs = [text for text in section.paragraphs if text.strip()]
        section.bullets = [text for text in section.bullets if text.strip()]
    document.sections = [
        section
        for section in document.sections
        if section.heading and (section.paragraphs or section.bullets)
    ]
    if not document.sections:
        _LOGGER.warning("The document answer has no sections.")
        return None
    return document
