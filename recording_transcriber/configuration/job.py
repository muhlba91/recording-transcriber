"""A job folder."""

import json
import logging
import re
from pathlib import Path

from docx import Document

from ..models.job_context import JobContext
from ..utils.const import (
    AGENDA_FILE,
    ATTENDEES_FILE,
    AUDIO_DIR,
    AUDIO_EXTENSIONS,
    CONTEXT_DIR,
    EXAMPLES_DIR,
    EXAMPLES_GLOB,
    FALLBACK_ENCODING,
    GLOSSARY_FILE,
    MAX_EXAMPLE_CHARS,
    MAX_GLOSSARY_TERMS,
    OUT_DIR,
    TRANSCRIPT_MD,
    WORK_DIR,
)

_LOGGER = logging.getLogger(__name__)


class Job:
    """A job folder: recordings and optional context files, work/ (cache), out/.

    Recordings and context files may sit in audio/ and context/ or directly in the
    folder: a missing subfolder falls back to the folder itself."""

    def __init__(self, root: Path):
        """Initialize the job.

        root: the job folder"""
        self.root = root
        self.work_dir = root / WORK_DIR
        self.out_dir = root / OUT_DIR

    @property
    def audio_dir(self) -> Path:
        """Return audio/ if it holds recordings, else the job folder."""
        folder = self.root / AUDIO_DIR
        return folder if _audio_in(folder) else self.root

    @property
    def context_dir(self) -> Path:
        """Return context/ if it holds anything, else the job folder."""
        folder = self.root / CONTEXT_DIR
        return folder if folder.is_dir() and any(folder.iterdir()) else self.root

    @property
    def finished(self) -> bool:
        """Return whether the transcript exists."""
        return (self.out_dir / TRANSCRIPT_MD).is_file()

    @property
    def name(self) -> str:
        """Return the job name (the folder name)."""
        return self.root.name

    def audio_files(self) -> list[Path]:
        """Return the recordings in natural order ("Teil 2" before "Teil 10")."""
        return sorted(_audio_in(self.audio_dir), key=_natural_key)

    def context(self, glossary_global: Path | None = None) -> JobContext:
        """Read the context files; missing files are skipped.

        glossary_global: the shared glossary, merged first (default: None)"""
        # only the warning depends on the condition
        if glossary_global and not glossary_global.is_file():  # pragma: no mutate
            _LOGGER.warning(
                "Global glossary %s not found; skipping it.", glossary_global
            )
        terms = []
        for path in (glossary_global, self.context_dir / GLOSSARY_FILE):
            text = _read(path)
            if text is not None:
                terms.extend(line.strip() for line in text.splitlines())
        # drop blanks and duplicates, keep the first spelling
        glossary = list(dict.fromkeys(term for term in terms if term))
        # only the warning depends on the condition
        if len(glossary) > MAX_GLOSSARY_TERMS:  # pragma: no mutate
            _LOGGER.warning(
                "Glossary has %s terms; using the first %s.",
                len(glossary),
                MAX_GLOSSARY_TERMS,
            )
        return JobContext(
            glossary=glossary[:MAX_GLOSSARY_TERMS],
            attendees=_read(self.context_dir / ATTENDEES_FILE),
            agenda=_read(self.context_dir / AGENDA_FILE),
        )

    def examples(self) -> list[str]:
        """Return the text of the example protocols in context/examples/*.docx.

        Headings are marked with "## " to keep the structure. Unreadable files are
        skipped; the total is capped at MAX_EXAMPLE_CHARS."""
        folder = self.context_dir / EXAMPLES_DIR
        if not folder.is_dir():
            return []
        examples = []
        budget = MAX_EXAMPLE_CHARS
        for path in sorted(folder.glob(EXAMPLES_GLOB), key=_natural_key):
            # Word keeps "~$name.docx" lock files next to open documents
            if path.name.startswith("~$"):
                continue
            text = _docx_text(path)
            if text is None:
                continue
            if budget <= 0:  # pragma: no mutate (threshold)
                _LOGGER.warning("Skipping example %s: size limit reached.", path.name)
                # break would skip the same examples
                continue  # pragma: no mutate
            # only the warning depends on the condition
            if len(text) > budget:  # pragma: no mutate
                _LOGGER.warning("Example %s is cut to fit the size limit.", path.name)
            examples.append(text[:budget])
            budget -= len(text)
        return examples

    def cached(self, name: str) -> dict | None:
        """Return a cached JSON object from work/, or None if missing or broken.

        name: the file name"""
        path = self.work_dir / name
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _LOGGER.warning("Ignoring unreadable cache file %s: %s.", path, error)
            return None
        return data if isinstance(data, dict) else None

    def cache(self, name: str, data: dict) -> None:
        """Write a JSON document to work/.

        name: the file name
        data: the JSON document"""
        _write(self.work_dir / name, json.dumps(data, ensure_ascii=False, indent=1))

    def write_output(self, name: str, content: str | bytes) -> None:
        """Write a file to out/.

        name: the file name
        content: the file content"""
        _write(self.out_dir / name, content)


def _read(path: Path | None) -> str | None:
    """Return the file content without # comment lines, or None if missing or empty.

    path: the file to read"""
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    try:
        # utf-8-sig drops the byte order mark Windows tools put in front
        content = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        _LOGGER.warning(
            "%s is not UTF-8; reading it as %s.", path.name, FALLBACK_ENCODING
        )
        content = data.decode(FALLBACK_ENCODING, errors="replace")
    lines = content.splitlines()
    text = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    return text.strip() or None


def _audio_in(folder: Path) -> list[Path]:
    """Return the recordings directly in a folder (none if it does not exist).

    folder: the folder"""
    if not folder.is_dir():
        return []
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def _docx_text(path: Path) -> str | None:
    """Return a Word file's paragraphs, headings marked "## ", or None if unreadable.

    Tables are not read: the examples only show structure and tone.

    path: the .docx file"""
    try:
        document = Document(str(path))
    # a broken upload can fail anywhere in python-docx or lxml (XMLSyntaxError is a
    # SyntaxError); an example is optional, so skip it whatever the cause
    except Exception as error:  # noqa: BLE001
        _LOGGER.warning("Skipping unreadable example %s: %s.", path.name, error)
        return None
    lines = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style = paragraph.style
        heading = style is not None and style.name.startswith(("Heading", "Title"))
        lines.append(f"## {text}" if heading else text)
    return "\n".join(lines) or None


def _natural_key(path: Path) -> list:
    """Return a sort key that orders embedded numbers by value.

    path: the file"""
    return [
        # lower and upper case sort the same
        int(part) if part.isdigit() else part.lower()  # pragma: no mutate
        for part in re.split(r"(\d+)", path.name)
    ]


def _write(path: Path, content: str | bytes) -> None:
    """Write a file via a temporary file, so a disconnect never leaves half a file.

    path: the file to write
    content: the file content"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if isinstance(content, bytes):
        temporary.write_bytes(content)
    else:
        temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
