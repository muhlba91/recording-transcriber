"""Prompt templates (Markdown files with $placeholders)."""

from importlib.resources import files
from pathlib import Path
from string import Template

_SUFFIX = ".md"


def names() -> list[str]:
    """Return the names of the packaged templates, without .md."""
    folder = files(__name__)  # pragma: no mutate (files(None) finds this package too)
    return sorted(
        path.name.removesuffix(_SUFFIX)
        for path in folder.iterdir()
        if path.name.endswith(_SUFFIX)
    )


def template(name: str, directory: Path | None = None) -> str:
    """Return a template's text: the file in directory if it has one, else the
    packaged one.

    name: the template name without .md
    directory: the folder with replacement templates (default: None, packaged only)"""
    if directory is not None:
        override = directory / f"{name}{_SUFFIX}"
        if override.is_file():
            return override.read_text(encoding="utf-8")
    return files(__name__).joinpath(f"{name}{_SUFFIX}").read_text(encoding="utf-8")


def placeholders(text: str) -> set[str]:
    """Return the placeholder names of a template; raise ValueError if it is invalid.

    text: the template text"""
    parsed = Template(text)
    if not parsed.is_valid():
        # pragma: no mutate start (message)
        raise ValueError("invalid placeholder (a $ must be doubled or name a value)")
        # pragma: no mutate end
    return set(parsed.get_identifiers())


def render(name: str, directory: Path | None = None, **values: str) -> str:
    """Render a prompt template; a missing value raises KeyError.

    name: the template name without .md
    directory: the folder with replacement templates (default: None, packaged only)
    values: the placeholder values"""
    return Template(template(name, directory)).substitute(values)
