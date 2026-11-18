"""Render prompts with the configured language, meeting and replacement templates."""

from .. import prompts
from ..configuration.config import Config


class PromptRenderer:
    """Fill the templates with the settings every prompt shares.

    A template in the configuration's prompts_dir replaces the packaged one."""

    def __init__(self, config: Config | None = None):
        """Initialize the renderer.

        config: the configuration (default: None, the defaults)"""
        config = config or Config()
        self.directory = config.prompts_dir
        self._values = {
            **config.transcript.prompt_values(),
            **config.meeting.prompt_values(),
        }

    def template(self, name: str) -> str:
        """Return a template's text as used, for cache hashes.

        name: the template name without .md"""
        return prompts.template(name, self.directory)

    def settings(self) -> str:
        """Return the shared values as one string, for cache hashes."""
        # pragma: no mutate start (any separator keeps the keys distinct)
        return "|".join(self._values.values())
        # pragma: no mutate end

    def render(self, name: str, **values: str) -> str:
        """Render a template with the shared values and its own.

        name: the template name without .md
        values: the template's own placeholder values"""
        return prompts.render(name, self.directory, **self._values, **values)
