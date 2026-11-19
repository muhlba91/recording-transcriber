"""Transcriber configuration (config.yaml)."""

import logging
from pathlib import Path

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from .. import prompts
from ..utils.const import (
    API_KEY_ENVS,
    DEFAULT_MEETING_TYPE,
    DEFAULT_WRITER_ROLE,
    PROMPT_PLACEHOLDER,
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_TRANSCRIBE,
    PROVIDERS,
    REGION_PROVIDERS,
    SPEECH_MAX_CHUNK_MINUTES,
    TASK_AGENDA,
    TASK_POLISH,
    TASK_PROTOCOL,
    TASK_TRANSCRIBE,
    TASKS,
    TEXT_TASKS,
    TRANSCRIBE_ONLY_PROVIDERS,
)

_LOGGER = logging.getLogger(__name__)


# the sub-sections below only exist as parts of Config, so they share its file
class ProviderConfig(BaseModel):
    """A custom OpenAI-compatible endpoint for the text tasks."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(min_length=1)
    # the environment variable that holds the key (a Colab secret of the same name)
    api_key_env: str = Field(min_length=1)


class TaskConfig(BaseModel):
    """The provider and model of one pipeline task."""

    # a typo in a key must fail loudly instead of silently using the default
    model_config = ConfigDict(extra="forbid")

    provider: str = PROVIDER_GEMINI
    model: str = Field(min_length=1)


class TasksConfig(BaseModel):
    """Provider and model per pipeline task."""

    model_config = ConfigDict(extra="forbid")

    # evaluation winner: far ahead of the other Gemini 4 models on words (design §5)
    transcribe: TaskConfig = TaskConfig(model="gemini-4-neon")
    # the polish pass only runs for verbatim speech-to-text transcribe providers
    polish: TaskConfig = TaskConfig(model="gemini-4-carbon")
    reconcile: TaskConfig = TaskConfig(model="gemini-4-carbon")
    protocol: TaskConfig = TaskConfig(model="gemini-4-carbon")
    agenda: TaskConfig = TaskConfig(model="gemini-4-carbon")


class MeetingConfig(BaseModel):
    """What kind of meeting the recording is, as the prompts phrase it."""

    model_config = ConfigDict(extra="forbid")

    # "a recorded <type>"
    type: str = Field(default=DEFAULT_MEETING_TYPE, min_length=1)
    # "You are <role>", with the article ("a medical writer")
    writer_role: str = Field(default=DEFAULT_WRITER_ROLE, min_length=1)

    def prompt_values(self) -> dict[str, str]:
        """Return the placeholders every prompt template may use."""
        return {"meeting_type": self.type, "writer_role": self.writer_role}


class TranscriptConfig(BaseModel):
    """The transcript language and the words the transcript itself uses."""

    model_config = ConfigDict(extra="forbid")

    # BCP-47; speech-to-text models get the primary subtag ("de"), Gemini the full code
    language: str = "en"
    # dialect, vocabulary and orthography hints added to the prompts
    notes: str = ""
    unclear_marker: str = Field(default="[unclear]", min_length=1)
    speaker_label: str = "Speaker {number}"
    # checks, speakers and the evaluated accuracy at the top of the transcript
    quality_note: bool = True

    def prompt_values(self) -> dict[str, str]:
        """Return the placeholders every prompt template may use."""
        return {
            "language": self.language,
            "language_notes": self.notes or PROMPT_PLACEHOLDER,
            "unclear_marker": self.unclear_marker,
        }

    @field_validator("speaker_label")
    @classmethod
    def _numbered(cls, value: str) -> str:
        """Reject a label that cannot number speakers (it would merge them all).

        value: the label template"""
        try:
            numbered = value.format(number=1) != value.format(number=2)
        except (KeyError, IndexError, ValueError, AttributeError, TypeError) as error:
            raise ValueError(f"invalid speaker_label: {error}") from error
        if not numbered:
            raise ValueError("speaker_label needs {number}")
        return value


class Price(BaseModel):
    """Price in USD per 1M tokens (thinking counts as output) or per audio minute."""

    model_config = ConfigDict(extra="forbid")

    input: float = 0
    output: float = 0
    per_minute: float = 0


class Config(BaseModel):
    """The transcriber configuration."""

    model_config = ConfigDict(extra="forbid")

    # read by the Colab notebook: the version to install, and whether a wheel in the
    # app folder wins over PyPI
    app_version: str | None = None
    local_wheels: bool = True
    # custom OpenAI-compatible endpoints, by name; for the text tasks only
    providers: dict[str, ProviderConfig] = {}
    tasks: TasksConfig = TasksConfig()
    transcript: TranscriptConfig = TranscriptConfig()
    meeting: MeetingConfig = MeetingConfig()
    chunk_minutes: int = Field(default=25, ge=10, le=120)
    denoise: bool = False
    # protocol and agenda after the transcript; the evaluation turns it off
    write_protocol: bool = True
    # Gemini only: the other providers have no such setting
    thinking_level: str = "low"
    glossary_global: Path | None = None
    # templates that replace the packaged prompts of the same file name
    prompts_dir: Path | None = None
    local_dir: Path | None = None
    prices: dict[str, Price] = {}
    # per transcription model: the evaluated word accuracy, quoted in the quality note
    accuracy_estimates: dict[str, str] = {}
    # the folder of the loaded config.yaml (the app folder); never written to
    _folder: Path | None = PrivateAttr(default=None)

    @property
    def folder(self) -> Path | None:
        """Return the folder the configuration was loaded from (None: defaults)."""
        return self._folder

    def with_folder(self, folder: Path | None) -> "Config":
        """Return a copy that counts as loaded from a folder (validating a changed
        configuration drops it).

        folder: the folder of the loaded config.yaml"""
        copy = self.model_copy()
        copy._folder = folder
        return copy

    @model_validator(mode="after")
    def _providers_fit_tasks(self) -> "Config":
        """Reject unknown providers, custom ones for transcription, speech-to-text
        ones for text tasks, and a bare language code for Gemini's transcribe model,
        which needs a region (de-DE); otherwise every call would fail."""
        clash = sorted(set(self.providers) & set(PROVIDERS))
        if clash:
            raise ValueError(f"custom providers named like built-in ones: {clash}")
        for task in TASKS:
            provider = getattr(self.tasks, task).provider
            if provider not in PROVIDERS and provider not in self.providers:
                raise ValueError(f"tasks.{task}: unknown provider {provider}")
            if task == TASK_TRANSCRIBE and provider in self.providers:
                raise ValueError(f"tasks.{task}: {provider} is for text tasks only")
            if task in TEXT_TASKS and provider in TRANSCRIBE_ONLY_PROVIDERS:
                raise ValueError(f"tasks.{task}: {provider} only transcribes audio")
        transcribe = self.tasks.transcribe.provider
        if transcribe in REGION_PROVIDERS and "-" not in self.transcript.language:
            raise ValueError(f"{transcribe} needs a region code like de-DE")
        return self

    @model_validator(mode="after")
    def _prompts_fit_templates(self) -> "Config":
        """Reject replacement prompts the packaged ones cannot stand in for: unknown
        file names and placeholders the packaged prompt does not offer. Placeholders
        the replacement leaves out are only logged."""
        if self.prompts_dir is None:
            return self
        if not self.prompts_dir.is_dir():
            raise ValueError(f"prompts_dir {self.prompts_dir} is not a folder")
        known = prompts.names()
        for path in sorted(self.prompts_dir.iterdir()):
            if path.suffix != ".md":
                _LOGGER.warning(
                    "prompts_dir: %s is ignored (not a .md file).", path.name
                )
                continue
            if path.stem not in known:
                raise ValueError(
                    f"prompts_dir: {path.name} replaces no prompt ({known})"
                )
            try:
                used = prompts.placeholders(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                raise ValueError(f"prompts_dir: {path.name}: {error}") from error
            offered = prompts.placeholders(prompts.template(path.stem))
            if used - offered:
                raise ValueError(
                    f"prompts_dir: {path.name} uses unknown placeholders "
                    f"{sorted(used - offered)}; offered: {sorted(offered)}"
                )
            if offered - used:
                _LOGGER.warning(
                    "Prompt %s does not use the placeholders %s.",
                    path.name,
                    sorted(offered - used),
                )
        return self

    @property
    def speech_to_text(self) -> bool:
        """Return whether the transcribe provider is a speech-to-text model (no
        prompt)."""
        return self.tasks.transcribe.provider != PROVIDER_GEMINI

    @property
    def target_chunk_minutes(self) -> int:
        """Return the chunk target; capped for speech-to-text models (design §4)."""
        if self.speech_to_text:
            return min(self.chunk_minutes, SPEECH_MAX_CHUNK_MINUTES)
        return self.chunk_minutes

    def key_variable(self, provider: str) -> str:
        """Return the environment variable that holds a provider's key.

        provider: a built-in or custom provider name"""
        if provider in self.providers:
            return self.providers[provider].api_key_env
        return API_KEY_ENVS[provider]

    def endpoint(self, provider: str) -> str:
        """Return the base URL of a custom provider, or "" for a built-in one (for
        cache hashes: the same name may point elsewhere after a config change).

        provider: a built-in or custom provider name"""
        if provider in self.providers:
            return self.providers[provider].base_url
        return ""

    def used_tasks(self) -> list[str]:
        """Return the tasks a run executes: the polish pass only follows the
        diarized Gemini transcribe model (the other speech-to-text models answer
        clean text), and the protocol steps can be off."""
        skipped = []
        if self.tasks.transcribe.provider != PROVIDER_GEMINI_TRANSCRIBE:
            skipped.append(TASK_POLISH)
        if not self.write_protocol:
            skipped += [TASK_PROTOCOL, TASK_AGENDA]
        return [task for task in TASKS if task not in skipped]

    def key_variables(self) -> list[str]:
        """Return the environment variables with the keys of the providers the used
        tasks run on, without duplicates."""
        return list(
            dict.fromkeys(
                self.key_variable(getattr(self.tasks, task).provider)
                for task in self.used_tasks()
            )
        )

    @staticmethod
    def load(path: Path | None) -> "Config | None":
        """Load the configuration from a YAML file; use the defaults without a file.

        Relative glossary_global, prompts_dir and local_dir paths resolve against the
        file's folder.

        path: the config.yaml to read (default: None)"""
        if path is None:
            return Config()
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            # resolved first: the prompts in it are checked while validating
            if isinstance(data, dict) and isinstance(data.get("prompts_dir"), str):
                data["prompts_dir"] = path.parent / data["prompts_dir"]
            config = Config.model_validate(data)
        except (OSError, yaml.YAMLError, ValidationError) as error:
            _LOGGER.error("Could not load the configuration %s: %s.", path, error)
            return None
        if config.glossary_global is not None:
            config.glossary_global = path.parent / config.glossary_global
        if config.local_dir is not None:
            config.local_dir = path.parent / config.local_dir
        config._folder = path.parent
        return config
