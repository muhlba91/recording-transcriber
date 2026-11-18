"""The candidate manifest (eval/candidates.yaml)."""

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ..configuration.config import Config, TaskConfig

_LOGGER = logging.getLogger(__name__)


# Candidate only exists as part of Manifest, so it shares its file
class Candidate(BaseModel):
    """One candidate: a transcribe provider and model and the settings it overrides."""

    model_config = ConfigDict(extra="forbid")

    label: str
    transcribe: TaskConfig
    chunk_minutes: int | None = None
    denoise: bool | None = None
    # may win production (design §5)
    eligible: bool = True
    # gives speaker labels; without them the speaker score is 0
    speakers: bool = True
    # third-party vendor: production needs a per-deployment sign-off (design D6)
    signoff: bool = False


class Manifest(BaseModel):
    """The candidates of an evaluation and the legacy baseline."""

    model_config = ConfigDict(extra="forbid")

    legacy: Path | None = None
    candidates: dict[str, Candidate]

    @staticmethod
    def load(path: Path) -> "Manifest | None":
        """Load a manifest; the legacy path resolves against the file's folder.

        path: the manifest file"""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            manifest = Manifest.model_validate(data)
        except (OSError, yaml.YAMLError, ValidationError) as error:
            _LOGGER.error("Could not load the manifest %s: %s.", path, error)
            return None
        if manifest.legacy is not None:
            manifest.legacy = path.parent / manifest.legacy
        return manifest

    def config_for(self, identifier: str, base: Config) -> Config | None:
        """Return the base configuration with a candidate's overrides.

        identifier: the candidate id
        base: the base configuration"""
        candidate = self.candidates[identifier]
        data = base.model_dump()
        data["tasks"]["transcribe"] = candidate.transcribe.model_dump()
        # the evaluation compares transcripts only
        data["write_protocol"] = False
        if candidate.chunk_minutes is not None:
            data["chunk_minutes"] = candidate.chunk_minutes
        if candidate.denoise is not None:
            data["denoise"] = candidate.denoise
        try:
            config = Config.model_validate(data)
        except ValidationError as error:
            _LOGGER.error("Invalid settings for candidate %s: %s.", identifier, error)
            return None
        # the pipeline's app-folder guard needs the folder
        return config.with_folder(base.folder)
