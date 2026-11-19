"""Tests for the configuration."""

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from recording_transcriber import prompts
from recording_transcriber.configuration.config import Config

_ROOT = Path(__file__).parents[2]


_SPEECH = {"provider": "gemini-transcribe", "model": "gemini-3.5-transcribe"}
_CUSTOM = {"litellm": {"base_url": "https://x/v1", "api_key_env": "LITELLM_KEY"}}


class TestConfig:
    def test_load_defaults(self):
        config = Config.load(None)
        assert config.model_dump() == Config().model_dump()
        assert config.folder is None

    def test_load_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "chunk_minutes: 60\nglossary_global: g.txt\nlocal_dir: scratch\n"
            "prices:\n  m: {input: 1, output: 2}\n"
        )
        config = Config.load(path)
        assert config.chunk_minutes == 60
        assert config.glossary_global == tmp_path / "g.txt"
        assert config.local_dir == tmp_path / "scratch"
        assert config.prices["m"].output == 2

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("")
        config = Config.load(path)
        assert config.model_dump() == Config().model_dump()
        assert config.folder == tmp_path

    def test_load_invalid_chunk_minutes(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("chunk_minutes: 3\n")
        assert Config.load(path) is None

    def test_load_unknown_key(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("tasks: {transcrib: {model: gemini-4-carbon}}\n")
        assert Config.load(path) is None

    def test_load_tasks(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            "tasks:\n  transcribe: {provider: mistral, model: v}\n"
            "  agenda: {provider: litellm, model: big}\n"
            "providers:\n  litellm: {base_url: 'https://x/v1', api_key_env: LITELLM_KEY}\n"
            "prices:\n  v: {per_minute: 0.004}\n"
        )
        config = Config.load(path)
        assert config.tasks.transcribe.provider == "mistral"
        assert config.tasks.agenda.provider == "litellm"
        assert config.tasks.protocol.provider == "gemini"
        assert config.prices["v"].per_minute == 0.004

    @pytest.mark.parametrize(
        "tasks",
        [
            {"transcribe": {"provider": "whisper", "model": "m"}},
            {"reconcile": {"provider": "gemini-transcribe", "model": "m"}},
            {"polish": {"provider": "gemini-transcribe-smart", "model": "m"}},
            {"transcribe": {"provider": "litellm", "model": "m"}},
            {"transcribe": {"model": ""}},
        ],
    )
    def test_invalid_tasks(self, tasks):
        providers = {"litellm": {"base_url": "https://x", "api_key_env": "K"}}
        with pytest.raises(ValidationError):
            Config(tasks=tasks, providers=providers, transcript={"language": "de-DE"})

    def test_custom_provider_named_like_builtin(self):
        with pytest.raises(ValidationError, match="built-in"):
            Config(providers={"openai": {"base_url": "https://x", "api_key_env": "K"}})

    def test_speaker_label_needs_number(self):
        with pytest.raises(ValidationError):
            Config(transcript={"speaker_label": "Speaker"})
        with pytest.raises(ValidationError):
            Config(transcript={"speaker_label": "Speaker {nr}"})
        with pytest.raises(ValidationError):
            Config(transcript={"speaker_label": "Speaker {number.x}"})

    def test_unclear_marker_not_empty(self):
        with pytest.raises(ValidationError):
            Config(transcript={"unclear_marker": ""})

    def test_gemini_transcribe_needs_region(self):
        with pytest.raises(ValidationError):
            Config(tasks={"transcribe": _SPEECH})
        config = Config(tasks={"transcribe": _SPEECH}, transcript={"language": "de-DE"})
        assert config.speech_to_text
        # a text task on Gemini needs no region
        assert not Config().speech_to_text

    def test_target_chunk_minutes(self):
        assert Config(chunk_minutes=60).target_chunk_minutes == 60
        config = Config(tasks={"transcribe": {"provider": "mistral", "model": "v"}})
        assert config.target_chunk_minutes == 25

    def test_prompt_values(self):
        values = Config().transcript.prompt_values()
        assert values == {
            "language": "en",
            "language_notes": "(none)",
            "unclear_marker": "[unclear]",
        }

    def test_load_missing(self):
        assert Config.load(Path("/nonexistent/config.yaml")) is None

    def test_meeting_prompt_values(self):
        assert Config().meeting.prompt_values() == {
            "meeting_type": "meeting",
            "writer_role": "a professional minute-taker",
        }

    def test_key_variables(self):
        assert Config().key_variables() == ["GEMINI_API_KEY"]
        config = Config(
            providers=_CUSTOM,
            tasks={
                "transcribe": {"provider": "mistral", "model": "v"},
                "reconcile": {"provider": "openai", "model": "g"},
                "agenda": {"provider": "litellm", "model": "m"},
            },
        )
        assert config.key_variables() == [
            "MISTRAL_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "LITELLM_KEY",
        ]

    def test_endpoint(self):
        config = Config(
            providers={"lab": {"base_url": "https://x/v1", "api_key_env": "K"}}
        )
        assert config.endpoint("lab") == "https://x/v1"
        assert config.endpoint("gemini") == ""

    def test_key_variables_only_used_tasks(self):
        # polish only follows gemini-transcribe; protocol and agenda can be off
        tasks = {"polish": {"provider": "openai", "model": "g"}}
        assert Config(tasks=tasks).key_variables() == ["GEMINI_API_KEY"]
        config = Config(
            tasks={**tasks, "transcribe": _SPEECH},
            transcript={"language": "de-DE"},
            write_protocol=False,
        )
        assert config.used_tasks() == ["transcribe", "polish", "reconcile"]
        assert config.key_variables() == ["GEMINI_API_KEY", "OPENAI_API_KEY"]
        assert Config(write_protocol=False).used_tasks() == ["transcribe", "reconcile"]


class TestPrompts:
    @pytest.fixture
    def folder(self, tmp_path):
        folder = tmp_path / "prompts"
        folder.mkdir()
        yield folder

    def test_valid_replacement(self, folder):
        (folder / "agenda.md").write_text(
            "Write the agenda of $meeting_type in $language_notes $language $writer_role\n"
            "$transcript $attendees $agenda $glossary $unclear_marker"
        )
        assert Config(prompts_dir=folder).prompts_dir == folder

    def test_other_files_are_ignored(self, folder):
        (folder / "notes.txt").write_text("ignored")
        assert Config(prompts_dir=folder).prompts_dir == folder

    def test_missing_placeholder_is_accepted(self, folder):
        (folder / "polish.md").write_text("Clean $turns")
        assert Config(prompts_dir=folder).prompts_dir == folder

    @pytest.mark.parametrize(
        ("name", "text"),
        [
            ("unknown.md", "x"),
            ("polish.md", "Clean $turns $bogus"),
            ("polish.md", "costs $ 5"),
        ],
    )
    def test_invalid_replacement(self, folder, name, text):
        (folder / name).write_text(text)
        with pytest.raises(ValidationError):
            Config(prompts_dir=folder)

    def test_unreadable_replacement(self, folder):
        (folder / "polish.md").write_bytes(b"\xff\xfe")
        with pytest.raises(ValidationError, match="polish.md"):
            Config(prompts_dir=folder)

    def test_missing_folder(self, tmp_path):
        with pytest.raises(ValidationError, match="not a folder"):
            Config(prompts_dir=tmp_path / "nope")

    def test_load_resolves_prompts_dir(self, folder, tmp_path):
        (folder / "polish.md").write_text("$turns $glossary $language $language_notes")
        path = tmp_path / "config.yaml"
        path.write_text("prompts_dir: prompts\n")
        assert Config.load(path).prompts_dir == folder

    def test_load_invalid_prompts(self, folder, tmp_path):
        (folder / "other.md").write_text("x")
        path = tmp_path / "config.yaml"
        path.write_text("prompts_dir: prompts\n")
        assert Config.load(path) is None


class TestExamples:
    @pytest.mark.parametrize("folder", sorted((_ROOT / "examples").iterdir()))
    def test_example_config_and_prompts_are_valid(self, folder):
        config = Config.load(folder / "config.yaml")
        assert config is not None
        assert config.glossary_global.is_file()
        if config.prompts_dir is not None:
            assert list(config.prompts_dir.glob("*.md"))

    @pytest.mark.parametrize(
        "path",
        [_ROOT / "config.example.yaml", *(_ROOT / "examples").glob("*/config.yaml")],
    )
    def test_app_version_pins_the_release(self, path):
        # release-please keeps the pins and pyproject.toml in step (release-please-config.json)
        pyproject = tomllib.loads(
            (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        config = Config.load(path)
        assert config.app_version == pyproject["tool"]["poetry"]["version"]

    def test_medical_example(self):
        folder = _ROOT / "examples" / "medical-advisory-board"
        config = Config.load(folder / "config.yaml")
        assert config.prompts_dir == folder / "prompts"
        assert config.glossary_global == folder / "glossary_global.txt"
        assert config.tasks == Config().tasks
        assert config.local_wheels
        # every packaged prompt has a specialized replacement that keeps its placeholders
        assert sorted(path.stem for path in config.prompts_dir.glob("*.md")) == (
            prompts.names()
        )

    def test_reference_config(self):
        config = Config.load(_ROOT / "config.example.yaml")
        assert config is not None
        assert config.tasks == Config().tasks
        assert config.meeting == Config().meeting
        assert config.prompts_dir is None
