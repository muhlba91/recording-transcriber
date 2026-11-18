"""Tests for the candidate manifest."""

from pathlib import Path

from recording_transcriber.configuration.config import Config
from recording_transcriber.evaluation.manifest import Manifest


class TestManifest:
    def test_load(self, tmp_path):
        path = tmp_path / "candidates.yaml"
        path.write_text(
            "legacy: legacy.txt\n"
            "candidates:\n"
            "  C1: {label: a, transcribe: {model: gemini-4-argon}}\n"
            "  C7: {label: b, transcribe: {model: m}, chunk_minutes: 60, denoise: true}\n"
            "  C9: {label: c, transcribe: {provider: openai, model: gpt-transcribe}}\n"
        )
        manifest = Manifest.load(path)
        assert manifest.legacy == tmp_path / "legacy.txt"
        assert manifest.candidates["C1"].eligible
        base = Config(chunk_minutes=30)
        assert manifest.config_for("C1", base).chunk_minutes == 30
        assert not manifest.config_for("C1", base).write_protocol
        whole = manifest.config_for("C7", base)
        assert (whole.chunk_minutes, whole.denoise) == (60, True)
        openai = manifest.config_for("C9", base)
        assert openai.tasks.transcribe.provider == "openai"
        assert openai.tasks.transcribe.model == "gpt-transcribe"
        assert base.tasks.transcribe.provider == "gemini"

    def test_load_without_legacy(self, tmp_path):
        path = tmp_path / "candidates.yaml"
        path.write_text("candidates: {}\n")
        assert Manifest.load(path).legacy is None

    def test_load_invalid(self, tmp_path):
        path = tmp_path / "candidates.yaml"
        path.write_text(
            "candidates:\n  C1: {label: a, transcribe: {model: m}, typo: 1}\n"
        )
        assert Manifest.load(path) is None
        assert Manifest.load(tmp_path / "missing.yaml") is None

    def test_config_for_invalid_override(self):
        manifest = Manifest.model_validate(
            {
                "candidates": {
                    "C1": {
                        "label": "a",
                        "transcribe": {"provider": "whisper", "model": "m"},
                    }
                }
            }
        )
        assert manifest.config_for("C1", Config()) is None

    def test_config_for_keeps_folder(self, tmp_path):
        (tmp_path / "config.yaml").write_text("chunk_minutes: 30\n")
        base = Config.load(tmp_path / "config.yaml")
        manifest = Manifest.model_validate(
            {"candidates": {"C1": {"label": "a", "transcribe": {"model": "m"}}}}
        )
        assert manifest.config_for("C1", base).folder == tmp_path

    def test_repository_manifest_is_valid(self):
        path = Path(__file__).parents[2] / "eval" / "candidates.yaml"
        manifest = Manifest.load(path)
        # optional, so the repository manifest leaves it out
        assert manifest.legacy is None
        base = Config(transcript={"language": "de-DE"})
        for identifier in manifest.candidates:
            assert manifest.config_for(identifier, base) is not None
