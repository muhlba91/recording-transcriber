"""Tests for the prompt templates."""

import pytest

from recording_transcriber import prompts
from recording_transcriber.configuration.config import Config
from recording_transcriber.prompts.prompt_renderer import PromptRenderer

_CONFIG = Config(
    transcript={"language": "de-DE", "notes": "Austrian German"},
    meeting={"type": "board meeting", "writer_role": "an editor"},
)
_VALUES = {
    **_CONFIG.transcript.prompt_values(),
    **_CONFIG.meeting.prompt_values(),
}


def test_render_reconcile():
    text = prompts.render(
        "reconcile_speakers", attendees="Dr. A", rosters="R", transcript="T", **_VALUES
    )
    assert "Dr. A" in text and "de-DE" in text and "$" not in text


@pytest.mark.parametrize(
    ("name", "values"),
    [
        ("polish", {"glossary": "G", "turns": "0: x"}),
        (
            "transcribe",
            {
                "part_number": "1",
                "part_count": "2",
                "part_start": "00:00:00",
                "part_duration": "25:00",
                "attendees": "A",
                "agenda": "B",
                "glossary": "G",
                "roster": "R",
                "previous": "P",
            },
        ),
    ],
)
def test_render_uses_transcript_settings(name, values):
    text = prompts.render(name, **values, **_VALUES)
    assert "de-DE" in text and "Austrian German" in text and "$" not in text


@pytest.mark.parametrize(
    ("name", "values"),
    [
        (
            "protocol_part",
            {"part_number": "1", "part_count": "2", "transcript": "T"},
        ),
        ("protocol", {"examples": "E", "parts": "P"}),
        ("agenda", {"transcript": "T"}),
    ],
)
def test_render_documents(name, values):
    text = prompts.render(
        name, attendees="A", agenda="B", glossary="G", **values, **_VALUES
    )
    assert "de-DE" in text and "$" not in text


def test_render_missing_value():
    with pytest.raises(KeyError):
        prompts.render("reconcile_speakers", attendees="Dr. A")


def test_render_meeting_settings():
    text = prompts.render(
        "agenda", attendees="A", agenda="B", glossary="G", transcript="T", **_VALUES
    )
    assert "an editor" in text and "board meeting" in text


def test_names():
    assert prompts.names() == [
        "agenda",
        "polish",
        "protocol",
        "protocol_part",
        "reconcile_speakers",
        "transcribe",
    ]


def test_placeholders():
    assert prompts.placeholders("Hi $name, ${other}x $$") == {"name", "other"}
    with pytest.raises(ValueError, match="invalid placeholder"):
        prompts.placeholders("costs $ 5")


class TestOverride:
    @pytest.fixture
    def folder(self, tmp_path):
        (tmp_path / "polish.md").write_text("Polish $language: $turns")
        yield tmp_path

    def test_template(self, folder):
        assert prompts.template("polish", folder) == "Polish $language: $turns"
        assert prompts.template("agenda", folder) == prompts.template("agenda")

    def test_render(self, folder):
        assert prompts.render("polish", folder, language="de", turns="0: x") == (
            "Polish de: 0: x"
        )

    def test_renderer(self, folder):
        renderer = PromptRenderer(Config(prompts_dir=folder))
        text = renderer.render("polish", glossary="G", turns="0: x")
        assert text == "Polish en: 0: x"
        assert renderer.template("polish") == "Polish $language: $turns"
        # the settings identify the placeholder values (cache key)
        other = PromptRenderer(Config(prompts_dir=folder, meeting={"type": "board"}))
        assert renderer.settings() != other.settings()

    def test_renderer_defaults(self):
        assert PromptRenderer().directory is None
