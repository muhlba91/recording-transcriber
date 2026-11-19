# Recording Transcriber

[![License](https://img.shields.io/github/license/muhlba91/recording-transcriber?style=for-the-badge)](LICENSE)
[![Release workflow](https://img.shields.io/github/actions/workflow/status/muhlba91/recording-transcriber/release.yml?style=for-the-badge)](https://github.com/muhlba91/recording-transcriber/actions/workflows/release.yml)
[![Coverage](https://img.shields.io/coveralls/github/muhlba91/recording-transcriber?style=for-the-badge)](https://github.com/muhlba91/recording-transcriber/)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/muhlba91/recording-transcriber/badge?style=for-the-badge)](https://scorecard.dev/viewer/?uri=github.com/muhlba91/recording-transcriber)
[![Release](https://img.shields.io/github/v/release/muhlba91/recording-transcriber?style=for-the-badge)](https://github.com/muhlba91/recording-transcriber/releases)
[![Release date](https://img.shields.io/github/release-date/muhlba91/recording-transcriber?style=for-the-badge)](https://github.com/muhlba91/recording-transcriber/releases)
[![All contributors](https://img.shields.io/github/all-contributors/muhlba91/recording-transcriber?color=ee8449&style=for-the-badge)](#contributors)
<a href="https://www.buymeacoffee.com/muhlba91" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="28" width="150"></a>

This repository contains a **recording transcriber** that turns recordings such as medical advisory boards and symposia into **clean transcripts with consistent, named speakers**, plus a **protocol** and an **agenda**, using **Google Gemini**. It runs in **Google Colab** (`colab/recording-transcriber.ipynb`) or locally.

> [!NOTE]
> This project was co-developed with the help of AI coding assistants. All changes are reviewed and tested by the maintainer.

The transcript, protocol and agenda use a configurable language; everything else (checklists, logs, evaluation report) is English. Speech-to-text models from Google, OpenAI and Mistral were evaluated and lost against Gemini (`docs/design.md` §5); they stay available as providers.

> [!NOTE]
> Design decisions, evaluation results and cost: `docs/design.md`. Evaluation how-to: `docs/eval.md`. User guide for Colab: `how-to.md`.

---

## Features

- **Silence-aligned chunks**: ~25 min chunks cut at the longest pause, with no overlap and no tiny last chunk.
- **Speakers without input**: each chunk sees the voices found so far and the last turns before it. A final text-only pass merges drifted ids and names people only from evidence in the recording (introductions, being addressed). Unnamed voices keep a numbered label (`transcript.speaker_label`).
- **Attendee list optional**: it only corrects spellings and links name forms ("Anna" and "Dr. X"). `speakers.md` lists every form a person was called and the attendees no voice was matched to.
- **Clean verbatim**: in the configured language, no fillers, an unclear marker (`transcript.unclear_marker`) and `word (?)`, `[hh:mm:ss] Name:` paragraphs.
- **Protocol and agenda**: a descriptive protocol written per part and joined, and an overview (agenda items, attendees by organization, core topics, next steps), both in the transcript language as Markdown and Word. Example protocols in the job folder's `examples/*.docx` (not the repository's `examples/`) set the style.
- **Provider per task**: `transcribe`, `polish`, `reconcile`, `protocol` and `agenda` each have their own provider and model (`tasks`): Gemini by default, speech-to-text models (`gemini-transcribe`, OpenAI, Mistral) for `transcribe`, and custom OpenAI-compatible endpoints (`providers`) for the text tasks. Speech-to-text speaker labels are linked by the speaker pass; their verbatim output gets a polish text pass.
- **Generic prompts, configurable**: the packaged prompts fit any meeting; `meeting.type` and `meeting.writer_role` fill them, and `prompts_dir` replaces single prompts per deployment.
- **Examples**: `examples/` holds complete deployment configurations with a glossary template and prompts.
- **Evaluation harness**: runs candidate models, aligns their transcripts, and builds an offline report where a reviewer judges disagreements and speakers blind; the answers produce a leaderboard (`docs/eval.md`).
- **Resumable**: every finished chunk, the speaker pass and every protocol call are cached in `work/`, so a rerun after a disconnect continues where it stopped. Changed audio, chunk settings, model, prompts, context files or transcript settings invalidate the cache.
- **Private by default**: apart from the Gemini upload, audio stays in Drive and on the local machine (never in `work/`). Uploads to Gemini are deleted after each call, and local audio copies are deleted after a successful run.

## Job Folder

```text
<job>/
  *.mp3 ...   recordings (mp3, m4a, wav, aac, ogg, opus, flac; joined in name order), or in audio/
  *.txt       optional: glossary.txt, attendees.txt, agenda.txt (# lines are comments), or in context/
  examples/   optional: example protocols (.docx); in the job folder, or in context/examples/
  work/       resume cache (JSON) and run.log
  out/        transcript.*, protocol.*, agenda.*, flags.md, speakers.md, run.json
```

| Output | Content |
|---|---|
| `transcript.md` / `.docx` / `.json` | The transcript; Markdown and Word start with the quality notes |
| `flags.md` | Review checklist: unclear passages, uncertain words, unnamed speakers, technical warnings |
| `speakers.md` | Speakers with name forms heard, ids, confidence, evidence, voice; unmatched attendees |
| `protocol.md` / `.docx` | Descriptive protocol in the order of the meeting (missing if a call failed; see `run.json`) |
| `agenda.md` / `.docx` | Agenda items, attendees, core topics, next steps, emphases (missing if the call failed) |
| `run.json` | Provider, model, endpoint, logging setting, settings, tokens and cost estimate per step, warnings |

## Configuration

`config.yaml` (`config.example.yaml` lists every key; `examples/medical-advisory-board/` is the production configuration), passed with `--config`. Every key is optional; unknown keys are rejected. A job folder and `local_dir` must not be inside the config's folder (the app folder only holds the maintainer's files).

| Option | Description | Default |
|---|---|---|
| `tasks.<task>.provider` | Provider per task (`transcribe`, `polish`, `reconcile`, `protocol`, `agenda`): `gemini` (general model; transcription with the prompt and all text tasks), `gemini-transcribe`, `gemini-transcribe-smart` (speech-to-text, transcribe only; target capped at 25 min, so chunks stay at 28 min or less), `openai`, `mistral` (speech-to-text for `transcribe`, chat models for the text tasks; evaluation only, see the provider note below) or a custom provider (text tasks only). `polish` only runs after `gemini-transcribe` | `gemini` |
| `tasks.<task>.model` | Model per task | `gemini-4-neon` for `transcribe`, `gemini-4-carbon` for the others |
| `providers.<name>.base_url` | Custom OpenAI-compatible endpoint (chat completions with JSON-schema output) for text tasks | required per provider |
| `providers.<name>.api_key_env` | Environment variable that holds its key | required per provider |
| `write_protocol` | Write the protocol and the agenda after the transcript (the evaluation turns it off) | `true` |
| `transcript.language` | Transcript language (BCP-47); speech-to-text APIs get the primary subtag, Gemini's transcribe model needs a region (`de-DE`) | `en` |
| `transcript.notes` | Dialect, vocabulary and orthography hints for the prompts | none |
| `transcript.unclear_marker` | Marker for unintelligible speech | `[unclear]` |
| `transcript.speaker_label` | Label for unnamed speakers | `Speaker {number}` |
| `transcript.quality_note` | Quality notes at the top of the transcript: check results, named and unnamed speakers, the accuracy estimate, flag counts | `true` |
| `meeting.type` | What the recording is, free text; fills `$meeting_type` in the prompts (the packaged ones read "a recorded TYPE"; replaced prompts decide their own phrasing) | `meeting` |
| `meeting.writer_role` | The role the protocol and agenda are written in, with the article; fills `$writer_role` verbatim ("You are $writer_role") | `a professional minute-taker` |
| `prompts_dir` | Folder, relative to the config file, with `<name>.md` files that replace the packaged prompts of the same name (`agenda`, `polish`, `protocol`, `protocol_part`, `reconcile_speakers`, `transcribe`). An unknown file name or a placeholder the packaged prompt lacks is an error; a placeholder it has but the replacement omits, and a file that is not `.md`, are logged as warnings. The cache covers the prompt text | none |
| `chunk_minutes` | Target chunk length (10-120) | `25` |
| `denoise` | Apply a denoise filter before transcription | `false` |
| `thinking_level` | Gemini thinking level (Gemini only; other providers ignore it) | `low` |
| `glossary_global` | Shared glossary, relative to the config file | none |
| `local_dir` | Folder for local audio copies; never inside a job folder | system temp |
| `app_version` | Version the Colab notebook installs, quoted (`"1.0.0"`, `"1.1.0rc1"`); pin it when installing from PyPI | the highest wheel in the app folder, else the latest on PyPI |
| `local_wheels` | A wheel in the app folder wins over PyPI (e.g. a release candidate); with none found or `false`, the notebook installs from PyPI | `true` |
| `accuracy_estimates` | Per transcription model: the evaluated word accuracy quoted in the quality notes (omitted for other models) | none |
| `prices` | Per model: USD per 1M tokens (`input`, `output`) or per audio minute (`per_minute`), for `run.json` | none |

The protocol, the agenda and the transcript are always written in `transcript.language`. The packaged prompts are generic; the medical-advisory-board example replaces them with specialized ones.

One key per provider, only for the providers a run uses (all missing keys are reported at once). In Colab the notebook reads each key from a secret of the same name.

| Environment variable | Description |
|---|---|
| `GEMINI_API_KEY` | Paid-tier Google AI Studio key for `gemini*` providers (never a free-tier key). In Colab the notebook also accepts the secret `GOOGLE_API_KEY` |
| `OPENAI_API_KEY` | For `openai` |
| `MISTRAL_API_KEY` | For `mistral` (EU endpoint) |
| `providers.<name>.api_key_env` | The variable you name for a custom provider |

**Provider note**: OpenAI and Mistral are for evaluation only, for any task: production use needs a sign-off per deployment. Mistral runs on its EU endpoint; OpenAI's EU endpoint needs a data-residency project.

**Attention**: use a key from a Google Cloud project with **billing enabled**. Free-tier data may be used by Google to improve its products.

## Usage

```bash
recording-transcriber run --job runs/sample --config runs/config.yaml
recording-transcriber run --job runs/sample --model gemini-4-hydrogen   # override tasks.transcribe.model
recording-transcriber run --job runs/sample --provider mistral --model voxtral-mini-transcribe-3-0
```

Every command takes `--verbose` for debug logging.

### Colab

Two drives (`docs/design.md` D1):

- **Shared drive** `audio_transcription/` (the app; the user has Viewer access): `config.yaml`, `glossary_global.txt`, `prompts/`, `recording-transcriber.ipynb`, `how-to.md`, `CHANGELOG.md` (and a wheel for a release candidate). Nothing from a job is ever written there.
- **The user's My Drive** `Transcriptions/`: one folder per job, created by the user; the notebook creates `Transcriptions/` with a `README.txt` on first use.

The notebook mounts Drive, installs the transcriber (see below), reads the Colab secrets for the keys the config needs (`GEMINI_API_KEY`, or `GOOGLE_API_KEY`, for Gemini), picks the named job or the newest one without a transcript, and runs it; "Run all" again resumes. User guide: `how-to.md`.

**Installing the transcriber**: with `local_wheels: true` (the default), the notebook installs the wheel in the app folder (`app_version`, else the highest). Without one, or with `local_wheels: false`, it installs `recording-transcriber` from PyPI (`app_version`, else the latest) and prints the source and version. A local wheel is the way to try a release candidate in the same notebook.

**Attention**: the package is not on PyPI yet. Until the first release, build the app folder with `--wheel` (see Releases and deployments).

### Releases and deployments

A **release** (release-please, on merge of its pull request) is a tag, the `CHANGELOG.md`, the wheel and sdist on [PyPI](https://pypi.org/project/recording-transcriber/) and one asset on the GitHub release: `recording-transcriber-<version>-generic.zip`, the starter folder `examples/generic/` as an app folder. The examples are not released. The same step bumps `version` in `pyproject.toml` and the `app_version` pin in `config.example.yaml` and in every `config.yaml` under `examples/`, so each example pins the newest release.

A **deployment** is an app folder on the shared drive, built by hand from any example folder (or your own copy of one, anywhere):

```bash
mise run package                                   # examples/generic
mise run package examples/medical-advisory-board   # the production deployment
mise run package --wheel ../my-deployment          # with the local wheel
```

The task fills `release/` (the notebook, `how-to.md`, `CHANGELOG.md`, `config.yaml`, `glossary_global.txt` and `prompts/` if the folder has them) and writes `dist/recording-transcriber-<version>-<folder name>.zip` holding it as `audio_transcription/`. Upload the files or the zip to the shared drive; edit the example in the repository, not on the drive. The uploaded config pins `app_version`, so the notebook installs that release from PyPI. A new version takes effect in a fresh Colab session; roll back by pinning an older `app_version` (quoted, e.g. `"1.0.0"`). `--wheel` adds the local wheel to the folder: use it to test a release candidate (pin its version) or while PyPI does not serve the version.

### Evaluation

`recording-transcriber eval run|report|score` compares candidate models from `eval/candidates.yaml`, builds an offline report for a blind review and scores the answers. How it works, the commands, outputs and adding a candidate: `docs/eval.md`. It needs the `eval` extra (`poetry install --all-extras`). The report holds confidential audio: never publish it.

## Maintenance

- **Add a model**: add one line to the config's `tasks` (`provider`, `model`). For cost estimates add the model to `prices`, and for a transcription model with an evaluated word accuracy add it to `accuracy_estimates`. To compare it with others first, add a candidate (`docs/eval.md`).
- **Change a deployment's prompts**: edit the example's `prompts/` (e.g. `examples/medical-advisory-board/prompts/`), not the packaged generic ones in `recording_transcriber/prompts/`; change those only for every deployment. The file names are the ones `prompts_dir` accepts. Rebuild the app folder afterwards (`mise run package examples/medical-advisory-board`).
- **Releases**: see Releases and deployments.

---

## Development

The project uses [poetry](https://python-poetry.org/). Tool versions (Python, uv, Poetry and the linters) are managed with [mise](https://mise.jdx.dev): run `mise trust` and `mise install` once, then install all dependencies:

```bash
poetry install --all-extras
```

Python is pinned to the Google Colab runtime (3.13). Secrets go into `mise.local.toml` (git-ignored; template `mise.local.toml.example`).

Contribution rules and mutation testing (mutmut): `CONTRIBUTING.md`.

### Testing

1. Install all dependencies as shown above.
2. Run `pytest` (coverage must stay at or above 95 %):

   ```bash
   poetry run pytest
   ```

### Linting and Code Style

Every file type is linted: [ruff](https://github.com/astral-sh/ruff) (Python), [markdownlint-cli2](https://github.com/DavidAnson/markdownlint-cli2) (Markdown), [shellcheck](https://www.shellcheck.net/) (shell scripts) and [yamllint](https://github.com/adrienverge/yamllint) (YAML), also using [pre-commit](https://pre-commit.com/).

1. Install all dependencies as shown above.
2. (Optional) Install pre-commit hooks:

   ```bash
   poetry run pre-commit install --hook-type pre-commit --hook-type commit-msg
   ```

3. Run the linters:

   ```bash
   mise run lint
   ```

### Releasing

See Releases and deployments above. To check and build an app folder locally (CHANGELOG, lint, tests, then `release/` and the zip; takes the same `[--wheel] [folder]` as `mise run package`):

```bash
mise run release
mise run release --wheel examples/medical-advisory-board
```

### Commit Message

This project follows [Conventional Commits](https://www.conventionalcommits.org/), and your commit message must also
adhere to the additional rules outlined in `.conform.yaml`.

---

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://muehlbachler.io/"><img src="https://avatars.githubusercontent.com/u/653739?v=4?s=100" width="100px;" alt="Daniel Mühlbachler-Pietrzykowski"/><br /><sub><b>Daniel Mühlbachler-Pietrzykowski</b></sub></a><br /><a href="#maintenance-muhlba91" title="Maintenance">🚧</a> <a href="https://github.com/muhlba91/recording-transcriber/commits?author=muhlba91" title="Code">💻</a> <a href="https://github.com/muhlba91/recording-transcriber/commits?author=muhlba91" title="Documentation">📖</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Supporting

If you enjoy the application and want to support my efforts, please feel free to buy me a coffee. :)

<a href="https://www.buymeacoffee.com/muhlba91" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="75" width="300"></a>
