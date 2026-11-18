# Design

Why the Recording Transcriber works the way it does: decisions, pipeline, evaluation, cost and risks. Usage and configuration: `README.md`.

## 1. Context, goal, constraints

**Context.** Medical writers record advisory boards and symposia: 1-3 h, 4-8 speakers, one phone or table recorder, echo and distance, medical terms, often in a regional variant of the language with dialect. The writer turns the recording into a transcript, a protocol and an agenda.

**Goal.** Good transcripts, above all **consistent speakers with real names** across the whole recording; protocol and agenda as a bonus. Comparing new models must stay cheap.

**Constraints.**

- A few jobs a year, run by a non-technical user: simple, near-zero upkeep, nothing running 24/7.
- Audio stays in the user's Google Drive; one paid-tier Google AI Studio key in production, no Vertex AI.
- No reference transcript exists, and the reviewer's time is limited.
- Project-local tools only (mise, `.venv`); secrets only in the environment.

**Earlier approach** (`legacy/audio_transcription.ipynb`): each 20 min chunk was transcribed in isolation, so names drifted between chunks; no timestamps, fillers kept, misspelled terms; no context files; uploads never deleted.

## 2. Decisions

**D1. Runtime: Colab with a thin notebook; the code is a versioned wheel on a shared drive, the jobs live in the user's My Drive.**

- No infrastructure, Drive mounted with the user's own login, audio stays inside Google. Session limits (12 h, idle timeout) are handled by resumable runs.
- Rejected: a service watching Drive (service accounts, runs all year), Cloud Run (new cloud surface), a local script (Python and ffmpeg on a non-technical laptop), Apps Script (6 min per run, no ffmpeg).
- Two drives. The shared drive `audio_transcription/` is the app: only the maintainer's files (list: README, Colab section), built and uploaded by hand (README, Releases and deployments); the user has Viewer access, so nothing of theirs can land there (Colab opens the notebook read-only; the code also refuses jobs inside the app folder). The user's My Drive `Transcriptions/` holds one folder per job.
- The notebook has five steps and is never edited: mount Drive and find the app folder, `pip install` the wheel from the app folder if there is one (`app_version`, else the highest; none or `local_wheels: false`: the pinned `app_version` from PyPI; README, Colab), read one Colab secret per key the config needs (`GEMINI_API_KEY` for the default; `GOOGLE_API_KEY` is also accepted), pick the named job or the newest one without a transcript, run with a log copy in `work/run.log`. Notebook outputs are private (never saved into the file). Update by uploading an app folder with a new `app_version` pin (it takes effect in a fresh Colab session); roll back by pinning the older version.

**D2. Everything goes directly to Gemini.** A proxy would add a hop for the audio and does not support the Files API. Production uses Gemini for the text steps (speakers, polish, protocol, agenda) too: a second provider would mean a second key for the user. Each task has its own provider and model (`tasks`); OpenAI, Mistral and custom OpenAI-compatible endpoints (`providers`) can serve the text tasks through a chat-completions backend, one key per provider, but stay subject to D6.

**D3. Pipeline: chunks with carried-over context, structured output, one text-only speaker pass** (§4).

**D4. Evaluation without a reference: blind judgment of disagreements** (§5). The reviewer judges short clips where candidates differ instead of correcting a whole recording.

**D5. Output: English everywhere except the transcript, protocol and agenda (`transcript.language`); the transcript is "clean verbatim".**

- Checklists (`flags.md`, `speakers.md`), model notes and evidence, logs, the evaluation report and the how-to are English.
- Fillers and false starts are removed, wording is kept. Unclear passages get `transcript.unclear_marker`, uncertain words `word (?)`, unnamed speakers `transcript.speaker_label`. Paragraphs start `[hh:mm:ss] Name:`.
- `examples/medical-advisory-board/` is the production configuration: Austrian German (`de-DE`, notes on Austrian vocabulary and dialect, `[unklar]`, `Sprecher {number}`), `meeting` set to a medical advisory board, a glossary template and specialized prompts. The packaged prompts are generic and configurable per deployment (`meeting`, `prompts_dir`); the README has the mechanics and `config.example.yaml` every key.
- Protocol and agenda: transcript language, no preamble; what the transcript does not say is left out, never a placeholder.

**D6. Privacy: audio only goes to providers whose API terms exclude training; any other vendor needs an explicit OK (per vendor for evaluation, per deployment for production).**

- **Google, paid key:** allowed for all recordings. The user creates the key in Google AI Studio in a project with billing enabled. With a paid key, Google does not use prompts and files for training; abuse-monitoring logs are kept up to 55 days; developer logs are on by default for the Interactions API (up to 55 days, adjustable). Uploads are deleted after each call (Google deletes them after 48 h anyway). No EU residency, no zero retention. A free-tier key trains on data and is never used.
- **OpenAI, Mistral:** evaluation only; production needs a per-deployment sign-off (README, provider note).
- Audio never goes to `work/` or the job folder; `out/run.json` records provider, model, endpoint and logging setting.

**D7. Local environment.**

- `mise.toml` and `pyproject.toml` pin Python 3.13, the Colab runtime. Bump both when Colab moves.
- mise installs Python, uv, poetry and the linters; `poetry install` fills `.venv`; `poetry build` makes the wheel. ffmpeg comes from `imageio-ffmpeg` (system ffmpeg if present).
- `mise.local.toml` (git-ignored) sets `GEMINI_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`. Keys are never printed, logged or committed.

## 3. Architecture

```text
recording_transcriber/    # flat package, packages by role
  cli.py, __main__.py     # recording-transcriber run | eval run|report|score
  notebook.py             # Colab helpers: workspace, job picker, run with log copy
  configuration/          # config.py (config.yaml, tasks, prompt checks), job.py (job folder, context, work/ cache)
  audio/                  # ffmpeg.py (prepare, silences, cut), chunker.py
  models/                 # pydantic models, one per file
  backends/               # base.py (protocols), gemini.py, gemini_speech.py, speech_api.py,
                          # chat_api.py, openai_speech.py, mistral_speech.py, registry.py
  prompts/                # generic transcribe.md, polish.md, reconcile_speakers.md,
                          # protocol_part.md, protocol.md, agenda.md, schemas.py, prompt_renderer.py
  pipeline/               # pipeline.py, chunk_transcriber.py, speech_transcriber.py,
                          # turn_polisher.py, speaker_reconciler.py, protocol_writer.py
  render/                 # paragraphs.py, markdown.py, word.py, records.py, review.py, quality.py
  evaluation/             # manifest, runner, legacy, words, alignment, spans, spot_checks,
                          # report_data, report_writer (+ report_page.html), scoring, leaderboard, commands
  utils/                  # const.py, checks.py, cost.py, retry.py, roster.py, serialize.py, timestamps.py
tests/                    # mirrors the package; fake backends replay hand-built responses
colab/recording-transcriber.ipynb  # the notebook shipped to the app folder
```

Drive (D1):

```text
Shared drive  audio_transcription/  the app: config, prompts, notebook, how-to, optional wheel (read-only)
My Drive      Transcriptions/       README.txt, one folder per job:
              Transcriptions/<job>/ recordings (or audio/), optional attendees.txt, agenda.txt,
                                    glossary.txt (or context/), examples/*.docx (in the job folder) · work/ (cache, run.log) · out/
```

## 4. Pipeline

1. **Prepare.** Join the recordings into one 16 kHz mono mp3 (48 kbps, `loudnorm`). Google downsamples to 16 kbps mono anyway.
2. **Chunk.** Target `chunk_minutes` (25), cut in the longest silence within ±3 min, no overlap, no tiny last chunk. Speech-to-text backends are capped at a 25 min target (chunks ≤28 min).
3. **Context.** Global plus job glossary (≤300 terms, "prefer when acoustically plausible, never insert"), attendees and agenda if given, `transcript.language` and `notes`. Everything works without context. The attendee list may be wrong: it only fixes spellings and links name forms; it never names a voice on its own.
4. **Transcribe chunks in order.** Each call gets the audio (Files API), the context, the roster so far (id, name, evidence, voice) and the last 8 turns. Structured output: turns, roster, unclear passages. Ids are reused for known voices; a name needs a self-introduction or direct address, quoted as evidence. Streaming, `thinking_level: low`.
   - Retries: 4 attempts with 10/20/40 s backoff; one re-ask on a schema error; a truncated answer splits the chunk.
   - Sanity checks (retry once, then flag): last turn within 120 s of the chunk end, timestamps in order, no 5 min without a turn, no repetition loop; words per minute far off the median are flagged (from 3 chunks).
   - Each finished chunk is cached in `work/`; a rerun resumes. The cache key covers audio, chunk settings, provider, model, prompt text, context and transcript settings.
   - Speech-to-text backends instead return segments with per-call labels; verbatim output gets a polish text pass.
5. **Reconcile speakers.** One text call over all rosters and turns merges drifted ids and names people from introductions and forms of address, with confidence, evidence and every name form heard. Low confidence keeps the numbered label. The code applies the mapping.
6. **Render** to `out/`: `transcript.md`, `.docx` (both start with quality notes: check results, named and unnamed speakers with the unnamed share of words, the model's evaluated accuracy from `accuracy_estimates`, flag counts; no per-job score, because models mark almost none of their own errors), `.json`; `flags.md` (review checklist); `speakers.md` (name forms, evidence, voice, unmatched attendees); `run.json` (models, endpoints, tokens and cost per step, warnings). Uploads and local audio copies are deleted.
7. **Protocol and agenda** (`write_protocol`, on by default; off in the evaluation). Written after the transcript files, so a failure never costs the transcript.
   - A descriptive protocol per part (chunk) from its paragraphs with the reconciled names, then one call joins them (merging topics split at a boundary). Optional `examples/*.docx` in the job folder (or `context/examples/`; ≤30,000 characters) are style references for the join only.
   - The agenda (agenda items, attendees by organization, core topics, next steps, emphases) comes from the full transcript in one call.
   - Plain-text structured output (title, sections with paragraphs and bullets) rendered to `protocol.md`/`.docx` and `agenda.md`/`.docx`.
   - Each answer is cached in `work/` under a hash of its prompt, provider, model and thinking level. If a part fails, there is no protocol (a gap would look complete) and older protocol files are removed; failures are retried on the next run and listed in `run.json`.

## 5. Evaluation

**Method** (`recording-transcriber eval run|report|score`, candidates in `eval/candidates.yaml`; how to re-run it: `docs/eval.md`):

- Every candidate transcribes the same recording with the same context. Transcripts are normalized (case, punctuation, fillers, number words, decimals) and aligned to a reference: the most typical transcript (lowest mean WER) among eligible candidates with speakers and real timestamps.
- Disqualified: a dropped passage over 30 s (how truncation shows) or a repetition loop.
- The report (`report/`, offline, blind): 40 disagreement spots (2-6 variants, medical terms first, spread over the recording), optional speaker clips, then the start of the top two candidates rated for readability.
- Scores (0-100): words = share of spots matching the reviewer's choice (medical counted double); speakers = pairwise same/different agreement; cost = cheapest $/h over the candidate's. Pre-score 0.53 words + 0.35 speakers + 0.12 cost; the final score adds readability (0.45/0.30/0.15/0.10). Within 3 points the cheaper wins.

**Results** on a 53 min test recording in Austrian German (2026-11, ~$3.90 for all runs):

| ID | Candidate | Words (of 40) | Measured $/h | Note |
|---|---|---|---|---|
| C2 | Carbon | 20 | 0.47 | |
| C1 / C7 | Argon / Argon whole file | 19 / 19 | 0.47 / 0.45 | |
| **C3** | **Hydrogen** | **18** | **0.44** | **winner** |
| C9 / C8 | OpenAI `gpt-transcribe` / Argon denoised | 15 / 15 | 0.30 / 0.47 | C9: no speakers |
| C0 | Legacy notebook | 14 | | |
| C4 | 3.8 Flash | 10 | 0.19 | |
| C5b | `gemini-3.5-transcribe`, 10 min chunks | 6 | 0.55 | |
| C10 | Mistral Voxtral Mini Transcribe 3 | 4 | 0.28 | |
| C5 / C6 | `gemini-3.5-transcribe`, 25 min chunks | | | disqualified: silent truncation |

- The four Gemini 4 runs tie (they differ on 9-10 spots). Hydrogen wins by the cost rule.
- `chunk_minutes` stays 25, `denoise` stays off (it hurt). Dedicated speech-to-text models are clearly worse on these recordings.
- Speakers and readability were not scored; they are judged on real jobs via `speakers.md` and the transcript.
- **Accuracy estimate** (`accuracy_estimates`): word disagreement with Argon and Carbon is 5.8 % for Hydrogen and 11.2-11.6 % for the legacy output; two top runs disagreeing by ~4 % puts each at ≥2 % errors. Estimated word accuracy: Hydrogen ~94-97 %, Argon/Carbon ~95-98 %, legacy ~88-92 %. Errors all models share (e.g. a misheard surname) are invisible.

## 6. Cost

Measured per audio hour, at the text rate (Google may bill audio higher): Hydrogen ~$0.44, Argon and Carbon ~$0.47 (double from 2026-12-01), 3.8 Flash ~$0.19. The speaker pass adds about $0.04 per job, protocol and agenda about $0.22 per audio hour. These figures come from the 53 min runs and do not add up exactly for longer recordings: a 3 h job cost $1.71 in total (transcript, speakers, protocol, agenda), about $2-2.50 once Argon (speaker pass, protocol, agenda) doubles. Colab's free CPU runtime suffices.

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Speakers drift or stay unnamed | Speaker pass; if names prove unreliable, pass short voice clips per named speaker to later chunks (voice enrollment) |
| Audio billed above the text rate | Visible in `run.json`; 3.8 Flash is the cheaper fallback |
| Glossary causes invented terms | "Only if acoustically plausible" rule |
| Silent truncation | Coverage checks and the 120 s tail check |
| Colab disconnect | Resume from `work/`; the notebook and how-to say "Run all again" |
| A free-tier key (data used for training) | The how-to requires billing; the maintainer checks the project once |
| Model deprecated or renamed | Ids in `config.yaml`; evaluate a new candidate |
| Developer logs keep audio up to 55 days | Stated in the how-to |

## 8. Roadmap

- Voice enrollment, if names prove unreliable on real jobs.
- Out of scope unless all API models fail: self-hosted Whisper/pyannote (needs a GPU).
