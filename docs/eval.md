# Evaluation

How candidate models are compared and how to re-evaluate. Results and decisions: `docs/design.md` §5 and D6.

## What it measures

Every candidate transcribes the same recording with the same context; only the `transcribe` task differs (`write_protocol` is off). The harness then:

1. **Normalizes** each transcript to timed words (`evaluation/words.py`): case, punctuation, fillers, number words and decimals, markers such as `[unclear]` and `word (?)` removed.
2. **Aligns** the candidates to a reference (`alignment.py`): the most typical transcript (lowest mean WER) among eligible candidates that have speakers and real timestamps.
3. **Disqualifies** a candidate with a dropped passage over 30 s (how truncation shows) or a repetition loop.
4. **Picks disagreement spans** (`spans.py`): 40 places where candidates wrote different words, medical terms first, spread over the recording.
5. **Picks speaker spot checks** (`spot_checks.py`): short clips of one voice with the label each candidate gave it, some after the first chunk boundary where ids may drift.
6. **Builds an offline report** (`report_writer.py`) in which a reviewer judges spans and speakers blind and rates the start of the top candidates for readability.
7. **Scores** the exported answers (`scoring.py`) and writes a leaderboard (`leaderboard.py`): words, speakers, readability and cost, combined with fixed weights; within 3 points the cheaper candidate wins. Weights and the exact rules are in the code (`utils/const.py`) and `docs/design.md` §5.

### Language-specific normalization

The language is `transcript.language` of the base config; its primary subtag picks the table:

| Language | Fillers dropped | Number words | Decimal handling |
|---|---|---|---|
| `de` | `äh`, `ähm`, `hm`, ... | `null` to `zwölf`, `zwanzig`, `dreißig`, `hundert`; `%` as `prozent` | comma and point are the same number (`2,5` = `2.5`) |
| `en` | `uh`, `um`, `er`, `hm`, ... | `zero` to `twenty`, the tens up to `ninety`, `hundred`, `thousand`; `%` as `percent` | the comma groups thousands (`1,000` = `1000`), the point is decimal |
| other | none | none | comma and point are the same number |

Only these common number words are normalized (no compounds such as `einundzwanzig`). For another language the evaluation still runs, but fillers and number words count as differences. Add a table in `evaluation/words.py` (`_TABLES`) for a language you evaluate often.

## Candidates

`eval/candidates.yaml` (`--manifest`) lists them. Each runs as its own job on top of the base config (`--config`, e.g. `examples/medical-advisory-board/config.yaml`):

```yaml
legacy: ../path/to/transcription.txt   # optional: a previous transcript, imported as is (C0); relative to this file
candidates:
  C3: { label: "Hydrogen, pipeline", transcribe: { model: gemini-4-hydrogen } }
  C10:
    label: "Mistral Voxtral Mini Transcribe 3"
    transcribe: { provider: mistral, model: voxtral-mini-transcribe-3-0 }
    signoff: true
```

| Key | Meaning | Default |
|---|---|---|
| `label` | Name in the report and leaderboard | required |
| `transcribe` | `{ provider, model }` of the transcribe task; the provider defaults to `gemini` | required |
| `chunk_minutes`, `denoise` | Override the base config | base config |
| `eligible` | May win production | `true` |
| `speakers` | The candidate gives speaker labels (otherwise its speaker score is 0) | `true` |
| `signoff` | Third-party vendor; production needs a sign-off (README, provider note) | `false` |

Unknown keys are rejected. A custom provider must be declared under `providers` in the base config.

**Adding a candidate**: add an entry with a new id, then `eval run --only <id>`. Rebuild the report afterwards; the reviewer has to judge it again, because a rebuild makes earlier answers invalid.

## Commands

Install the `eval` extra first (`poetry install --all-extras`). All subcommands take `--out` (default `runs/eval`) and `--verbose`. `run` and `report` also take `--manifest` (default `eval/candidates.yaml`), `--job` (required) and `--config` (the base config; defaults apply without it). `run` takes `--only`, `report` takes `--report`, `score` requires `--answers`.

```bash
recording-transcriber eval run --job runs/sample --config examples/medical-advisory-board/config.yaml [--only C1,C5]
recording-transcriber eval report --job runs/sample --config examples/medical-advisory-board/config.yaml [--report report]
recording-transcriber eval score --answers answers.json
```

- `run`: runs the candidates (and imports `legacy` as `C0`). The job folder needs `audio/` and `context/` (the candidates link to them, nothing is copied). A candidate with `out/run.json` is finished and skipped; delete its folder to run it again. A failed one resumes from its `work/`.
- `report`: aligns the finished candidates and writes the report folder (`--report`, default `report/`) and `<out>/report_data.json`, the key the scores need.
- `score`: reads `<out>/report_data.json` and the reviewer's `answers.json`, and writes `<out>/leaderboard.md` and `<out>/leaderboard.json`. It refuses answers that belong to another report.

### Outputs

| Path | Content |
|---|---|
| `runs/eval/<id>/` | One job per candidate (`work/`, `out/transcript.*`, `out/run.json` with tokens and cost) |
| `runs/eval/report_data.json` | Spans, spot checks and the answer key |
| `report/` | `index.html` and audio clips of the recording |
| `runs/eval/leaderboard.md` / `.json` | Scores and decisions |

`report/` works offline and holds confidential audio: share it only privately, never publish it. A rebuild replaces the folder (only if it holds a report). The reviewer's answers stay in the browser until exported as `answers.json`. `runs/` and `report/` are git-ignored.

## Keys and cost

One key per provider the candidates use, read from the environment (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `MISTRAL_API_KEY`, or the variable of a custom provider); all missing keys are reported at once. Check a provider's data-use terms before sending real recordings to it.

Each candidate transcribes the whole recording and the speaker pass runs per candidate, so a run costs roughly the sum of the candidates' prices for that recording. `out/run.json` of each candidate holds the estimate (from `prices` in the config); `docs/design.md` §6 has measured figures. Use `--only` while iterating and a short recording while testing a new candidate.
