# Recording Transcriber how-to

The Recording Transcriber turns a meeting recording into a transcript with named speakers, a protocol and an agenda. It runs in Google Colab in your browser. Time and cost depend on the length of the recording and on the configuration; `run.json` in `out/` records the actual cost of each run.

## 1. One-time setup

**Your Gemini API key.** You need a key from Google AI Studio in a project with **billing enabled**. Without billing the key is on the free tier, and Google may use free-tier data to improve its products: never use such a key for recordings.

1. Sign in to Chrome with your work Google account. Open the shared drive folder `audio_transcription`, double-click `recording-transcriber.ipynb` and choose **Open with Google Colaboratory**.
2. Click the **key icon** (Secrets) in the left sidebar.
3. Click **Gemini API keys**, then import an existing key or create a new one in Google AI Studio. In AI Studio, choose your project with billing and click **Create API key**.
4. Back in Colab, make sure the secret is called `GEMINI_API_KEY` or `GOOGLE_API_KEY` (the import creates one of them) and switch on **Notebook access**.
5. Your administrator (who set up the app folder) checks once that the project is on a paid plan, not the free tier.

The key stays in your Colab account; it is never written into the notebook or a file.

**Your jobs folder.** In your My Drive, create a folder named `Transcriptions` (or let the notebook create it on its first run). Keep the name: the notebook looks for it.

## 2. Every job

1. In `Transcriptions`, create a new folder for the meeting, e.g. `2026-12 Advisory Board Vienna`, and upload the recording(s) into it (mp3, m4a, wav, aac, ogg, opus or flac). Several files are joined in name order ("Part 1", "Part 2", ...).
2. Optional, in the same folder:
   - `attendees.txt`: one person per line, e.g. `Prof. Dr. Anna Example, Example University Hospital`. It corrects the spelling of names; it never assigns a name to a voice on its own.
   - `agenda.txt`: the planned agenda.
   - `glossary.txt`: one term per line (drugs, studies, abbreviations) in the right spelling.
   - `examples/`: one or two earlier protocols as Word files; the new protocol follows their style.
3. Open the notebook and choose **Runtime > Run all**. The first time in a session, Colab asks for permission to access your Google Drive: allow it with your work account. The notebook picks the most recently changed folder with recordings and without a transcript; to pick another one, type its name into the `JOB` field in step 1 first.
4. When step **5. Run** prints "Done", the results are in the meeting folder under `out/`.

Colab may say that you cannot save the notebook: that is expected. Just close it; everything you do happens in your My Drive.

## 3. The results (`out/`)

| File | Content |
|---|---|
| `transcript.docx` / `.md` | The transcript, starting with quality notes |
| `protocol.docx` / `.md` | The protocol, in the order of the meeting |
| `agenda.docx` / `.md` | Agenda, attendees, core topics, next steps |
| `flags.md` | What to check: unclear passages, uncertain words, unnamed speakers |
| `speakers.md` | Every speaker, every form of their name heard, and why the name was given |
| `run.json` | Settings, usage and cost of the run |

How to read the **quality notes** at the top of the transcript:

- "Automatic checks: passed" means no gaps, repetitions or timestamp problems were found. "Check recommended" names the part to look at.
- Unnamed speakers are numbered ("Speaker 1", "Speaker 2", ... or the label your configuration sets). A name is only given when the recording supports it (an introduction or being addressed); `speakers.md` shows the evidence.
- If the note quotes a word accuracy, it is an estimate from an evaluation of the model, not a measurement of this recording. Names and technical terms are the most likely errors, even when `flags.md` lists nothing.

## 4. If something goes wrong

- **The connection dropped or the browser closed**: open the notebook and choose **Run all** again. It continues where it stopped; only the part that was interrupted is done again.
- **"Missing secret(s)"** or **"Allow access to the secret"**: repeat part 1, steps 2-4.
- **"No job to do"**: there is no new folder, or it has a transcript already. Type the folder name into `JOB` to redo it.
- **"The app folder audio_transcription was not found"** or **"Installing the transcriber ... failed"** (twice in a row): ask your administrator.
- **Anything else**: run it once more. If it fails again, email your administrator the file `work/run.log` from the meeting folder.

To redo a meeting from scratch, delete its `out/` and `work/` folders.

## 5. Privacy

- The recordings stay in your My Drive. For transcription, each part is uploaded to Google's Gemini API and the upload is deleted right after (Google deletes leftovers after 48 hours anyway).
- With a paid key, Google does not use the data for training. Google may keep the requests, including the audio, in logs for up to 55 days (abuse monitoring and API logs).
- Colab keeps a temporary copy of the audio while it runs and deletes it when the transcript is done; the Colab machine is discarded after the session.
- Nothing from your jobs is ever written to the shared drive.
