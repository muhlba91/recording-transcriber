"""Transcriber constants."""

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# job folder layout
AUDIO_DIR = "audio"
CONTEXT_DIR = "context"
WORK_DIR = "work"
OUT_DIR = "out"

# accepted audio uploads
AUDIO_EXTENSIONS = (".mp3", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".flac")

# context files in context/
GLOSSARY_FILE = "glossary.txt"
ATTENDEES_FILE = "attendees.txt"
AGENDA_FILE = "agenda.txt"
# example protocols (Word) in context/examples/: style references for the protocol
EXAMPLES_DIR = "examples"
EXAMPLES_GLOB = "*.docx"
# style only; longer examples just cost tokens
MAX_EXAMPLE_CHARS = 30000
# files saved by Windows tools may not be UTF-8
FALLBACK_ENCODING = "cp1252"

# longer glossaries dilute the prompt without helping
MAX_GLOSSARY_TERMS = 300

# resume cache in work/
PLAN_FILE = "chunks.json"
CHUNK_FILE = "chunk_{index:02d}.json"
CHUNK_GLOB = "chunk_*.json"
MAPPING_FILE = "mapping.json"
PROTOCOL_PART_FILE = "protocol_part_{index:02d}.json"
# heading of each part in the join prompt
PROTOCOL_PART_HEADING = "=== Part {number} ==="
PROTOCOL_FILE = "protocol.json"
AGENDA_CACHE_FILE = "agenda.json"

# outputs in out/
TRANSCRIPT_MD = "transcript.md"
TRANSCRIPT_DOCX = "transcript.docx"
TRANSCRIPT_JSON = "transcript.json"
FLAGS_MD = "flags.md"
SPEAKERS_MD = "speakers.md"
PROTOCOL_MD = "protocol.md"
PROTOCOL_DOCX = "protocol.docx"
AGENDA_MD = "agenda.md"
AGENDA_DOCX = "agenda.docx"
RUN_JSON = "run.json"
# a copy of the console log in work/, so a failed run can be debugged remotely
RUN_LOG = "run.log"

# the notebook's workspace in My Drive: one folder per job; names starting with these
# are not jobs
WORKSPACE_README = "README.txt"
IGNORED_JOB_PREFIXES = (".", "_")
WORKSPACE_README_TEXT = """\
One folder per meeting. Put the recordings (mp3, m4a, wav, ...) into a new folder
here, then open the transcriber notebook and choose Runtime > Run all.

Optional files in the same folder: attendees.txt (one person per line), agenda.txt,
glossary.txt (one term per line) and examples/ with example protocols (.docx).
The results appear in out/; work/ is the resume cache. See how-to.md.
"""

# local audio copies (never in the job folder)
PREPARED_FILE = "prepared.mp3"
PREPARED_PARTIAL_FILE = "prepared.partial.mp3"

# prepared audio: Google downsamples to 16 kbps mono anyway, so nothing is lost
SAMPLE_RATE = 16000
BITRATE = "48k"
DENOISE_FILTER = "afftdn=nf=-25"

# chunking: cut at the longest silence within this window around the target
CHUNK_WINDOW_S = 180
SILENCE_NOISE = "-35dB"
SILENCE_MIN_S = 0.3

# model calls: 10 + 20 + 40 s backoff rides out short 429/503 bursts
MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 10
# a truncated response is split in half at most this often
MAX_SPLIT_DEPTH = 2
# streaming lifts the 32k cap that Argon/Carbon have for non-streaming calls
MAX_OUTPUT_TOKENS = 65536
# long audio calls take minutes; the SDK expects milliseconds
REQUEST_TIMEOUT_MS = 15 * 60 * 1000
UPLOAD_POLL_S = 2
UPLOAD_TIMEOUT_S = 120
PREVIOUS_TURNS = 8
PROMPT_PLACEHOLDER = "(none)"

# model response states
STATUS_COMPLETED = "completed"
STATUS_INCOMPLETE = "incomplete"
STATUS_UNKNOWN = "unknown"

# sanity checks
MAX_TAIL_GAP_S = 120
# turns are requested every 60-90 s, so 5 min without one suggests dropped audio
MAX_TURN_GAP_S = 300
# tolerance for timestamps that go backwards or past the chunk end
TIMESTAMP_SLACK_S = 5
LOOP_NGRAM = 8
LOOP_REPEATS = 4
WPM_MIN_RATIO = 0.5
WPM_MAX_RATIO = 2.5
WPM_MIN_CHUNKS = 3

# warning codes (run.json; the user-facing ones are translated in flags.md)
WARNING_NO_TURNS = "NO_TURNS"
WARNING_TAIL_GAP = "TAIL_GAP"
WARNING_TURN_GAP = "TURN_GAP"
WARNING_TIMESTAMPS = "TIMESTAMPS"
WARNING_LOOP = "REPETITION_LOOP"
WARNING_WPM = "WPM_OUTLIER"
WARNING_SPLIT = "SPLIT"
WARNING_RECONCILIATION = "RECONCILIATION_FAILED"
WARNING_POLISH = "POLISH_FAILED"
WARNING_PROTOCOL = "PROTOCOL_FAILED"
WARNING_AGENDA = "AGENDA_FAILED"

# speaker confidence
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"
NAMED_CONFIDENCE = (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM)
# consecutive turns of one person merge into one paragraph spanning at most this
MERGE_GAP_S = 90

# flags.md and speakers.md
WARNING_TEXT = {
    WARNING_NO_TURNS: "Part {part} has no text.",
    WARNING_TAIL_GAP: "The end of part {part} may be missing.",
    WARNING_TURN_GAP: "A passage in part {part} may be missing.",
    WARNING_TIMESTAMPS: "The timestamps in part {part} are unreliable.",
    WARNING_LOOP: "Part {part} has suspicious repetitions.",
    WARNING_WPM: "Part {part} has unusually much or little text.",
    WARNING_POLISH: "Part {part} is not polished (fillers kept).",
}
RECONCILIATION_TEXT = (
    "Speaker reconciliation failed; names come straight from the transcription."
)
EXCERPT_CHARS = 60

# quality note at the top of the transcript (transcript.quality_note)
QUALITY_HEADING = "Quality notes"
QUALITY_PASSED = (
    "Automatic checks: passed (no gaps, repetitions or timestamp problems found)."
)
QUALITY_CHECK = "Check recommended: {hints}"
QUALITY_ALL_NAMED = "Speakers: all {named} named."
QUALITY_UNNAMED = "Speakers: {named} named; {unnamed} unnamed ({share} % of the words)."
QUALITY_SMALL_SHARE = "<1"
QUALITY_ESTIMATE = (
    "Typical word accuracy for this setup, estimated from the evaluation and not "
    "measured on this recording: {estimate}."
)
QUALITY_FLAGS = (
    "The model marked {unclear} unclear passage(s) and {uncertain} uncertain "
    "word(s) (see flags.md); unmarked errors remain likely, most often in names "
    "and technical terms."
)
# a model note and an unclear marker this close are the same passage
UNCLEAR_DEDUP_S = 120

# providers (config.yaml "providers" and "tasks"); gemini is a general model that takes
# the prompt, the gemini-transcribe ones are speech-to-text models for the transcribe task
PROVIDER_GEMINI = "gemini"
PROVIDER_GEMINI_TRANSCRIBE = "gemini-transcribe"
PROVIDER_GEMINI_TRANSCRIBE_SMART = "gemini-transcribe-smart"
PROVIDER_OPENAI = "openai"
PROVIDER_MISTRAL = "mistral"
PROVIDERS = (
    PROVIDER_GEMINI,
    PROVIDER_GEMINI_TRANSCRIBE,
    PROVIDER_GEMINI_TRANSCRIBE_SMART,
    PROVIDER_OPENAI,
    PROVIDER_MISTRAL,
)
# providers that only transcribe audio
TRANSCRIBE_ONLY_PROVIDERS = (
    PROVIDER_GEMINI_TRANSCRIBE,
    PROVIDER_GEMINI_TRANSCRIBE_SMART,
)
# providers that need a region code (de-DE) as the language
REGION_PROVIDERS = TRANSCRIBE_ONLY_PROVIDERS

# pipeline tasks, each with its own provider and model
TASK_TRANSCRIBE = "transcribe"
TASK_POLISH = "polish"
TASK_RECONCILE = "reconcile"
TASK_PROTOCOL = "protocol"
TASK_AGENDA = "agenda"
TEXT_TASKS = (TASK_POLISH, TASK_RECONCILE, TASK_PROTOCOL, TASK_AGENDA)
TASKS = (TASK_TRANSCRIBE, *TEXT_TASKS)

# speech-to-text models degrade or truncate silently past 30 min; with the 3 min cut
# window a 25 min target keeps every chunk at 28 min or less
SPEECH_MAX_CHUNK_MINUTES = 25
# custom_vocabulary takes up to 1,000 terms; Google recommends at most 100
SPEECH_MAX_TERMS = 100
# speech-to-text segments merge into one turn until this long, then break at a
# sentence end, so turns stay short enough for timestamps and sanity checks
SPEECH_TURN_S = 20
SENTENCE_END = (".", "?", "!")
# a speaker label is only valid within one call, so ids carry the part number
SPEECH_SPEAKER_ID = "P{part}-{label}"
SPEECH_NO_SPEAKER = "X"

# secrets come only from the environment: one variable per provider; custom providers
# name theirs
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
API_KEY_ENVS = {
    PROVIDER_GEMINI: GEMINI_API_KEY_ENV,
    PROVIDER_GEMINI_TRANSCRIBE: GEMINI_API_KEY_ENV,
    PROVIDER_GEMINI_TRANSCRIBE_SMART: GEMINI_API_KEY_ENV,
    PROVIDER_OPENAI: OPENAI_API_KEY_ENV,
    PROVIDER_MISTRAL: MISTRAL_API_KEY_ENV,
}

# provider metadata for run.json
GEMINI_PROVIDER = "google-gemini-developer-api"
GEMINI_ENDPOINT = "generativelanguage.googleapis.com (Interactions API)"
GEMINI_LOGGING = "developer logs: project default (up to 55 days)"
# OpenAI's EU endpoint needs a data-residency project; ours has none (design D6)
OPENAI_PROVIDER = "openai"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_LOGGING = "no training; retention up to 30 days (evaluation only)"
MISTRAL_PROVIDER = "mistral"
MISTRAL_BASE_URL = "https://api.eu.mistral.ai/v1"
MISTRAL_LOGGING = "no training; EU endpoint; retention not stated (evaluation only)"
CUSTOM_LOGGING = "set by the endpoint's operator (custom provider)"
SPEECH_ROUTE = "/audio/transcriptions"
CHAT_ROUTE = "/chat/completions"
MODELS_ROUTE = "/models"
# the json_schema response format needs a name
CHAT_SCHEMA_NAME = "answer"

# the prompts the meeting settings fill in (config.yaml "meeting")
DEFAULT_MEETING_TYPE = "meeting"
DEFAULT_WRITER_ROLE = "a professional minute-taker"

# evaluation (design §5)
LEGACY_ID = "C0"
CANDIDATE_SOURCE_DIRS = (AUDIO_DIR, CONTEXT_DIR)
# a speech-to-text candidate needs at least this many timed turns per minute to be the
# reference or the time anchor (untimed answers have one turn per chunk)
MIN_TURNS_PER_MINUTE = 1
# speech without a turn after it: assume this speaking rate to time its words
WORDS_PER_SECOND = 2.5
# disagreement spans: up to this many reference words, two to six distinct variants
SPAN_COUNT = 40
SPAN_MAX_TOKENS = 12
SPAN_MERGE_GAP = 2
SPAN_MAX_VARIANTS = 6
SPAN_CONTEXT_TOKENS = 12
SPAN_SPACING_S = 15
# at least this many spans after minute 25, where the second chunk starts
LATE_SPAN_COUNT = 10
LATE_SPAN_S = 25 * 60
# clips: padding absorbs timestamp errors of a few seconds
CLIP_PAD_BEFORE_S = 5
CLIP_PAD_AFTER_S = 6
CLIP_MAX_S = 30
# a word counts as medical with a glossary match, an acronym or a Latin/Greek ending
MEDICAL_RATIO = 85
MEDICAL_MIN_CHARS = 9
# endings like -in or -on would also match everyday words (Patientin, Diskussion)
MEDICAL_SUFFIXES = (
    "itis",
    "ose",
    "ämie",
    "urie",
    "pathie",
    "mab",
    "tid",
    "pril",
    "sartan",
    "gliflozin",
    "statin",
    "olol",
)
# speaker spot-checks: 15 s of one voice, spread out, some after the first chunk
SPOT_COUNT = 12
SPOT_S = 15
SPOT_SPACING_S = 60
LATE_SPOT_COUNT = 4
# readability: the first minutes of the top two candidates
SAMPLE_S = 5 * 60
READABILITY_COUNT = 2
# a dropped passage this long disqualifies a candidate
MAX_DROPPED_S = 30
# a tail gap is not enough: trailing silence or applause trips it; the alignment
# finds real drops
DISQUALIFYING_WARNINGS = (WARNING_LOOP,)
# scores (0-100): pre-score picks the top two, the final score adds readability
PRE_WEIGHTS = {"word": 0.53, "speaker": 0.35, "cost": 0.12}
FINAL_WEIGHTS = {"word": 0.45, "speaker": 0.30, "readability": 0.15, "cost": 0.10}
MEDICAL_WEIGHT = 2
TIE_POINTS = 3
WHOLE_FILE_POINTS = 5
H1_WORD_POINTS = 10
# fixed candidate roles in the decision rules (eval/candidates.yaml); a disqualified
# C6, C7 or C8 decides nothing (a truncated C6 forfeits the H1 test)
ROLE_ARGON = "C1"
ROLE_WORDS_ONLY_PROBE = "C6"
ROLE_WHOLE_FILE = "C7"
ROLE_DENOISED = "C8"
# answers.json choices
ANSWER_NONE = "none"
ANSWER_UNCLEAR = "unclear"
# speaker answers for voices the reviewer cannot name
UNKNOWN_PERSON = "Unknown person"
UNKNOWN_PERSON_COUNT = 8
REPORT_PAGE = "index.html"
REPORT_CLIPS = "clips"
REPORT_DATA_FILE = "report_data.json"
LEADERBOARD_MD = "leaderboard.md"
LEADERBOARD_JSON = "leaderboard.json"
