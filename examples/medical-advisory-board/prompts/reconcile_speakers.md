You check the speaker labels of a transcript (language: $language) of a recorded meeting, such as $meeting_type. It was transcribed in parts. Each part reused speaker ids from a shared roster, but a voice may have received more than one id ("drift").

## Task
Group the speaker ids into people and name them where the recording supports it.

- Put ids into the same group only if the transcript shows they are the same person: a continued argument, a reply addressed to them, a matching name, or a matching roster voice description next to a part boundary.
- Ids like `P2-speaker_1` come from a speech-to-text model that labels voices per part only, without voice descriptions: the same person has a different id in every part, and one id may even cover two people with similar voices. Link them across parts from the content.
- Every id listed in the roster must appear in exactly one group.
- `name`: only from evidence in the recording (self-introduction, a direct address the named person answers, a moderator handing over). Include titles as spoken ("Dr.", "Prof.", "Priv.-Doz."). Use the most complete form that was said (e.g. introduced as "Dr. Maria Huber", later addressed as "Maria": name is "Dr. Maria Huber"). Leave `name` empty when unsure.
- `aliases`: every form the person was called in the recording ("Maria", "Frau Dr. Huber"). Empty if never named.
- Attendee list: it may be missing, incomplete or wrong. Use it only to correct the spelling of a name that was actually said, or to link a first name and a last name that were both said to the same person. Never name someone only because the list has a matching role or a similar name.
- `confidence`:
  - `high`: an explicit introduction or address with an answer.
  - `medium`: consistent indirect evidence.
  - `low`: a guess. Low-confidence names are not shown in the transcript.
- `evidence`: one short English sentence with the timestamp(s) that support the grouping and the name. Say if the attendee list was used to correct a spelling.

## Attendees (spelling only; may be missing, incomplete or wrong)
$attendees

## Roster (per part)
$rosters

## Transcript (`[hh:mm:ss] id: text`)
$transcript
