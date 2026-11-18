You clean up one part of a verbatim speech-to-text transcript of a recorded meeting, such as $meeting_type.

## Language
- Transcript language: $language. Keep the text in this language. Do not translate.
- Notes on dialect, vocabulary and orthography: $language_notes

## Task: "clean verbatim"
- Return every turn with its `index`, in the same order. Never merge, split, drop or reorder turns.
- Remove fillers, stutters, repeated words and false starts. Keep the wording and grammar otherwise. Do not summarise, shorten or rephrase.
- Fix punctuation and capitalisation, using the standard orthography of the transcript language.
- Write numbers, units and drug names the way $writer_role would.
- Glossary: correct the spelling of a term only if the transcript clearly contains that term misspelt. Never insert a term that is not there.
- A turn that holds only fillers becomes an empty string.

## Glossary
$glossary

## Turns (`index: text`)
$turns
