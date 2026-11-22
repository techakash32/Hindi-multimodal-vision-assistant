# ADR 0001: Hinglish on screen, Devanagari for the voice

- Status: accepted
- Date: 2026-11-21

## Context

The users speak Hindi. The Piper Hindi voice reads Devanagari only and mangles Roman text. Sighted helpers and developers read Roman-script Hindi with English loanwords ("Hinglish") far more easily than Devanagari, and that is how people type Hindi on phones.

## Decision

Spoken text is Devanagari Hindi with English loanwords written in Devanagari ("असिस्टेंट तैयार है"). Screen text is Hinglish. Fixed messages have both versions written by hand (`strings.py`); dynamic text (scene descriptions, text read from signs) goes through the rule-based transliterator in `hinglish.py`, with `tts.speak(display=...)` carrying the screen version alongside the audio.

## Consequences

Two renderings must stay in step, so every spoken key needs a display twin and a test enforces both are script-pure. The transliterator is heuristic: `tests/golden/hinglish.tsv` (200+ hand-written pairs) pins its behaviour, and the golden file already caught real bugs (flap consonants, schwa deletion, final vowels). OpenCV cannot draw Devanagari, so the OpenCV window shows boxes only and the browser dashboard draws labels.
