# What changed from the original project

**Bugs fixed**
- `requirements.txt` was UTF-16 and had no `deep-translator`, so a fresh install crashed on import. Replaced with clean, split requirement files.
- Detection boxes were only drawn on every second frame (flicker). They are now drawn on every frame.
- When the speech queue was full the *newest* alert was dropped. Now the oldest is.
- Windows-only camera backend (`CAP_DSHOW`) broke Linux/macOS. Now chosen per platform; Linux audio playback falls back across `aplay`, `paplay`, `ffplay`.
- `test_*.py` hardware scripts would have been collected (and failed) by pytest. Moved to `tools/`.
- A real `.env` with API keys was in the zip. It is excluded and `.env.example` added. **Rotate those keys.**

**Structure**: flat scripts and globals became an installable package with a UI-agnostic engine, typed config with env overrides, rotating logs, thread-safe metrics and an event bus.

**New**: web dashboard, redesigned OpenCV HUD, `visionassist` CLI (`run`, `check`, `download-models`, `mics`), demo mode, repeat command, configurable VLM model list, 96 tests, CI, Makefile.

**Language and UI (latest)**
- Spoken output is Hindi only (Devanagari, natural everyday wording with English loanwords). The Hindi Piper voice cannot read Roman text.
- On-screen text is Hinglish (Roman-script Hindi) via `hinglish.py` (curated phrases plus a rule-based transliterator) and the tables in `strings.py`. The transliteration is heuristic.
- Dashboard: strictly black and white, serif headings, new "Aas-paas ka view" radar (distance rings, left/centre/right), progress bar, "Abhi kya bola" card.
- The OpenCV window shows boxes only, because OpenCV cannot render Devanagari.
