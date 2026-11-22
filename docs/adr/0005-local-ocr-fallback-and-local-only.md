# ADR 0005: Tesseract as a local OCR path, and a LOCAL_ONLY mode

- Status: accepted
- Date: 2026-11-21

## Context

The default OCR (OCR.space) and the VLM (OpenRouter) receive camera frames. For an assistive camera that is a privacy decision the user should be able to refuse. OCR.space also has quotas and no confidence output.

## Decision

`OCR_ENGINE=tesseract` runs the local binary and returns its mean word confidence. `LOCAL_ONLY=true` forces Tesseract, disables cloud scene description and translation, and the dashboard shows which mode is active.

## Consequences

Offline mode gives no scene description and weaker OCR on poor images. The evaluation can run on any machine with Tesseract installed. Hindi OCR needs the Tesseract Hindi language data, which is not bundled.
