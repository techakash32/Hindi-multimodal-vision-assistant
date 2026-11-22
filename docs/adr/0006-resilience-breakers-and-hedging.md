# ADR 0006: Circuit breakers and hedged VLM requests

- Status: accepted
- Date: 2026-11-21

## Context

Free OCR and VLM services fail, rate-limit and stall. A blind user waiting in silence is the worst failure.

## Decision

Each OCR service and each VLM model has a circuit breaker (pause after repeated failures). If the preferred VLM has not answered after `HEDGE_DELAY_S` the next one is also asked and the first answer wins; a slow model is sorted to the back next time. The speech queue drops the oldest item, not the newest.

## Consequences

More requests are sent than strictly needed when hedging, which costs free-tier quota. Behaviour is unit-tested with fakes; not tested against real outages. See the degradation table in the README.
