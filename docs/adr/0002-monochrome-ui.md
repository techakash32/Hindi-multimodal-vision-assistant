# ADR 0002: Strict black and white dashboard

- Status: accepted
- Date: 2026-11-21

## Context

The dashboard is a secondary surface for sighted helpers and reviewers, but low-vision users may use it too. Colour carries meaning poorly for colour-blind users and adds contrast risk.

## Decision

Only `#000` and `#fff`. Proximity is encoded by chip style (solid, outlined, dashed) and line weight; the radar uses ring weight. Serif headings, sans body. A test rejects any other colour or any `opacity` rule outside animations, so contrast stays 21:1.

## Consequences

No colour cues to lean on, so every state needs a shape or text cue. Simple to audit for WCAG. Designers lose a tool; the trade is intentional. A real screen-reader pass has not been done: the primary interface for a blind user is the headless voice mode.
