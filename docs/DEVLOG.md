# Dev log

A short, honest log of what changed and what was learned. Newest first. One entry per day you work on the project; skip days with no real change.

**How to add an entry** (2 minutes):

```bash
# 1. add a block at the top of the list below (copy the template)
# 2. commit it together with the work it describes
git add -A && git commit -m "feat/fix/docs: what you did today" && git push
```

**Template**

```markdown
## YYYY-MM-DD
- Did: one line on what changed
- Learned: one line on what surprised you or broke
- Next: the one thing for tomorrow
```

---

## 2026-11-22
- Did: rewrote the README (FAQ, comparison, detected classes, alert rules, performance, roadmap); deployed the live demo on Render; added `docs/DEPLOY_RENDER.md`, `CITATION.cff`, `CONTRIBUTING.md`, `llms.txt`, `.env.example`; added the book PDF.
- Learned: one laptop report had been run with a scripted fake OCR, so its numbers were not real. Marked it "do not quote" and added `reports/README.md` saying which report to trust.
- Next: record a 20 to 30 second demo video; replace the Chapter 11 sentence in the book; run the full 80-scenario evaluation with real YOLO, OCR and a VLM.
