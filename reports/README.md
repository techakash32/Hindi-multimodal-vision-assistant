# Reports: which one to trust

Every number in this repository comes from one of these files. Each is labelled with how it was measured.

| File | What it is | Real components | Quote it? |
| --- | --- | --- | --- |
| [`eval_report.md`](eval_report.md) | 80-scenario evaluation, Linux sandbox | Real Tesseract OCR. Stand-in detector (colour blobs) and VLM (template). No audio. | **Yes, with the synthetic-data caveat.** Re-run and reproduced on 2026-11-22 (17 of 80 not fully correct). |
| [`eval_report_2026-11-22.md`](eval_report_2026-11-22.md) | Same 80 scenarios on the author's laptop | **None**: scripted fake OCR, stand-in detector and VLM | **No.** See the correction at the top of the file. Its Latency section is from the Linux sandbox. |
| [`latency_2026-11-22.md`](latency_2026-11-22.md) | Latency on the author's Windows laptop (12-core Intel, RTX 2050) | Real YOLOv8n, real OCR.space, real free VLM. ASR skipped. | **Yes.** Small n (5 to 10); p95 is near the maximum. |
| [`latency_summary.md`](latency_summary.md) | Latency on the Linux sandbox | Real Tesseract only; detection, VLM and ASR skipped | Only for the stages it lists |
| [`calibration.md`](calibration.md) | Abstention thresholds and the holdout result | Synthetic images | Quote the holdout, not the dev half |
| [`kfold_abstention.md`](kfold_abstention.md) | 5-fold cross-validation of abstention | Synthetic images | Yes, wide spread |
| [`adversarial_report.md`](adversarial_report.md), [`adversarial_noguardrails.md`](adversarial_noguardrails.md) | 20 degraded images, with and without guardrails | Synthetic images | Yes |

## Not yet done

- A full 80-scenario run with real YOLO, real OCR and a real VLM.
- Any run on real photographs or real Hindi speech.
- Any test with blind or low-vision users.

## Corrections to the book

The PDF in [`docs/VisionAssist-Book.pdf`](../docs/VisionAssist-Book.pdf) says (Chapter 11, "Read with care") that a later laptop run "with real components" left 9 of 80 scenarios not fully correct. That sentence is wrong: that run used stand-ins and a scripted fake OCR, as the correction at the top of `eval_report_2026-11-22.md` explains. Suggested replacement: *"A later run on the author's laptop used stand-ins and a scripted OCR and is not evidence of real-component accuracy; its report is marked accordingly. The laptop was used only to measure latency with real YOLOv8n and cloud services."*
