# CS-Audit prototype (CSS 410, Assignment 12)

Code-switch detection and E1–E6 error classification for Kazakh / Kazakh–Russian ASR output.

| File | What it does |
|---|---|
| `csaudit.py` | detector, aligner, classifier, simulator; `python3 csaudit.py` runs the 20-seed experiment → `results.json`, `fig_results.png` |
| `soniox_transcribe.py` | sends audio to the Soniox async API, saves tokens to `soniox_raw/` |
| `soniox_eval.py` | real check on 20 recorded phrases (`references.tsv`) → `real_results.json` |
| `demo.py` | demo; `python3 demo.py --live raw/memo_b.wav` transcribes live (needs `SONIOX_API_KEY`) |

Requirements: Python 3.9+, `requests`; `matplotlib` only for the figure.
Words in `[brackets]` in the data are Russian(-origin) words (gold labels for the detector).
