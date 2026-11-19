# llm-watermark-ip

A streaming hardware IP that watermarks the text a language model generates, and
detects that watermark. Design project for NYU ECE-GY 6463, Advanced Hardware
Design, Fall 2026.

**Status (2026-11-19): planning.** Nothing is built yet. This README describes
what exists; it will be rewritten to describe what was actually built.

| path | what |
|---|---|
| `plans/initial_plan.md` | the initial plan — graded, 20 points |
| `src/golden/wmark_ref.py` | bit-exact Python reference for every block in the plan; `python3 src/golden/wmark_ref.py` runs a demo: on synthetic logits the detector reports z ≈ +14 with the watermark on and ≈ 0 off |

Team: Fahad Tajiki and **TODO: partner**.
