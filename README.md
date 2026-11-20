# llm-watermark-ip

A hardware sampler for language-model text that embeds a statistical watermark,
and a detector that finds it. One keyed hash drives two published schemes:
Gumbel (Aaronson; distortion-free) and green list (Kirchenbauer). Design project
for NYU ECE-GY 6463, Advanced Hardware Design, Fall 2026.

Team: Fahad Tajiki (fxt2021) and Krish Mehta (km6152).

**Status (2026-11-20): planning.** The golden model runs; no hardware is built
yet. This README describes what exists and will be rewritten to describe what was
built.

| path | what |
|---|---|
| `plans/initial_plan.md` | the initial plan, graded |
| `plans/design.md` | working detail: formats, commands, errors, verification, timeline |
| `plans/design_review.md` | the two original plans, their pros and cons, and how they were merged |
| `src/golden/wm_ref.py` | bit-exact Python reference for every block |

    python3 src/golden/wm_ref.py     # self-test: ROM statistics, χ² sampling checks, detection on both schemes

Needs `numpy`; `scipy` is optional and used for exact p-values when present.

The two plans that were merged are kept as written on branches `green-list` and
`gumbel`.
