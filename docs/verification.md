# Verification and evaluation

Every claim here names the file it came from. All files in `results/` are
written by a script in `scripts/` and are reproduced by the Makefile targets in
the README.

## Method

The reference is the golden model `src/golden/wm_ref.py`, written before the
RTL. It has one function per hardware block, and both ROM images are generated
from it (`scripts/gen_roms.py`), so the RTL and the model share no hand-typed
constants. Correctness is **bit-exact** equality with the model: the IP's
arithmetic is all integer, so any tolerance would hide a real bug. The
statistical properties the model promises are checked separately, through the
RTL, with χ² tests.

| layer | what runs | correctness check | file |
|---|---|---|---|
| 1. golden model | `python3 src/golden/wm_ref.py` | ROM noise moments, χ² of both schemes, detection of both schemes | console |
| 2. unit tests | `tb_units.v`: fmix32, fmix64, pf_index, ctx_fold, one vector per cycle | bit-exact at each block's latency | results/unit_tests.md |
| 3. top level | `tb_wm_top.v` through the real AXI4-Lite and AXI4-Stream ports | every response word, its TLAST, the status registers and the CTRL bits, bit-exact | results/rtl_tests.md |
| 4. flow control and pipeline depth | layer 3 again with 35 % input bubbles and 45 % output backpressure, then rebuilt at `MUL_REG` = 0, 2, 3 | identical results | results/rtl_tests.md |
| 5. test strength | 29 planted bugs, one at a time | the layer-3 suite must fail for each | results/mutations.md |
| 6. distribution | 20,000 SAMPLEs per scheme through the RTL | χ² against the exact target distribution | results/stats_rtl.md |
| 7. real model | Qwen3-Coder-30B-A3B steps replayed through the RTL with the full 151,936-token vocabulary | RTL picks the token the text contains | results/llm_rtl_replay.md |
| 8. performance | latency and throughput at the ports across sizes | constant overhead shows II = 1 | results/perf.md |
| 9. detection quality | 90 texts from the real model, both schemes, k = 1, 2, 4, three temperatures | detection rate, tokens needed, false positives | results/llm_eval.md |
| 10. synthesis | Vivado flow in `syn/` | utilization and timing reports | not run (see below) |

## Results

### Functional correctness

| check | result | file |
|---|---|---|
| fmix32, fmix64, pf_index, ctx_fold, 20,080 vectors each, including 0, 1, 2^31 ± 1, 2^32 − 1, every power of two and the ROM bank boundaries | 4 of 4 blocks PASS, every vector equal | results/unit_tests.md |
| 50 activations: 12 scenario activations over 11 cases (both schemes, k = 1–4, V = 1 to 64, argmax ties in every lane position, pad slots, the θ boundary, extreme δ and logits, DETECT with T = 1, T = k, all skipped, T = 1,001, two 40-step closed-loop generations) and 19 error activations, including the error order, each followed by a fresh one | 50 of 50 PASS | results/rtl_tests.md |
| the same 50 with random input bubbles and output backpressure, and at `MUL_REG` = 0, 2, 3 | 50 of 50 PASS in all four | results/rtl_tests.md |
| large vocabularies, V = 1,024 to 151,936, and DETECT up to T = 16,384 | every response equal to the golden model | results/perf.md |
| 29 planted bugs: tie-breaking, thresholds, index stride, noise select, δ gating, context order, constants, seed timing, every TLAST check, parameter limits, cross-activation state, response packing | 29 of 29 caught | results/mutations.md |

Writing the mutation sweep found two weak tests, which were then fixed. Both
missed bugs were caught afterwards:

- The test host padded unused logit slots with −32768, which can never win, so a
  bug that counted the pad slot as a token went unseen. The pad slots now carry
  32767.
- The θ-boundary token reached only one detect lane. It now reaches both.

### Sampling distribution, through the RTL

| scheme | target | χ² (4 d.o.f.) | p | file |
|---|---|---|---|---|
| GUMBEL, 20,000 samples, fresh context each | softmax(ℓ) | 6.14 | 0.19 | results/stats_rtl.md |
| GREEN, γ 0.4, δ 2.0, 20,000 samples | softmax(ℓ + δ·green) | 1.97 | 0.74 | results/stats_rtl.md |

### Real language model

Qwen3-Coder-30B-A3B-Instruct (V = 151,936), served by llama.cpp on the build
machine, generated 90 texts of up to 200 tokens: both schemes at k = 1, 2, 4 and
temperatures 0.7 and 1.0, k = 2 also at 1.5, and plain sampling at all three
temperatures. The golden model chose every watermarked token. Two of those texts
were then replayed step by step through the RTL with the full vocabulary.

| check | result | file |
|---|---|---|
| RTL replay: 80 real-model SAMPLE steps, 37,984 beats each, plus DETECT on both texts | 80 of 80 tokens match the text; every response word equals the golden model | results/llm_rtl_replay.md |
| GREEN steers as designed: expected against observed green rate, 7 settings, 8,400 steps | agree to within 0.006 in every setting | results/llm_eval.md |
| false positives: 2,058 null p-values from wrong keys and plain text | Gumbel score: 2 below 10⁻³, against 2.1 expected; Kolmogorov–Smirnov p 0.75. Green count: 0 below 10⁻³, conservative | results/llm_eval.md |
| detection, GUMBEL, k = 2, temperature 1.5 (0.85 nats/token) | 4 of 6 single texts below 10⁻⁶; about 36 tokens needed | results/llm_eval.md |
| detection, GUMBEL, k = 2, temperature 1.0 (0.37 nats/token) | 2 of 6 single texts; six texts pooled p = 2·10⁻²³; about 206 tokens needed | results/llm_eval.md |
| detection, GREEN, δ = 2, k = 2, temperature 1.0 | 0 of 6 single texts; pooled p = 2·10⁻⁶; about 1,072 tokens needed | results/llm_eval.md |

The main finding is about the model, not the hardware. This instruct model is
very confident, at 0.2 to 0.4 nats of entropy per token at temperatures 0.7 and
1.0, which leaves little room to steer. Most 200-token answers stay above the
10⁻⁶ bar, and detection needs a few hundred to a few thousand tokens. GUMBEL
needs fewer tokens than GREEN in every setting and leaves the distribution
unbiased. Synthetic high-entropy logits detect in well under 200 tokens: the
golden model's self-test gives p = 10⁻²⁰⁴ at 200 tokens.

### Module-to-test map

| module | unit test | top-level coverage | mutations planted |
|---|---|---|---|
| `fmix32_pipe`, `fmix64_pipe`, `ctx_fold`, `pf_index` | results/unit_tests.md, 20,080 vectors each | every SAMPLE and DETECT | constants, shift, leading-zero count, ROM bank |
| `rom_dp` and the two ROM images | through `pf_index` and every response value | every SAMPLE and DETECT, all 16,896 addresses reachable | upper-bank placement |
| `seed_unit` | — | every SAMPLE, free and keyed | free-seed constant, feed timing |
| `sample_core` | — | ties in every lane position, pad slots, θ boundary, δ extremes, χ² | 8, from tie rules to δ gating |
| `detect_core` | — | T = 1, T = k, skips in both lanes, T = 16,384 | 6, from context order to skip bits |
| `wm_top` controller | — | 19 error activations, CONFIG replacement, cross-activation state, TLAST of every response | 9, from TLAST checks to response packing |
| `axil_ctrl` | — | every activation polls CTRL and reads all status registers | through the error and state mutations |

### Performance against the plan's goals

| goal (initial plan) | target | measured | verdict | file |
|---|---|---|---|---|
| SAMPLE latency, V = 50,257 | ⌈V/4⌉ = 12,565 cycles, about 126 µs at 100 MHz | 12,591 cycles, 125.9 µs | met: the 26 extra cycles are the header, seed and pipeline fill | results/perf.md |
| SAMPLE initiation interval | one 64-bit beat (4 logits) per cycle | latency − beats = 26 at every V from 4 to 151,936 | met | results/perf.md |
| DETECT throughput | 2 tokens per cycle | 1.995 tokens per cycle at T = 16,384 | met | results/perf.md |
| SAMPLE at the test model's V = 151,936 | — | 38,010 cycles, 380 µs | bandwidth-bound, as planned | results/perf.md |
| faster than software | the plan compared against the Zynq's Cortex-A9 | a gcc -O3 loop on one desktop core: 135 µs (V = 50,257), 294 µs (V = 151,936); the IP at 100 MHz: 126 µs and 380 µs | comparable: the IP takes 0.94× the desktop core's time at V = 50,257 and 1.29× at V = 151,936, at about 1/50 of its clock; the A9 was not measured | results/baseline.md |
| DSP48E1 | about 110–150 of 220 | about 167 by structural count, not synthesized | open: over the plan's estimate; two reductions listed in docs/architecture.md | docs/architecture.md |
| BRAM | about 27 of 140 RAMB36 | about 25.5 by count, not synthesized | open | docs/architecture.md |
| clock | 100 MHz | not measured | open: needs `make synth-vivado` | — |

The IP's advantage is not raw speed against a desktop processor. It is that a
100 MHz fabric block keeps pace with one, inside the Zynq's power budget, without
the ARM cores doing the per-token pass over the whole vocabulary. SAMPLE is
bound by the stream: every logit crosses it once. The next step up is a wider
stream, or sitting the IP beside an on-chip LM head so that the logits never
cross a bus at all.

### Synthesis: not run

No synthesis tool is installed on the machine this was built on, and installing
one was not possible during the build. `make synth-vivado` runs out-of-context
synthesis, placement and routing on `xc7z020clg400-1` at 100 MHz and writes
utilization and timing to `results/synth/vivado/`. Until it has run, the
resource figures in `docs/architecture.md` are a structural count, and the
100 MHz clock is a target, not a result.
