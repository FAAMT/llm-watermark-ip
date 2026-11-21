> **Historical.** Working detail written before the build. The interface as
> built is `docs/interface.md` (its error codes and DETECT rules differ from the
> table below), the verification as run is `docs/verification.md`, and the
> resource estimate as built is in `docs/architecture.md`.

# Design: Watermarking Sampler, working detail

The detail behind `plans/initial_plan.md`: wire formats, error codes, word
widths, the verification plan, the efficiency plan and the timeline. Where this
file and the golden model `src/golden/wm_ref.py` disagree, the golden model wins
and this file is wrong.

## Word formats

| Quantity | Format | Range and notes |
|---|---|---|
| logit ℓ | int16 Q8.8, natural-log units | host subtracts the max, so ℓ ≤ 0; clamped at −128.0 |
| Gumbel noise G | int16 Q8.8 | −3.13 to 22.87 (−801 to 5,856 LSB) |
| δ | int16 Q8.8 | 0 < δ < 16.0; default 2.0 |
| ℓ + δ·green + G | int18 | −33,569 to 9,951 LSB, so 18 bits never overflow |
| θ | uint32 | 2^32 − ⌊γ·2^32⌋, with 0 < γ < 1 |
| detection score | uint16 Q5.11 | 0 to 22.87 |
| S | uint32 Q21.11 | wraps after about 2 million average tokens, 91,000 at the maximum score; T is far smaller |
| g, N | uint32 | |
| token id | 24 bits in a 32-bit word | bit 31 is the skip flag in DETECT |

## Commands

The command-response pattern (course Lab 5): every header carries a tx_id; the
kernel runs until END or an error; on an error it sets the AXI-Lite status
registers, closes any output burst with TLAST, and returns.

| Command | Header fields | Data after the header | Response |
|---|---|---|---|
| CONFIG (0) | K, V, k, nonce, scheme (0 GUMBEL, 1 GREEN), θ, δ | none | none |
| SAMPLE (1) | free bit, step, k context ids | ⌈V/4⌉ beats of 4 int16 logits; logit_unpack forces slots past V to −32768; TLAST on the last | header, then one beat: token id and winning value |
| DETECT (2) | T | ⌈T/2⌉ beats of 2 words: bits 23:0 token id, bit 31 skip; TLAST on the last | header, then two beats: S and N, then g |
| END (3) | none | none | none; the kernel returns |
| MASK (4), stretch | k context ids | none | header, then ⌈V/64⌉ beats; bit b of beat w is token 64w + b; bits past V are 0 |

| Error code | Meaning |
|---|---|
| 1 | header TLAST early or missing |
| 2 | unknown command, including MASK before it is built |
| 3 | SAMPLE, DETECT or MASK before CONFIG |
| 4 | bad parameter: V = 0 or V > 2^24, k outside 1–4, T ≤ k, scheme > 1, θ = 0, δ ≤ 0 or δ ≥ 16.0 |
| 5 | data TLAST early |
| 6 | data TLAST missing |

All parameter checks run on the header, before any payload is read.

## Who does what between host and IP

- **Host:** runs the model; draws a fresh nonce for each generation, so GREEN
  text varies between runs of the same prompt; scales, max-subtracts, clamps and quantises the
  logits; keeps the step counter; sets the free bit with a hash set of contexts
  already seen (`repeat_flags`); turns S, g and N into p-values (`p_gumbel`,
  `p_green`).
- **IP:** seeds, hashes, noise, green test, argmax, and the detection sums. It
  keeps no state between SAMPLE commands, so the host sends the k context ids
  each time.

## Verification plan

1. **Unit tests**, each block's C testbench against its Python function:
   seed_unit ↔ `step_seed`/`free_seed`, hash_lanes ↔ `uniforms` and `green`,
   rom_index ↔ `pf_index`, add_argmax ↔ `sample`, detect path ↔ `detect`.
   Edge cases: u = 0, 1, 2^31 − 1, 2^31, 2^32 − 1; u = θ − 1 and θ; argmax ties;
   V not a multiple of 4.
2. **Bit-exact:** C simulation and co-simulation against the model. Co-sim runs
   V = 1,024 for a few steps; full V runs in C simulation.
3. **Distribution checks (χ²):** GUMBEL sample frequencies against softmax(ℓ) over
   random contexts; GREEN against softmax(ℓ + δ·green) with the context fixed and
   the step varied.
4. **Detector calibration:** p_S and p_g on unwatermarked GPT-2 text are uniform
   (Kolmogorov–Smirnov), and the false-positive rate at 10⁻³ matches.
5. **Detection power:** tokens needed to reach p < 10⁻⁶, for both schemes and both
   statistics, at temperatures 0.7 and 1.0, k = 1, 2 and 4. Also the fraction of
   free steps for each k.
6. **Error handling:** every error code in its own activation, then a fresh
   activation that passes.
7. **Automation:** one script builds the ROMs, runs the model, C simulation,
   co-simulation and synthesis, and compares everything.

Already run on the golden model (`python3 src/golden/wm_ref.py`, synthetic
logits):

| Check | Result |
|---|---|
| ROM noise mean / variance, 2M draws | 0.5783 / 1.6481 (exact 0.5772 / 1.6449) |
| GUMBEL vs softmax(ℓ), χ², 50k samples | 2.39 (95% cutoff 9.49) |
| GREEN vs softmax(ℓ + δ·green), χ², 50k samples | 2.56 (95% cutoff 9.49) |
| unwatermarked text, 200 tokens | p_S 0.78, p_g 0.63 |
| GUMBEL text, 200 tokens | p_S 2·10⁻²⁰⁴, p_g 1·10⁻⁹⁷ |
| GREEN text, 200 tokens | p_S 6·10⁻²⁰, p_g 7·10⁻³⁶ |

## Efficiency plan

| Item | Estimate | Measured by |
|---|---|---|
| SAMPLE, W = 64, P = 4 | 12,565 cycles, about 126 µs per token at 100 MHz | co-sim latency, II |
| SAMPLE, W = 128, P = 8, if Waveflow allows | 6,283 cycles, about 63 µs | the same |
| DETECT, P_d = 2 | 2 tokens per clock | the same |
| MASK (stretch), 8 keyed lanes | 6,283 cycles, about 63 µs; 6.3 KB out | the same |
| DSP48E1 | sample lanes about 48 (two fmix32 each), detect lanes about 52 plus up to 40 for the context and token-id constant multiplies, seed_unit about 10: about 110–150 of 220 | synthesis |
| BRAM36 | gumbel_rom about 9 per copy × 2, one shared score ROM about 9: about 27 of 140; P = 8 doubles the gumbel_rom share to about 45 | synthesis |
| ROM mantissa bits M | 6, 7, 8 trade BRAM against accuracy | BRAM count and the χ² test per M |
| software baseline | NumPy sampling and detection | time per step |

SAMPLE is limited by input bandwidth, not compute: every logit crosses the stream
once, so P = W/16. A wider stream, or an IP placed beside an on-chip LM head,
would raise P. MASK is the compute-bound case, which is why it is the stretch
that demonstrates lane scaling.

## Timeline (4 weeks)

| Week | Work |
|---|---|
| 1 | golden model and ROMs (done), recorded GPT-2 logits, topic approval |
| 2 | controller, seed_unit, hash_lanes, rom_index, add_argmax; GUMBEL SAMPLE at P = 1, then P = 4, matching the model in C simulation |
| 3 | DETECT with both statistics; GREEN scheme (second fmix32, comparator, δ); co-simulation; synthesis |
| 4 | statistical tests, detection-power plots, automation, results; MASK if time allows |

## Open questions

1. Waveflow stream width: 64 bits, or 128?
2. Default k: 1 is the most robust to edits; 4 gives fewer repeated contexts. We
   sweep both.
3. Professor approval for the topic.
