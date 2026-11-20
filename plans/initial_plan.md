# Initial Plan: Watermarking Sampler for LLM Text

Repository: https://github.com/FAAMT/llm-watermark-ip

## Project Team

- Fahad Tajiki (fxt2021)
- Krish Mehta (km6152)

## IP Definition

### What the IP does

A language model picks each next token by sampling from `softmax(ℓ)` over the
logits ℓ of a V-token vocabulary. This IP is that sampler, with its randomness
derived from a secret key and the previous k tokens. The text therefore carries a
statistical watermark that anyone holding the key can detect from the tokens
alone, without the model.

One keyed uniform u per candidate token drives two published schemes, chosen at
CONFIG:

- **GUMBEL** (default): Aaronson's scheme, also called exponential-minimum
  sampling (Kuditipudi et al.). The IP returns `argmax_t (ℓ_t + G(u_t))` with
  Gumbel noise G. Over a random key this is a softmax sample up to ROM
  quantisation (the Gumbel-max trick), so the text distribution is unchanged.
- **GREEN**: the soft red/green list of Kirchenbauer et al. Tokens whose u falls
  in the top γ fraction are green and gain δ on their logit; the IP then samples
  with Gumbel noise from an unkeyed seed. It shifts the distribution slightly, and
  in the MarkMyWords benchmark (Piet et al.) it is detectable in under 100 tokens.

The detector recomputes u for each token actually chosen and returns two
statistics in one pass: the Gumbel score S and the green count g. Green means
"high u", and Gumbel sampling favours high u, so each statistic responds to both
schemes; detection power falls with the entropy of the text, as for any
sampling watermark.

- **SAMPLE:** context and V logits in, sampled token out.
- **DETECT:** token sequence in; S, g and the count N out. The host turns them
  into p-values.
- **MASK** (stretch goal): context in, the V-bit green mask out, for a host that
  samples on its own.

The model (GPT-2 small, V = 50,257) runs in Python with Hugging Face. Its logits
are recorded and replayed by the testbench, and the model weights are never
changed.

### Mathematical operations

CONFIG fixes a 64-bit key K, V, the context length k (1–4), a 64-bit nonce, the
scheme, γ (sent as the threshold θ = 2^32 − ⌊γ·2^32⌋) and δ (Q8.8). Logits are
int16 Q8.8 in natural-log units; the host scales by 1/temperature, subtracts the
maximum and clamps. fmix64 and fmix32 are the MurmurHash3 finalisers;
C_1..C_4 are the odd 64-bit constants `C_CTX` in `src/golden/wm_ref.py`.

1. **Seeds**, once per step: `s = fmix64(K ⊕ ⊕_j ((t_{i−j} + 1)·C_j mod 2^64))`
   from the previous k tokens, and the unkeyed `s_f = fmix64(nonce ⊕ ((step + 1)·C_4 mod 2^64))`.
2. **Uniforms**, per index t: `u_t = fmix32(s[31:0] ⊕ (t·0x9E3779B1 mod 2^32))`,
   read as `ũ = (u + 0.5)/2^32`; `u'_t` is the same from `s_f`.
3. **Noise:** `G(ũ) = −ln(−ln ũ)`, read from a ROM in Q8.8 (range −3.13 to 22.87).
4. **Sample**, ties to the smallest t:
   - GUMBEL: `argmax_t (ℓ_t + G(u_t))`, with `u'_t` instead on a repeated context.
   - GREEN: `argmax_t (ℓ_t + δ·[u_t ≥ θ] + G(u'_t))`.
5. **Detect**, over tokens i ≥ k not flagged as repeated:
   `S = Σ −ln(1 − ũ)` (ROM, Q5.11), `g = Σ [u ≥ θ]`, N = tokens scored. For
   unwatermarked text each score is Exp(1), so `S ~ Gamma(N, 1)` and
   `p_S = gammaincc(N, S)` (Fernandez et al.); and `g ~ Binomial(N, γ)`, so
   `p_g = P[Binomial(N, γ) ≥ g]`. Text is watermarked if p < 10⁻⁶.
6. **Repeated contexts:** when a k-token context has already appeared in the text,
   the host sets a free bit. Generation uses `s_f` and detection skips the token,
   so noise never repeats and both p-values stay exact.

**ROM address.** Take w, the distance of u from the nearer end of [0, 2^32); e,
its leading-zero count; m, the 8 bits after its leading one. The address is
`(half·33 + e)·256 + m`, 16,896 entries. This works like a small floating-point
number and stays accurate in both tails, where winning tokens come from. A table
on the top 16 bits of u would clip u > 1 − 2^−16, which on average 0.77 tokens
out of 50,257 reach at every step, and those are the likely winners.

Outline of the golden model; the bit-exact file is `src/golden/wm_ref.py`:

```python
def sample(ctx, logits, free, step):                            # SAMPLE
    u  = uniforms(step_seed(K, ctx), range(V))                  # keyed: hash_lanes
    uf = uniforms(free_seed(nonce, step), range(V))             # unkeyed: hash_lanes
    if scheme == GUMBEL:
        v = logits + G_ROM[pf_index(uf if free else u)]
    else:                                                       # GREEN
        v = logits + delta * (u >= theta) + G_ROM[pf_index(uf)]
    return argmax(v)                                            # first maximum

def detect(tokens, skip):                                       # DETECT
    S = g = N = 0
    for i in range(k, len(tokens)):
        if skip[i]: continue
        u = uniforms(step_seed(K, tokens[i-k:i]), [tokens[i]])
        S += S_ROM[pf_index(u)]; g += (u >= theta); N += 1
    return S, g, N                     # host: gammaincc(N, S/2048) and binom.sf(g-1, N, gamma)
```

### Why it suits hardware acceleration

- **Sampling without softmax.** V additions and one running maximum replace V
  exponentials, a sum, a normalisation and a search: one integer-only streaming
  pass, the same pass for both schemes.
- **Highly parallel.** The V hashes and ROM lookups are independent; P lanes run
  at II = 1 and feed a P-input max tree.
- **Latency-critical.** It runs once per generated token, inside the decode loop,
  so its latency adds to every token.
- **Small, fixed state.** The key, two seeds and a running maximum. The ROMs are
  fixed and nothing grows with V.
- **Parallel detection.** Every token's seed depends only on input tokens, so P_d
  detector lanes score P_d tokens per clock.

### How it talks to the host

The course command-response pattern: one 64-bit AXI4-Stream in for commands and
data, one 64-bit AXI4-Stream out for responses, and AXI-Lite for status only
(halted, error code, tx_id). On PYNQ an AMD AXI DMA drives both streams.

| Command | Header | Data after the header | Response |
|---|---|---|---|
| CONFIG (0) | K, V, k, nonce, scheme, θ, δ | none | none |
| SAMPLE (1) | free bit, step, k context ids | ⌈V/4⌉ beats of 4 int16 logits, TLAST on the last | header, then token id and its winning value |
| DETECT (2) | T | ⌈T/2⌉ beats of 2 words (bits 23:0 token id, bit 31 skip), TLAST on the last | header, then S (Q21.11) and N, then g |
| END (3) | none | none | none; the kernel returns |
| MASK (4), stretch | k context ids | none | header, then ⌈V/64⌉ beats; bit b of beat w is token 64w + b |

## IP Architecture

```
SAMPLE  in ─► controller ─► seed_unit ─s, s_f─► hash_lanes[P] ─u, u', green─► rom_index[P] ─► gumbel_rom ─G─┐
                 └─► logit_unpack ─ℓ, 4 per beat──────────────────────────────────────────────► add_argmax ─► resp ─► out
DETECT  in ─► controller ─► ctx_shift ─► detect_lanes[P_d] (fmix64 ► fmix32 ► rom_index ► score_rom, green) ─► score_acc ─► resp ─► out
```

### Sub-modules

1. **controller:** parses CONFIG, SAMPLE, DETECT and END headers, checks errors in
   the course's order, holds the configuration and writes the AXI-Lite status.
2. **seed_unit:** computes the keyed seed s and the unkeyed seed s_f once per
   SAMPLE with a multi-cycle fmix64, off the critical path.
3. **hash_lanes (P copies):** lane p handles index t = cP + p. Two pipelined
   fmix32 per lane give u and u'. The index multiply is an accumulator adding
   P·0x9E3779B1 each cycle, and one comparator gives the green bit `u ≥ θ`.
4. **rom_index (P copies):** a mux picks u or u' from the scheme and the free bit,
   then a leading-zero count and a shift give the ROM address. Logic only.
5. **gumbel_rom:** the 16,896 × 16-bit table, about 9 BRAM36. Two lanes share each
   dual-port copy.
6. **logit_unpack:** splits each 64-bit beat into 4 int16 logits and forces
   indices past V to −32768, whatever the host sent. A pad cannot win, because
   the host's max subtraction puts one real logit at 0.
7. **add_argmax:** P additions `ℓ + δ·green + G` in 18 bits (δ is gated by the
   scheme), a P-input max tree, and a running maximum with its index across beats.
8. **ctx_shift + detect_lanes (P_d copies):** a shift register gives each lane its
   token and k previous tokens. Each lane has a pipelined fmix64 and fmix32,
   rom_index and the green comparator; the two lanes share one dual-port score
   ROM (Q5.11). Skipped tokens add 0.
9. **score_acc:** adder trees over the P_d lanes and 32-bit accumulators for S, g
   and N.
10. **resp:** builds the response header and payload beats.

The MASK stretch adds **mask_packer**, which packs green bits into 64-bit beats.

### Interfaces

| From → To | Type | What it carries |
|---|---|---|
| PS → controller | AXI4-Stream in, 64-bit | command headers with tx_id, then logits or token words, ending in TLAST |
| resp → PS | AXI4-Stream out, 64-bit | response header (tx_id, word count), then token and value, or S, g and N |
| controller → PS | AXI-Lite | halted, error code, tx_id |
| controller → seed_unit | FIFO (`hls::stream`) | K, nonce, k context ids, step and free bit, once per SAMPLE |
| seed_unit → rom_index | FIFO, once per SAMPLE | the free bit, which with the scheme selects u or u' |
| controller → hash_lanes, rom_index, add_argmax, ctx_shift, detect_lanes | registers, static during a command | K (to detect_lanes), scheme, θ, δ, V, k |
| seed_unit → hash_lanes | FIFO | s[31:0] and s_f[31:0] |
| hash_lanes → rom_index | FIFO, P per cycle | u, u' (32-bit each) and the green bit |
| rom_index → gumbel_rom | BRAM read port, one per lane, two lanes per dual-port copy | 15-bit address in, 16-bit G (Q8.8) out after a fixed latency |
| gumbel_rom → add_argmax | FIFO, P per cycle | G, with the green bit delayed alongside it |
| controller → logit_unpack → add_argmax | FIFO | 64-bit beats, then P int16 logits per cycle |
| controller → ctx_shift → detect_lanes | FIFO | token words, then (context, token, skip) per lane |
| detect_lanes → score_acc | FIFO, P_d per cycle | score (uint16 Q5.11) and green bit |
| add_argmax, score_acc → resp | FIFO | result words |

No shared memory is used and the IP has no AXI master. The top level is an HLS
dataflow region with ready/valid flow control throughout, so a stalled output
stream stops the lanes without dropping data.

### Why this decomposition

- **One front end, two schemes.** seed_unit and hash_lanes serve both schemes and
  the detector. GREEN costs a second fmix32 and a comparator per lane and one
  adder input, not a second datapath.
- **One golden function per block:** seed_unit ↔ `step_seed`/`free_seed`,
  hash_lanes ↔ `uniforms`, rom_index ↔ `pf_index`, add_argmax ↔ `sample`,
  detect_lanes and score_acc ↔ `detect`. Each gets its own C testbench.
- **Incremental build.** GUMBEL SAMPLE at P = 1 runs end to end first, then
  P = 4, then DETECT, then GREEN, then the MASK stretch. Each step leaves a
  working IP.
- **Scaling knobs.** The stream width sets both: P = W/16 logits and P_d = W/32
  tokens per beat. The DSP budget is the ceiling on each.

### Expected performance

- **SAMPLE** is bandwidth-bound: 4 logits per 64-bit beat, so P = 4 and
  ⌈50,257/4⌉ = 12,565 cycles, about 126 µs per token at 100 MHz, for either
  scheme.
- **DETECT** takes 2 tokens per beat with P_d = 2 at II = 1.
- **MASK** (stretch) is compute-bound. It runs all 8 fmix32 units on the keyed
  seed (a seed mux, an index stride of 8 and a comparator on each unit), taking ⌈50,257/8⌉ = 6,283 cycles, about 63 µs, and sends 6.3 KB out
  instead of 100 KB of logits in.
- **Resource estimate** on the `xc7z020`, to be replaced by synthesis: about
  110–150 of 220 DSP48E1 (detect lanes also multiply each context id and the
  token id by constants) and 27 of 140 BRAM36.
- **Verification:** bit-exact C simulation and co-simulation against the golden
  model; a χ² test that samples match softmax(ℓ), or softmax(ℓ + δ·green) in
  GREEN; uniform p-values on unwatermarked text; a NumPy software baseline.

### Existing IP

AMD AXI DMA for the two streams, and the AXI4-Stream Data FIFO for decoupling if
needed. We considered the AMD Floating-Point Operator (log) and CORDIC for G(u)
and rejected them: two logarithms per lane cost more DSPs and latency than one
ROM read. We know of no AMD core for the keyed hash.

## References

- S. Aaronson, "My AI Safety Lecture for UT Effective Altruism," *Shtetl-Optimized*, 2022.
- R. Kuditipudi, J. Thickstun, T. Hashimoto, P. Liang, "Robust Distortion-free Watermarks for Language Models," TMLR 2024, arXiv:2307.15593.
- J. Kirchenbauer et al., "A Watermark for Large Language Models," ICML 2023, arXiv:2301.10226.
- P. Fernandez et al., "Three Bricks to Consolidate Watermarks for Large Language Models," WIFS 2023, arXiv:2308.00113.
- J. Piet et al., "Mark My Words: Analyzing and Evaluating Language Model Watermarks," arXiv:2312.00273.
- A. Appleby, MurmurHash3 (public domain): the fmix32 and fmix64 finalisers.

The design history (two plans and their merge) is in `plans/design_review.md`.
