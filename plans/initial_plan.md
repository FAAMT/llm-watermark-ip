# Initial Plan: Distortion-Free LLM Watermark Sampler

Repository: **TODO**

## Project Team

- Krish Mehta (km6152)
- Fahad Tajiki (NetID **TODO**)

## IP Definition

### What the IP does

A language model chooses each next token by sampling from `softmax(ℓ)` over the
logits ℓ of a V-token vocabulary. Our IP performs that sampling with **keyed
randomness**, which is Aaronson's Gumbel watermark, also called
exponential-minimum sampling (Kuditipudi et al.). It works in three steps:

1. For each candidate token t, hash (secret key, previous k tokens, t) to get a
   uniform number u.
2. Turn u into Gumbel noise.
3. Return `argmax(ℓ + noise)`.

Over a random key this is exactly a softmax sample (the Gumbel-max trick), so
text quality is unchanged. Anyone holding the key can still detect the
watermark, because the chosen tokens have unusually large u.

- **SAMPLE:** context and V logits in, sampled token out.
- **DETECT:** token sequence in, score S and count N out. The host computes a
  p-value.

The language model (GPT-2 small, V = 50,257) runs in Python. Its logits are
recorded and replayed by the testbench, and the model weights are never changed.

### Mathematical operations

The bit-exact golden model is in `src/golden/gumbel_ref.py`. In outline:

```python
# CONFIG: 64-bit key K, vocabulary size V, context length k (1..4), 64-bit nonce
# Logits: int16 Q8.8 in natural-log units; the host applies temperature and subtracts the max
def seed(ctx):                         # once per step; fmix64 = MurmurHash3 64-bit finalizer
    x = K
    for j, t in enumerate(reversed(ctx)):
        x ^= ((t + 1) * C[j]) % 2**64      # C[0..3]: fixed odd 64-bit constants
    return fmix64(x)

def u(s, t):                           # keyed uniform; fmix32 = 32-bit finalizer
    return fmix32((s % 2**32) ^ ((t * 0x9E3779B1) % 2**32))   # read as (u + 0.5) / 2**32

def sample(ctx, logits):               # SAMPLE; ties go to the smallest t
    s = seed(ctx)
    return argmax(logits[t] + G_ROM[addr(u(s, t))] for t in range(V))   # G = -ln(-ln u), Q8.8

def detect(tokens, skip):              # DETECT
    S = N = 0
    for i in range(k, len(tokens)):
        if not skip[i]:
            S += S_ROM[addr(u(seed(tokens[i-k:i]), tokens[i]))]       # -ln(1 - u), Q5.11
            N += 1
    return S, N                        # host: p = gammaincc(N, S / 2048); watermarked if p < 1e-6
```

- **ROM address:** `addr(u)` takes the distance of u from the nearer end of
  [0, 2^32) and returns its leading-zero count plus the next 8 bits. It works
  like a small floating-point number and gives 16,896 entries. This keeps both
  tails accurate, which is where the winning tokens come from. A plain table on
  the top bits of u would clip the tails and bias the sampling.
- **Exact p-value:** for unwatermarked text each score is Exp(1), so S follows
  Gamma(N, 1) (Fernandez et al., 2023).
- **Repeated contexts:** when a k-token context repeats within one text, the
  host sets a *free* bit. Generation then uses the unkeyed seed
  `fmix64(nonce ⊕ step)`, and detection skips that token. The noise never
  repeats, so the p-value stays valid.

### Why it suits hardware acceleration

- **Sampling without softmax:** V additions and one maximum replace V
  exponentials, a sum, a normalization and a search. It's a single streaming,
  integer-only pass.
- **Parallel:** the V hashes and table lookups are independent. P lanes run at
  II = 1 and feed a max tree.
- **Latency-critical:** it runs once per generated token, inside the decode
  loop.
- **Small state:** the key, one seed and a running maximum. The ROMs are fixed,
  and nothing grows with V.
- **Fast detection:** each token's seed depends only on input tokens, so the
  detector scores several tokens per clock.

## IP Architecture

```
SAMPLE  in ─► controller ─► seed_unit ─► hash_lanes[P] ─► rom_index[P] ─► gumbel_rom ─┐
                 └─► logit_unpack ─────────────────────────────────────────────► add_argmax ─► out
DETECT  in ─► controller ─► detect_lanes[P_d] (seed, hash, rom_index, score_rom) ─► score_acc ─► out
```

### Sub-modules

1. **controller:** parses the commands CONFIG, SAMPLE, DETECT and END using the
   course command-response pattern. It checks errors, holds the configuration
   and reports status over AXI-Lite.
2. **seed_unit:** computes fmix64 of the key and context, or of the nonce and
   step for a free step. It runs once per SAMPLE, so it's off the critical path.
3. **hash_lanes (P):** lane p computes u for t = cP + p with a pipelined fmix32.
   The index multiply becomes an accumulator.
4. **rom_index (P):** a leading-zero count and a shift give the ROM address.
   Logic only.
5. **gumbel_rom:** a 16,896 × 16-bit BRAM ROM. It's dual-port, so two lanes
   share each copy.
6. **logit_unpack:** splits each 64-bit beat into 4 int16 logits and pads
   indices past V with −32768.
7. **add_argmax:** P additions, a P-input max tree, and a running maximum with
   its index across beats.
8. **detect_lanes (P_d):** a context shift register feeds each lane. Each lane
   has a pipelined fmix64, an fmix32, rom_index and a score ROM. Skipped tokens
   add 0.
9. **score_acc:** an adder tree and 32-bit accumulators for S and N.

### Interfaces

| From → To | Type | What it carries |
|---|---|---|
| PS → controller | AXI4-Stream, 64-bit | command headers with tx_id, then logits (4 int16 per beat) or token words (2 per beat: id and skip bit), ending in TLAST |
| add_argmax, score_acc → PS | AXI4-Stream, 64-bit | response header, then token + value, or S + N |
| controller ↔ PS | AXI-Lite | status: halted, error code, tx_id |
| controller → seed_unit | FIFO | key or nonce, context ids, step, free bit |
| seed_unit → hash_lanes | FIFO | 32-bit seed |
| hash_lanes → rom_index → gumbel_rom → add_argmax | FIFO, P per cycle | u, then the ROM address, then G (Q8.8) |
| logit_unpack → add_argmax | FIFO, P per cycle | int16 logits |
| detect_lanes → score_acc | FIFO, P_d per cycle | scores (Q5.11) |

No shared memory is used. The top level is an HLS dataflow region, and on PYNQ
an AXI DMA drives the streams.

### Expected performance

SAMPLE is limited by input bandwidth: 4 logits per 64-bit beat, so P = 4. For
GPT-2 that's ⌈50,257/4⌉ = 12,565 cycles, about 126 µs per token at 100 MHz. A
wider stream allows a larger P. DETECT handles 2 tokens per beat at II = 1. We
will report DSP, BRAM, LUT, clock and II from synthesis, sweep the ROM precision
against BRAM, and compare with a NumPy software baseline.

Verification: bit-exact C simulation and co-simulation against the golden
model, a χ² test that samples match softmax, and a check that p-values on
unwatermarked text are uniform.

## Alternatives considered

| Decision | Options | Chosen |
|---|---|---|
| Watermark | Green list (Kirchenbauer): biases the logits and distorts the text · **Gumbel**: distortion-free · key-sequence inverse transform (Kuditipudi): detection needs an alignment search | Gumbel |
| Computing G(u) | Plain top-bits table: clips the tails · **floating-point-style ROM** · CORDIC logarithm units: more DSPs and latency | ROM |
| Repeated contexts | Ignore them: invalid p-value · **free seed + skip** · hardware hash set | free seed + skip |
| Detection test | z-score approximation · **exact Gamma p-value** | Gamma |

## References

- S. Aaronson, "My AI Safety Lecture for UT Effective Altruism," *Shtetl-Optimized*, 2022.
- R. Kuditipudi et al., "Robust Distortion-free Watermarks for Language Models," TMLR 2024, arXiv:2307.15593.
- P. Fernandez et al., "Three Bricks to Consolidate Watermarks for LLMs," WIFS 2023, arXiv:2308.00113.
- J. Kirchenbauer et al., "A Watermark for Large Language Models," ICML 2023, arXiv:2301.10226.
