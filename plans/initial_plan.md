# Initial Plan: LLM Text-Watermark Accelerator

Repository: **TODO — GitHub URL of this repo**

## Project Team

- Fahad Tajiki (fxt2021)
- Krish Mehta (km6152)

## IP Definition

### What the IP does

A language model emits text one token at a time by sampling from a distribution
over a vocabulary of V tokens. A statistical text watermark nudges that sampling so
the chosen tokens carry a hidden, machine-detectable signal while the text reads
normally; detection needs only the token sequence and the secret key, not the
model. Because each token's mark depends only on the k tokens before it, copy-paste
and light edits damage one window each, while a full rewrite removes it.

The IP is a streaming hardware block for the soft red/green-list watermark of
Kirchenbauer et al. (ICML 2023). Google DeepMind's SynthID-Text (Dathathri et al.,
Nature 2024) uses the same keyed context hash but steers the sampler by tournament
rather than biasing logits. The IP sits between the model's output logits and the
sampler, with three modes:

- **Mode 0, mask** — given the previous k token ids, emit the V-bit green mask for
  the next step. This is the deployment path: 6.3 KB out per step, and the host
  adds δ in float where the mask is set.
- **Mode 1, bias** — also take the V logits as int16, add δ to the green ones,
  return them. The whole pipeline in hardware, for demonstration.
- **Mode 2, detect** — take a finished token sequence, return the green count and
  the length; the host computes the z-score.

The IP stores neither the vocabulary nor the model and is parameterised by V. The
demonstration target is GPT-2 small (V = 50257), with logits recorded offline from
the Hugging Face model and replayed by the PS, or by the C testbench in simulation.

### Mathematical operations

Fixed per deployment: a 64-bit secret key K, green fraction γ (default 0.25), bias
δ (default 2.0), context length k (default 1, up to 4). Token ids are 32-bit
unsigned. Logits are int16 in Q8.8 after the host subtracts the maximum (softmax is
shift-invariant) and clamps.

1. **Step seed**, once per generated token i:
   `s_i = fmix64( K ⊕ ⊕_{j=1..k} ((t_{i−j} + 1) · C_j mod 2^64) )`
   fmix64 is the MurmurHash3 64-bit finaliser (two multiplies by constants, three
   xor-shifts); C_1..C_4 are fixed odd 64-bit constants.
2. **Green test**, once per vocabulary index t ∈ [0, V):
   `g_i(t) = 1[ fmix32( s_i[31:0] ⊕ (t · 0x9E3779B1 mod 2^32) ) < ⌊γ · 2^32⌋ ]`
   fmix32 is the 32-bit finaliser: two constant multiplies, three xor-shifts, one
   unsigned compare. The V tests share no data.
3. **Bias** (mode 1): `ℓ'_t = sat16( ℓ_t + δ_Q8.8 · g_i(t) )`; the host samples
   `t_i ~ softmax(ℓ')` or takes the argmax.
4. **Detection** (mode 2) over T tokens, N = T − k:
   `g = Σ_{i>k} g_i(t_i)`,  `z = (g − γN) / sqrt(N · γ · (1 − γ))`.
   z ≥ 4 declares the text watermarked; the one-sided false-positive rate under the
   null is about 3·10⁻⁵.

One deliberate deviation: Kirchenbauer et al. take the green list as the first γV
entries of a seeded random permutation of the vocabulary; we hash each index
independently. The detection statistic is unchanged, since each token is green with
probability γ independently of its logit; only the green count per step becomes
Binomial(V, γ) instead of exactly γV, which the z-test does not use. In exchange
the computation becomes parallel: a permutation needs O(V) memory and a serial
pass, the hash needs neither.

Golden model (full file `src/golden/wmark_ref.py`; a numpy array stands in for the
lane array):

```python
M32, M64 = (1 << 32) - 1, (1 << 64) - 1
C_IDX = 0x9E3779B1                                   # odd, so t -> t*C_IDX is a bijection mod 2^32
C_CTX = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D)

def fmix64(k):                                       # MurmurHash3 64-bit finaliser
    k ^= k >> 33; k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33; k = (k * 0xC4CEB9FE1A85EC53) & M64
    return k ^ (k >> 33)

def fmix32_vec(h):                                   # 32-bit finaliser on a uint32 array = P lanes
    h = h ^ (h >> np.uint32(16)); h = h * np.uint32(0x85EBCA6B)
    h = h ^ (h >> np.uint32(13)); h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))

def step_seed(key, ctx):                             # seed_unit: ctx = previous k ids, oldest first
    x = key
    for j, t in enumerate(reversed(ctx)):
        x ^= ((t + 1) * C_CTX[j]) & M64
    return fmix64(x)

def green_mask(seed, V, gamma):                      # hash_lanes: V independent hashes -> V bits
    thr, s_lo = np.uint32(int(gamma * 2**32)), np.uint32(seed & M32)
    t = np.arange(V, dtype=np.uint32)
    return fmix32_vec((t * np.uint32(C_IDX)) ^ s_lo) < thr

def bias_logits(q, mask, delta_q88):                 # logit_biaser: saturating int16 add
    return np.clip(q.astype(np.int32) + delta_q88 * mask, -32768, 32767).astype(np.int16)

def detect(key, tokens, V, gamma, k):                # detect_acc, then the host's z-score
    g = sum(green_mask(step_seed(key, tokens[i-k:i]), V, gamma)[tokens[i]]
            for i in range(k, len(tokens)))
    N = len(tokens) - k
    return g, N, (g - gamma * N) / math.sqrt(N * gamma * (1 - gamma))
```

### Why it suits hardware acceleration

- **Highly parallel.** Each step needs V ≈ 50k independent keyed hashes with no
  data dependence. P lanes evaluate P indices per clock at an initiation interval
  of 1. At 100 MHz, P = 16 takes ⌈V/16⌉ = 3,142 cycles ≈ 31 µs per step; P = 32
  takes 1,571 cycles ≈ 16 µs and about 192 DSP48E1 of the xc7z020's 220 (two 32-bit
  constant multiplies per lane, three DSPs each), the device's ceiling. The index
  multiply is not a multiplier: each lane adds P·C to an accumulator every cycle.
- **Latency-critical.** The watermark runs inside the autoregressive decode loop,
  once per generated token, so its latency adds directly to per-token latency. A
  software pass of 50k 32-bit hashes on the Zynq's Cortex-A9 is estimated at about
  a millisecond per token (measured, not assumed, in milestone 1). The cycle counts
  above are mode-0 figures; mode 1 is bounded by the 64-bit stream at
  ⌈V/4⌉ = 12,565 beats ≈ 126 µs per step for any P ≥ 4, which is why mode 0 is the
  deployment path.
- **Integer-only and stateless.** Every operation is a multiply by a constant,
  XOR, shift, compare or 16-bit saturating add. The state is the key, one seed and
  two counters; no memory scales with V.

### How it interacts with the PS

The PS configures the IP over AXI4-Lite and moves bulk data with the AMD AXI DMA
over two 64-bit AXI4-Stream channels. In modes 0 and 1 the PS writes the previous
token ids, pulses start, and runs one DMA packet per step: logits in and biased
logits out in mode 1, the mask out in mode 0. In mode 2 the PS streams the token
sequence in and reads the green count and length from registers once done is set.
Completion is a status bit and an optional interrupt.

## IP Architecture

```
mode 0  PS ─AXI4-Lite─► ctrl_regs ─► seed_unit ─s_lo─► hash_lanes[P] ─P bits─► mask_packer ─► axis_out ═DMA S2MM═► DDR
mode 1  DDR ═DMA MM2S═► axis_in ─logits─► logit_biaser ◄─4 bits/beat─ hash_lanes[P];  logit_biaser ─► axis_out ═► DDR
mode 2  DDR ═DMA MM2S═► axis_in ─token id─► hash_lanes[0] (s_lo reseeded per token by seed_unit) ─1 bit─► detect_acc ─G, N─► ctrl_regs ─► PS
```

### Sub-modules

1. **ctrl_regs** — AXI4-Lite register file: `KEY` (2 × 32, write-only),
   `GAMMA_THR` (⌊γ·2^32⌋), `DELTA` (Q8.8), `VOCAB`, `CTX_K`, `MODE`, `CTX0..3`
   (previous token ids), `CTRL` (start), `STATUS` (busy, done), `G_COUNT`,
   `N_COUNT`. No datapath.
2. **seed_unit** — folds K with the context tokens and runs fmix64 once per step,
   multi-cycle, off the critical path. In mode 2 it shifts each incoming token id
   into a k-deep context register and reseeds, holding the input stream for the
   six or so cycles this takes.
3. **hash_lanes** — P identical pipelined lanes. Lane p in cycle c evaluates index
   t = cP + p: accumulator step, XOR with `s_lo`, fmix32, compare with
   `GAMMA_THR`, one green bit; lanes with t ≥ `VOCAB` output zero. In mode 2 lane 0
   takes its index from the incoming token id instead of the counter. P is a
   compile-time parameter set from the resource report.
4. **mask_packer** (mode 0) — packs P bits per cycle into 64-bit beats, zero-pads
   the final beat, asserts TLAST.
5. **logit_biaser** (mode 1) — unpacks four int16 logits per beat, adds `DELTA`
   with saturation where the four matching green bits are set, repacks. One beat
   per cycle, so it takes a 4-bit slice of the lane results per cycle and the lanes
   stall otherwise.
6. **detect_acc** (mode 2) — per token, takes lane 0's green bit under that
   position's seed, accumulates `G_COUNT` and `N_COUNT`, raises done at TLAST.
7. **axis_in / axis_out** — AXI4-Stream slave and master ports with skid buffers
   and TLAST framing.

### Interfaces

| From → To | Interface | What it carries |
|---|---|---|
| PS → ctrl_regs | AXI4-Lite, 32-bit | key, γ threshold, δ, V, k, mode, previous token ids, start |
| ctrl_regs → PS | AXI4-Lite reads, IRQ | busy / done, `G_COUNT`, `N_COUNT` |
| ctrl_regs → every block | static registers | `GAMMA_THR`, `VOCAB`, `DELTA`, `MODE`, `CTX_K` |
| ctrl_regs → seed_unit | register bus, start pulse | K, `CTX0..3` |
| seed_unit → hash_lanes | 32-bit `s_lo` + valid | the step seed |
| DDR → axis_in → logit_biaser | AXI4-Stream slave, 64-bit, AXI DMA MM2S | mode 1: 4 × int16 Q8.8 logits per beat, ⌈V/4⌉ beats, TLAST per step |
| DDR → axis_in → seed_unit, hash_lanes | same channel, one 32-bit id per beat + valid | mode 2: the token to score, T beats, TLAST per sequence; seed_unit shifts it into the context, lane 0 hashes it under the previous seed |
| hash_lanes → mask_packer | `hls::stream`, P bits per cycle, ready/valid | green bits for indices cP … cP+P−1 |
| hash_lanes → logit_biaser | `hls::stream`, 4 bits per beat (P:4 slice) | green bits for the four logits in the beat |
| hash_lanes → detect_acc | 1 bit + valid | g_i(t_i) for the current token |
| mask_packer / logit_biaser → axis_out → DDR | AXI4-Stream master, 64-bit, AXI DMA S2MM | mode 0: ⌈V/64⌉ beats, bit b of beat w is token 64w+b, bits ≥ V zero · mode 1: biased logits in the input format |

Flow control is ready/valid end to end: if the DMA cannot accept output, TREADY
falls and the lanes hold their counters; nothing is dropped and no buffer grows.
logit_biaser accepts one beat per cycle, the stream's maximum. detect_acc accepts
one token per reseed, about six cycles, holding TREADY meanwhile; at roughly
15 Mtoken/s a 10k-token document scores in under a millisecond, so the reseed, not
the bus, bounds mode 2. In mode 1 the host pads the input to a whole beat and
ignores the surplus lanes. The IP has no AXI master; the AXI DMA does all memory
movement.

### Why this decomposition

- **One scaling knob, one hash.** Only hash_lanes grows with P; the other six
  blocks are small and fixed, and it is the only place the hash function appears.
  MurmurHash3 is a mixer, not a PRF, so the key in its write-only register is a
  convenience, not a security property; if secrecy matters, SipHash is a
  one-module swap.
- **Every block has a line in the golden model**: seed_unit ↔ `step_seed`,
  hash_lanes ↔ `green_mask`, logit_biaser ↔ `bias_logits`, detect_acc ↔ `detect`.
  Each is one C++ function with its own C testbench against the matching Python
  function; the top level is an HLS dataflow region joining them with
  `hls::stream`.
- **The modes build incrementally.** ctrl_regs + seed_unit + hash_lanes +
  mask_packer is a complete mode-0 IP that runs end to end first; mode 1 adds
  logit_biaser, mode 2 adds detect_acc. Rejected on the way: a permutation-based
  green list (O(V) memory, serial) and softmax plus multinomial sampling on chip
  (doubles the design for no gain in watermark correctness; argmax on chip is a
  stretch goal).

### Existing IP

AMD AXI DMA for the two stream channels, AXI SmartConnect for the AXI4-Lite port,
the AXI4-Stream Data FIFO if the DMA needs decoupling, and the AXI4-Stream Data
Width Converter in place of mask_packer's own packing if P is byte-aligned. We know
of no AMD core for the hash lanes; if the grader suggests one we will evaluate it.

### Scope

Target `xc7z020clg400-1` (PYNQ-Z2) at 100 MHz, Vitis HLS C++, bit-exact C
simulation against the Python golden model; PYNQ deployment is a stretch. Out of
scope: sampling on chip beyond argmax, a multi-bit payload, running the language
model on the PS.

### References

- J. Kirchenbauer, J. Geiping, Y. Wen, J. Katz, I. Miers, T. Goldstein, "A
  Watermark for Large Language Models," ICML 2023. arXiv:2301.10226.
- J. Kirchenbauer et al., "On the Reliability of Watermarks for Large Language
  Models," 2023. arXiv:2306.04634 (k = 1 seeding is the most edit-robust).
- S. Dathathri et al., "Scalable watermarking for identifying large language
  model outputs," Nature 634, 818–823 (2024). SynthID-Text.
- A. Appleby, MurmurHash3, public domain: the `fmix32` and `fmix64` finalisers.
