# Design review: green list, Gumbel, and the merge

Written 2026-11-20 to settle the design before the initial plan was graded. Two
plans existed:

- **Option A, green list**, Fahad Tajiki: branch `green-list`.
- **Option B, Gumbel**, Krish Mehta: branch `gumbel`, his documents as received.

`main` holds the merge. This file records what each plan did well, what did not
fit together, and where each piece of the merged design came from.

## Option A: green list (Kirchenbauer)

The IP hashes the key and context into a green list. Mode 0 returns the V-bit
mask, mode 1 returns logits with δ added to green tokens, and mode 2 counts green
tokens for a z-score. The host samples.

| Pros | Cons |
|---|---|
| Simplest datapath: hash, compare, add | Biases the distribution by δ, so text quality changes |
| Mask mode is compute-bound: 6.3 KB out instead of 100 KB of logits in, so adding lanes speeds it up | The IP never samples: mode 1 returns logits, and an argmax on chip would be greedy decoding |
| Piet et al. single the scheme out as detectable in under 100 tokens with no perceivable quality loss | AXI-Lite registers plus DMA, not the course's command-response pattern |
| Sampling stays on the host, so the same prompt gives different text | Detection reseeds serially, about 6 cycles per token |
| | No handling of repeated contexts; a repeated k-gram repeats its green list and inflates the z-score |
| | z-score approximation rather than an exact tail |

## Option B: Gumbel (Aaronson / exponential-minimum)

The same keyed hash feeds Gumbel noise from a ROM; the IP samples with an argmax
and detects with an exact Gamma p-value.

| Pros | Cons |
|---|---|
| Distortion-free: an exact softmax sample over a random key | Bandwidth-bound: P = W/16 = 4, so extra lanes do not help and the parallelism story is thin |
| True sampling on chip with no softmax | Deterministic: the same prompt and key always give the same text |
| Course command-response pattern with a full error table | ROM costs about 9 BRAM36 per copy, and the addressing scheme is new |
| Exact Gamma p-value; repeated contexts handled by a free seed and a skip | |
| Pipelined detection at 2 tokens per clock | |
| Pseudo-float ROM addressing keeps both tails accurate with no logarithm hardware | |
| Complete verification plan and timeline; prototype checks already run | |

## Feature by feature

| Feature | A | B | Fit | Merged design |
|---|---|---|---|---|
| Seed hash: fmix64 over key and context, same constants | yes | yes | identical | one seed_unit |
| Per-index hash: fmix32, index multiply as accumulator | yes | yes | identical | one hash_lanes |
| Logits int16 Q8.8 on a 64-bit stream | yes | yes | identical | kept |
| Green test | comparator on u | none | compatible | kept, on the top γ of u |
| Sampling | on the host | Gumbel argmax on chip | B supersedes | add_argmax samples for both schemes |
| Mode 1, return biased logits | yes | none | superseded | dropped; GREEN SAMPLE adds δ and samples on chip |
| Mode 0, return the green mask | yes | none | compatible | MASK command, stretch |
| Control | AXI-Lite registers plus DMA | command-response stream, AXI-Lite status | conflict | B |
| Detection | serial reseed, z-score | pipelined lanes, Gamma p-value | B supersedes | B's lanes also count green bits; exact binomial p for g |
| Repeated contexts | not handled | free seed plus skip | compatible | B's rule, applied to both statistics |
| Noise for GREEN sampling | none | unkeyed free seed | compatible | second fmix32 per lane |
| Same prompt, same text | no | yes | conflict | the scheme decides: GUMBEL is deterministic; GREEN varies because the host draws a fresh nonce per generation |
| Key secrecy claim | made, then withdrawn | none | not a property | dropped |

## The choice: option C, the hybrid, built Gumbel-first

1. **The front ends are already the same.** Krish built B on A's seed and hash, so
   the hybrid adds a comparator, a second fmix32 per lane, a mux and one adder
   input. It does not add a second datapath.
2. **One u, two tests.** Defining green as the top γ of u, where Gumbel sampling
   also concentrates, makes each detector catch both schemes. The golden model
   confirms it on synthetic logits (V = 4,096, 200 tokens); real text with less
   entropy will need more tokens: GUMBEL text gives a green-count p of 10⁻⁹⁷, and GREEN text gives a
   Gumbel p of 10⁻²⁰.
3. **A results section neither plan had.** It measures quality against detection
   strength for both published schemes on the same hardware.
4. **The risk is contained by the build order.** GUMBEL SAMPLE runs end to end in
   week 2; GREEN arrives in week 3, and MASK only if time allows. Each step leaves
   a working IP.

## Corrections made while merging

- **GREEN needs two hashes per lane.** Krish's hybrid sketch put the green test
  "on the same u" as the noise. Noise drawn from the u that decides greenness
  would be correlated with it, so GREEN draws its noise from the unkeyed u'.
- **Green is the top γ of u.** Neither plan fixed the orientation; this one makes
  the two statistics agree.
- **The ROM is about 9 BRAM36, not 8.** 16,896 × 16 bits fills 2K × 18 blocks
  8.25 times.
- **One benchmark claim was dropped.** B's comparison said Gumbel is weaker at low
  temperature; the MarkMyWords abstract does not say that, so the plan no longer
  claims it.
- **Repeat skipping covers the green count too**, as Fernandez et al. recommend.
- **A's earlier fixes stand:** its mode-1 latency is stream-bound, P = 64 lanes do
  not fit the xc7z020's DSPs, and the key is not secret behind a non-cryptographic
  hash.

## Where each piece came from

| Piece | From |
|---|---|
| fmix64 context seed, fmix32 per-index hash, constants, Q8.8 logits | A, kept by B |
| Green test, δ bias, MASK output format | A |
| Gumbel noise, pseudo-float ROM and its addressing, add_argmax | B |
| Command-response protocol, error codes, AXI-Lite status | B |
| Free seed, repeat skip, exact Gamma p-value | B |
| Pipelined detect lanes | B, extended with the green counter |
| Two-hash GREEN lane, green as top γ, exact binomial p for g, cross-detection | the merge |
| Verification plan and timeline | B, extended for GREEN |

## Git

| Branch | Holds |
|---|---|
| `green-list` | option A as committed on 2026-11-19 and 2026-11-20 |
| `gumbel` | Krish's two documents as received, and his golden model as `src/golden/gumbel_ref.py` |
| `main` | the merge commit of both branches, rewritten to the hybrid; the branch the grader reads |
