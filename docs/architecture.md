# Architecture

How the IP is built: the modules, the pipelines with their latencies, where
parallelism comes from, and what it costs in resources. Source: `src/rtl/`.

```
 s_axis ─► wm_top controller ──CONFIG regs──────────────────────────────────────────────┐
 (64-bit)    │ header parse, error checks, burst count                                      │
             ├─SAMPLE─► seed_unit ─s, s_f─► sample_core ─token, value─┐                     │
             │          ctx_fold+fmix64      4 lanes × (2 fmix32, green test,              │
             │                               pf_index, Gumbel ROM) → add → max tree         │
             │                               → running max                                  ├─► response words ─► m_axis
             └─DETECT─► detect_core ─S, g, N──────────────────────────────────────────────┘    (64-bit)
                        ctx_shift → 2 lanes × (ctx_fold, fmix64, fmix32, green, pf_index)
                        → shared score ROM → score_acc
 s_axil ◄─► axil_ctrl: ap_start, ap_done, halted, error, tx_id
```

## Modules

| module | file | does | latency | golden function |
|---|---|---|---|---|
| `wm_top` | `wm_top.v` | controller FSM: header parse, error checks in course order, CONFIG storage, burst counting, sequencing, response | — | the command layer of `scripts/wmtb.py` |
| `axil_ctrl` | `axil_ctrl.v` | AXI4-Lite control and status registers | 1 | — |
| `seed_unit` | `seed_unit.v` | keyed seed s and unkeyed seed s_f once per SAMPLE, sharing one fmix64 | ready about 9 cycles after start | `step_seed`, `free_seed` |
| `sample_core` | `sample_core.v` | hash_lanes, rom_index, Gumbel ROM, add_argmax for P = 4 lanes | 10 cycles from beat to running max | `uniforms`, `green`, `pf_index`, `sample` |
| `detect_core` | `detect_core.v` | ctx_shift, two detect lanes, shared score ROM, score_acc | 15 cycles from beat to accumulators | `detect` |
| `fmix32_pipe` | `wm_util.v` | MurmurHash3 32-bit finaliser | 3 + 2·MUL_REG = 5 | `fmix32` |
| `fmix64_pipe` | `wm_util.v` | MurmurHash3 64-bit finaliser | 5 | `fmix64` |
| `ctx_fold` | `wm_util.v` | key ⊕ Σ (c_j + 1)·C_j over the k newest tokens | 2 | `step_seed` before its fmix64 |
| `pf_index` | `wm_util.v` | pseudo-float ROM address: leading-zero count and shift | 1 | `pf_index` |
| `rom_dp` | `wm_util.v` | 16,896 × 16-bit dual-port ROM in block RAM | 1 | `G_ROM`, `S_ROM` |
| `delay`, `delay_rst` | `wm_util.v` | alignment registers for data and valid flags | N | — |

## The SAMPLE pipeline

One 64-bit beat carries four logits, so four lanes (P = 64/16 = 4) each handle
one vocabulary index per cycle. Lane p in beat b handles index t = 4b + p.

| cycle after the beat | lane work |
|---|---|
| 0 | index hash input `acc_p ⊕ s` and `acc_p ⊕ s_f`, where `acc_p = t·0x9E3779B1 mod 2^32` is kept by an adder that steps by 4·0x9E3779B1, so there is no index multiplier |
| 1–5 | two pipelined fmix32 per lane: keyed u and unkeyed u′ |
| 5 | green bit `u ≥ θ`; noise select: u′ if GREEN or a free step, else u |
| 6 | `pf_index`: ROM address |
| 7 | Gumbel ROM read; lanes 0/1 share one dual-port copy, lanes 2/3 another |
| 8 | `v = ℓ + δ·green + G` in 18 bits; lane valid if t < V |
| 9 | 4-input max tree, lower index wins ties |
| 10 | running maximum across beats, replaced only by a strictly greater value |

The logits and the index base travel alongside in delay lines, so nothing
waits on anything else: the pipeline accepts a beat every cycle (II = 1) and
runs freely, with no stall logic. This works because SAMPLE's only output is one
response after the last beat; output backpressure therefore stalls only the
response, never the datapath.

## The DETECT pipeline

A beat carries two tokens, so two lanes (P_d = 64/32 = 2). `ctx_shift` keeps the
last four tokens; lane a's context is that history, lane b's is lane a's token
plus the history.

| cycle after the beat | lane work |
|---|---|
| 0 | token × 0x9E3779B1; position check (scored if k ≤ i < T and not skipped) |
| 1–2 | `ctx_fold` of the lane's own context |
| 3–7 | fmix64: the seed |
| 8–12 | fmix32 of (token·C ⊕ seed): u |
| 13 | green bit and ROM address |
| 14 | score ROM read (one dual-port copy shared by both lanes) |
| 15 | `score_acc`: S += scores, g += green bits, N += scored tokens |

Every token's seed depends only on tokens already in the input, so the two lanes
never wait on each other and DETECT also runs at II = 1.

## Parallelism and its limit

- **SAMPLE is bandwidth-bound.** Every logit must cross the 64-bit stream once,
  four per beat, so four lanes saturate it. More lanes would sit idle; a 128-bit
  stream would allow P = 8 and halve the latency. Measured: ⌈V/4⌉ + 26 cycles,
  12,591 cycles for GPT-2's V = 50,257 (results/perf.md), against the plan's
  12,565-cycle estimate.
- **DETECT is bandwidth-bound too**, at two tokens per beat: 1.995 tokens per
  cycle measured at T = 16,384.
- **Pipelining** puts a register after every multiply (`MUL_REG = 1`) so the
  synthesiser can absorb it into the DSP48E1's M and P registers; `MUL_REG` is a
  parameter, raised if timing needs more.
- **Memory:** the two 16,896-entry tables live in block RAM. Dual ports let two
  lanes share one copy, so P = 4 needs two Gumbel ROM copies, and DETECT's two
  lanes need one score ROM copy.

## Resources

Synthesis has not run: neither Vivado nor Yosys is installed on the machine
this was built on. `make synth-vivado` and `make synth-yosys` produce the reports
in `results/synth/`. Until then, this is a structural count from the RTL.

| multiplier (variable × constant) | where | count | DSP48E1 each (estimate) | DSP48E1 |
|---|---|---|---|---|
| 32 × 32 → 32 | fmix32: 16 in sample lanes, 4 in detect lanes | 20 | 3 | 60 |
| 64 × 64 → 64 | fmix64: 4 in detect lanes, 2 in seed_unit | 6 | 8 | 48 |
| 24 × 64 → 64 | ctx_fold: 8 in detect lanes, 4 in seed_unit | 12 | 4 | 48 |
| 32 × 64 → 64 | free seed (step + 1)·C_4 | 1 | 7 | 7 |
| 24 × 32 → 32 | token × 0x9E3779B1 in detect lanes | 2 | 2 | 4 |
| **total** | | **41** | | **≈ 167 of 220** |

The per-multiply figures count 24-bit × 17-bit unsigned partial products and drop
the ones that land entirely above the output width.

| memory | entries × width | copies | RAMB36 each | RAMB36 |
|---|---|---|---|---|
| Gumbel ROM | 16,896 × 16 | 2 | 8.5 | 17 |
| score ROM | 16,896 × 16 | 1 | 8.5 | 8.5 |
| **total** | | | | **≈ 25.5 of 140** |

The DSP estimate is above the plan's 110–150 because the plan left out the
context folds in seed_unit and the detect lanes. Two reductions are available if
synthesis confirms the overrun:

- **seed_unit could share detect lane a's fold and fmix64.** SAMPLE and DETECT
  never run at the same time, which saves about 32 DSPs.
- **DETECT could run one lane at one token per cycle.** That still scores 100
  million tokens per second and saves about 40 DSPs.

## Choices that differ from the plan

| plan | built | why |
|---|---|---|
| Vitis HLS C++ | Verilog RTL | the build machine has no Vitis; RTL can be simulated cycle-accurately with Icarus Verilog here, and the course accepts RTL or HLS |
| error 1 = header TLAST early or missing | TLAST checked on data bursts only; codes renumbered BAD_CMD, BAD_PARAM, BAD_SIZE, NO_CONFIG, TLAST_EARLY, TLAST_MISSING | the course's command-response rules check TLAST on data only and order errors code → size → configuration |
| DETECT T ≤ k is an error | allowed; returns N = 0 | nothing to score is a valid answer, and the check needed CONFIG state inside the header check |
| GPT-2 logits | Qwen3-Coder-30B-A3B logits from the local llama.cpp server | GPT-2 weights and the transformers library could not be installed on the build machine |
| MASK command (stretch) | not built | time went to verification; GUMBEL and GREEN SAMPLE and DETECT are complete |
| (c + 1)·C in the fold | c·C + C | same value, keeps the multiplier operand at 24 bits |
