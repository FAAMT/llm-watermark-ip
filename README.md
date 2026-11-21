# llm-watermark-ip

A hardware sampler for language-model text that embeds a statistical
watermark, and a detector that finds it. NYU ECE-GY 6463, Advanced Hardware
Design, Fall 2026.

Team: Fahad Tajiki (fxt2021) and Krish Mehta (km6152).

## What it does

A language model picks each next token by sampling from softmax(ℓ) over its V
logits. This IP is that sampler, with its randomness derived from a secret key
and the previous k tokens, so anyone holding the key can later detect the
watermark from the tokens alone. One keyed hash per candidate token drives two
published schemes:

- **GUMBEL** (Aaronson; Kuditipudi et al.): the IP returns argmax(ℓ + G(u)) with
  keyed Gumbel noise G. This is an exact softmax sample up to ROM rounding, so
  the text distribution is unchanged.
- **GREEN** (Kirchenbauer et al.): the top γ of tokens by u get +δ, and the IP
  samples with unkeyed noise.

`DETECT` streams a token sequence back and returns both statistics in one pass:
the Gumbel score S, whose p-value is a Gamma tail, and the green count g, whose
p-value is a binomial tail.

## How it is built

Verilog RTL in `src/rtl/`, verified in cycle-accurate simulation (Icarus
Verilog) against a bit-exact Python golden model, `src/golden/wm_ref.py`.

| document | contents |
|---|---|
| [docs/interface.md](docs/interface.md) | ports, AXI4-Lite registers, every command and response word, error codes |
| [docs/architecture.md](docs/architecture.md) | modules, the two pipelines cycle by cycle, parallelism, resource count, changes from the plan |
| [docs/verification.md](docs/verification.md) | the verification method and every result, with the file it came from |
| [plans/](plans/) | the initial plan as graded, the working design notes and the merge of the two original plans; all historical |

| path | what |
|---|---|
| `src/rtl/` | `wm_top.v` (controller), `seed_unit.v`, `sample_core.v`, `detect_core.v`, `axil_ctrl.v`, `wm_util.v` (hash, fold, ROM address, ROM); `rom/` holds the two ROM images |
| `src/tb/` | `tb_wm_top.v` (top level through the real ports), `tb_units.v` (arithmetic blocks) |
| `src/golden/wm_ref.py` | the golden model: every block's reference function and the host's p-values |
| `src/sw/wm_baseline.c` | the CPU baseline, the same computation as a C loop |
| `scripts/` | test, measurement and experiment scripts; each writes one file in `results/` |
| `syn/` | Vivado out-of-context synthesis and implementation flow, and a Yosys flow |
| `results/` | every generated result, in Markdown |

## Results against the plan's goals

| goal from the initial plan | target | result | file |
|---|---|---|---|
| bit-exact with the golden model | every word | unit tests 4 of 4 blocks, 80,320 vectors; scenario and error suite 50 of 50 activations in five configurations; 80 real-model steps at V = 151,936 | results/unit_tests.md, results/rtl_tests.md, results/llm_rtl_replay.md |
| tests that catch bugs | — | 29 of 29 planted bugs caught | results/mutations.md |
| GUMBEL samples softmax(ℓ) | χ² passes | p = 0.19 over 20,000 RTL samples | results/stats_rtl.md |
| GREEN samples softmax(ℓ + δ·green) | χ² passes | p = 0.74 over 20,000 RTL samples | results/stats_rtl.md |
| SAMPLE latency, V = 50,257 | 12,565 cycles, about 126 µs at 100 MHz | 12,591 cycles, 125.9 µs | results/perf.md |
| initiation interval | one beat (4 logits or 2 tokens) per cycle | met: latency − beats is constant from V = 4 to 151,936 | results/perf.md |
| DETECT throughput | 2 tokens per cycle | 1.995 at T = 16,384 | results/perf.md |
| detector false positives | uniform null p-values | 2 of 2,058 below 10⁻³, against 2.1 expected | results/llm_eval.md |
| detection on real text | — | GUMBEL detects in about 36 to 2,000 tokens depending on the model's entropy; GREEN needs 1.6 to 6.3 times more | results/llm_eval.md |
| DSP48E1, BRAM | about 110–150 DSP, 27 RAMB36 | about 167 DSP and 25.5 RAMB36 by structural count; **synthesis not run** | docs/architecture.md |
| clock | 100 MHz | **not verified**: no synthesis tool on the build machine; `make synth-vivado` is ready | syn/ |

Software comparison: a gcc -O3 loop on one desktop core takes 135 µs for the same
V = 50,257 SAMPLE (results/baseline.md). The IP at 100 MHz matches it at about
1/50 of the clock, and frees the processor from the per-token pass over the
whole vocabulary.

## What changed from the plan

The plan said Vitis HLS. The build machine has no Vitis, so the IP was written as
Verilog RTL, which the course accepts and which could be simulated cycle by
cycle here. The error codes follow the course's command-response rules rather
than the plan's table. The test model is Qwen3-Coder-30B-A3B from the local
llama.cpp server, because GPT-2 could not be installed. The MASK stretch command
was not built. Details: docs/architecture.md, "Choices that differ from the
plan".

## Reproduce

Needs Python 3 with NumPy and SciPy, Icarus Verilog 12 and gcc.

    make test        # golden self-test, unit tests, scenario and error suite, χ² through the RTL, latency
    make baseline    # C and NumPy timing
    make mutate      # planted-bug sweep (about 10 minutes)
    make llm         # real-model experiment: needs a llama.cpp server on 127.0.0.1:8081
    make replay      # recorded real-model steps through the RTL (after make llm)
    make synth-vivado    # synthesis, placement, routing and timing: needs Vivado on PATH
