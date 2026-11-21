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

RESULTS_TABLE

## Reproduce

Needs Python 3 with NumPy and SciPy, Icarus Verilog 12 and gcc.

    make test        # golden self-test, unit tests, scenario and error suite, χ² through the RTL, latency
    make baseline    # C and NumPy timing
    make mutate      # planted-bug sweep (about 10 minutes)
    make llm         # real-model experiment: needs a llama.cpp server on 127.0.0.1:8081
    make replay      # recorded real-model steps through the RTL (after make llm)
    make synth-vivado    # synthesis, placement, routing and timing: needs Vivado on PATH
