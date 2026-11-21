# Interface

What the host and the IP exchange, word by word. The executable form of every
layout here is `scripts/wmtb.py` (encoder) and `src/golden/wm_ref.py` (expected
responses); the RTL is `src/rtl/wm_top.v`.

## Ports

| port | type | width | role |
|---|---|---|---|
| `ap_clk`, `ap_rst_n` | clock, synchronous active-low reset | 1 | target 100 MHz |
| `s_axis_*` | AXI4-Stream slave: `tdata`, `tvalid`, `tready`, `tlast` | 64 | commands and their data |
| `m_axis_*` | AXI4-Stream master: `tdata`, `tvalid`, `tready`, `tlast` | 64 | responses |
| `s_axil_*` | AXI4-Lite slave | 6-bit address, 32-bit data | control and status only |

On PYNQ an AMD AXI DMA drives both streams (MM2S into `s_axis`, S2MM from `m_axis`).

## Control and status registers (AXI4-Lite)

| offset | name | access | meaning |
|---|---|---|---|
| 0x00 | CTRL | write bit 0 = 1 | `ap_start`: begin an activation (ignored while busy) |
| 0x00 | CTRL | read | bit 0 busy, bit 1 `ap_done` (cleared on read), bit 2 idle |
| 0x10 | STATUS | read | bit 0 `halted`: the activation ended in an error |
| 0x14 | ERROR | read | error code, 0 = none |
| 0x18 | TX_ID | read | tx_id of the last command read; on an error, the failing command's |
| 0x1C | NCMD | read | commands completed in this activation |

## Activation

The course command-response pattern: the host writes `ap_start`, streams
commands, ends with `END`, and waits for `ap_done`. Nothing carries over
between activations: CONFIG, status and counters are cleared at `ap_start`, so
an activation must begin with CONFIG. On an error the IP sets STATUS, ERROR and
TX_ID, reads nothing more, and returns; the host resets the stream path before
the next activation.

## Command header, word 0 (all commands)

| bits | field |
|---|---|
| 7:0 | command: 0 CONFIG, 1 SAMPLE, 2 DETECT, 3 END |
| 8 | free bit (SAMPLE only): this step's context already appeared in the text |
| 31:16 | tx_id, echoed in the response |
| 63:32 | CONFIG: V · SAMPLE: step · DETECT: T · END: unused |

## Commands

| command | header words | then | response |
|---|---|---|---|
| CONFIG | 4: word 0; word 1 = key K; word 2 = nonce; word 3 = θ (31:0), δ int16 Q8.8 (47:32), k (50:48), scheme (57:56: 0 GUMBEL, 1 GREEN) | nothing | none |
| SAMPLE | 3: word 0; word 1 = c0 (23:0), c1 (55:32); word 2 = c2 (23:0), c3 (55:32), where c0 is the newest context token | ⌈V/4⌉ beats; beat b carries logits 4b … 4b+3 as int16 Q8.8 in bits 15:0, 31:16, 47:32, 63:48; slots past V are ignored; TLAST on the last beat | 2 words: header, then token id (23:0) and its winning value ℓ + δ·green + G, sign-extended (63:32) |
| DETECT | 1 | ⌈T/2⌉ beats; each 32-bit half is one token: id (23:0), skip bit (31); the upper half of the last beat is unused when T is odd; TLAST on the last beat | 3 words: header, then S (31:0) and N (63:32), then g (31:0) |
| END | 1 | nothing | none; `ap_done` follows |

Response header: command (7:0), tx_id (31:16), number of payload words (63:32);
TLAST is on the last payload word.

Value formats: logits int16 Q8.8 in natural-log units, after the host has scaled
by 1/temperature, subtracted the maximum and clamped at −128.0; the winning
value is int18; S is uint32 Q21.11; g and N are uint32.

## Errors

| code | name | when |
|---|---|---|
| 1 | BAD_CMD | unknown command code |
| 2 | BAD_PARAM | CONFIG with V = 0 or V ≥ 2^24, k outside 1–4, scheme > 1, θ = 0, δ ≤ 0 or δ ≥ 16.0 |
| 3 | BAD_SIZE | DETECT with T = 0 or T > 2^24 |
| 4 | NO_CONFIG | SAMPLE or DETECT before any CONFIG in this activation |
| 5 | TLAST_EARLY | a data burst ends before its declared length |
| 6 | TLAST_MISSING | the last word of a data burst has no TLAST |

The first error in stream order wins; within one header the order is command
code, then size or parameters, then NO_CONFIG. TLAST is checked on data bursts
only, not on headers. A failing command writes no response, so there is never
an open output burst to close.

## Timing (measured, results/perf.md)

| command | latency from its first header word to its response's last word |
|---|---|
| SAMPLE | ⌈V/4⌉ + 26 cycles |
| DETECT | ⌈T/2⌉ + 19 cycles |
| CONFIG | four header words, no response |

Within a data burst the IP accepts one beat per cycle whenever `s_axis_tvalid` is
high. `s_axis_tready` is low between a SAMPLE header and its first logit beat
(12 cycles: the seeds, then the core's start cycle), for one cycle before a
DETECT burst, and while a response is being written. It never drops data under
backpressure: the response waits in place until `m_axis_tready` rises.

## Host side (intended integration, not built)

On a PYNQ-Z2, an AMD AXI DMA would carry the streams: one MM2S transfer per
header and one per data burst, since the IP expects TLAST on the last word of
each, and one S2MM transfer per response. The testbench models exactly this
packetisation. The PYNQ driver and block design were not built: the project was
verified in simulation, which the course accepts.

S, g and N add across texts, so a host scoring a long document can split it
into several DETECT commands and sum the results. That also keeps S below its
2^32 wrap, which a single DETECT reaches after about 2 million tokens of
average score.
