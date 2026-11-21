#!/usr/bin/env python3
"""Mutation testing: plant one bug at a time in a copy of the RTL and confirm the
top-level suite (scripts/test_rtl.py) fails. A mutation that passes means the
tests cannot see that bug.

    python3 scripts/mutation_test.py     -> results/mutations.md
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MUTATIONS = [   # (file, original text, mutated text, what the bug is)
    ("sample_core.v", "(base_g + i) < V", "(base_g + i) <= V", "pad slot past V counted as a token"),
    ("sample_core.v", "(uk[p] >= theta)", "(uk[p] > theta)", "green test off by one at the threshold"),
    ("sample_core.v", "v[1] > v[0]", "v[1] >= v[0]", "tie in lanes 0/1 goes to the higher index"),
    ("sample_core.v", "v[3] > v[2]", "v[3] >= v[2]", "tie in lanes 2/3 goes to the higher index"),
    ("sample_core.v", "bv > best_val", "bv >= best_val", "tie across beats goes to the later beat"),
    ("sample_core.v", "acc[i] + C_IDX * P", "acc[i] + C_IDX * (P + 1)", "index accumulator steps by the wrong stride"),
    ("sample_core.v", "noise_free ? uf[p] : uk[p]", "uk[p]", "GREEN and free steps use keyed noise"),
    ("sample_core.v", "((green_en && green_g[i]) ? delta : 16'sd0)", "(green_g[i] ? delta : 16'sd0)", "delta added in GUMBEL mode"),
    ("detect_core.v", "ga <= (ua >= theta)", "ga <= (ua > theta)", "detect lane a: green off by one"),
    ("detect_core.v", "gb <= (ub >= theta)", "gb <= (ub > theta)", "detect lane b: green off by one"),
    ("detect_core.v", "(pos >= k)", "(pos > k)", "first scorable position skipped"),
    ("detect_core.v", "!in_data[63]", "1'b1", "lane b ignores its skip bit"),
    ("detect_core.v", "h2 <= h0; h3 <= h1;", "h2 <= h1; h3 <= h0;", "context shift register out of order"),
    ("detect_core.v", ".c0(ta), .c1(h0), .c2(h1), .c3(h2)", ".c0(h0), .c1(ta), .c2(h1), .c3(h2)", "lane b context misordered"),
    ("seed_unit.v", "64'h2545F4914F6CDD1D", "64'h2545F4914F6CDD1F", "wrong free-seed constant"),
    ("seed_unit.v", "st[2] ? xk : xf_d", "st[3] ? xk : xf_d", "keyed seed fed one cycle late"),
    ("wm_util.v", "if (v[i]) clz32 = 6'd31 - i[5:0];", "if (v[i]) clz32 = 6'd30 - i[5:0];", "leading-zero count off by one"),
    ("wm_util.v", "half ? 7'd33 : 7'd0", "half ? 7'd32 : 7'd0", "upper-half ROM bank misplaced"),
    ("wm_util.v", "64'hFF51AFD7ED558CCD", "64'hFF51AFD7ED558CCF", "wrong fmix64 constant"),
    ("wm_util.v", "p1d ^ (p1d >> 13)", "p1d ^ (p1d >> 12)", "wrong fmix32 shift"),
    ("wm_top.v", "else state <= S_SWAIT;\n                end else if (s_axis_tlast) begin\n                    fail(E_TLAST_EARLY);",
     "else state <= S_SWAIT;\n                end else if (s_axis_tlast) begin\n                    state <= state;", "SAMPLE: early TLAST ignored"),
    ("wm_top.v", "else state <= S_DWAIT;\n                end else if (s_axis_tlast) begin\n                    fail(E_TLAST_EARLY);",
     "else state <= S_DWAIT;\n                end else if (s_axis_tlast) begin\n                    state <= state;", "DETECT: early TLAST ignored"),
    ("wm_top.v", "if (!s_axis_tlast) fail(E_TLAST_MISSING);\n                    else state <= S_SWAIT;",
     "if (1'b0) fail(E_TLAST_MISSING);\n                    else state <= S_SWAIT;", "SAMPLE: missing TLAST ignored"),
    ("wm_top.v", "if (!s_axis_tlast) fail(E_TLAST_MISSING);\n                    else state <= S_DWAIT;",
     "if (1'b0) fail(E_TLAST_MISSING);\n                    else state <= S_DWAIT;", "DETECT: missing TLAST ignored"),
    ("wm_top.v", "(state == S_SDATA && !samp_start)", "(state == S_SDATA)", "first beat accepted during the core's start pulse"),
    ("wm_top.v", "(w3_delta >= 16'sd4096)", "(w3_delta > 16'sd4096)", "delta = 16.0 accepted"),
    ("wm_top.v", "else if (!cfg_valid)\n                            fail(E_NO_CONFIG);", "else if (1'b0)\n                            fail(E_NO_CONFIG);", "DETECT before CONFIG accepted"),
    ("wm_top.v", "S_IDLE: if (ap_start) begin\n                cfg_valid <= 1'b0;", "S_IDLE: if (ap_start) begin\n                cfg_valid <= cfg_valid;", "CONFIG carried over between activations"),
    ("wm_top.v", "resp[1] <= {dN, dS};", "resp[1] <= {dS, dN};", "DETECT response fields swapped"),
]


def main():
    rows, missed = [], 0
    for f, old, new, what in MUTATIONS:
        with tempfile.TemporaryDirectory() as tmp:
            t = pathlib.Path(tmp)
            for d in ("src", "scripts"):
                shutil.copytree(ROOT / d, t / d, ignore=shutil.ignore_patterns("__pycache__"))
            (t / "results").mkdir()
            src = t / "src" / "rtl" / f
            s = src.read_text()
            if s.count(old) != 1:
                rows.append((f, what, "not applied (text not found)")); continue
            src.write_text(s.replace(old, new))
            try:
                p = subprocess.run([sys.executable, "scripts/test_rtl.py"], cwd=t, capture_output=True,
                                   text=True, timeout=600, env={**os.environ, "WM_QUICK": "1"})
                caught = p.returncode != 0
                how = (p.stdout.splitlines() or ["compile error"])[0]
                if p.returncode != 0 and "Traceback" in p.stderr:
                    how = "simulation did not build: " + p.stderr.strip().splitlines()[-1][:80]
            except subprocess.TimeoutExpired:
                caught, how = True, "timed out"
            missed += not caught
            if how.startswith("clean: FAIL (simulation timed out)") or how == "timed out":
                how = "the simulation hung (the IP waited forever)"
            elif how.startswith("clean: FAIL"):
                how = "wrong responses or status: " + how.split("(", 1)[1].rstrip(")") + " correct"
            rows.append((f, what, ("caught: " if caught else "MISSED: ") + how))
            print(rows[-1])
    out = ["# Mutation testing", "",
           "Generated by `python3 scripts/mutation_test.py`. Each row plants one bug in a copy of the",
           "RTL and runs the top-level suite (its clean and bubbles-and-backpressure runs). Caught means the",
           "suite failed.", "",
           "| file | planted bug | result |", "|---|---|---|"]
    out += [f"| `{f}` | {w} | {r} |" for f, w, r in rows]
    applied = [r for r in rows if not r[2].startswith("not applied")]
    out += ["", f"{len(applied) - missed} of {len(applied)} planted bugs caught."]
    (ROOT / "results" / "mutations.md").write_text("\n".join(out) + "\n")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
