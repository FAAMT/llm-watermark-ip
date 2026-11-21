#!/usr/bin/env python3
"""Top-level RTL verification: every scenario and error case in one testbench
run, compared word for word with the golden model, then the same stimulus again
with random input bubbles and output backpressure.

    python3 scripts/test_rtl.py        -> results/rtl_tests.md, exit 1 on any mismatch
"""
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import wmtb as T  # noqa: E402
from wmtb import W  # noqa: E402

KEY, NONCE = 0x0123456789ABCDEF, 0x5EED5EED12345678
rng = np.random.default_rng(2026)


def rt(V):
    return [int(x) for x in rng.integers(V, size=4)]


def scenarios():
    S = []   # (name, what it checks, [activations])

    a = T.Activation().config(1, 37, KEY, NONCE, k=1)
    for s in range(8):
        a.sample(100 + s, T.rand_logits(rng, 37), rt(37), step=s, free=(s % 3 == 2))
    a.detect(200, [int(x) for x in rng.integers(37, size=9)], skip=[0, 0, 1, 0, 0, 0, 0, 1, 0])
    a.detect(201, [int(x) for x in rng.integers(37, size=10)])
    S.append(("gumbel_k1", "GUMBEL, k = 1, V = 37 (not a multiple of 4), free steps, DETECT odd and even T with skips", [a.end(300)]))

    a = T.Activation().config(1, 64, KEY, NONCE, gamma=0.5, delta=3.0, k=4, scheme=W.GREEN)
    for s in range(8):
        a.sample(100 + s, T.rand_logits(rng, 64), rt(64), step=s)
    a.detect(200, [int(x) for x in rng.integers(64, size=21)])
    S.append(("green_k4", "GREEN, k = 4, gamma 0.5, delta 3.0", [a.end(300)]))

    acts = []
    for k, V in ((2, 5), (3, 6)):
        a = T.Activation().config(1, V, KEY ^ k, NONCE, k=k, gamma=0.25)
        for s in range(6):
            a.sample(100 + s, T.rand_logits(rng, V), rt(V), step=s)
        a.detect(200, [int(x) for x in rng.integers(V, size=12)])
        acts.append(a.end(300))
    S.append(("k2_k3", "k = 2 and 3, tiny vocabularies, two activations back to back", acts))

    a = T.Activation()
    for V in (1, 4, 5):
        a.config(10 + V, V, KEY, NONCE, k=1)
        a.sample(20 + V, T.rand_logits(rng, V), [0], step=V)
    S.append(("vocab_edges", "V = 1, 4, 5; CONFIG replaced mid-activation", [a.end(99)]))

    # ties: logits chosen so every candidate scores the same; the smallest index must win
    a = T.Activation().config(1, 23, KEY, NONCE, k=1)
    for s in range(3):
        ctx = [s]
        u = W.uniforms(W.step_seed(KEY, ctx), range(23))
        G = W.G_ROM[W.pf_index(u)].astype(np.int64)
        a.sample(100 + s, (-2000 - G).astype(np.int16), ctx, step=s)
        assert a.last_sample[0] == 0
    # ties placed on chosen indices: inside lanes 2/3, lanes 0/1, and across beats
    for s, (top, want) in enumerate((((2, 3), 2), ((1, 0), 0), ((9, 5, 13), 5), ((22, 21), 21))):
        ctx = [10 + s]
        u = W.uniforms(W.step_seed(KEY, ctx), range(23))
        G = W.G_ROM[W.pf_index(u)].astype(np.int64)
        lg = -3000 - G
        lg[list(top)] += 1000
        a.sample(120 + s, lg.astype(np.int16), ctx, step=s)
        assert a.last_sample[0] == want, (top, a.last_sample)
    a.config(2, 23, KEY, NONCE, k=1, scheme=W.GREEN, gamma=0.25, delta=2.0)
    ctx = [7]
    uf = W.uniforms(W.free_seed(NONCE, 5), range(23))
    green = W.green(W.uniforms(W.step_seed(KEY, ctx), range(23)), W.green_thr(0.25))
    G = W.G_ROM[W.pf_index(uf)].astype(np.int64)
    a.sample(110, (-2000 - G - 512 * green).astype(np.int16), ctx, step=5)
    assert a.last_sample[0] == 0
    S.append(("argmax_ties", "all candidates tied, and ties placed in lanes 2/3, lanes 0/1, across beats and in the last beat: the smallest index wins", [a.end(300)]))

    a = T.Activation().config(1, 13, KEY, NONCE, k=1, scheme=W.GREEN, gamma=0.75, delta_q=4095)
    lg = np.full(13, -32768, np.int16); lg[9] = 0
    a.sample(100, lg, [3], step=0)
    a.sample(101, np.zeros(13, np.int16), [4], step=1)
    a.sample(102, np.full(13, -32768, np.int16), [5], step=2)
    S.append(("extremes", "delta at its maximum, logits at -32768 and 0, all logits at -32768", [a.end(300)]))

    # pad slots: V = 13, all logits equal; contexts where the first pad index (13)
    # would have the largest noise, so a pad wrongly counted as a token would win
    a = T.Activation().config(1, 13, KEY, NONCE, k=1)
    found = 0
    for c in range(5000):
        G = W.G_ROM[W.pf_index(W.uniforms(W.step_seed(KEY, [c]), range(16)))].astype(np.int64)
        if G[13] > G[:13].max() and found < 2:
            a.sample(130 + found, np.full(13, -5000, np.int16), [c], step=found, pad=32767)
            assert a.last_sample[0] < 13
            found += 1
    assert found == 2
    S.append(("pad_slots", "V = 13, pad slots past V sent as 32767 with the largest noise: they must still lose", [a.end(300)]))

    # green threshold boundary: theta equal to one token's u exactly (u >= theta is green)
    c, t5 = 11, 5
    u = W.uniforms(W.step_seed(KEY, [c]), range(40))
    theta = int(u[t5])
    uf = W.uniforms(W.free_seed(NONCE, 0), range(40))
    Gf = W.G_ROM[W.pf_index(uf)].astype(np.int64)
    lg = -3100 - Gf
    lg[t5] += 50
    a = T.Activation().config(1, 40, KEY, NONCE, k=1, scheme=W.GREEN, theta=theta, delta=2.0)
    assert (u >= theta).sum() >= 2
    a.sample(140, lg.astype(np.int16), [c], step=0)
    assert a.last_sample[0] == t5
    a.detect(141, [c, t5])                     # boundary token in lane b
    assert a.last_detect[1] == 1
    a.detect(142, [9, c, t5])                  # and in lane a
    assert a.last_detect[1] == 1
    S.append(("theta_boundary", "u exactly equal to theta counts as green, in SAMPLE and in DETECT", [a.end(300)]))

    a = T.Activation().config(1, 50, KEY, NONCE, k=4)
    a.detect(200, [7])                                     # T = 1 < k
    a.detect(201, [1, 2, 3, 4])                            # T = k: nothing scored
    a.detect(202, [int(x) for x in rng.integers(50, size=9)], skip=[1] * 9)
    a.detect(203, [int(x) for x in rng.integers(50, size=1001)])
    a.detect(204, [5] * 40)                                # one repeated context
    S.append(("detect_edges", "T = 1, T = k, all skipped, T = 1001, a repeated token", [a.end(300)]))

    # generation as the host runs it: context from the IP's own outputs, repeat flags
    for scheme in (W.GUMBEL, W.GREEN):
        a = T.Activation().config(1, 40, KEY, NONCE, k=2, scheme=scheme)
        toks = [3, 17]
        seen = {tuple(toks)}
        for s in range(40):
            free = tuple(toks[-2:]) in seen and s > 0
            a.sample(100 + s, T.rand_logits(rng, 40, scale=1.0), toks[-2:], step=s, free=free)
            seen.add(tuple(toks[-2:]))
            toks.append(a.last_sample[0])
        a.detect(200, toks, skip=W.repeat_flags(toks, 2))
        S.append((f"closed_loop_{'gumbel' if scheme == W.GUMBEL else 'green'}",
                  "40-step generation feeding the IP its own tokens, then DETECT on the text", [a.end(300)]))

    return S


def errors():
    E = []

    def fresh():
        a = T.Activation().config(1, 9, KEY, NONCE, k=1)
        a.sample(2, T.rand_logits(rng, 9), [1], step=0)
        return a.end(3)

    a = T.Activation().config(1, 9, KEY, NONCE, k=1)
    a.sample(2, T.rand_logits(rng, 9), [1], step=0)
    a.raw([(1, 1, T.hdr0(7, 55, 0))]).error("BAD_CMD", 55)
    a.raw([(1, 1, T.hdr0(T.CMD_END, 56, 0))])
    E.append(("BAD_CMD", "unknown command 7 after two good commands", a))

    for label, kw in (("V = 0", dict(V=0)), ("V = 2^24", dict(V=1 << 24)), ("k = 0", dict(k=0)),
                      ("k = 5", dict(k=5)), ("scheme = 2", dict(scheme=2)), ("theta = 0", dict(theta=0)),
                      ("delta = 0", dict(delta_q=0)), ("delta < 0", dict(delta_q=-5)),
                      ("delta = 16.0", dict(delta_q=4096))):
        args = dict(V=9, k=1, scheme=W.GUMBEL); args.update(kw)
        a = T.Activation().config(9, args.pop("V"), KEY, NONCE, valid=False, **args).error("BAD_PARAM", 9)
        a.raw([(1, 1, T.hdr0(T.CMD_END, 10, 0))])
        E.append(("BAD_PARAM", f"CONFIG with {label}", a))

    for label, Tn in (("T = 0", 0), ("T = 2^24 + 1", (1 << 24) + 1)):
        a = T.Activation().config(1, 9, KEY, NONCE, k=1)
        a.raw([(1, 1, T.hdr0(T.CMD_DETECT, 12, Tn))]).error("BAD_SIZE", 12)
        E.append(("BAD_SIZE", f"DETECT with {label}", a))

    a = T.Activation()
    a.raw([(1, 1, T.hdr0(T.CMD_DETECT, 13, 0))]).error("BAD_SIZE", 13)
    E.append(("BAD_SIZE", "DETECT with T = 0 before any CONFIG: the size check comes before NO_CONFIG", a))

    a = T.Activation()
    a.sample(4, T.rand_logits(rng, 9), [1], predict=False); a.error("NO_CONFIG", 4)
    E.append(("NO_CONFIG", "SAMPLE before any CONFIG", a))
    a = T.Activation()
    a.detect(5, [1, 2, 3], predict=False); a.error("NO_CONFIG", 5)
    E.append(("NO_CONFIG", "DETECT before any CONFIG", a))

    def corrupt(act, start, idx, last):
        w = act.words[start + idx]
        act.words[start + idx] = (w[0], last, w[2])

    a = T.Activation().config(1, 37, KEY, NONCE, k=1)
    n0 = len(a.words) + 3
    a.sample(6, T.rand_logits(rng, 37), [1], predict=False); corrupt(a, n0, 3, 1); a.error("TLAST_EARLY", 6)
    E.append(("TLAST_EARLY", "SAMPLE burst of 10 beats with TLAST on beat 4", a))
    a = T.Activation().config(1, 37, KEY, NONCE, k=1)
    n0 = len(a.words) + 1
    a.detect(7, [1, 2, 3, 4, 5, 6, 7, 8, 9], predict=False); corrupt(a, n0, 1, 1); a.error("TLAST_EARLY", 7)
    E.append(("TLAST_EARLY", "DETECT burst of 5 beats with TLAST on beat 2", a))
    a = T.Activation().config(1, 37, KEY, NONCE, k=1)
    n0 = len(a.words) + 3
    a.sample(8, T.rand_logits(rng, 37), [1], predict=False); corrupt(a, n0, 9, 0); a.error("TLAST_MISSING", 8)
    E.append(("TLAST_MISSING", "SAMPLE burst whose last beat has no TLAST", a))
    a = T.Activation().config(1, 37, KEY, NONCE, k=1)
    n0 = len(a.words) + 1
    a.detect(9, [1, 2, 3, 4, 5], predict=False); corrupt(a, n0, 2, 0); a.error("TLAST_MISSING", 9)
    E.append(("TLAST_MISSING", "DETECT burst whose last beat has no TLAST", a))
    return E, fresh


def main():
    S = scenarios()
    E, fresh = errors()
    acts, index = [], []
    for name, what, al in S:
        for a in al:
            index.append(("scenario", name, what)); acts.append(a)
    for code, what, a in E:
        index.append(("error", code, what)); acts.append(a)
        index.append(("fresh", code, "fresh activation after the error")); acts.append(fresh())

    runs = [("clean", dict(seed=1, bubble=0, stall=0)),
            ("bubbles 35 %, backpressure 45 %", dict(seed=7, bubble=35, stall=45)),
            ("MUL_REG = 0", dict(seed=1, bubble=0, stall=0, mul_reg=0)),
            ("MUL_REG = 2", dict(seed=3, bubble=20, stall=20, mul_reg=2)),
            ("MUL_REG = 3", dict(seed=5, bubble=20, stall=20, mul_reg=3))]
    if os.environ.get("WM_QUICK"):                      # mutation sweep: the first two runs suffice
        runs = runs[:2]
    results = []
    for label, kw in runs:
        try:
            ok, rows = T.run(acts, f"rtl_tests_{kw['seed']}_{kw.get('mul_reg')}", maxcyc=2_000_000, **kw)
        except RuntimeError:                            # a hung simulation is a failure
            print(f"{label}: FAIL (simulation timed out)"); sys.exit(1)
        results.append((label, ok, rows))
        print(f"{label}: {'PASS' if ok else 'FAIL'} ({sum(r['resp_ok'] and r['status_ok'] for r in rows)}/{len(rows)} activations)")

    out = ["# RTL verification: scenarios and errors", "",
           "Generated by `python3 scripts/test_rtl.py`. Every activation is compared word for word",
           "with the golden model `src/golden/wm_ref.py`, its status registers with the expected status, and",
           "the TLAST of every response word with its expected position. Every run uses",
           "the same stimulus: the second adds random gaps in the input stream and random backpressure on the",
           "output stream, and the last three rebuild the RTL with a different MUL_REG pipeline depth. The",
           "status column also requires CTRL to read busy or done on the first poll and only idle after it.", "",
           "| run | result | activations |", "|---|---|---|"]
    for label, ok, rows in results:
        out.append(f"| {label} | {'PASS' if ok else 'FAIL'} | {sum(r['resp_ok'] and r['status_ok'] for r in rows)} of {len(rows)} |")
    out += ["", "| # | kind | case | what it checks | responses | status (halted, error, tx_id, commands) | clean | bubbles |",
            "|---|---|---|---|---|---|---|---|"]
    for i, (kind, name, what) in enumerate(index):
        r0, r1 = results[0][2][i], results[1][2][i]
        st = r0["got_status"]
        out.append(f"| {i} | {kind} | {name} | {what} | {r0['nresp']} words | {st} | "
                   f"{'PASS' if r0['resp_ok'] and r0['status_ok'] else 'FAIL'} | "
                   f"{'PASS' if r1['resp_ok'] and r1['status_ok'] else 'FAIL'} |")
    pathlib.Path(T.ROOT / "results" / "rtl_tests.md").write_text("\n".join(out) + "\n")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
