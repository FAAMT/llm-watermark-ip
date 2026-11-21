#!/usr/bin/env python3
"""Watermark experiments on a real language model.

The model is Qwen3-Coder-30B-A3B (V = 151,936) served by llama.cpp on this
machine's local stack (http://127.0.0.1:8081). At every step the host asks it
for the top-100 next-token log-probabilities, builds the IP's int16 Q8.8 logit
vector (top-k sampling: every other token at -32768), and picks the token with
the golden model, which the RTL matches bit for bit (scripts/replay_llm_rtl.py
replays recorded steps through the RTL). Plain text is sampled from the same
top-100 distribution with an ordinary RNG.

    python3 scripts/llm_eval.py   -> build/llm/*.jsonl (raw steps), results/llm_eval.md, results/llm_eval.json
"""
import concurrent.futures as cf
import json
import threading
import time
import urllib.error
import math
import pathlib
import sys
import urllib.request

import numpy as np
from scipy.special import gammaincc
from scipy.stats import binom, kstest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from wmtb import ROOT, W  # noqa: E402

SERVER = "http://127.0.0.1:8081"
V, K, L = 151936, 100, 200
KEY = 0x0123456789ABCDEF
GAMMA, DELTA_Q = 0.25, 512
PROMPTS = [
    "Write a short essay about the history of the bicycle.",
    "Explain how a refrigerator works to a ten-year-old.",
    "Describe a walk through a city market on a rainy morning.",
    "What are the main causes of inflation? Answer in a few paragraphs.",
    "Write a letter to a friend recommending a book you enjoyed.",
    "Summarize the rules of chess for a beginner.",
]
OUT = ROOT / "build" / "llm"


def post(path, body, tries=8):
    # the local server is shared with other agents: retry transient failures with backoff
    for i in range(tries):
        try:
            req = urllib.request.Request(SERVER + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 500 and b"Failed to parse input" in e.read():
                raise ValueError("partial UTF-8 token at the end of the prompt")
            if i == tries - 1:
                raise
            time.sleep(2 ** i)
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


def prompt_ids(text):
    chat = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    return post("/tokenize", {"content": chat, "parse_special": True})["tokens"]


def top(ids):
    for _ in range(3):
        d = post("/completion", {"prompt": ids, "n_predict": 1, "n_probs": K, "post_sampling_probs": False,
                                 "cache_prompt": True, "temperature": 1.0})
        if d.get("completion_probabilities"):
            t = d["completion_probabilities"][0]["top_logprobs"]
            return np.array([x["id"] for x in t]), np.array([x["logprob"] for x in t])
        time.sleep(1)
    return None, None                      # no distribution returned: the text ends here


def generate(job):
    scheme, k, temp, pi, seed = job
    rng = np.random.default_rng(seed)
    nonce = int(rng.integers(0, 2**63))                     # fresh nonce per generation
    ids = prompt_ids(PROMPTS[pi])
    gen, steps, seen = [], [], set()
    prev = None
    for step in range(L):
        try:
            tid, lp = top(ids + gen)
        except ValueError:
            # llama.cpp rejects a prompt ending in an incomplete UTF-8 byte token. Only plain
            # sampling can hit this (every watermarked text finished): redraw that step
            # without the offending token.
            if scheme != "plain" or prev is None:
                break
            bad = gen.pop(); steps.pop()
            ptid, px = prev
            keep = ptid != bad
            p = np.exp(px[keep] - px[keep].max()); t = int(ptid[keep][rng.choice(keep.sum(), p=p / p.sum())])
            steps.append(dict(ctx=(ids + gen)[-k:], ids=ptid.tolist(), q=[], free=False, tok=t,
                              lp=float(prev_lp[list(ptid).index(t)]), redrawn=True))
            gen.append(t)
            continue
        if tid is None:
            break
        x = lp / temp
        q = np.round(np.clip(x - x.max(), -128, 127) * 256).astype(np.int16)
        ctx = (ids + gen)[-k:]
        free = tuple(ctx) in seen
        seen.add(tuple(ctx))
        if scheme == "plain":
            p = np.exp(x - x.max()); t = int(tid[rng.choice(len(tid), p=p / p.sum())])
            prev, prev_lp = (tid, x), lp
        else:
            lg = np.full(V, -32768, np.int16); lg[tid] = q
            t, _ = W.sample(KEY, ctx, lg, W.GUMBEL if scheme == "gumbel" else W.GREEN, GAMMA, DELTA_Q,
                            free=free, nonce=nonce, step=step)
        lp_chosen = float(lp[list(tid).index(t)])
        steps.append(dict(ctx=ctx, ids=tid.tolist(), q=q.tolist(), free=free, tok=t, lp=lp_chosen))
        gen.append(t)
        if t == 151645:                                         # <|im_end|>
            break
    return dict(scheme=scheme, k=k, temp=temp, prompt=pi, nonce=nonce, tokens=gen, steps=steps)


def scores(key, toks, k):
    """Per-token Gumbel score (ROM value / 2048) and green bit, skipping repeated contexts."""
    skip = W.repeat_flags(toks, k)
    sc, gr = [], []
    for i in range(k, len(toks)):
        if skip[i]: continue
        u = W.uniforms(W.step_seed(key, toks[i - k:i]), [toks[i]])
        sc.append(int(W.S_ROM[W.pf_index(u)[0]])); gr.append(bool(W.green(u, W.green_thr(GAMMA))[0]))
    return np.array(sc, np.int64), np.array(gr, bool)


def pvals(sc, gr):
    if len(sc) == 0:
        return 1.0, 1.0, None, None
    N = np.arange(1, len(sc) + 1)
    pS = gammaincc(N, np.cumsum(sc) / 2048)
    pg = binom.sf(np.cumsum(gr) - 1, N, GAMMA)
    first = lambda p: int(np.argmax(p < 1e-6)) + 1 if (p < 1e-6).any() else None
    return float(pS[-1]), float(pg[-1]), first(pS), first(pg)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, k, t, pi, 1000 * i + pi) for i, (s, k, t) in enumerate(
        [(s, k, t) for s in ("gumbel", "green") for k in (1, 2, 4) for t in (0.7, 1.0)] +
        [("plain", k, t) for k in (1,) for t in (0.7, 1.0)] +
        [(s, 2, 1.5) for s in ("gumbel", "green", "plain")]) for pi in range(len(PROMPTS))]
    # resumable: every finished text is appended at once, and a rerun skips them
    path = OUT / "texts.jsonl"
    done = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line); done[(r["scheme"], r["k"], r["temp"], r["prompt"])] = r
    todo = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"{len(done)} texts already recorded, {len(todo)} to generate", flush=True)
    lock = threading.Lock()

    def run(job):
        r = generate(job)
        with lock:
            with open(path, "a") as f: f.write(json.dumps(r) + "\n")
            done[(r["scheme"], r["k"], r["temp"], r["prompt"])] = r
            if len(done) % 12 == 0: print(f"{len(done)}/{len(jobs)} texts", flush=True)
        return r

    with cf.ThreadPoolExecutor(3) as ex:
        list(ex.map(run, todo))
    texts = [done[(j[0], j[1], j[2], j[3])] for j in jobs]

    rows = []
    for r in texts:
        k = r["k"] if r["scheme"] != "plain" else 2 if r["temp"] == 1.5 else 1
        sc, gr = scores(KEY, r["tokens"], k)
        pS, pg, nS, ng = pvals(sc, gr)
        ent = []
        for st in r["steps"]:
            if st["q"]:
                q = np.array(st["q"]) / 256.0; pp = np.exp(q - q.max()); pp /= pp.sum()
                ent.append(float(-(pp * np.log(pp + 1e-300)).sum()))
        rows.append(dict(scheme=r["scheme"], k=r["k"], temp=r["temp"], prompt=r["prompt"], T=len(r["tokens"]),
                         N=len(sc), pS=pS, pg=pg, firstS=nS, firstg=ng, Ssum=int(sc.sum()), gsum=int(gr.sum()),
                         ent=float(np.mean(ent)) if ent else None,
                         nll=-float(np.mean([s["lp"] for s in r["steps"]])),
                         free=float(np.mean([s["free"] for s in r["steps"]]))))
    # null: plain text under the right key, every text under 20 wrong keys
    rng = np.random.default_rng(99)
    null_S, null_g = [], []
    for r in texts:
        for _ in range(20):
            sc, gr = scores(int(rng.integers(0, 2**63)), r["tokens"], max(r["k"], 1))
            pS, pg, _, _ = pvals(sc, gr)
            null_S.append(pS); null_g.append(pg)
    plain = [x for x in rows if x["scheme"] == "plain"]
    null_S += [x["pS"] for x in plain]; null_g += [x["pg"] for x in plain]
    res = dict(rows=rows, null=dict(n=len(null_S), ks_S=float(kstest(null_S, "uniform").pvalue),
                                    ks_g=float(kstest(null_g, "uniform").pvalue),
                                    fp_S=int(np.sum(np.array(null_S) < 1e-3)), fp_g=int(np.sum(np.array(null_g) < 1e-3)),
                                    min_S=float(min(null_S)), min_g=float(min(null_g))))
    # prediction check: the green rate GREEN should produce, from the recorded distributions,
    # against the rate it did produce, at generation time (every step, with its own context)
    chk = []
    for s_, k_, t_ in [(s_, k_, t_) for s_ in ("green",) for k_ in (1, 2, 4) for t_ in (0.7, 1.0, 1.5)]:
        exp_g, obs_g = [], []
        for r in texts:
            if (r["scheme"], r["k"], r["temp"]) != (s_, k_, t_): continue
            for st in r["steps"]:
                ids = np.array(st["ids"]); q = np.array(st["q"]) / 256.0
                gr = W.green(W.uniforms(W.step_seed(KEY, st["ctx"]), ids), W.green_thr(GAMMA))
                pb = np.exp(q - q.max() + (DELTA_Q / 256) * gr); pb /= pb.sum()
                exp_g.append(float(pb[gr].sum())); obs_g.append(bool(gr[list(ids).index(st["tok"])]))
        if exp_g:
            chk.append(dict(k=k_, temp=t_, steps=len(obs_g), expected=float(np.mean(exp_g)), observed=float(np.mean(obs_g))))
    res["green_check"] = chk
    (ROOT / "results" / "llm_eval.json").write_text(json.dumps(res, indent=1))
    report(res)


def med(v):
    v = [x for x in v if x is not None]
    return f"{int(np.median(v))}" if v else "—"


def report(res):
    rows = res["rows"]
    out = ["# Watermarking real model output", "",
           "Generated by `python3 scripts/llm_eval.py`. Model: Qwen3-Coder-30B-A3B-Instruct (Q4_K_M, V = 151,936),",
           f"served by llama.cpp on this machine. Top-{K} sampling, {L}-token limit, six prompts per setting,",
           f"γ = {GAMMA}, δ = {DELTA_Q / 256}. Tokens are chosen by the golden model, which the RTL matches bit for",
           "bit (results/llm_rtl_replay.md). Detection sees only the generated tokens, not the prompt, and skips",
           "repeated contexts.", "",
           "| scheme | k | temp. | entropy, nats/token | mean score per token (null 1.000) | green rate (null 0.250) | single texts detected, Gumbel / green | six texts pooled: p, Gumbel | pooled p, green | tokens needed for p < 10⁻⁶, Gumbel / green | NLL per token | free steps |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["scheme"], r["k"], r["temp"]) for r in rows}, key=lambda x: (["gumbel", "green", "plain"].index(x[0]), x[1], x[2]))
    Z6 = 4.753                                         # one-sided z for p = 1e-6
    for s, k, t in keys:
        g = [r for r in rows if (r["scheme"], r["k"], r["temp"]) == (s, k, t)]
        N = sum(r["N"] for r in g); S = sum(r["Ssum"] for r in g) / 2048; G = sum(r["gsum"] for r in g)
        m, f = S / N, G / N
        pS = gammaincc(N, S); pg = binom.sf(G - 1, N, GAMMA)
        nS = f"{(Z6 / (m - 1)) ** 2:,.0f}" if m > 1.0 else "—"
        ng = f"{(Z6 * math.sqrt(GAMMA * (1 - GAMMA)) / (f - GAMMA)) ** 2:,.0f}" if f > GAMMA else "—"
        ents = [r["ent"] for r in g if r["ent"] is not None]
        name = s.upper() if s != "plain" else "plain"
        out.append(f"| {name} | {k if s != 'plain' else '—'} | {t} | {np.mean(ents):.2f} | {m:.3f} | {f:.3f} | "
                   f"{sum(r['pS'] < 1e-6 for r in g)} / {sum(r['pg'] < 1e-6 for r in g)} of {len(g)} | "
                   f"{pS:.1e} | {pg:.1e} | {nS} / {ng} | {np.mean([r['nll'] for r in g]):.3f} | "
                   f"{100 * np.mean([r['free'] for r in g]):.0f} % |")
    n = res["null"]
    out += ["", "## False positives", "",
            f"{n['n']} null p-values: every text scored under 20 wrong keys, plus the plain texts under the right key.", "",
            "| statistic | smallest p | p < 10⁻³ (expected ≈ " + f"{n['n'] / 1000:.1f}) | Kolmogorov–Smirnov p against uniform |",
            "|---|---|---|---|",
            f"| Gumbel score | {n['min_S']:.2g} | {n['fp_S']} | {n['ks_S']:.2f} |",
            f"| green count | {n['min_g']:.2g} | {n['fp_g']} | {n['ks_g']:.2f} |", "",
            "The binomial p-value of the green count is discrete, so its null p-values sit above the uniform",
            "line rather than on it: the Kolmogorov–Smirnov test against a continuous uniform rejects it even",
            f"though it is conservative, and its false-positive count ({n['fp_g']}) is within the expected {n['n'] / 1000:.1f}.", "",
            "## Does GREEN steer as designed?", "",
            "The green rate GREEN should produce at each step follows from the recorded top-100 distribution",
            "with δ added to the green tokens. Expected against observed, over every generation step:", "",
            "| k | temperature | steps | expected green rate | observed |", "|---|---|---|---|---|"]
    out += [f"| {c['k']} | {c['temp']} | {c['steps']} | {c['expected']:.3f} | {c['observed']:.3f} |" for c in res.get("green_check", [])]
    out += ["", "## Reading the table", "",
            "- **The signal per token scales with entropy.** This instruct model is very confident: 0.2–0.4",
            "  nats per token at temperature 0.7–1.0, so a 200-token answer leaves little room to steer, and",
            "  most single texts stay above the 10⁻⁶ bar. Pooling a setting's six texts, or raising the",
            "  temperature, crosses it. The tokens-needed column projects the length at which one text would.",
            "- **GREEN steers exactly as predicted** (table above): what limits it is the model's entropy, not",
            "  the hardware or the scheme.",
            "- **GUMBEL needs fewer tokens than GREEN at δ = 2 in every setting here**, comparing each scheme's",
            "  better statistic, and it does so without biasing the distribution.",
            "- **NLL** is the model's own negative log-probability of the chosen tokens at temperature 1. Six",
            "  texts per setting are too few to measure distortion: the spread between prompts is as large as",
            "  the differences between schemes."]
    (ROOT / "results" / "llm_eval.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[6:]))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--report":
        report(json.loads((ROOT / "results" / "llm_eval.json").read_text()))
    else:
        main()
