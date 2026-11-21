"""Shared harness: encode commands into stream words, predict every response
with the golden model, run the Icarus testbench and compare word for word.

Word layouts are specified in docs/interface.md; this file is their
executable form."""
import math
import os
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "golden"))
import wm_ref as W  # noqa: E402

BUILD = ROOT / "build"
CMD_CONFIG, CMD_SAMPLE, CMD_DETECT, CMD_END = 0, 1, 2, 3
ERR = {"BAD_CMD": 1, "BAD_PARAM": 2, "BAD_SIZE": 3, "NO_CONFIG": 4, "TLAST_EARLY": 5, "TLAST_MISSING": 6}
M32, M64 = (1 << 32) - 1, (1 << 64) - 1


def hdr0(cmd, txid, p32, free=0):
    return (cmd & 0xFF) | ((free & 1) << 8) | ((txid & 0xFFFF) << 16) | ((p32 & M32) << 32)


class Activation:
    """One ap_start .. ap_done: the words to send and the responses and status
    the golden model says must come back."""

    def __init__(self):
        self.words = []          # (hdr, last, data)
        self.expect = []         # response words, in order
        self.cfg = None
        self.ncmd = 0
        self.status = None       # (halted, err, txid, ncmd), set by end() or error()
        self.meta = []           # per command: (kind, txid, nbeats) for timing

    # ---- commands
    def config(self, txid, V, key, nonce, gamma=0.25, delta=2.0, k=1, scheme=W.GUMBEL,
               theta=None, delta_q=None, valid=True):
        theta = W.green_thr(gamma) if theta is None else theta
        delta_q = int(round(delta * 256)) if delta_q is None else delta_q
        w3 = (theta & M32) | ((delta_q & 0xFFFF) << 32) | ((k & 7) << 48) | ((scheme & 3) << 56)
        self.words += [(1, 0, hdr0(CMD_CONFIG, txid, V)), (0, 0, key & M64), (0, 0, nonce & M64), (0, 1, w3)]
        if valid:
            self.cfg = dict(V=V, key=key, nonce=nonce, gamma=gamma, delta_q=delta_q, k=k, scheme=scheme,
                            theta=theta)
            self.ncmd += 1
        self.meta.append(("CONFIG", txid, 0))
        return self

    def sample(self, txid, logits_q88, ctx_oldest_first, step=0, free=False, predict=True, pad=-32768):
        c = self.cfg
        V = len(logits_q88)
        ctx = list(ctx_oldest_first)[-4:]
        newest = list(reversed(ctx)) + [0] * (4 - len(ctx))
        self.words += [(1, 0, hdr0(CMD_SAMPLE, txid, step, int(free))),
                       (0, 0, (newest[0] & 0xFFFFFF) | ((newest[1] & 0xFFFFFF) << 32)),
                       (0, 1, (newest[2] & 0xFFFFFF) | ((newest[3] & 0xFFFFFF) << 32))]
        q = np.asarray(logits_q88, dtype=np.int64)
        nb = (V + 3) // 4
        pad = np.full(nb * 4, pad, dtype=np.int64); pad[:V] = q    # slots past V: the IP must ignore them
        u16 = (pad & 0xFFFF).reshape(nb, 4)
        data = u16[:, 0] | (u16[:, 1] << 16) | (u16[:, 2] << 32) | (u16[:, 3] << 48)
        for i, d in enumerate(data):
            self.words.append((0, int(i == nb - 1), int(d)))
        if predict and c is not None:
            t, v = W.sample(c["key"], ctx[-c["k"]:] if c["k"] else [], np.asarray(logits_q88, np.int16),
                            c["scheme"], c["gamma"], c["delta_q"], free=free, nonce=c["nonce"], step=step,
                            theta=c["theta"])
            self.expect += [hdr0(CMD_SAMPLE, txid, 1), (t & 0xFFFFFF) | ((v & M32) << 32)]
            self.ncmd += 1
            self.last_sample = (t, v)
        self.meta.append(("SAMPLE", txid, nb))
        return self

    def detect(self, txid, tokens, skip=None, predict=True, T=None):
        c = self.cfg
        skip = [False] * len(tokens) if skip is None else list(skip)
        T = len(tokens) if T is None else T
        self.words.append((1, 1, hdr0(CMD_DETECT, txid, T)))
        nb = (len(tokens) + 1) // 2
        for b in range(nb):
            lo = (tokens[2 * b] & 0xFFFFFF) | (int(skip[2 * b]) << 31)
            hi = 0
            if 2 * b + 1 < len(tokens):
                hi = (tokens[2 * b + 1] & 0xFFFFFF) | (int(skip[2 * b + 1]) << 31)
            self.words.append((0, int(b == nb - 1), lo | (hi << 32)))
        if predict and c is not None:
            S, g, N = W.detect(c["key"], tokens, skip, c["k"], c["gamma"], theta=c["theta"])
            self.expect += [hdr0(CMD_DETECT, txid, 2), (S & M32) | ((N & M32) << 32), g & M32]
            self.ncmd += 1
            self.last_detect = (S, g, N)
        self.meta.append(("DETECT", txid, nb))
        return self

    def end(self, txid):
        self.words.append((1, 1, hdr0(CMD_END, txid, 0)))
        self.ncmd += 1
        self.status = (0, 0, txid, self.ncmd)
        self.meta.append(("END", txid, 0))
        return self

    def error(self, code, txid):
        """The activation must stop on this error; nothing after it is read."""
        self.status = (1, ERR[code], txid, self.ncmd)
        return self

    def raw(self, words):
        self.words += words
        return self


def compile_tb(force=False):
    BUILD.mkdir(exist_ok=True)
    out = BUILD / "tb_wm_top.vvp"
    srcs = sorted((ROOT / "src" / "rtl").glob("*.v")) + [ROOT / "src" / "tb" / "tb_wm_top.v"]
    if force or not out.exists() or any(s.stat().st_mtime > out.stat().st_mtime for s in srcs):
        subprocess.run(["iverilog", "-g2012", "-o", str(out), "-s", "tb_wm_top", *map(str, srcs)],
                       check=True, cwd=ROOT)
    return out


def run(acts, name, seed=1, bubble=0, stall=0, maxcyc=20_000_000):
    """Simulate the activations; return per-activation results and a pass flag."""
    vvp = compile_tb()
    fin, fout = BUILD / f"{name}.in.txt", BUILD / f"{name}.out.txt"
    with open(fin, "w") as f:
        for a, act in enumerate(acts):
            for hdr, last, data in act.words:
                f.write(f"{a} {hdr} {last} {data & M64:016x}\n")
    p = subprocess.run(["vvp", "-n", str(vvp), f"+in={fin}", f"+out={fout}", f"+seed={seed}",
                        f"+bubble={bubble}", f"+stall={stall}", f"+maxcyc={maxcyc}"], cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0 or "TIMEOUT" in p.stdout:
        raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
    got = {a: dict(resp=[], H=[], L=[], Z=[], status=None) for a in range(len(acts))}
    for line in open(fout):
        f = line.split()
        a = int(f[1])
        if f[0] == "R":
            got[a]["resp"].append(int(f[3], 16))
        elif f[0] == "H":
            got[a]["H"].append((int(f[2]), int(f[3], 16)))
        elif f[0] in ("L", "Z"):
            got[a][f[0]].append(int(f[2]))
        elif f[0] == "S":
            got[a]["status"] = tuple(int(x) for x in f[2:6])
            got[a]["cycles"] = int(f[6])
    rows, ok_all = [], True
    for a, act in enumerate(acts):
        g = got[a]
        resp_ok = g["resp"] == [w & M64 for w in act.expect]
        st_ok = g["status"] == act.status
        ok_all &= resp_ok and st_ok
        rows.append(dict(act=a, resp_ok=resp_ok, status_ok=st_ok, got_status=g["status"],
                         want_status=act.status, nresp=len(g["resp"]), H=g["H"], L=g["L"], Z=g["Z"],
                         cycles=g.get("cycles"), got=g["resp"], want=act.expect))
    return ok_all, rows


def rand_logits(rng, V, scale=3.0):
    return W.to_q88(rng.normal(0, scale, V))
