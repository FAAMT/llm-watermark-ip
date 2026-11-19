#!/usr/bin/env python3
"""Golden model for the LLM text-watermark IP.

Every function is the bit-exact reference for one block in plans/initial_plan.md;
the HLS testbench compares against this file.

    seed_unit     <-> step_seed
    hash_lanes    <-> green_mask      (a uint32 numpy array stands in for the P lanes)
    logit_biaser  <-> bias_logits
    detect_acc    <-> detect          (the count; the z-score stays on the host)

Running the file watermarks a synthetic token stream and detects it:

    python3 src/golden/wmark_ref.py [--V 50257] [--T 200] [--gamma 0.25] [--delta 2.0]

Synthetic logits stand in for the model until GPT-2 logits are recorded (milestone 1).
"""
import argparse
import math

import numpy as np

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1
C_IDX = 0x9E3779B1                                   # odd, so t -> t*C_IDX is a bijection mod 2^32
C_CTX = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9,     # one odd constant per context position
         0x94D049BB133111EB, 0x2545F4914F6CDD1D)


def fmix64(k):
    """MurmurHash3 64-bit finaliser (Appleby, public domain)."""
    k ^= k >> 33; k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33; k = (k * 0xC4CEB9FE1A85EC53) & M64
    return k ^ (k >> 33)


def fmix32_vec(h):
    """MurmurHash3 32-bit finaliser on a uint32 array: one lane per element.
    numpy uint32 arithmetic wraps mod 2^32 exactly like the hardware multiplier."""
    h = h ^ (h >> np.uint32(16)); h = h * np.uint32(0x85EBCA6B)
    h = h ^ (h >> np.uint32(13)); h = h * np.uint32(0xC2B2AE35)
    return h ^ (h >> np.uint32(16))


def step_seed(key, ctx):
    """seed_unit: fold the previous k token ids (oldest first) into the key."""
    x = key
    for j, t in enumerate(reversed(list(ctx))):        # j = 0 is the newest token
        x ^= ((int(t) + 1) * C_CTX[j]) & M64
    return fmix64(x)


def gamma_thr(gamma):
    """GAMMA_THR register: floor(gamma * 2^32), clamped to 32 bits."""
    return min(int(gamma * (1 << 32)), M32)


def green_mask(seed, V, gamma):
    """hash_lanes: V independent keyed hashes -> V green bits (bool array)."""
    thr = np.uint32(gamma_thr(gamma))
    s_lo = np.uint32(seed & M32)
    t = np.arange(V, dtype=np.uint32)
    return fmix32_vec((t * np.uint32(C_IDX)) ^ s_lo) < thr


def to_q88(logits):
    """Host side: subtract the max (softmax is shift-invariant), clamp, quantise to int16 Q8.8."""
    x = np.asarray(logits, dtype=np.float64)
    x = np.clip(x - x.max(), -128.0, 127.0)
    return np.round(x * 256.0).astype(np.int16)


def bias_logits(q, mask, delta_q88):
    """logit_biaser: saturating int16 add of delta on green indices."""
    out = q.astype(np.int32) + int(delta_q88) * mask.astype(np.int32)
    return np.clip(out, -32768, 32767).astype(np.int16)


def detect(key, tokens, V, gamma, k):
    """detect_acc: green count over positions k..T-1; then the host's z-score."""
    tokens = [int(t) for t in tokens]
    thr = gamma_thr(gamma)
    g = 0
    for i in range(k, len(tokens)):
        seed = step_seed(key, tokens[i - k:i])
        lane = np.array([(tokens[i] * C_IDX) & M32], dtype=np.uint32) ^ np.uint32(seed & M32)
        g += int(fmix32_vec(lane)[0] < thr)
    N = len(tokens) - k
    z = (g - gamma * N) / math.sqrt(N * gamma * (1 - gamma))
    return g, N, z


def _demo(V, T, gamma, delta, k, key, rng_seed):
    rng = np.random.default_rng(rng_seed)
    delta_q = int(round(delta * 256))

    def generate(watermark):
        toks = [int(rng.integers(V))]                  # one start token
        for _ in range(T):
            q = to_q88(rng.normal(0.0, 3.0, V))        # synthetic model: Gaussian logits
            if watermark:
                q = bias_logits(q, green_mask(step_seed(key, toks[-k:]), V, gamma), delta_q)
            p = np.exp(q.astype(np.float64) / 256.0)
            toks.append(int(rng.choice(V, p=p / p.sum())))
        return toks

    print(f"V={V} T={T} gamma={gamma} delta={delta} k={k}")
    for wm in (False, True):
        g, N, z = detect(key, generate(wm), V, gamma, k)
        print(f"  watermark={'on ' if wm else 'off'}  green {g:4d}/{N}  z = {z:+6.2f}")

    # the detector agrees with the mask path on every position
    toks = generate(True)
    for i in range(k, len(toks)):
        assert green_mask(step_seed(key, toks[i - k:i]), V, gamma)[toks[i]] == \
            (fmix32_vec(np.array([(toks[i] * C_IDX) & M32], dtype=np.uint32)
                        ^ np.uint32(step_seed(key, toks[i - k:i]) & M32))[0] < gamma_thr(gamma))
    print("  detect path == mask path on every token: ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--V", type=int, default=50257)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--key", type=lambda s: int(s, 0), default=0x0123456789ABCDEF)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    _demo(a.V, a.T, a.gamma, a.delta, a.k, a.key, a.seed)
