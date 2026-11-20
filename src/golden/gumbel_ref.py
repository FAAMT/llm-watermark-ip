"""Krish Mehta's Gumbel golden model, extracted verbatim from plans/plan_gumbel.md."""
import math, numpy as np
M32, M64 = (1 << 32) - 1, (1 << 64) - 1
C_IDX = 0x9E3779B1                       # odd: t -> t*C_IDX is a bijection mod 2^32
C_CTX = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D)

def fmix64(k):                           # MurmurHash3 64-bit finalizer (seed_unit)
    k ^= k >> 33; k = (k * 0xFF51AFD7ED558CCD) & M64
    k ^= k >> 33; k = (k * 0xC4CEB9FE1A85EC53) & M64
    return k ^ (k >> 33)

def fmix32(h):                           # MurmurHash3 32-bit finalizer on a uint64 array
    h = h ^ (h >> 16); h = (h * 0x85EBCA6B) & M32
    h = h ^ (h >> 13); h = (h * 0xC2B2AE35) & M32
    return h ^ (h >> 16)

def step_seed(key, ctx):                 # ctx = previous k token ids, oldest first
    x = key
    for j, t in enumerate(reversed(ctx)):
        x ^= ((t + 1) * C_CTX[j]) & M64
    return fmix64(x)

def free_seed(nonce, step):              # seed for a step whose context repeats
    return fmix64(nonce ^ (((step + 1) * C_CTX[3]) & M64))

def uniforms(seed, idx):                 # hash_lanes: one uint32 u per token index
    t = np.asarray(idx, dtype=np.uint64)
    return fmix32(((t * C_IDX) & M32) ^ np.uint64(seed & M32))

def pf_index(u):                         # rom_index: pseudo-float address of u
    u = np.asarray(u, dtype=np.uint64)
    half = u >> np.uint64(31)                                    # 1 if u >= 2^31
    w = np.where(half == 1, np.uint64(M32) - u, u)               # distance to nearer end, < 2^31
    e = (32 - np.frexp(w.astype(np.float64))[1]).astype(np.uint64)  # leading zeros (32 if w == 0)
    m = ((w << (e + np.uint64(1))) & np.uint64(M32)) >> np.uint64(24)  # 8 bits after leading one
    return (half * np.uint64(33) + e) * np.uint64(256) + m

def build_roms():                        # generated once; the C++ #includes the same tables
    G = np.zeros(2 * 33 * 256, np.int16)     # Gumbel noise, Q8.8
    S = np.zeros(2 * 33 * 256, np.uint16)    # detection score, Q5.11
    for half in (0, 1):
        for e in range(1, 33):
            for m in range(256):
                if e == 32:
                    if m: continue
                    lo = hi = 0
                elif e <= 23:
                    lo = (256 + m) << (23 - e); hi = lo + (1 << (23 - e)) - 1
                else:
                    if m & ((1 << (e - 23)) - 1): continue   # address never produced
                    lo = hi = (256 + m) >> (e - 23)
                x = ((lo + hi) / 2 + 0.5) / 2**32             # distance of u~ to nearer end
                neg_ln_u = -math.log1p(-x) if half else -math.log(x)       # -ln(u~)
                neg_ln_1mu = -math.log(x) if half else -math.log1p(-x)     # -ln(1-u~)
                a = (half * 33 + e) * 256 + m
                G[a] = round(256 * -math.log(neg_ln_u))
                S[a] = round(2048 * neg_ln_1mu)
    return G, S
G_ROM, S_ROM = build_roms()

def sample(key, ctx, logits_q88, free=False, nonce=0, step=0):    # SAMPLE command
    s = free_seed(nonce, step) if free else step_seed(key, ctx)
    v = logits_q88.astype(np.int32) + G_ROM[pf_index(uniforms(s, range(len(logits_q88))))]
    t = int(np.argmax(v))                # first maximum: smallest index wins ties
    return t, int(v[t])

def detect(key, tokens, skip, k):        # DETECT command: S in Q21.11, N tokens scored
    S = N = 0
    for i in range(k, len(tokens)):
        if skip[i]: continue
        u = uniforms(step_seed(key, tokens[i - k:i]), [tokens[i]])
        S = (S + int(S_ROM[pf_index(u)[0]])) & M32; N += 1
    return S, N

def p_value(S, N):                       # host: score ~ Gamma(N, 1) for unwatermarked text
    from scipy.special import gammaincc
    return gammaincc(N, S / 2048)

def repeat_flags(tokens, k):             # host: True where the k-token window was seen before
    seen, out = set(), []
    for i in range(len(tokens)):
        w = tuple(tokens[max(0, i - k):i])
        out.append(i >= k and w in seen); seen.add(w)
    return out

