/* Software baseline: the same SAMPLE and DETECT as the IP, bit-exact with
 * src/golden/wm_ref.py, as a straightforward scalar C loop. Used only to time
 * a CPU doing the IP's job (scripts/baseline.py builds it with gcc -O3).
 *
 *   wm_baseline <gumbel_rom.hex> <score_rom.hex> <V> <reps> <seed>
 * prints:  sample <token> <value> <ns per SAMPLE>
 *          detect <S> <g> <N> <ns per token>
 *   wm_baseline <gumbel_rom.hex> <score_rom.hex> check <logits.txt> <tokens.txt>
 * runs one SAMPLE (context [7], k = 1) and one DETECT (k = 2, gamma 0.25) on the
 * given inputs, for the bit-exactness check against the golden model.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int16_t G_ROM[16896];
static uint16_t S_ROM[16896];
static const uint64_t C_CTX[4] = {0x9E3779B97F4A7C15ull, 0xBF58476D1CE4E5B9ull,
                                  0x94D049BB133111EBull, 0x2545F4914F6CDD1Dull};

static uint64_t fmix64(uint64_t k) {
    k ^= k >> 33; k *= 0xFF51AFD7ED558CCDull;
    k ^= k >> 33; k *= 0xC4CEB9FE1A85EC53ull;
    return k ^ (k >> 33);
}
static uint32_t fmix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}
static uint32_t pf_index(uint32_t u) {
    uint32_t half = u >> 31, w = half ? ~u : u;
    uint32_t e = w ? (uint32_t)__builtin_clz(w) : 32;
    uint32_t m = e >= 32 ? 0 : (uint32_t)(((uint64_t)w << (e + 1)) & 0xFFFFFFFFull) >> 24;
    return (half * 33 + e) * 256 + m;
}
static uint64_t step_seed(uint64_t key, const uint32_t *ctx_newest, int k) {
    uint64_t x = key;
    for (int j = 0; j < k; j++) x ^= ((uint64_t)ctx_newest[j] + 1) * C_CTX[j];
    return fmix64(x);
}
static void load(const char *path, void *rom, int sixteen_signed) {
    FILE *f = fopen(path, "r"); unsigned v; int i = 0;
    if (!f) { perror(path); exit(1); }
    while (i < 16896 && fscanf(f, "%x", &v) == 1) {
        if (sixteen_signed) ((int16_t *)rom)[i++] = (int16_t)v; else ((uint16_t *)rom)[i++] = (uint16_t)v;
    }
    fclose(f);
}
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

/* GUMBEL sample: argmax_t (l_t + G(u_t)), first maximum */
static uint32_t sample(uint64_t key, const uint32_t *ctx, int k, const int16_t *lg, uint32_t V, int32_t *val) {
    uint32_t s = (uint32_t)step_seed(key, ctx, k), best = 0;
    int32_t bv = INT32_MIN;
    for (uint32_t t = 0; t < V; t++) {
        int32_t v = lg[t] + G_ROM[pf_index(fmix32((t * 0x9E3779B1u) ^ s))];
        if (v > bv) { bv = v; best = t; }
    }
    *val = bv;
    return best;
}

static uint32_t *read_ints(const char *path, uint32_t *n) {
    FILE *f = fopen(path, "r"); long v; uint32_t cap = 1024, i = 0;
    uint32_t *a = malloc(cap * sizeof(uint32_t));
    if (!f) { perror(path); exit(1); }
    while (fscanf(f, "%ld", &v) == 1) {
        if (i == cap) a = realloc(a, (cap *= 2) * sizeof(uint32_t));
        a[i++] = (uint32_t)v;
    }
    fclose(f); *n = i; return a;
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    load(argv[1], G_ROM, 1); load(argv[2], S_ROM, 0);
    if (argv[3][0] == 'c') {                       /* check mode */
        uint32_t V, Tn; uint32_t *raw = read_ints(argv[4], &V), *tk = read_ints(argv[5], &Tn);
        int16_t *l = malloc(V * sizeof(int16_t));
        for (uint32_t t = 0; t < V; t++) l[t] = (int16_t)(int32_t)raw[t];
        uint32_t c[4] = {7, 0, 0, 0}; int32_t v; uint32_t t = sample(0x0123456789ABCDEFull, c, 1, l, V, &v);
        uint32_t S = 0, g = 0, N = 0;
        for (uint32_t i = 2; i < Tn; i++) {
            uint32_t cc[2] = {tk[i - 1], tk[i - 2]};
            uint32_t u = fmix32((tk[i] * 0x9E3779B1u) ^ (uint32_t)step_seed(0x0123456789ABCDEFull, cc, 2));
            S += S_ROM[pf_index(u)]; g += (u >= 0xC0000000u); N++;
        }
        printf("%u %d %u %u %u\n", t, v, S, g, N);
        return 0;
    }
    uint32_t V = (uint32_t)atoi(argv[3]); int reps = atoi(argv[4]); srand((unsigned)atoi(argv[5]));
    int16_t *lg = malloc(V * sizeof(int16_t));
    for (uint32_t t = 0; t < V; t++) lg[t] = (int16_t)(-(rand() % 2048));
    uint32_t ctx[4] = {1, 2, 3, 4}, tok = 0; int32_t val = 0;
    const uint64_t key = 0x0123456789ABCDEFull;
    double t0 = now_ns();
    for (int r = 0; r < reps; r++) { ctx[0] = r; tok = sample(key, ctx, 1, lg, V, &val); }
    double ns = (now_ns() - t0) / reps;
    ctx[0] = 0; tok = sample(key, ctx, 1, lg, V, &val);
    printf("sample %u %d %.0f\n", tok, val, ns);

    /* DETECT over a 1,000,000-token pseudo-random text, k = 2 */
    const uint32_t Tn = 1000000; uint32_t *toks = malloc(Tn * sizeof(uint32_t));
    for (uint32_t i = 0; i < Tn; i++) toks[i] = (uint32_t)(rand() % V);
    uint32_t S = 0, g = 0, N = 0, thr = 0xC0000000u;
    t0 = now_ns();
    for (uint32_t i = 2; i < Tn; i++) {
        uint32_t c[2] = {toks[i - 1], toks[i - 2]};
        uint32_t u = fmix32((toks[i] * 0x9E3779B1u) ^ (uint32_t)step_seed(key, c, 2));
        S += S_ROM[pf_index(u)]; g += (u >= thr); N++;
    }
    ns = (now_ns() - t0) / N;
    printf("detect %u %u %u %.2f\n", S, g, N, ns);
    return 0;
}
