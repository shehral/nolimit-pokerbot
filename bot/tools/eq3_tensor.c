// Three-way preflop all-in tensors for the 169 starting-hand classes (Monte Carlo, with card removal), used by
// bot/tools/solve_icm_pushfold.py for the over-call branch. The evaluator and class enumeration below match the
// heads-up equity tool behind bot_tables.EQ_TABLE, which is not in this repo (same class order as bot_tables.EQ_LABELS).
//   cc -O2 -o eq3_tensor bot/tools/eq3_tensor.c && ./eq3_tensor OUTDIR TRIALS_PER_TRIPLE
// Card c in 0..51: rank c>>2 (0 = deuce .. 12 = ace), suit c&3.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

static uint64_t rs = 0x9E3779B97F4A7C15ULL;
static inline uint64_t rng(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static inline int rnd(int n) { return (int)((rng() >> 11) % (uint64_t)n); }

static inline int straight_high(uint32_t m) {
  for (int r = 12; r >= 4; r--) if (((m >> (r - 4)) & 31u) == 31u) return r;
  if ((m & 0x100Fu) == 0x100Fu) return 3;
  return -1;
}
static inline uint32_t topk(uint32_t m, int k) { // pack top k ranks of mask into nibbles
  uint32_t out = 0; int n = 0;
  for (int r = 12; r >= 0 && n < k; r--) if (m >> r & 1u) { out = (out << 4) | (uint32_t)r; n++; }
  while (n < k) { out <<= 4; n++; }
  return out;
}
static uint32_t eval7(const int *c, int n) {
  uint32_t sm[4] = {0, 0, 0, 0}; int rc[13]; memset(rc, 0, sizeof rc);
  for (int i = 0; i < n; i++) { sm[c[i] & 3] |= 1u << (c[i] >> 2); rc[c[i] >> 2]++; }
  uint32_t all = sm[0] | sm[1] | sm[2] | sm[3];
  int fs = -1;
  for (int s = 0; s < 4; s++) if (__builtin_popcount(sm[s]) >= 5) fs = s;
  if (fs >= 0) { int h = straight_high(sm[fs]); if (h >= 0) return (8u << 20) | (uint32_t)h; }
  int q = -1, t1 = -1, t2 = -1, p1 = -1, p2 = -1;
  for (int r = 12; r >= 0; r--) {
    if (rc[r] == 4) { if (q < 0) q = r; }
    else if (rc[r] == 3) { if (t1 < 0) t1 = r; else if (t2 < 0) t2 = r; }
    else if (rc[r] == 2) { if (p1 < 0) p1 = r; else if (p2 < 0) p2 = r; }
  }
  if (q >= 0) return (7u << 20) | ((uint32_t)q << 16) | (topk(all & ~(1u << q), 1) << 12);
  if (t1 >= 0 && (t2 >= 0 || p1 >= 0)) { int pr = t2 > p1 ? t2 : p1; return (6u << 20) | ((uint32_t)t1 << 16) | ((uint32_t)pr << 12); }
  if (fs >= 0) return (5u << 20) | topk(sm[fs], 5);
  int st = straight_high(all);
  if (st >= 0) return (4u << 20) | (uint32_t)st;
  if (t1 >= 0) return (3u << 20) | ((uint32_t)t1 << 16) | (topk(all & ~(1u << t1), 2) << 8);
  if (p2 >= 0) return (2u << 20) | ((uint32_t)p1 << 16) | ((uint32_t)p2 << 12) | (topk(all & ~(1u << p1) & ~(1u << p2), 1) << 8);
  if (p1 >= 0) return (1u << 20) | ((uint32_t)p1 << 16) | (topk(all & ~(1u << p1), 3) << 4);
  return topk(all, 5);
}

// class index: pairs 0..12 (rank), suited / offsuit by (hi, lo)
static int ncls = 0; static char lab[169][4]; static int chi[169], clo[169], csu[169]; // csu: 0 pair, 1 suited, 2 offsuit
static int combos[169][12][2]; static int ncomb[169];
static const char *RK = "23456789TJQKA";
static void build_classes(void) {
  for (int hi = 12; hi >= 0; hi--) for (int lo = hi; lo >= 0; lo--) {
    for (int t = 0; t < (hi == lo ? 1 : 2); t++) {
      int k = ncls++; chi[k] = hi; clo[k] = lo; csu[k] = hi == lo ? 0 : (t == 0 ? 1 : 2);
      lab[k][0] = RK[hi]; lab[k][1] = RK[lo]; lab[k][2] = csu[k] == 0 ? 0 : (csu[k] == 1 ? 's' : 'o'); lab[k][3] = 0;
      int n = 0;
      for (int s1 = 0; s1 < 4; s1++) for (int s2 = 0; s2 < 4; s2++) {
        if (csu[k] == 0) { if (s2 <= s1) continue; }
        else if (csu[k] == 1) { if (s1 != s2) continue; }
        else { if (s1 == s2) continue; }
        combos[k][n][0] = hi * 4 + s1; combos[k][n][1] = lo * 4 + s2; n++;
      }
      ncomb[k] = n;
    }
  }
}

// Full 3-way tensors for the 169 classes (Monte Carlo, N trials per unordered triple, with card removal).
// Writes OUTDIR/t3.f32: float32[6][169][169][169], T[o][x][a][b] = P(finish order o) with o over role orders
//   0 = x>a>b, 1 = x>b>a, 2 = a>x>b, 3 = a>b>x, 4 = b>x>a, 5 = b>a>x (ties split evenly over consistent orders)
// and OUTDIR/w3.f32: float32[169][169][169], W3[x][a][b] = compatible (a, b) combo pairs per x combo.
static float *T3, *W3;
#define IDX(x, a, b) (((size_t)(x) * 169 + (a)) * 169 + (b))
int main(int argc, char **argv) {
  const char *od = argv[1]; int N = atoi(argv[2]);
  build_classes();
  size_t K3 = (size_t)169 * 169 * 169;
  T3 = (float *)calloc(6 * K3, sizeof(float)); W3 = (float *)calloc(K3, sizeof(float));
  int perm[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
  for (int p = 0; p < 169; p++) for (int q = p; q < 169; q++) for (int r = q; r < 169; r++) {
    int cls[3] = {p, q, r};
    // exact compatibility count
    long cnt = 0;
    for (int i = 0; i < ncomb[p]; i++) for (int j = 0; j < ncomb[q]; j++) {
      int *x = combos[p][i], *y = combos[q][j];
      if (x[0] == y[0] || x[0] == y[1] || x[1] == y[0] || x[1] == y[1]) continue;
      for (int k = 0; k < ncomb[r]; k++) {
        int *z = combos[r][k];
        if (z[0] == x[0] || z[0] == x[1] || z[0] == y[0] || z[0] == y[1] ||
            z[1] == x[0] || z[1] == x[1] || z[1] == y[0] || z[1] == y[1]) continue;
        cnt++;
      }
    }
    double P[6] = {1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6, 1.0/6};   // P[o] over orders of positions (0,1,2) = (p,q,r)
    if (cnt > 0) {
      for (int o = 0; o < 6; o++) P[o] = 0;
      int done = 0;
      while (done < N) {
        int *x = combos[p][rnd(ncomb[p])], *y = combos[q][rnd(ncomb[q])], *z = combos[r][rnd(ncomb[r])];
        uint64_t used = 0; int bad = 0; int cs[6] = {x[0], x[1], y[0], y[1], z[0], z[1]};
        for (int i = 0; i < 6; i++) { if (used >> cs[i] & 1ULL) bad = 1; used |= 1ULL << cs[i]; }
        if (bad) continue;
        int h[3][7]; for (int i = 0; i < 3; i++) { h[i][0] = cs[2*i]; h[i][1] = cs[2*i+1]; }
        int nb = 0;
        while (nb < 5) { int cc = rnd(52); if (used >> cc & 1ULL) continue; used |= 1ULL << cc; for (int i = 0; i < 3; i++) h[i][2+nb] = cc; nb++; }
        uint32_t s[3]; for (int i = 0; i < 3; i++) s[i] = eval7(h[i], 7);
        int ok[6]; int nok = 0;
        for (int o = 0; o < 6; o++) { ok[o] = s[perm[o][0]] >= s[perm[o][1]] && s[perm[o][1]] >= s[perm[o][2]]; nok += ok[o]; }
        for (int o = 0; o < 6; o++) if (ok[o]) P[o] += 1.0 / nok;
        done++;
      }
      for (int o = 0; o < 6; o++) P[o] /= N;
    }
    // fill every role assignment: roles (x, a, b) = positions (rx, ra, rb), a permutation of (0, 1, 2)
    for (int ra = 0; ra < 6; ra++) {
      int px = perm[ra][0], pa = perm[ra][1], pb = perm[ra][2];
      int cx = cls[px], ca = cls[pa], cb = cls[pb];
      size_t id = IDX(cx, ca, cb);
      W3[id] = (float)((double)cnt / ncomb[cx]);
      // role order o (over roles x,a,b) -> position order: first = role perm[o][0] -> position
      int rolepos[3] = {px, pa, pb};
      for (int o = 0; o < 6; o++) {
        int f = rolepos[perm[o][0]], s2 = rolepos[perm[o][1]], t = rolepos[perm[o][2]];
        // find position-order index with (f, s2, t)
        for (int po = 0; po < 6; po++) if (perm[po][0] == f && perm[po][1] == s2 && perm[po][2] == t) {
          T3[(size_t)o * K3 + id] = (float)P[po]; break; }
      }
    }
  }
  char path[512]; FILE *fo;
  snprintf(path, sizeof path, "%s/t3.f32", od); fo = fopen(path, "wb"); fwrite(T3, sizeof(float), 6 * K3, fo); fclose(fo);
  snprintf(path, sizeof path, "%s/w3.f32", od); fo = fopen(path, "wb"); fwrite(W3, sizeof(float), K3, fo); fclose(fo);
  return 0;
}
