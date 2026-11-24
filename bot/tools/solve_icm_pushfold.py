"""ICM push/fold equilibria for 2, 3 and 4 players left (no antes), by damped fictitious play over the 169 classes.

    .venv/bin/python bot/tools/solve_icm_pushfold.py --cache DIR [--workers 4] [--payouts TOP3,LINEAR,WTA]
        [--n 2,3,4] [--iters 400] [--tol 0.002] [--out bot/out/icm_nash] [--emit bot/icm_nash_tables.py]
    .venv/bin/python bot/tools/solve_icm_pushfold.py --cache DIR --point 4 TOP3 10,10,10,10 --verbose

THE GAME (the standard push/fold model, with one over-call). n players, stacks in big blinds, ordered by position
(SB, BB, BTN, CO; heads-up the SB is the button). Blinds 0.5 / 1, no antes. Preflop order: CO, BTN, SB, BB (3-handed
BTN, SB, BB; heads-up SB, BB). Each player, when everybody before has folded, jams or folds (BB wins the SB's blind
if all fold). After a jam each later player calls or folds; after one call each later player may over-call or fold;
after two callers the rest fold (no four-way all-ins; they are rarer still). Payoff = exact Malmuth-Harville ICM
(bot/brain.py make_icm, the bot's own function) of the stacks after the hand, for the announced payout over the
places still to be decided; players busted in the same hand are ranked by their starting stacks.

EQUITIES. Heads-up: the 169 x 169 class-vs-class all-in equity and card-removal weights (the matrices behind
bot_tables.EQ_TABLE), read from sim/assets/sng_icm/out/eq/. That folder is not in this repo, so this solver cannot run
from this copy; it is kept to document how bot/icm_nash_tables.py was made. Three-way: a Monte Carlo tensor of
finish-order probabilities for every class triple (bot/tools/eq3_tensor.c, 500 trials per unordered triple, exact
class-level removal weights), contracted against the ranges, so the over-call branch is exact up to that noise (side
pots resolved exactly per finish order). Removal between the hero's class and each opponent's range is exact at the
class level; removal between two opponents is ignored except inside the three-way tensor. Earlier folds do not
condition the later players' hands (the usual push/fold simplification).

SOLVER. Simultaneous fictitious play: at each iteration every decision node (first-in jam, call vs each pusher,
over-call vs each pusher and caller) gets its best response against the others' AVERAGE strategies, then
avg += (BR - avg) * DAMP / (t + DAMP) (DAMP = 2). Each player acts at most once per hand, so a player's best-response
gain is the sum over its nodes of reach x sum_h P(h) x (max EV - EV under its average): the per-player gains sum to
the NashConv of the average profile, reported in payout POINTS (the prize pool is 100) per hand. The emitted pure
sets are {class : average share >= 0.5}; the exploitability of that purified profile is reported too.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot.brain import make_icm, pad_payouts, LABELS, PUSH_PCT   # noqa: E402

EQDIR = os.path.join(ROOT, "sim", "assets", "sng_icm", "out", "eq")
C_SRC = os.path.join(ROOT, "bot", "tools", "eq3_tensor.c")
GRID = (3, 5, 7, 10, 13, 16, 20)
POS = {2: ("SB", "BB"), 3: ("SB", "BB", "BTN"), 4: ("SB", "BB", "BTN", "CO")}
ACT = {2: (0, 1), 3: (2, 0, 1), 4: (3, 2, 0, 1)}          # preflop action order in vector positions
PAYOUTS = {"TOP3": (50.0, 30.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
           "LINEAR": tuple(x * 100.0 / 36.0 for x in (8, 7, 6, 5, 4, 3, 2, 1)),
           "WTA": (100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
PERM = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))   # role finish orders (eq3_tensor.c)
K = 169
SUPP_EPS = 2e-3                                           # range entries below this are dropped in 3-way sums


def nodes_of(n):
    """Canonical decision-node list for n players: ("push", p), ("call", i, j), ("over", i, j, k) in positions."""
    act = ACT[n]
    out = [("push", p) for p in act[:-1]]
    for a, i in enumerate(act[:-1]):
        for j in act[a + 1:]:
            out.append(("call", i, j))
    for a, i in enumerate(act[:-1]):
        for b in range(a + 1, n):
            for k in act[b + 1:]:
                out.append(("over", i, act[b], k))
    return out


# ----------------------------------------------------------------------------- data
class Eq:
    """Equity data shared by every solve (heads-up matrices + the 3-way tensor, memory-mapped)."""

    def __init__(self, cache):
        comb = []
        for line in open(os.path.join(EQDIR, "classes.txt")):
            _, lab, c = line.split()
            comb.append(int(c))
        labs = [line.split()[1] for line in open(os.path.join(EQDIR, "classes.txt"))]
        assert labs == list(LABELS), "class order of the heads-up equity matrices differs from bot_tables"
        self.comb = np.array(comb, dtype=float)
        self.E = np.loadtxt(os.path.join(EQDIR, "eq_hu.txt"))
        self.W = np.loadtxt(os.path.join(EQDIR, "w_hu.txt"))
        self.WE = self.W * self.E
        self.T7 = np.load(ensure_tensor(cache), mmap_mode="r").view(np.ndarray)   # [b][7][x][a]: 6 orders x W3, then W3


def ensure_tensor(cache, trials=500):
    """Build (once) the [b][o][x][a] tensor: planes 0-5 = P(order o) x W3, plane 6 = W3 (float32, 135 MB)."""
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, "t7_bxoa.npy")
    if os.path.exists(path):
        return path
    t3p, w3p = os.path.join(cache, "t3.f32"), os.path.join(cache, "w3.f32")
    if not (os.path.exists(t3p) and os.path.exists(w3p)):
        exe = os.path.join(cache, "eq3_tensor")
        subprocess.check_call(["cc", "-O2", "-o", exe, C_SRC])
        subprocess.check_call([exe, cache, str(trials)])
    t3 = np.fromfile(t3p, dtype=np.float32).reshape(6, K, K, K)      # [o][x][a][b]
    w3 = np.fromfile(w3p, dtype=np.float32).reshape(K, K, K)         # [x][a][b]
    T7 = np.empty((K, 7, K, K), dtype=np.float32)
    for o in range(6):
        T7[:, o] = (t3[o] * w3).transpose(2, 0, 1)
    T7[:, 6] = w3.transpose(2, 0, 1)
    tmp = path + ".tmp.npy"
    np.save(tmp, T7)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------------- outcomes
def resolve(s, post, involved, order):
    """Final stacks after a hand: `involved` all-in (each puts in min(own, biggest other involved)), others lose
    their posted blinds; pots resolved layer by layer, each layer to the best-ranked eligible player in `order`."""
    n = len(s)
    c = list(post)
    if len(involved) == 1:
        i = involved[0]
        fin = [s[p] - post[p] for p in range(n)]
        fin[i] = s[i] + sum(post[p] for p in range(n) if p != i)
        return fin
    for p in involved:
        c[p] = min(s[p], max(s[q] for q in involved if q != p))
    dead = sum(c[p] for p in range(n) if p not in involved)
    win = [0.0] * n
    levels = sorted(set(c[p] for p in involved))
    prev = 0.0
    for L in levels:
        amt = sum(min(c[p], L) - min(c[p], prev) for p in involved)
        if prev == 0.0:
            amt += dead
        for p in order:
            if c[p] >= L:
                win[p] += amt
                break
        prev = L
    return [s[p] - c[p] + win[p] for p in range(n)]


def utility(icm, pays, start, fin):
    """ICM equity vector (payout points) of final stacks; busted players get the places after the live ones, the
    bigger starting stack the better place (ties share)."""
    n = len(fin)
    fin = [x if x > 1e-9 else 0.0 for x in fin]
    live = [p for p in range(n) if fin[p] > 0]
    u = [0.0] * n
    for p in live:
        u[p] = icm(fin, p)
    bust = sorted((p for p in range(n) if fin[p] <= 0), key=lambda p: -start[p])
    place = len(live)
    q = 0
    while q < len(bust):
        r = q
        while r + 1 < len(bust) and start[bust[r + 1]] == start[bust[q]]:
            r += 1
        share = sum(pays[place + t] for t in range(r - q + 1)) / (r - q + 1)
        for t in range(q, r + 1):
            u[bust[t]] = share
        place += r - q + 1
        q = r + 1
    return np.array(u)


class Outcomes:
    def __init__(self, n, stacks, pays):
        self.n = n
        s = [float(x) for x in stacks]
        post = [0.0] * n
        post[0] = min(0.5, s[0])
        post[1] = min(1.0, s[1])
        pays = pad_payouts(pays, max(8, len(pays)))
        icm = make_icm(pays)
        U = lambda fin: utility(icm, pays, s, fin)   # noqa: E731
        self.walk = U(resolve(s, post, [1], [1]))
        self.steal = {i: U(resolve(s, post, [i], [i])) for i in range(n)}
        self.hu = {}
        for i, j in itertools.permutations(range(n), 2):
            if i < j:
                wi = U(resolve(s, post, [i, j], [i, j]))
                wj = U(resolve(s, post, [i, j], [j, i]))
                self.hu[(i, j)] = (wi, wj)
                self.hu[(j, i)] = (wj, wi)
        self.three = {}
        for tri in itertools.combinations(range(n), 3):
            for order in itertools.permutations(tri):
                self.three[order] = U(resolve(s, post, list(tri), list(order)))

    def hu_win(self, i, j):
        """(utility vector if i beats j, if j beats i)."""
        return self.hu[(i, j)]

    def three_roles(self, x, a, b):
        """(6, n) utilities by ROLE finish order (PERM) for roles (x, a, b) = these players."""
        pl = (x, a, b)
        return np.stack([self.three[tuple(pl[r] for r in PERM[o])] for o in range(6)])


# ----------------------------------------------------------------------------- the solver
class Solver:
    def __init__(self, eq, n, stacks, pays):
        self.eq = eq
        self.n = n
        self.stacks = tuple(stacks)
        self.out = Outcomes(n, stacks, pays)
        self.act = ACT[n]
        self.nodes = nodes_of(n)
        self.after = {p: list(self.act[self.act.index(p) + 1:]) for p in self.act}
        pc = PUSH_PCT_ARR
        self.avg = {}
        for nd in self.nodes:
            if nd[0] == "push":
                self.avg[nd] = (pc <= 0.30).astype(float)
            elif nd[0] == "call":
                self.avg[nd] = (pc <= 0.12).astype(float)
            else:
                self.avg[nd] = (pc <= 0.025).astype(float)
        self.t = 0
        self.hist = []

    # ---- range helpers
    def P(self, r):
        return float(self.eq.comb @ r) / 1326.0

    def pc_h(self, r):
        return (self.eq.W @ r) / 1225.0

    def eq_h(self, r):
        den = self.eq.W @ r
        num = self.eq.WE @ r
        return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.5)

    def eq_rr(self, r1, r2):
        w1 = r1 * self.eq.comb
        den = w1 @ self.eq.W @ r2
        return float(w1 @ self.eq.WE @ r2 / den) if den > 1e-12 else 0.5

    def three(self, A, B):
        """Hero-level 3-way finish-order probabilities, roles (x = hero class, a ~ A, b ~ B): (6, 169) with rows
        summing to 1 where defined; sum_{a,b} A[a] B[b] W3[x,a,b] T[o][x,a,b] / sum A[a] B[b] W3[x,a,b]."""
        T7 = self.eq.T7
        Aa = np.where(A > SUPP_EPS, A, 0.0).astype(np.float32)
        acc = np.zeros((7, K))
        for b in np.nonzero(B > SUPP_EPS)[0]:
            acc += float(B[b]) * (T7[b] @ Aa)          # contiguous (7, 169, 169) slab: no gather copy
        den = acc[6]
        ok = den > 1e-12
        out = np.full((6, K), 1.0 / 6.0)
        out[:, ok] = acc[:6, ok] / den[ok]
        return out, den

    # ---- range-level continuation values (vectors over players)
    def V_over(self, i, j, L, cache):
        v = np.zeros(self.n)
        q = 1.0
        for k in L:
            o = self.avg[("over", i, j, k)]
            po = self.P(o)
            if po > 0.0:
                v += q * po * cache["U3range"][(i, j, k)]
            q *= 1.0 - po
        e = cache["e_rr"][(i, j)]
        wi, wj = self.out.hu_win(i, j)
        return v + q * (e * wi + (1.0 - e) * wj)

    def V_resp(self, i, L, cache):
        v = np.zeros(self.n)
        q = 1.0
        for idx, j in enumerate(L):
            pc = self.P(self.avg[("call", i, j)])
            if pc > 0.0:
                v += q * pc * self.V_over(i, j, L[idx + 1:], cache)
            q *= 1.0 - pc
        return v + q * self.out.steal[i]

    def V_first(self, idx, cache):
        if idx >= self.n - 1:
            return self.out.walk
        p = self.act[idx]
        pp = self.P(self.avg[("push", p)])
        return pp * self.V_resp(p, self.after[p], cache) + (1.0 - pp) * self.V_first(idx + 1, cache)

    # ---- one iteration: EVs of both actions at every node (vectors over the hero's class)
    def evaluate(self, avg=None):
        if avg is not None:
            saved, self.avg = self.avg, avg
        try:
            return self._evaluate()
        finally:
            if avg is not None:
                self.avg = saved

    def _evaluate(self):
        n, act, out = self.n, self.act, self.out
        cache = {"e_rr": {}, "U3range": {}}
        three_h = {}
        for nd in self.nodes:
            if nd[0] == "call":
                _, i, j = nd
                cache["e_rr"][(i, j)] = self.eq_rr(self.avg[("push", i)], self.avg[nd])
        # over nodes: the over-caller's own hero-level 3-way table (roles k, i, j) also gives the range-level orders
        for nd in self.nodes:
            if nd[0] != "over":
                continue
            _, i, j, k = nd
            o_avg = self.avg[nd]
            pr, den = self.three(self.avg[("push", i)], self.avg[("call", i, j)])
            three_h[("k", i, j, k)] = pr
            w = o_avg * self.eq.comb * den
            if w.sum() > 1e-12:
                prob_roles = (pr * w).sum(1) / w.sum()          # roles (k, i, j)
            else:
                prob_roles = pr.mean(1)
            cache["U3range"][(i, j, k)] = prob_roles @ out.three_roles(k, i, j)
        ev = {}
        for idx, p in enumerate(act[:-1]):
            # first-in jam by p
            nd = ("push", p)
            L = self.after[p]
            fold = self.V_first(idx + 1, cache)[p]
            val = np.zeros(K)
            q = np.ones(K)
            for a, j in enumerate(L):
                c = self.avg[("call", p, j)]
                pcj = self.pc_h(c)
                e = self.eq_h(c)
                wi, wj = out.hu_win(p, j)
                vj = np.zeros(K)
                qq = np.ones(K)
                for k in L[a + 1:]:
                    o = self.avg[("over", p, j, k)]
                    if o.max() > SUPP_EPS:
                        pok = self.pc_h(o)
                        pr, _ = self.three(c, o)                   # roles (p, j, k)
                        u3 = out.three_roles(p, j, k)[:, p]
                        vj += qq * pok * (u3 @ pr)
                        qq = qq * (1.0 - pok)
                vj += qq * (e * wi[p] + (1.0 - e) * wj[p])
                val += q * pcj * vj
                q = q * (1.0 - pcj)
            val += q * out.steal[p][p]
            ev[nd] = (val, np.full(K, fold))
        for nd in self.nodes:
            if nd[0] == "call":
                _, i, j = nd
                L = self.after[i]
                Lk = L[L.index(j) + 1:]
                push = self.avg[("push", i)]
                fold = self.V_resp(i, Lk, cache)[j]
                e = self.eq_h(push)
                wi, wj = out.hu_win(i, j)
                val = np.zeros(K)
                qq = np.ones(K)
                for k in Lk:
                    o = self.avg[("over", i, j, k)]
                    if o.max() > SUPP_EPS:
                        pok = self.pc_h(o)
                        pr, _ = self.three(push, o)                # roles (j, i, k)
                        u3 = out.three_roles(j, i, k)[:, j]
                        val += qq * pok * (u3 @ pr)
                        qq = qq * (1.0 - pok)
                val += qq * (e * wj[j] + (1.0 - e) * wi[j])
                ev[nd] = (val, np.full(K, fold))
            elif nd[0] == "over":
                _, i, j, k = nd
                L = self.after[i]
                Lk = L[L.index(k) + 1:]
                fold = self.V_over(i, j, Lk, cache)[k]
                pr = three_h[("k", i, j, k)]
                u3 = out.three_roles(k, i, j)[:, k]
                ev[nd] = (u3 @ pr, np.full(K, fold))
        return ev

    def reach(self):
        """Range-level probability that each node is reached (no card removal)."""
        r = {}
        f = 1.0
        for p in self.act[:-1]:
            r[("push", p)] = f
            f *= 1.0 - self.P(self.avg[("push", p)])
        for nd in self.nodes:
            if nd[0] == "call":
                _, i, j = nd
                L = self.after[i]
                x = r[("push", i)] * self.P(self.avg[("push", i)])
                for jj in L[:L.index(j)]:
                    x *= 1.0 - self.P(self.avg[("call", i, jj)])
                r[nd] = x
        for nd in self.nodes:
            if nd[0] == "over":
                _, i, j, k = nd
                L = self.after[i]
                x = r[("call", i, j)] * self.P(self.avg[("call", i, j)])
                for kk in L[L.index(j) + 1:L.index(k)]:
                    x *= 1.0 - self.P(self.avg[("over", i, j, kk)])
                r[nd] = x
        return r

    def gains(self, ev, strat):
        """Per-node best-response gain (points) of each node's owner against `strat` (reach under self.avg)."""
        r = self.reach()
        wh = self.eq.comb / 1326.0
        g = {}
        for nd, (a, b) in ev.items():
            s = strat[nd]
            g[nd] = r[nd] * float(wh @ (np.maximum(a, b) - (s * a + (1.0 - s) * b)))
        return g

    def owner(self, nd):
        return nd[1] if nd[0] == "push" else nd[2] if nd[0] == "call" else nd[3]

    def nashconv(self, g):
        per = [0.0] * self.n
        for nd, x in g.items():
            per[self.owner(nd)] += x
        return sum(per), max(per)

    def solve(self, iters=400, tol=0.002, damp=2.0, check=10, min_iters=40, t_start=1):
        """t_start > 1 continues a warm start (the averages then move with the smaller steps of iteration t_start)."""
        t0 = time.process_time()
        nc = None
        for t in range(t_start, iters + 1):
            ev = self.evaluate()
            if t % check == 0 or t == iters:
                g = self.gains(ev, self.avg)
                nc, mx = self.nashconv(g)
                self.hist.append((t, nc, mx))
                if t - t_start + 1 >= min_iters and nc < tol:
                    break
            step = damp / (t + damp)
            for nd, (a, b) in ev.items():
                br = (a > b).astype(float)
                self.avg[nd] += step * (br - self.avg[nd])
        self.t = t
        pure = {nd: (v >= 0.5).astype(float) for nd, v in self.avg.items()}
        # purified profile: exploitability of the sets the bot plays
        evp = self.evaluate(pure)
        saved = self.avg
        self.avg = pure
        gp = self.gains(evp, pure)
        self.avg = saved
        ncp, mxp = self.nashconv(gp)
        return {"iters": t, "nashconv": nc, "nashconv_pure": ncp, "max_player_pure": mxp,
                "cpu_s": time.process_time() - t0, "pure": pure, "avg": self.avg}


PUSH_PCT_ARR = np.array(PUSH_PCT)
WARM_T = 20             # a warm-started solve begins at this iteration's step size (2 / 22)


def mask_of(v):
    m = 0
    for c in np.nonzero(v >= 0.5)[0]:
        m |= 1 << int(c)
    return m


def pct_of(v, comb):
    return float((v >= 0.5) @ comb) / 1326.0


# ----------------------------------------------------------------------------- grid driver
_EQ = None


def _init(cache):
    global _EQ
    _EQ = Eq(cache)


def _solve_chunk(task):
    n, kind, pays, points, iters, tol = task
    rows = []
    prev = None
    for st in points:
        s = Solver(_EQ, n, st, pays)
        t_start = 1
        if prev is not None:                       # warm start from the neighbouring point's average
            for nd in s.nodes:
                s.avg[nd] = prev[nd].copy()
            t_start = WARM_T
        r = s.solve(iters=iters, tol=tol, t_start=t_start)
        prev = r["avg"]
        rows.append({"n": n, "kind": kind, "stacks": list(st), "iters": r["iters"], "nashconv": r["nashconv"],
                     "nashconv_pure": r["nashconv_pure"], "max_player_pure": r["max_player_pure"],
                     "cpu_s": r["cpu_s"],
                     "masks": [mask_of(r["pure"][nd]) for nd in s.nodes],
                     "pcts": [pct_of(r["pure"][nd], _EQ.comb) for nd in s.nodes]})
    return rows


def grid_points(n):
    return list(itertools.product(GRID, repeat=n))


def run_grid(cache, ns, kinds, workers, iters, tol, out_dir, log):
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(cache,)) as pool:
        for n in ns:
            for kind in kinds:
                path = os.path.join(out_dir, f"solve_n{n}_{kind}.json")
                if os.path.exists(path):
                    with open(path) as f:
                        res[(n, kind)] = json.load(f)
                    log(f"loaded {path}")
                    continue
                pts = grid_points(n)
                # chunks of consecutive points (last coordinate varies fastest): warm starts stay local
                step = len(GRID) if n > 2 else len(pts)
                tasks = [(n, kind, PAYOUTS[kind], pts[a:a + step], iters, tol) for a in range(0, len(pts), step)]
                t0 = time.time()
                rows = []
                for r in pool.imap(_solve_chunk, tasks):
                    rows.extend(r)
                nc = [r["nashconv"] for r in rows]
                ncp = [r["nashconv_pure"] for r in rows]
                log(f"n={n} {kind}: {len(rows)} points in {time.time() - t0:.0f} s wall, cpu "
                    f"{sum(r['cpu_s'] for r in rows):.0f} s; iters median {sorted(r['iters'] for r in rows)[len(rows) // 2]}; "
                    f"NashConv avg max {max(nc):.4f} median {sorted(nc)[len(nc) // 2]:.4f}; pure max {max(ncp):.4f} "
                    f"median {sorted(ncp)[len(ncp) // 2]:.4f} pts; load {os.getloadavg()[0]:.1f}")
                doc = {"n": n, "kind": kind, "grid": GRID, "nodes": [list(x) for x in nodes_of(n)], "rows": rows}
                with open(path + ".tmp", "w") as f:
                    json.dump(doc, f)
                os.replace(path + ".tmp", path)
                res[(n, kind)] = doc
    return res


# ----------------------------------------------------------------------------- emit the data module
def emit(res, path, kinds):
    """bot/icm_nash_tables.py: hex-packed 169-bit masks (class index = bit, bot_tables.EQ_LABELS order) in one
    palette, and per (n, kind) a hex string of 4-digit palette indices, row = grid point (base-7 bucket number,
    SB first), column = node (NODES[n])."""
    palette = {}
    pal = []
    rows_hex = {}
    for (n, kind), doc in sorted(res.items()):
        if kind not in kinds:
            continue
        nodes = nodes_of(n)
        assert [tuple(x) for x in doc["nodes"]] == nodes
        by_pt = {tuple(r["stacks"]): r for r in doc["rows"]}
        parts = []
        for st in grid_points(n):
            r = by_pt[st]
            for m in r["masks"]:
                idx = palette.get(m)
                if idx is None:
                    idx = palette[m] = len(pal)
                    pal.append(m)
                parts.append(f"{idx:04x}")
        rows_hex[(n, kind)] = "".join(parts)
    assert len(pal) < 65536
    lines = [
        '"""ICM push/fold equilibrium sets (GENERATED by bot/tools/solve_icm_pushfold.py; do not edit by hand).',
        "",
        "Pure data plus a nearest-bucket lookup. Stacks are in big blinds at the start of the hand, by position",
        "(SB, BB, BTN, CO); heads-up SB = button. Each stack is mapped to the nearest GRID value (no interpolation):",
        "the bucket error is at most half a grid gap (1 bb up to 7 bb, 1.5 bb up to 16 bb, 2 bb up to 20 bb) inside",
        "the grid; stacks above 20 bb use the 20 bb bucket and stacks below 3 bb the 3 bb bucket.",
        "A set is a 169-bit mask over the hand classes in bot_tables.EQ_LABELS order (bit c = class c in the set).",
        "Nodes: (\"push\", p) first-in jam of position p; (\"call\", i, j) j calls i's jam with nobody in yet;",
        "(\"over\", i, j, k) k over-calls after i jammed and j called. Positions 0 SB, 1 BB, 2 BTN, 3 CO.",
        "See ICM_NASH notes for the method, convergence and checks.",
        '"""',
        f"GRID = {GRID!r}",
        f"KINDS = {tuple(sorted(set(k for (_, k) in rows_hex)))!r}",
        f"PAYOUTS = {{{', '.join(f'{k!r}: {PAYOUTS[k]!r}' for k in sorted(set(k for (_, k) in rows_hex)))}}}",
        "NODES = {",
    ]
    for n in sorted(set(n for (n, _) in rows_hex)):
        lines.append(f"    {n}: {tuple(nodes_of(n))!r},")
    lines.append("}")
    lines.append("NODE_INDEX = {n: {nd: i for i, nd in enumerate(v)} for n, v in NODES.items()}")
    lines.append("_EDGES = tuple((GRID[i] + GRID[i + 1]) / 2.0 for i in range(len(GRID) - 1))")
    lines.append("")
    lines.append("")
    lines.append("def bucket(x_bb):")
    lines.append('    """Index of the nearest GRID stack (ties go up); outside the grid: the end buckets."""')
    lines.append("    i = 0")
    lines.append("    for e in _EDGES:")
    lines.append("        if x_bb < e:")
    lines.append("            return i")
    lines.append("        i += 1")
    lines.append("    return i")
    lines.append("")
    lines.append("")
    lines.append("def row_index(stacks_bb):")
    lines.append('    """Grid point number of a stack vector (SB first), base len(GRID), last position fastest."""')
    lines.append("    r = 0")
    lines.append("    for x in stacks_bb:")
    lines.append("        r = r * len(GRID) + bucket(x)")
    lines.append("    return r")
    lines.append("")
    lines.append("")
    lines.append("def lookup(n, kind, stacks_bb, node):")
    lines.append('    """The 169-bit mask of `node` at the nearest grid point, or None (no table for n / kind / node)."""')
    lines.append("    rows = ROWS.get((n, kind))")
    lines.append("    ni = NODE_INDEX.get(n, {}).get(node)")
    lines.append("    if rows is None or ni is None:")
    lines.append("        return None")
    lines.append("    p = (row_index(stacks_bb) * len(NODES[n]) + ni) * 4")
    lines.append("    q = int(rows[p:p + 4], 16) * 43")
    lines.append("    return int(MASKS[q:q + 43], 16)")
    lines.append("")
    lines.append("")
    lines.append("MASKS = (")
    blob = "".join(f"{m:043x}" for m in pal)
    for a in range(0, len(blob), 100):
        lines.append(f"    {blob[a:a + 100]!r}")
    lines.append(")")
    lines.append("ROWS = {")
    for key, h in sorted(rows_hex.items()):
        lines.append(f"    {key!r}: (")
        for a in range(0, len(h), 100):
            lines.append(f"        {h[a:a + 100]!r}")
        lines.append("    ),")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return len(pal), len(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True, help="directory for the 3-way tensor (135 MB, built once)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--payouts", default="TOP3,LINEAR,WTA")
    ap.add_argument("--n", default="2,3,4")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--tol", type=float, default=0.002, help="stop when NashConv (points) is below this")
    ap.add_argument("--out", default=os.path.join(ROOT, "bot", "out", "icm_nash"))
    ap.add_argument("--emit", default=os.path.join(ROOT, "bot", "icm_nash_tables.py"))
    ap.add_argument("--emit-kinds", default="TOP3,LINEAR,WTA")
    ap.add_argument("--point", nargs=3, metavar=("N", "KIND", "STACKS"))
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    def log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} (load {os.getloadavg()[0]:.1f}) {msg}"
        print(line, flush=True)

    if a.point:
        n, kind, st = int(a.point[0]), a.point[1], tuple(float(x) for x in a.point[2].split(","))
        eq = Eq(a.cache)
        s = Solver(eq, n, st, PAYOUTS[kind])
        r = s.solve(iters=a.iters, tol=a.tol)
        log(f"n={n} {kind} {st}: iters {r['iters']} NashConv {r['nashconv']:.5f} pure {r['nashconv_pure']:.5f} "
            f"pts, cpu {r['cpu_s']:.2f} s")
        if a.verbose:
            for t, nc, mx in s.hist:
                print(f"   iter {t:4d} NashConv {nc:.5f} max player {mx:.5f}")
        for nd in s.nodes:
            v = r["pure"][nd]
            labs = [LABELS[c] for c in np.argsort(PUSH_PCT_ARR) if v[c] >= 0.5]
            print(f"  {str(nd):<22} {pct_of(v, eq.comb) * 100:5.1f}%  {' '.join(labs[:40])}{' ...' if len(labs) > 40 else ''}")
        return
    ns = [int(x) for x in a.n.split(",")]
    kinds = a.payouts.split(",")
    log(f"solve n={ns} payouts={kinds} workers={a.workers} iters<={a.iters} tol={a.tol}")
    ensure_tensor(a.cache)
    res = run_grid(a.cache, ns, kinds, a.workers, a.iters, a.tol, a.out, log)
    if a.emit:
        npal, size = emit(res, a.emit, a.emit_kinds.split(","))
        log(f"emitted {a.emit}: {npal} distinct sets, {size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
