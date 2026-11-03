"""Jam/fold equilibrium solver, standard library port of an earlier research script (not published) (numpy removed,
class count and deck size generic). Same model: players act in order (UTG .. BTN, SB, BB); the first player who does
not fold must jam; after a jam each later player calls or folds; once one calls the rest fold. Strategies are vectors
over hand classes; card removal between hero and each single opponent is exact at the class level (W); removal
between two opponents is ignored; fictitious play.

Inputs (from build_tables' matrix stage): E[i][j] equity of class i vs class j, W[i][j] average number of class-j hands
compatible with one class-i hand, combos[i], total = number of dealt hands, Q = hands compatible with one hand
(C(N - n, n); 1225 for NLHE).

experiments(): the sng_icm runs behind the bot's fits (exp_hu.py, exp_multi.py, fit_rules.py): heads-up Nash by
stack -> push/call orders and the SB jam fit; 8-handed equal-stack Nash -> the k >= 2 jam fit and the call fit.
"""
import math
import time
from operator import mul


class Game:
    def __init__(self, E, W, combos, total, Q):
        self.C = len(combos)
        self.E = E
        self.W = W
        self.WE = [[w * e for w, e in zip(rw, re)] for rw, re in zip(W, E)]
        self.combos = list(combos)
        self.total = float(total)
        self.Q = float(Q)

    def pct(self, r):
        return sum(map(mul, r, self.combos)) / self.total

    @staticmethod
    def mv(M, v):
        return [sum(map(mul, row, v)) for row in M]


def icm_dp(stacks, payouts):
    n = len(stacks)
    live = [i for i in range(n) if stacks[i] > 1e-9]
    m = min(len(payouts), len(live))
    eq = [0.0] * n
    if m == 0:
        return eq
    total = float(sum(stacks[i] for i in live))
    cur = {0: (1.0, total)}
    for place in range(m):
        pay = payouts[place]
        last = place + 1 == m
        nxt = {}
        for mask, (p, rem) in cur.items():
            inv = p / rem
            for j in live:
                bit = 1 << j
                if mask & bit:
                    continue
                pj = inv * stacks[j]
                eq[j] += pj * pay
                if not last:
                    nm = mask | bit
                    v = nxt.get(nm)
                    nxt[nm] = (pj, rem - stacks[j]) if v is None else (v[0] + pj, v[1])
        cur = nxt
    return eq


class Spot:
    """A preflop jam/fold spot. stacks in big blinds, in preflop action order (..., SB, BB)."""

    def __init__(self, stacks, sb, bb, ante=0.0, payouts=None):
        self.s = [float(x) for x in stacks]
        self.n = len(stacks)
        self.sb, self.bb, self.ante = sb, bb, ante
        self.payouts = payouts
        n = self.n
        post = [min(self.s[i], ante) for i in range(n)]
        if n >= 2:
            post[n - 2] += min(self.s[n - 2] - post[n - 2], sb) if n > 2 else 0.0
            post[n - 1] += min(self.s[n - 1] - post[n - 1], bb)
            if n == 2:
                post[0] += min(self.s[0] - post[0], sb)
        self.post = post
        self._ucache = {}

    def util(self, final):
        key = tuple(round(x, 6) for x in final)
        u = self._ucache.get(key)
        if u is None:
            if self.payouts is None:
                u = list(final)
            else:
                u = icm_dp(final, self.payouts)
                k = sum(1 for x in final if x > 1e-9)
                busted = sorted((i for i in range(self.n) if self.s[i] > 0 and final[i] <= 1e-9),
                                key=lambda i: self.s[i])
                place = k + len(busted)
                for i in busted:
                    u[i] = self.payouts[place - 1] if place - 1 < len(self.payouts) else 0.0
                    place -= 1
            self._ucache[key] = u
        return u

    def outcome_steal(self, j):
        f = [self.s[i] - self.post[i] for i in range(self.n)]
        f[j] += sum(self.post)
        return self.util(f)

    def outcome_walk(self):
        return self.outcome_steal(self.n - 1)

    def outcome_allin(self, j, m, j_wins):
        n = self.n
        amt = min(self.s[j], self.s[m])
        f = [self.s[i] - self.post[i] for i in range(n)]
        dead = sum(self.post[i] for i in range(n) if i != j and i != m)
        f[j] = self.s[j] - amt
        f[m] = self.s[m] - amt
        f[j if j_wins else m] += 2 * amt + dead
        return self.util(f)


def solve(g, spot, iters=1500, deadline=None, info=None):
    """Fictitious play, same schedule as pushfold.solve. Returns (P, C): P[j] jam range of first-in seat j, C[j][m]
    call range of seat m facing a jam by seat j (fractions of each class's hands). Stops early at `deadline`
    (time.time() value) and records the iterations done in info["iters"]."""
    n = spot.n
    Cn = g.C
    W, WE, Q, cmb, mv = g.W, g.WE, g.Q, g.combos, g.mv
    P = [[0.3] * Cn for _ in range(n)]
    Cr = [[[0.15] * Cn for _ in range(n)] for _ in range(n)]
    U_steal = [spot.outcome_steal(j) for j in range(n)]
    U_walk = spot.outcome_walk()
    U_win = [[None] * n for _ in range(n)]
    U_lose = [[None] * n for _ in range(n)]
    for j in range(n):
        for m in range(j + 1, n):
            U_win[j][m] = spot.outcome_allin(j, m, True)
            U_lose[j][m] = spot.outcome_allin(j, m, False)
    done = 0
    for t in range(1, iters + 1):
        if deadline is not None and t > 10 and time.time() > deadline:
            break
        done = t
        lr = 1.0 / (t + 1) if t > 50 else 0.2
        cp = [g.pct(P[j]) for j in range(n)]
        qu = [[0.0] * n for _ in range(n)]
        rvr = [[0.5] * n for _ in range(n)]
        Wc = [[None] * n for _ in range(n)]
        WEc = [[None] * n for _ in range(n)]
        for j in range(n - 1):
            pc = [a * b for a, b in zip(P[j], cmb)]
            for m in range(j + 1, n):
                c = Cr[j][m]
                Wc[j][m] = wc = mv(W, c)
                WEc[j][m] = wec = mv(WE, c)
                qu[j][m] = g.pct(c)
                if qu[j][m] > 0 and cp[j] > 0:
                    den = sum(map(mul, pc, wc))
                    rvr[j][m] = sum(map(mul, pc, wec)) / den if den > 0 else 0.5

        def after_jam(j, m0):
            val = [0.0] * n
            reach = 1.0
            for m in range(m0, n):
                q = qu[j][m]
                if q > 0:
                    e = rvr[j][m]
                    uw, ul = U_win[j][m], U_lose[j][m]
                    f = reach * q
                    val = [v + f * (e * a + (1 - e) * b) for v, a, b in zip(val, uw, ul)]
                reach *= (1 - q)
            us = U_steal[j]
            return [v + reach * a for v, a in zip(val, us)]

        cont = [None] * (n + 1)
        cont[n] = U_walk
        cont[n - 1] = U_walk
        for j in range(n - 2, -1, -1):
            aj = after_jam(j, j + 1)
            cont[j] = [cp[j] * a + (1 - cp[j]) * b for a, b in zip(aj, cont[j + 1])]
        newP = []
        for j in range(n - 1):
            ev = [0.0] * Cn
            reach = [1.0] * Cn
            for m in range(j + 1, n):
                q = [x / Q for x in Wc[j][m]]
                winw = [x / Q for x in WEc[j][m]]
                uw, ul = U_win[j][m][j], U_lose[j][m][j]
                ev = [e + r * (ww * uw + (qq - ww) * ul) for e, r, ww, qq in zip(ev, reach, winw, q)]
                reach = [r * (1 - qq) for r, qq in zip(reach, q)]
            us = U_steal[j][j]
            evf = cont[j + 1][j]
            newP.append([1.0 if e + r * us > evf else 0.0 for e, r in zip(ev, reach)])
        newP.append([0.0] * Cn)
        newC = [[None] * n for _ in range(n)]
        for j in range(n - 1):
            p = P[j]
            den = mv(W, p)
            num = mv(WE, p)
            eqm = [a / b if b > 0 else 0.5 for a, b in zip(num, den)]
            for m in range(j + 1, n):
                ul, uw = U_lose[j][m][m], U_win[j][m][m]
                val = after_jam(j, m + 1)[m]
                newC[j][m] = [1.0 if e * ul + (1 - e) * uw > val else 0.0 for e in eqm]
        a = 1 - lr
        for j in range(n):
            P[j] = [a * x + lr * y for x, y in zip(P[j], newP[j])]
            for m in range(j + 1, n):
                Cr[j][m] = [a * x + lr * y for x, y in zip(Cr[j][m], newC[j][m])]
    if info is not None:
        info["iters"] = done
    return P, Cr


# ===================================================================== experiments and fits
HU_STACKS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25)
MULTI_STACKS = (5, 8, 10, 12, 15, 20)


def hu_task(args):
    """(E, W, combos, total, Q, S, iters, ante) -> dict for one heads-up stack."""
    E, W, combos, total, Q, S, iters, ante = args[:8]
    deadline = args[8] if len(args) > 8 else None
    g = Game(E, W, combos, total, Q)
    info = {}
    P, C = solve(g, Spot([S, S], 0.5, 1.0, ante=ante), iters, deadline, info)
    return {"S": S, "sb_jam_pct": g.pct(P[0]), "bb_call_pct": g.pct(C[0][1]), "P": P[0], "C": C[0][1],
            "iters": info.get("iters")}


def multi_task(args):
    """(E, W, combos, total, Q, n, S, iters, ante) -> dict for one n-handed equal-stack spot."""
    E, W, combos, total, Q, n, S, iters, ante = args[:9]
    deadline = args[9] if len(args) > 9 else None
    g = Game(E, W, combos, total, Q)
    info = {}
    P, C = solve(g, Spot([S] * n, 0.5, 1.0, ante=ante), iters, deadline, info)
    return {"n": n, "S": S, "jam": [g.pct(P[j]) for j in range(n - 1)],
            "bb_call": [g.pct(C[j][n - 1]) for j in range(n - 1)], "iters": info.get("iters")}


def lstsq(X, y):
    """Least squares by normal equations (tiny systems)."""
    k = len(X[0])
    A = [[sum(r[a] * r[b] for r in X) for b in range(k)] for a in range(k)]
    v = [sum(r[a] * yy for r, yy in zip(X, y)) for a in range(k)]
    for i in range(k):                       # Gaussian elimination with partial pivoting
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[p] = A[p], A[i]
        v[i], v[p] = v[p], v[i]
        for r in range(i + 1, k):
            f = A[r][i] / A[i][i]
            for c in range(i, k):
                A[r][c] -= f * A[i][c]
            v[r] -= f * v[i]
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        x[i] = (v[i] - sum(A[i][c] * x[c] for c in range(i + 1, k))) / A[i][i]
    return x


def fit(hu, multi8):
    """The fit_rules.py fits. hu: list of hu_task results; multi8: list of multi_task results with n = 8.
    Returns (JAM_FIT, CALL_FIT, details) in the bot's PARAMS format: jam% = a S^b k^c (k >= 2), a1 S^b1 (k = 1);
    call% = a J^b S^c."""
    rows = []
    for d in multi8:
        n = d["n"]
        for idx, (j, c) in enumerate(zip(d["jam"], d["bb_call"])):
            rows.append((float(d["S"]), n - 1 - idx, j, c))
    eps = 1e-6
    sel = [r for r in rows if r[1] >= 2 and r[2] > eps]
    X = [(1.0, math.log(r[0]), math.log(r[1])) for r in sel]
    a, b, c = lstsq(X, [math.log(r[2]) for r in sel])
    hs = [d for d in hu if d["S"] >= 4 and d["sb_jam_pct"] > eps]
    a1, b1 = lstsq([(1.0, math.log(d["S"])) for d in hs], [math.log(d["sb_jam_pct"]) for d in hs])
    selc = [r for r in rows if r[2] > eps and r[3] > eps]
    cc = lstsq([(1.0, math.log(r[2]), math.log(r[0])) for r in selc], [math.log(r[3]) for r in selc])
    jam_fit = (round(math.exp(a), 4), round(b, 4), round(c, 4), round(math.exp(a1), 4), round(b1, 4))
    call_fit = (round(math.exp(cc[0]), 4), round(cc[1], 4), round(cc[2], 4))
    err = [abs(min(1.0, math.exp(a) * r[0] ** b * r[1] ** c) - r[2]) for r in sel]
    return jam_fit, call_fit, {"jam_k2_log": (a, b, c), "jam_sb_log": (a1, b1), "call_log": tuple(cc),
                               "jam_fit_max_abs_err": max(err) if err else None, "rows": len(rows)}


def orders(hu, eq_random):
    """Push and call orders (class indices, best first) from heads-up Nash by stack: largest stack where the class
    is jammed (called) at least half the time, ties by equity vs random (exp_hu.py)."""
    C = len(eq_random)
    push_max = [0.0] * C
    call_max = [0.0] * C
    for d in sorted(hu, key=lambda d: d["S"]):
        for i in range(C):
            if d["P"][i] >= 0.5:
                push_max[i] = max(push_max[i], d["S"])
            if d["C"][i] >= 0.5:
                call_max[i] = max(call_max[i], d["S"])
    po = sorted(range(C), key=lambda i: (-push_max[i], -eq_random[i], i))
    co = sorted(range(C), key=lambda i: (-call_max[i], -eq_random[i], i))
    return po, co, push_max, call_max
