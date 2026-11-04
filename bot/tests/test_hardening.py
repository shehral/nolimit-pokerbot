"""Hardening pass (2026-11-03): the faster ICM code is bit-identical to the pre-change code, and the per-hand cache
of a seat's current ICM equity gives exactly the bubble factors of the uncached computation.

The reference functions below are the pre-change sources verbatim (bot/out/brain_pre_hardening.py keeps the whole
pre-change brain). Exact float equality is required, not a tolerance: NLHE decisions must not move.
"""
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import brain as B        # noqa: E402


def ref_icm_hero(stacks, i, pays):
    live = [s for s in stacks if s > 0]
    k, si = len(live), stacks[i]
    if si <= 0 or k == 0:
        return 0.0
    p = list(pays[:k]) + [0.0] * max(0, k - len(pays))
    if k == 1:
        return p[0]
    step, base = p[k - 2] - p[k - 1], p[k - 1]
    acc = 0.0
    for sj in live:
        acc += si / (si + sj)
    lin = base + step * (acc - 0.5)
    c0 = p[0] - (base + step * (k - 1))
    c1 = p[1] - (base + step * (k - 2)) if k > 1 else 0.0
    c2 = p[2] - (base + step * (k - 3)) if k > 2 else 0.0
    if not (c0 or c1 or c2):
        return lin
    T = float(sum(live))
    e2 = e3 = 0.0
    n = len(stacks)
    for j in range(n):
        sj = stacks[j]
        if j == i or sj <= 0:
            continue
        Tj = T - sj
        pj = sj / T
        e2 += pj * si / Tj
        if c2:
            for m in range(n):
                sm = stacks[m]
                if m != i and m != j and sm > 0:
                    e3 += pj * (sm / Tj) * si / (Tj - sm)
    return lin + c0 * (si / T) + c1 * e2 + c2 * e3


def ref_make_icm_exact(pays, max_players=10):
    C = [[1]]
    for n in range(1, max_players + 1):
        C.append([1] + [C[n - 1][t - 1] + C[n - 1][t] for t in range(1, n)] + [1])
    coef = []
    for m in range(max_players):
        p = list(pays[:m + 1]) + [0.0] * max(0, m + 1 - len(pays))
        row = []
        for b in range(m + 1):
            acc, sign = 0.0, 1.0
            for t in range(b + 1):
                acc += sign * C[b][t] * p[m - b + t]
                sign = -sign
            row.append(acc)
        coef.append(row)

    def icm(stacks, i):
        si = stacks[i]
        if si <= 0:
            return 0.0
        sums, sizes = [0.0], [0]
        for j, s in enumerate(stacks):
            if j != i and s > 0:
                sums += [x + s for x in sums]
                sizes += [b + 1 for b in sizes]
        row = coef[len(sizes).bit_length() - 1]
        ev = 0.0
        for x, b in zip(sums, sizes):
            ev += row[b] * si / (si + x)
        return ev
    return icm


LINEAR = tuple(x * 100.0 / 36.0 for x in (8, 7, 6, 5, 4, 3, 2, 1))     # the kit's: tail coefficients ~1e-15, not 0
CLOSED = [(50, 30, 20), LINEAR, (65, 35), (100,), (25, 20, 15, 12, 10, 8, 6, 4), (40, 30, 20, 10)]
EXACT = [(40, 30, 20, 10), (35, 25, 18, 12, 6, 4), (25, 20, 15, 12, 10, 8, 6, 4), (50, 30, 20)]


def _spots(rng, count):
    for _ in range(count):
        n = rng.choice((2, 3, 4, 5, 6, 7, 8, 9, 10))
        st = [rng.choice((0, rng.randint(1, 20000), rng.randint(1, 300))) for _ in range(n)]
        if rng.random() < 0.1:
            st = [float(x) + rng.random() for x in st]
        yield st


def test_icm_hero_bit_identical():
    rng = random.Random(11)
    for st in _spots(rng, 6000):
        pays = B.pad_payouts(rng.choice(CLOSED), max(8, len(st)))
        for i in range(len(st)):
            assert B.icm_hero(st, i, pays) == ref_icm_hero(st, i, pays), (st, i, pays)


def test_exact_icm_bit_identical():
    rng = random.Random(12)
    for pays in EXACT:
        p = B.pad_payouts(pays, 10)
        new, old = B.make_icm_exact(p, 10), ref_make_icm_exact(p, 10)
        for st in _spots(rng, 1500):
            for i in range(len(st)):
                assert new(st, i) == old(st, i), (st, i, pays)


def test_cached_now_gives_identical_bubble_factors():
    rng = random.Random(13)
    for pays in CLOSED + EXACT:
        p = B.pad_payouts(pays, 10)
        icm = B.make_icm(p)
        for st in _spots(rng, 800):
            n = len(st)
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            full = B.bubble_factor(icm, st, i, j, p)
            now = icm(st, i) if st[i] > 0 and min(st[i], st[j]) > 0 else None
            assert B.bubble_factor(icm, st, i, j, p, now=now) == full


def test_brain_bf_uses_cache_and_matches_module_function():
    br = B.Brain(0, {"n_seats": 8, "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "seed": 1}, version=2,
                 params={"GUARD_NS": 10 ** 15})
    st = [2600, 2100, 1900, 1700, 1500, 1300, 1100, 900]
    br.view_start, br.n_live, br.t0 = st, 8, 0
    br.bf_cache, br.icm_now, br.bf_budget = {}, {}, 10
    vals = [br.bf(0, j) for j in range(1, 8)]
    assert list(br.icm_now) == [0]                      # one ICM "now" per (hand, seat), reused for every opponent
    assert vals == [B.bubble_factor(br.icm, st, 0, j, br.pays, br.P["BF_MIN"], br.P["BF_MAX"], br.P["BF_NO_GAIN"])
                    for j in range(1, 8)]


def test_guard_default_bounds_bf_starts():
    """The default leaves room for one BF plus the decision's tail under 200 us: start limit = GUARD - 3 x cost."""
    br = B.Brain(0, {"n_seats": 8, "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "seed": 1}, version=2)
    assert br.P["GUARD_NS"] == 200_000
    assert br.bf_start_limit == br.P["GUARD_NS"] - br.P["BF_COST_SAFETY"] * br.bf_cost_ns
    assert br.bf_start_limit < 200_000
