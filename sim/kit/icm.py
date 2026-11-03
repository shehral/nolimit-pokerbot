"""Exact ICM (Malmuth-Harville) for the mock starter kit: the "ICM calculator" a starter kit ships.

Model: the chance a live player finishes first is its share of the chips; the remaining places follow the same rule
among the remaining players (Harville). A player's equity is the expected payout over finishing orders.

Three implementations, all exact (tests/test_icm.py checks them against each other and brute force for n <= 6):
- icm_equities(stacks, payouts): every seat at once, bitmask DP over subsets (positive terms only; numerically the
  most robust). The engine uses it to split the remaining points among survivors at the 200-hand cap.
- make_icm(payouts): returns icm(stacks, i) for one seat, ported from an earlier research script (not published)
  (exponential-race identity, coefficients precomputed once per payout vector).
- icm_brute(stacks, payouts): enumerates every finishing order (reference; fine for n <= 8).
Seats with a zero stack get 0. With k live players only payouts[0:k] are paid (places k+1.. are already decided).
Pure standard library.
"""
import itertools


def _pad(payouts, k):
    p = [float(x) for x in payouts[:k]]
    return p + [0.0] * (k - len(p))


def icm_equities(stacks, payouts):
    """Exact Malmuth-Harville equity of every seat (list, same length as stacks)."""
    n = len(stacks)
    live = [i for i in range(n) if stacks[i] > 0]
    eq = [0.0] * n
    k = len(live)
    if k == 0:
        return eq
    pays = _pad(payouts, k)
    m = k
    while m > 0 and pays[m - 1] == 0.0:   # trailing zero places add nothing
        m -= 1
    if k == 1:
        eq[live[0]] = pays[0]
        return eq
    total = float(sum(stacks[i] for i in live))
    cur = {0: (1.0, total)}   # mask of players already placed -> (probability, chips of players not placed)
    for place in range(m):
        pay = pays[place]
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


def make_icm(payouts, max_players=10):
    """Return icm(stacks, i): exact Malmuth-Harville equity of seat i for any payout vector.

    Port of an earlier research script (not published). Exponential-race identity:
    E_i = sum over subsets B of the live others of coef[m][|B|] * s_i / (s_i + s(B)), where m = number of live others
    and coef[m][b] = sum_t C(b, t) (-1)^t pays[m - b + t]. Coefficients are precomputed once."""
    C = [[1]]
    for n in range(1, max_players + 1):
        C.append([1] + [C[n - 1][t - 1] + C[n - 1][t] for t in range(1, n)] + [1])
    pays_all = [float(x) for x in payouts]
    coef = []
    for m in range(max_players):
        p = pays_all[:m + 1] + [0.0] * max(0, m + 1 - len(pays_all))
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


def icm_brute(stacks, payouts):
    """Reference: enumerate every finishing order of the live players (Harville probabilities)."""
    n = len(stacks)
    live = [i for i in range(n) if stacks[i] > 0]
    k = len(live)
    eq = [0.0] * n
    if k == 0:
        return eq
    pays = _pad(payouts, k)
    for order in itertools.permutations(live):
        p = 1.0
        rem = float(sum(stacks[i] for i in live))
        for j in order:
            p *= stacks[j] / rem
            rem -= stacks[j]
        for place, j in enumerate(order):
            eq[j] += p * pays[place]
    return eq
