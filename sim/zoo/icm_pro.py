"""ICM-aware jam/fold pro (ZOO #12, archetype A9, 6% of T1 and 13% of T6 seats in the book; 15-30% / 25-45% after the
realism review, refute_field_realism notes section 1.3).

Stands for: the competent team that uses the starter kit's ICM calculator and the standard sit-and-go recipe
(llm_field notes A9; sng_icm notes). Logic ported from an earlier research script (not published) (the tested
brain) without the per-seat typing and cap tricks: with an effective stack of 20 BB or less it jams first in with the
fitted chip-EV Nash width for the number of players behind, scaled by exact bubble factors (ICM loss/gain of the jam),
and calls jams with the fitted Nash calling width divided by BF^2.6. Above 20 BB it opens the top 12-45% (by players
behind) to 2.5 BB, 3-bets or jams the top 3%, and plays postflop by made-hand class (value bets 0.6 pot, calls with
top pair, bubble-factor-aware vs big bets). Jam and call orderings come from sim/zoo/data/orders.json (Nash-style,
169 classes). Bubble factors are cached per (stacks, payouts) so act() stays well under 1 ms.
"""
try:
    from sim.zoo._common import (ZooAgent, all_in, call, check, fold, PUSH_PCT, CALL_PCT, label, hole_ints, board_ints,
                                 post_class, preflop_behind, top_bet, raise_amt, check_else, fold_or_check,
                                 call_or_check)
except ImportError:
    from _common import (ZooAgent, all_in, call, check, fold, PUSH_PCT, CALL_PCT, label, hole_ints, board_ints,
                         post_class, preflop_behind, top_bet, raise_amt, check_else, fold_or_check, call_or_check)

JAM_A, JAM_BS, JAM_BK = 3.29, -0.717, -0.917      # chip-EV Nash first-in jam fit (sng_brain.py)
SB_A, SB_B = 1.41, -0.410
CALL_A, CALL_BJ, CALL_BS = 2.64, 0.971, -0.582
CALL_LAM = 2.6                                      # call% /= BF ** 2.6
JAM_OPP, JAM_HERO = 2.0, -1.5                       # jam% *= BF_callers ** 2.0 * BF_hero ** -1.5


def chip_jam(S, k):
    S = max(S, 1.0)
    if k <= 1:
        return min(1.0, SB_A * S ** SB_B)
    return min(1.0, JAM_A * S ** JAM_BS * k ** JAM_BK)


def chip_call(J, S):
    return min(1.0, CALL_A * max(J, 0.01) ** CALL_BJ * max(S, 1.0) ** CALL_BS)


def icm_hero(stacks, hero, pays):
    """Malmuth-Harville equity of one seat; O(n^2)-O(n^3) with the first three places exact and a linear tail."""
    live = [s for s in stacks if s > 0]
    k = len(live)
    if k == 0 or stacks[hero] <= 0:
        return 0.0
    p = list(pays[:k]) + [0.0] * max(0, k - len(pays))
    if k == 1:
        return p[0]
    step = p[k - 2] - p[k - 1]
    base = p[k - 1]
    si = stacks[hero]
    acc = 0.0
    for sj in live:
        acc += si / (si + sj)
    lin = base + step * (acc - 0.5)
    c1 = p[0] - (base + step * (k - 1))
    c2 = (p[1] - (base + step * (k - 2))) if k > 1 else 0.0
    c3 = (p[2] - (base + step * (k - 3))) if k > 2 else 0.0
    if c1 == 0 and c2 == 0 and c3 == 0:
        return lin
    T = sum(live)
    e1 = si / T
    e2 = e3 = 0.0
    n = len(stacks)
    for j in range(n):
        sj = stacks[j]
        if j == hero or sj <= 0:
            continue
        pj = sj / T
        Tj = T - sj
        e2 += pj * si / Tj
        if c3:
            for m in range(n):
                sm = stacks[m]
                if m == hero or m == j or sm <= 0:
                    continue
                e3 += pj * (sm / Tj) * si / (Tj - sm)
    return lin + c1 * e1 + c2 * e2 + c3 * e3


def bubble_factor(stacks, hero, vill, pays_all):
    """ICM loss / ICM gain of hero risking min(stacks) against vill (1.0 = chip EV)."""
    a = min(stacks[hero], stacks[vill])
    if a <= 0:
        return 1.0
    k = sum(1 for s in stacks if s > 0)
    pays = pays_all[:k]
    base = icm_hero(stacks, hero, pays)
    w = list(stacks); w[hero] += a; w[vill] -= a
    lo = list(stacks); lo[hero] -= a; lo[vill] += a
    kw = sum(1 for s in w if s > 0)
    gain = icm_hero(w, hero, pays_all[:kw]) - base
    el = pays_all[k - 1] if lo[hero] <= 0 and k - 1 < len(pays_all) else (0.0 if lo[hero] <= 0 else icm_hero(lo, hero, pays))
    loss = base - el
    if gain <= 1e-12:
        return 4.0
    return max(0.5, min(6.0, loss / gain))


class IcmPro(ZooAgent):
    PARAMS = {"short_bb": 20.0, "open_bb": 2.5, "call_lam": CALL_LAM, "jam_scale": 1.0, "call_scale": 1.0,
              "open_scale": 1.0}
    JITTER = {"short_bb": (15.0, 25.0), "call_lam": (1.8, 3.0), "jam_scale": (0.8, 1.2), "call_scale": (0.8, 1.2),
              "open_scale": (0.8, 1.25)}

    def __init__(self, seat, config, **params):
        super().__init__(seat, config, **params)
        self._bf = {}

    def bf(self, tot, hero, vill, pays):
        key = (tuple(tot), hero, vill)
        v = self._bf.get(key)
        if v is None:
            if len(self._bf) > 400:
                self._bf.clear()
            v = self._bf[key] = bubble_factor(list(tot), hero, vill, pays)
        return v

    def act(self, state):
        if self.new_hand(state):
            self.hand_state["pf_raiser"] = False
        return self.preflop(state) if state["street"] == "preflop" else self.postflop(state)

    def preflop(self, state):
        p, legal, bb, me, n = self.p, state["legal_actions"], state["bb"], state["seat"], state["n_seats"]
        lab = label(state["hole"])
        pp, cp = PUSH_PCT[lab], CALL_PCT[lab]
        tot = [state["stacks"][j] + state["bets"][j] for j in range(n)]
        opp = [j for j in range(n) if j != me and state["active"][j] and tot[j] > 0]
        if not opp:
            return check_else(state, call())
        S = min(tot[me], max(tot[j] for j in opp)) / bb
        pays = list(state["payouts"])
        top = max(state["bets"])
        if top <= bb:                                              # unopened (blinds only)
            behind = preflop_behind(state)
            k = len(behind)
            if k == 0:
                return check_else(state, call())
            if S <= p["short_bb"]:
                J = chip_jam(S, k)
                use = behind[:3]                                   # bound the cost: at most 3 bubble-factor pairs
                bfo = sum(self.bf(tot, j, me, pays) for j in use) / len(use)
                bfh = sum(self.bf(tot, me, j, pays) for j in use) / len(use)
                J = min(1.0, J * bfo ** JAM_OPP * bfh ** JAM_HERO * p["jam_scale"])
                return all_in(state) if pp <= J else fold_or_check(state)
            op = {7: 0.12, 6: 0.14, 5: 0.17, 4: 0.20, 3: 0.26, 2: 0.36, 1: 0.45}.get(k, 0.12) * p["open_scale"]
            if pp <= op and "raise" in legal:
                self.hand_state["pf_raiser"] = True
                return raise_amt(state, int(p["open_bb"] * bb))
            return fold_or_check(state)
        raiser = max((j for j in opp), key=lambda j: state["bets"][j])
        commit = state["to_call"] >= 0.4 * tot[me] or state["all_in"][raiser] or S <= p["short_bb"]
        if commit:
            Sr = tot[raiser] / bb
            order = [(state["bb_seat"] + 1 + i) % n for i in range(n)]
            kr = max(1, sum(1 for j in order[order.index(raiser) + 1:] if state["active"][j]))
            Jr = chip_jam(Sr, kr)
            C = chip_call(Jr, S) * p["call_scale"] / self.bf(tot, me, raiser, pays) ** p["call_lam"]
            if state["all_in"][raiser]:
                return call() if cp <= C else fold()
            return all_in(state) if cp <= 1.2 * C else fold()
        if pp <= 0.03:
            return all_in(state) if S <= 40 else raise_amt(state, 3 * top)
        if pp <= 0.10 and state["to_call"] <= 0.1 * tot[me]:
            return call()
        return fold()

    def postflop(self, state):
        p, legal, me, n = self.p, state["legal_actions"], state["seat"], state["n_seats"]
        pc = post_class(hole_ints(state), board_ints(state))
        pot, to_call = state["pot"], state["to_call"]
        opp = [j for j in range(n) if j != me and state["active"][j]]
        if to_call == 0:
            if pc >= 2 and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(0.6 * pot))
            heads_up = len(opp) == 1
            if self.hand_state.get("pf_raiser") and state["street"] == "flop" and heads_up and \
                    self.rng.random() < 0.5 and "raise" in legal:
                return raise_amt(state, top_bet(state) + int(0.5 * pot))
            return check()
        big = to_call >= 0.4 * state["stacks"][me]
        bfv = 1.0
        if big and opp:
            tot = [state["stacks"][j] + state["bets"][j] for j in range(n)]
            raiser = max(opp, key=lambda j: state["bets"][j])
            bfv = self.bf(tot, me, raiser, list(state["payouts"]))
        if pc == 3:
            return raise_amt(state, top_bet(state) * 3) if (not big or bfv < 2.0) and "raise" in legal else call()
        if pc == 2 and to_call <= pot and (not big or bfv < 1.3):
            return call()
        if pc == 1 and to_call <= 0.35 * pot and not big:
            return call()
        return fold()
