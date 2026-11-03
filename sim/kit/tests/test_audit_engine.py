"""Audit (2026-11-03): independent checks of the kit engine, written separately from the engine itself.

1. Hand-built scenarios with expectations computed by hand: three all-ins + a covering caller (three side pots),
   refunds (preflop, postflop, over-shove), short all-in that does not reopen (postflop), heads-up order and the
   3 -> 2 transition, split main pot with a side pot, 3-way odd-chip split, same-hand busts with a tie.
2. An independent pot resolver: rebuild every hand's contributions from the public event stream, evaluate hands
   with a separate brute-force 5-of-7 evaluator, split layered pots, and compare with the engine's awards.
3. An independent places/points recomputation from the hand log (bust order by starting stack, ties split, ICM by
   brute-force permutation at the cap) for many random tables, payouts of several lengths and table sizes 2-10.
"""
import itertools
import random

import pytest

from helpers import ALLIN, C, F, R, X, events_of, scripted_table
from sim.kit.engine import Table
from sim.kit.fixtures import AllInAgent, CheckCallAgent, CheckFoldAgent, PassiveRandomAgent, RandomAgent
from sim.kit.icm import icm_brute
from sim.kit.payouts import get_payouts

BOARD_DRY = ["2c", "5d", "8h", "9s", "3c"]


# ---------------------------------------------------------------- 1. hand-built scenarios
def test_three_all_ins_and_a_covering_caller():
    # button 3: SB 0 (300), BB 1 (600), UTG 2 (1000), button 3 (2000). UTG shoves, button calls, SB and BB call
    # all-in for less. Contributions 300 / 600 / 1000 / 1000: main 1200, side-1 900, side-2 800.
    holes = {0: ("As", "Ad"), 1: ("Ks", "Kd"), 2: ("Qs", "Qd"), 3: ("Jh", "Jc")}
    res, logs = scripted_table([300, 600, 1000, 2000], {2: [ALLIN], 3: [C], 0: [C], 1: [C]}, button=3,
                               holes=holes, board=BOARD_DRY)
    assert res.final_stacks == [1200, 900, 800, 1000]
    assert events_of(res, "refund") == []
    assert logs[0][0]["to_call"] == 290 and logs[0][0]["legal_actions"] == ["fold", "call"]


def test_three_all_ins_covering_caller_wins_all_and_bust_places():
    # same action, the covering caller holds the best hand: three players bust in one hand, ordered by starting
    # stack (300 -> 4th, 600 -> 3rd, 1000 -> 2nd); a 4-seat table pays (50, 30, 20, 0)
    holes = {0: ("Jh", "Jc"), 1: ("Qs", "Qd"), 2: ("Ks", "Kd"), 3: ("As", "Ad")}
    res, _ = scripted_table([300, 600, 1000, 2000], {2: [ALLIN], 3: [C], 0: [C], 1: [C]}, button=3,
                            holes=holes, board=BOARD_DRY)
    assert res.final_stacks == [0, 0, 0, 3900]
    assert res.places == [4, 3, 2, 1] and res.points == [0.0, 20.0, 30.0, 50.0]
    assert res.bust_hand == [1, 1, 1, None]


def test_same_hand_busts_with_a_tie_and_a_shorter_stack():
    # starting stacks 500 / 500 / 300 bust in one hand: the 300 takes 4th, the two 500s share 2nd and 3rd
    holes = {0: ("Jh", "Jc"), 1: ("Qs", "Qd"), 2: ("Ks", "Kd"), 3: ("As", "Ad")}
    # UTG (300) shoves, the button (2000) re-shoves, both blinds (500 each) call all-in; the button's 1500
    # above the 500s comes back
    res, _ = scripted_table([500, 500, 300, 2000], {2: [ALLIN], 3: [ALLIN], 0: [C], 1: [C]}, button=3,
                            holes=holes, board=BOARD_DRY)
    assert [(e["seat"], e["amount"]) for e in events_of(res, "refund")] == [(3, 1500)]
    assert res.final_stacks == [0, 0, 0, 3300]
    assert res.places == [2.5, 2.5, 4, 1] and res.points == [25.0, 25.0, 0.0, 50.0]


def test_postflop_raise_fold_refund():
    # heads-up, button 0 = SB. Preflop SB calls, BB checks. Flop: BB bets 100, SB raises to 500, BB folds:
    # SB's uncalled 400 comes back; SB wins the 40 + 100 + 100.
    res, _ = scripted_table([2000, 2000], {0: [C, R(500)], 1: [X, R(100), F]}, button=0)
    assert [(e["seat"], e["street"], e["amount"]) for e in events_of(res, "refund")] == [(0, "flop", 400)]
    assert res.final_stacks == [2120, 1880]


def test_over_shove_called_all_in_for_less_refund():
    # 3 seats, button 0, SB 1, BB 2. Button shoves 2000, SB calls all-in for 600 (AA), BB folds.
    holes = {0: ("Ks", "Kd"), 1: ("As", "Ad"), 2: ("7h", "Td")}
    res, _ = scripted_table([2000, 600, 2000], {0: [ALLIN], 1: [C], 2: [F]}, button=0, holes=holes,
                            board=BOARD_DRY)
    assert [(e["seat"], e["amount"]) for e in events_of(res, "refund")] == [(0, 1400)]
    assert res.final_stacks == [1400, 1220, 1980]     # SB wins 600 + 600 + the BB's dead 20


def test_postflop_short_all_in_does_not_reopen_but_full_one_does():
    # 3 seats, button 2: SB 0, BB 1. Everyone limps preflop. Flop order 0, 1, 2.
    # Seat 0 bets 100; seat 1 (130 behind after the 20 blind... it has 150 total) shoves 150 = +50 short raise;
    # seat 2 (not yet acted) may raise, min 150 + 100 = 250; it calls. Seat 0 may only fold or call 50.
    res, logs = scripted_table([2000, 170, 2000], {2: [C, C], 0: [C, R(100), C], 1: [X, ALLIN]}, button=2)
    s2 = [s for s in logs[2] if s["street"] == "flop"][0]
    assert s2["legal_actions"] == ["fold", "call", "raise"] and s2["min_raise_to"] == 250 and s2["to_call"] == 150
    s0 = [s for s in logs[0] if s["street"] == "flop"][1]
    assert s0["legal_actions"] == ["fold", "call"] and s0["to_call"] == 50
    # full-raise version: seat 1 has 220 total, shoves 200 on the flop (+100 = a full raise): seat 0 may raise
    res, logs = scripted_table([2000, 220, 2000], {2: [C, C], 0: [C, R(100), C], 1: [X, ALLIN]}, button=2)
    s0 = [s for s in logs[0] if s["street"] == "flop"][1]
    assert s0["legal_actions"] == ["fold", "call", "raise"] and s0["min_raise_to"] == 300


def test_heads_up_after_a_bust_button_is_small_blind():
    # 3 seats, button 0, SB 1, BB 2. Hand 1: seat 0 shoves, seat 1 calls all-in with less and loses, seat 2 folds.
    # Hand 2 (heads-up): the button moves to the next seat with chips (2); it posts the SB and acts first preflop.
    holes = {0: ("As", "Ad"), 1: ("Ks", "Kd"), 2: ("7h", "Td")}
    res, logs = scripted_table([2000, 500, 2000], {0: [ALLIN, X], 1: [C], 2: [F, C, X]}, button=0, holes=holes,
                               board=BOARD_DRY, max_hands=2)
    hs = events_of(res, "hand_start")
    assert hs[1]["button"] == 2 and hs[1]["sb_seat"] == 2 and hs[1]["bb_seat"] == 0
    acts = [(e["hand_no"], e["seat"], e["street"]) for e in events_of(res, "action") if e["hand_no"] == 2]
    assert acts[0] == (2, 2, "preflop") and acts[1] == (2, 0, "preflop")
    assert acts[2] == (2, 0, "flop")                    # BB acts first postflop heads-up


def test_split_main_pot_and_side_pot():
    # seat 0 (300, short) and seat 1 tie for the main pot with a broadway straight; seat 1 also beats seat 2 for the
    # side pot. Button 2: SB 0, BB 1. Button shoves, SB calls all-in for less, BB calls.
    board = ["As", "Ks", "Qd", "Jh", "2c"]
    holes = {0: ("Tc", "3d"), 1: ("Td", "4h"), 2: ("9c", "9d")}
    res, _ = scripted_table([300, 2000, 2000], {2: [ALLIN], 0: [C], 1: [C]}, button=2, holes=holes, board=board)
    # main 900 split 450 / 450; side (2000 - 300) x 2 = 3400 to seat 1
    assert res.final_stacks == [450, 3850, 0]


def test_three_way_split_odd_chip():
    # 4 seats, ante 1, button 3: UTG 2 calls, button folds, SB completes, BB checks; checked down on a royal board.
    # pot 4 + 60 = 64 = 3 x 21 + 1: the odd chip goes to seat 0 (first winner left of the button)
    board = ["As", "Ks", "Qs", "Js", "Ts"]
    holes = {0: ("2c", "3d"), 1: ("4c", "5d"), 2: ("6c", "7d"), 3: ("8c", "9d")}
    res, _ = scripted_table([2000] * 4, {2: [C, X, X, X], 3: [F], 0: [C, X, X, X], 1: [X, X, X, X]}, button=3,
                            holes=holes, board=board, schedule=((10, 20, 1),))
    assert res.final_stacks == [2001, 2000, 2000, 1999]


def test_bb_short_of_small_blind_sb_fold_refund():
    # heads-up, BB has 5 chips (posts 5 all-in), SB posts 10 and folds: the 5 the BB cannot cover comes back
    res, logs = scripted_table([2000, 5], {0: [F]}, button=0)
    s = logs[0][0]
    assert s["to_call"] == 10 and s["legal_actions"] == ["fold", "call"]
    assert res.final_stacks == [1995, 10]


# ---------------------------------------------------------------- 2. independent pot resolver
_RANKS = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def _rank5(cards):
    rs = sorted((_RANKS[c[0]] for c in cards), reverse=True)
    suits = [c[1] for c in cards]
    flush = len(set(suits)) == 1
    uniq = sorted(set(rs), reverse=True)
    straight_hi = 0
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_hi = uniq[0]
        elif uniq == [14, 5, 4, 3, 2]:
            straight_hi = 5
    counts = sorted(((rs.count(r), r) for r in uniq), reverse=True)
    shape = [c for c, _ in counts]
    order = [r for _, r in counts]
    if straight_hi and flush:
        return (8, straight_hi)
    if shape == [4, 1]:
        return (7, *order)
    if shape == [3, 2]:
        return (6, *order)
    if flush:
        return (5, *rs)
    if straight_hi:
        return (4, straight_hi)
    if shape == [3, 1, 1]:
        return (3, *order)
    if shape == [2, 2, 1]:
        return (2, *order)
    if shape == [2, 1, 1, 1]:
        return (1, *order)
    return (0, *rs)


def best7(cards):
    return max(_rank5(c) for c in itertools.combinations(cards, 5))


def resolve_hand(evs, n):
    """Rebuild one hand from its public events; return the awards an independent resolver gives."""
    hs = evs[0]
    contrib = [0] * n
    street_bet = [0] * n
    folded = [not a for a in hs["alive"]]
    for seat, tag, amt in hs["posts"]:
        if tag == "ante":
            contrib[seat] += amt
        else:
            street_bet[seat] = amt
    shown = None
    board = None
    won = None
    button = hs["button"]
    for ev in evs[1:]:
        t = ev["type"]
        if t == "street":
            for i in range(n):
                contrib[i] += street_bet[i]
                street_bet[i] = 0
        elif t == "action":
            if ev["action"] == "fold":
                folded[ev["seat"]] = True
            elif ev["action"] in ("call", "raise"):
                street_bet[ev["seat"]] = ev["amount"]
        elif t == "refund":
            street_bet[ev["seat"]] -= ev["amount"]
        elif t == "showdown":
            shown = dict(ev["shown"])
            board = list(ev["board"])
        elif t == "hand_end":
            won = dict(ev["won"])
    for i in range(n):
        contrib[i] += street_bet[i]
    pot = sum(contrib)
    contenders = [i for i in range(n) if not folded[i]]
    expect = {}
    if len(contenders) == 1:
        expect[contenders[0]] = pot
    else:
        assert shown is not None and sorted(shown) == sorted(contenders)
        score = {i: best7(list(shown[i]) + board) for i in contenders}
        # clockwise from the left of the button
        clock = sorted(contenders, key=lambda i: (i - button - 1) % n)
        prev = 0
        for lvl in sorted(set(c for c in contrib if c > 0)):
            sl = sum(min(c, lvl) - prev for c in contrib if c > prev)
            elig = [i for i in clock if contrib[i] >= lvl]
            assert elig, "a pot layer with no eligible contender"
            best = max(score[i] for i in elig)
            winners = [i for i in elig if score[i] == best]
            share, rem = divmod(sl, len(winners))
            for k, i in enumerate(winners):
                expect[i] = expect.get(i, 0) + share + (1 if k < rem else 0)
            prev = lvl
    return {k: v for k, v in expect.items() if v}, won, pot


class ShortShover(RandomAgent):
    """Random, but shoves 40% of the time: many all-ins for less, side pots, refunds."""

    def act(self, state):
        if self.rng.random() < 0.4:
            return ALLIN(state)
        return super().act(state)


@pytest.mark.parametrize("block", range(4))
def test_independent_pot_resolver_matches_engine(block):
    rng = random.Random(f"audit-pots-{block}")
    hands = showdowns = multi_layer = splits = 0
    for t in range(150):
        n = rng.randint(2, 9)
        pool = [ShortShover, ShortShover, RandomAgent, AllInAgent, CheckCallAgent, PassiveRandomAgent]
        facs = [rng.choice(pool) for _ in range(n)]
        stacks = [rng.choice([1, 3, 15, 40, 77, 300, 999, 2000, 2000, 3500]) for _ in range(n)]
        if sum(1 for s in stacks if s > 0) < 2:
            continue
        sched = rng.choice([((10, 20, 0),), ((7, 13, 0), (11, 23, 3)), ((10, 20, 5),), ((1, 2, 0),),
                            ((50, 100, 25),), ((15, 30, 0), (30, 60, 0))])
        tb = Table(facs, "TOP3", sched, seed=f"pots{block}:{t}", stacks=stacks, max_hands=rng.randint(3, 60),
                   hands_per_level=rng.choice([1, 5, 25]), time_limit_ms=None, record_events=True)
        res = tb.play()
        cur = []
        for only, ev in res.events:
            if only is not None:
                continue
            if ev["type"] == "hand_start":
                cur = [ev]
            elif cur:
                cur.append(ev)
                if ev["type"] == "hand_end":
                    expect, won, pot = resolve_hand(cur, n)
                    assert expect == won, (t, cur[0]["hand_no"], expect, won)
                    assert sum(won.values()) == pot
                    hands += 1
                    if any(e["type"] == "showdown" for e in cur):
                        showdowns += 1
                    if len(won) > 1:
                        splits += 1
                    cur = []
    assert hands > 1000 and showdowns > 300 and splits > 20


# ---------------------------------------------------------------- 3. places and points, recomputed
def icm_recursive(stacks, pays):
    """Malmuth-Harville by the textbook recursion over who finishes next (memoized on the remaining set)."""
    live = [i for i in range(len(stacks)) if stacks[i] > 0]
    memo = {}

    def rec(rem):
        if rem in memo:
            return memo[rem]
        place = len(live) - len(rem)
        out = {i: 0.0 for i in rem}
        if place < len(pays) and rem:
            tot = float(sum(stacks[i] for i in rem))
            for j in rem:
                p = stacks[j] / tot
                out[j] += p * pays[place]
                sub = rec(rem - {j})
                for i, v in sub.items():
                    out[i] += p * v
        memo[rem] = out
        return out

    eq = rec(frozenset(live))
    return [eq.get(i, 0.0) for i in range(len(stacks))]


def recompute_places_points(hand_log, final_stacks, pays, n):
    places = [None] * n
    points = [0.0] * n
    for (_h, _sb, _bb, _ante, _b, before, after) in hand_log:
        alive = sum(1 for s in before if s > 0)
        busted = sorted((i for i in range(n) if before[i] > 0 and after[i] == 0), key=lambda i: before[i])
        place = alive
        k = 0
        while k < len(busted):
            grp = [i for i in busted if before[i] == before[busted[k]]]
            g = len(grp)
            pts = sum(pays[q - 1] for q in range(place - g + 1, place + 1)) / g
            for i in grp:
                places[i] = (2 * place - g + 1) / 2.0
                points[i] = pts
            place -= g
            k += g
    surv = [i for i in range(n) if final_stacks[i] > 0]
    eq = icm_recursive([final_stacks[i] for i in surv], pays[:len(surv)])
    if len(surv) <= 6:
        eb = icm_brute([final_stacks[i] for i in surv], pays[:len(surv)])
        assert all(abs(a - b) < 1e-9 for a, b in zip(eq, eb))
    for j, i in enumerate(surv):
        points[i] = eq[j]
        places[i] = sum(1 for x in surv if final_stacks[x] > final_stacks[i]) + \
            (sum(1 for x in surv if final_stacks[x] == final_stacks[i]) + 1) / 2.0
    return places, points


@pytest.mark.parametrize("pay", ["TOP3", "LINEAR", "WTA", "TOP4", (60, 40), (1000, 500, 250), (7,),
                                 (30, 25, 20, 15, 10), (3, 3, 2, 2, 1, 1, 0, 0, 0, 0)])
def test_places_points_icm_recomputed(pay):
    rng = random.Random(f"audit-places-{pay}")
    checked = capped = ties = 0
    for t in range(60):
        n = rng.randint(2, 10)
        if not isinstance(pay, str) and len(pay) > n and any(pay[n:]):
            continue
        facs = [rng.choice([ShortShover, RandomAgent, AllInAgent, CheckFoldAgent, PassiveRandomAgent])
                for _ in range(n)]
        stacks = [rng.choice([20, 40, 500, 500, 2000]) for _ in range(n)]
        res = Table(facs, pay, "TURBO", seed=f"pl{pay}:{t}", stacks=stacks, max_hands=rng.choice([5, 20, 80, 200]),
                    time_limit_ms=None, record_hands=True).play()
        pays = list(get_payouts(pay, n, truncate=n < 8))
        places, points = recompute_places_points(res.hand_log, res.final_stacks, pays, n)
        assert [float(p) for p in res.places] == [float(p) for p in places], (t, res.places, places)
        assert all(abs(a - b) < 1e-9 for a, b in zip(res.points, points)), (t, res.points, points)
        assert abs(sum(res.points) - sum(pays)) < 1e-9
        assert abs(sum(res.places) - n * (n + 1) / 2) < 1e-9
        checked += 1
        capped += res.survivors > 1
        ties += any(isinstance(p, float) and p != int(p) for p in res.places)
    assert checked >= 25 and capped >= 5


def test_folded_antes_above_every_live_contribution_are_returned():
    # Audit bug (fixed in engine.py): 3 seats, ante 25, blinds 50/100. Seats 0 and 2 have 1 chip (all-in on the
    # ante); seat 1 antes 25, posts the SB and folds. Before the fix the 24 chips of seat 1's ante that nobody live
    # matched made a pot layer with no eligible player and the engine raised EngineError.
    holes = {0: ("As", "Ad"), 1: ("Ks", "Kd"), 2: ("7h", "Td")}
    res, _ = scripted_table([1, 300, 1], {1: [F]}, button=0, holes=holes, board=BOARD_DRY,
                            schedule=((50, 100, 25),))
    refunds = [(e["seat"], e["amount"]) for e in events_of(res, "refund")]
    assert refunds == [(1, 50), (1, 24)]
    assert res.final_stacks == [3, 299, 0]


# ---------------------------------------------------------------- strictness (rules s3.7), auditor's own agents
class _Bad:
    """Wants to raise all-in on every decision but errs: 'sleep' (2 ms), 'exc' (raises), 'illegal' (raise to
    max + 1, or a raise when raising is not legal). The engine must fold facing a bet and check when free, and
    never apply the raise."""

    def __init__(self, seat, config, mode):
        self.mode = mode

    def act(self, state):
        import time as _t
        if self.mode == "sleep":
            _t.sleep(0.002)
            if "raise" in state["legal_actions"]:
                return {"action": "raise", "amount": state["max_raise_to"]}
            return {"action": "call"} if "call" in state["legal_actions"] else {"action": "check"}
        if self.mode == "exc":
            raise ZeroDivisionError("audit")
        return {"action": "raise", "amount": (state["max_raise_to"] or 10 ** 9) + 1}


@pytest.mark.parametrize("mode,counter", [("sleep", "timeouts"), ("exc", "exceptions"), ("illegal", "illegal")])
def test_strict_errors_fold_or_check_and_never_apply(mode, counter):
    import functools
    facs = [CheckCallAgent] * 4
    facs[1] = functools.partial(_Bad, mode=mode)
    res = Table(facs, "TOP3", "TURBO", seed=f"audit-strict-{mode}", max_hands=40, record_events=True).play()
    st = res.stats[1]
    acts = [ev for only, ev in res.events if ev["type"] == "action" and ev["seat"] == 1]
    assert acts and st.decisions == len(acts)
    assert getattr(st, counter) == st.decisions
    free = [ev for ev in acts if ev["to_call"] == 0]
    facing = [ev for ev in acts if ev["to_call"] > 0]
    assert free and facing
    assert all(ev["action"] == "check" for ev in free)
    assert all(ev["action"] == "fold" for ev in facing)
    assert st.n_raise == 0 and st.n_call == 0
